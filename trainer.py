import numpy as np
import time
import torch
import os
import json

from tqdm import tqdm
import wandb
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryF1Score
from torchmetrics.classification import BinaryAUROC, BinaryConfusionMatrix, BinaryROC
from dataset import process_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from PIL import Image
from utils import reshape_transform


class Trainer:
    def __init__(self, config, model, criterion, optimizer, scheduler):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = config['device']
        self.global_step = 0
        self.global_epoch = 0
        self.best_auroc = 0

        self.metrics = MetricCollection([BinaryAccuracy(), BinaryRecall(), BinaryPrecision(),
                                         BinaryF1Score(), BinaryAUROC()])
        self.ys = []
        self.y_preds = []
        self.losses = []

    def update_metrics(self, y_pred, y, loss):
        # move variables to cpu
        y_pred = y_pred.detach().cpu()
        y = y.detach().cpu()

        # save predictions, targets and losses
        self.ys.append(y)
        self.y_preds.append(y_pred)
        self.losses.append(loss)

        # update metrics
        self.metrics(y_pred, y)

    def log_metrics(self, train):
        # log loss
        wandb.log({f"{train}_loss": np.mean(self.losses), 'epoch': self.global_epoch})

        # calculate metrics over all batches
        metrics = self.metrics.compute()

        # log metrics
        for metric in ['BinaryAccuracy', 'BinaryRecall', 'BinaryPrecision', 'BinaryF1Score', 'BinaryAUROC']:
            wandb.log({f"{train}_{metric}": metrics[metric], 'epoch': self.global_epoch})

        # TODO Finish Confusion Matrix and ROC curve
        # concat array of tensors to tensor
        # y_true = torch.cat(self.ys).tolist()
        # preds = torch.cat(self.y_preds).tolist()

        # log confusion matrix
        # wandb.log({'conf_mat': wandb.plot.confusion_matrix(y_true=y_true, probs=preds,
        #                                                    class_names=['no-surgery', 'surgery']),
        #            'epoch': self.global_epoch})

        # log roc
        # wandb.log({'roc': wandb.plot.roc_curve(y_true, preds)})

    def reset_metrics(self):
        self.ys = []
        self.y_preds = []
        self.losses = []
        self.metrics.reset()

    def train_epoch(self, data_loader):
        # set model to train mode
        self.model.train()

        for i, (X, y, mask) in enumerate(tqdm(data_loader)):

            # calculate y_pred and loss
            y_pred = self.model(X.to(self.device)).reshape(-1)
            loss = self.criterion(y_pred, y.to(self.device).float())

            # model optimization
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log batch loss
            wandb.log({'batch_loss': loss.item(), 'step': self.global_step})
            self.global_step += 1

        # log learning rate
        wandb.log({'learning_rate': self.optimizer.param_groups[0]['lr'], 'epoch': self.global_epoch})

        # update learning rate
        self.scheduler.step()

    def validate_epoch(self, data_loader):

        # set model to evaluation mode
        self.model.eval()

        for i, (X, y, mask) in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                # calculate y_pred
                y_pred = self.model(X.to(self.device)).reshape(-1)

                # move y to cuda
                y = y.to(self.device)

                # calculate loss
                loss = self.criterion(y_pred, y.float())

                # add batch predictions, ground truth and loss to metrics
                self.update_metrics(y_pred.sigmoid(), y, loss.item())

    def run(self, train_loader, valid_loader):
        since = time.time()

        for epoch in range(self.config['num_epochs']):
            print('Epoch:', epoch)

            # train epoch
            self.train_epoch(train_loader)

            # validate and log on train data
            self.validate_epoch(train_loader)
            self.log_metrics(train='train')
            self.reset_metrics()

            # validate and log on valid data
            self.validate_epoch(valid_loader)
            self.log_metrics(train='valid')
            self.reset_metrics()

            # save the model's weights if BinaryAUROC is higher than previous
            if wandb.run.summary['valid_BinaryAUROC'] > self.best_auroc:
                print(f'valid BinaryAUROC is improved {round(self.best_auroc, 4)} => '
                      f'{round(wandb.run.summary["valid_BinaryAUROC"], 4)}, '
                      f'saving the model, epoch {self.global_epoch}')

                torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, 'model.pth'))
                self.best_auroc = wandb.run.summary['valid_BinaryAUROC']

            self.global_epoch += 1

        # print training time
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def evaluate(self, data_loaders):
        self.model.eval()

        with open(self.config['data_path'] + 'segmentations2.json', 'r') as f:
            data = json.loads(f.read())

        # Construct the CAM object once, and then re-use it on many images
        if wandb.config['model_name'].startswith('vit'):
            target_layers = [self.model.blocks[-1].norm1]
            cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=True,
                          reshape_transform=reshape_transform)
        else:
            target_layers = [self.model.layer4[-1]]
            cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=True)

        for el in data:
            imgs, transformed_imgs, array_imgs, csegs, ksegs, ys = [], [], [], [], [], []

            for i in range(len(data[el])):
                # open images and convert to rgb if needed
                img = process_image(self.config['data_path'] + data[el][i][0])

                # append img, cseg path, kseg path and y
                imgs.append(img)
                csegs.append(self.config['data_path'] + data[el][i][1])
                ksegs.append(self.config['data_path'] + data[el][i][2])
                ys.append(data[el][i][3])

                # create tensor of image
                transformed_img = self.config['data_transforms']['valid'](image=np.asarray(img))['image']
                transformed_imgs.append(transformed_img)

                # create np array of image
                array_img = np.array(img.resize((224, 224))) / 255
                array_imgs.append(array_img)

            # stack input tensors
            input_tensor = torch.stack(transformed_imgs)
            input_tensor = input_tensor.to(self.config['device'])

            # calculate model prediction
            y_probs = self.model(input_tensor).sigmoid()
            y_preds = y_probs.data.gt(self.config['threshold'])

            # We have to specify the target we want to generate the Class Activation Maps for.
            targets = [BinaryClassifierOutputTarget(1) if y.item() else BinaryClassifierOutputTarget(0)
                       for y in y_preds]

            # Calculate GradCAMs
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # plot images
            fig, axes = plt.subplots(nrows=len(ys), ncols=4, figsize=(12.1, 3.1 * len(ys)))

            if len(ys) > 1:
                for i in tqdm(range(len(ys))):
                    # select grayscale_cam for the image in the batch:
                    visualization = show_cam_on_image(array_imgs[i], grayscale_cam[i, :], use_rgb=True)

                    # set titles
                    axes[i, 0].set_title(f'Original {"surgery" if ys[i] == 1 else "no surgery"}')
                    axes[i, 1].set_title('Cseg')
                    axes[i, 2].set_title('Kseg')
                    axes[i, 3].set_title(f'GradCAM {round(y_probs[i].item(), 2)}')

                    # plot images
                    axes[i, 0].imshow(imgs[i])
                    axes[i, 1].imshow(Image.open(csegs[i]))
                    axes[i, 2].imshow(Image.open(ksegs[i]))
                    axes[i, 3].imshow(Image.fromarray(visualization))
            else:
                # select grayscale_cam for the image in the batch:
                visualization = show_cam_on_image(array_imgs[0], grayscale_cam[0, :], use_rgb=True)

                # set titles
                axes[0].set_title(f'Original {"surgery" if ys[0] == 1 else "no surgery"}')
                axes[1].set_title('Cseg')
                axes[2].set_title('Kseg')
                axes[3].set_title(f'GradCAM {round(y_probs[0].item(), 2)}')

                # plot images
                axes[0].imshow(imgs[0])
                axes[1].imshow(Image.open(csegs[0]))
                axes[2].imshow(Image.open(ksegs[0]))
                axes[3].imshow(Image.fromarray(visualization))

            # just for plots to look good
            [axi.set_axis_off() for axi in axes.ravel()]
            plt.subplots_adjust(wspace=0.2, hspace=0.1)

            wandb.log({f'{el}': wandb.Image(plt)})

        # calculate and plot AUROC for other datasets
        for el in data_loaders:
            print(el)
            self.validate_epoch(data_loaders[el])
            self.log_metrics(train=el)
            self.reset_metrics()
