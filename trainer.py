import numpy as np
import time
import torch
import os

from tqdm import tqdm
import wandb
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryF1Score
from torchmetrics.classification import BinaryAUROC, BinaryConfusionMatrix, BinaryROC


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

        self.metrics = MetricCollection([BinaryAccuracy(), BinaryRecall(), BinaryPrecision(),
                                         BinaryF1Score(), BinaryAUROC()])
        self.ys = []
        self.y_preds = []
        self.losses = []
        self.best_auroc = 0

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

        # save the model's weights if train is valid and BinaryAUROC is higher than previous
        if train == 'valid' and metrics['BinaryAUROC'] > self.best_auroc:
            print(f'valid BinaryAUROC is improved, saving the model, epoch {self.global_epoch}')
            torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, 'model.pth'))

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
        self.best_auroc = 0

    def train_epoch(self, data_loader):
        # set model to train mode
        self.model.train()

        for i, (X, y) in enumerate(tqdm(data_loader)):

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

        for i, (X, y) in enumerate(tqdm(data_loader)):
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

            self.global_epoch += 1

        # print training time
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def evaluate(self, data_loaders):
        for el in data_loaders:
            print(el)

            self.validate_epoch(data_loaders[el])
            self.log_metrics(train=el)
            self.reset_metrics()
