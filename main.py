import wandb
import os

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from utils import build_model, get_transformations
from dataset import KidneyDataset
from trainer import Trainer


def run_experiment(model_name, pretrained):

    # set up random seed for reproducibility
    torch.manual_seed(42)

    # set up augmentation
    data_transforms = get_transformations()

    # all changeable parameters should be specified here
    wandb.config.update({
        'batch_size': 32,
        'num_epochs': 3,
        'threshold': 0.5,
        'loss': 'BCE',
        'model_name': model_name,   # 'swin'/'vit' + '_'  + 'small', 'tiny', 'base' || 'resnet' + '18'/'50'
        'lr': 0.000001,
        'weight_decay': 0.001,
        'gamma': 0.93,
        'freeze': False,
        'freeze_until': 'layer3',
        'device': 'cuda',
        'pretrained': pretrained,
        'data_transforms': data_transforms,
        'data_path': '/home/mrizhko/hn_miccai/data/'
    })

    # setup model
    model = build_model(wandb.config)
    model = model.to(wandb.config['device'])

    # create train and valid datasets
    train_dataset = KidneyDataset('SickKids_train_seg.csv', wandb.config['data_path'], data_transforms['train'])
    valid_dataset = KidneyDataset('SickKids_valid_seg.csv', wandb.config['data_path'], data_transforms['valid'])

    # pass datasets to pytorch loaders
    train_loader = DataLoader(train_dataset, batch_size=wandb.config['batch_size'])
    valid_loader = DataLoader(valid_dataset, batch_size=wandb.config['batch_size'])

    # set up loss function
    if wandb.config['loss'] == 'BCE':
        criterion = BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f'Unknown loss')

    # set up optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=wandb.config['lr'],
                                 weight_decay=wandb.config['weight_decay'])

    # set up scheduler
    scheduler = ExponentialLR(optimizer, gamma=wandb.config['gamma'])

    # create and run trainer
    Trainer(config=wandb.config, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler).run(
        train_loader, valid_loader
    )


if __name__ == '__main__':

    # init wandb
    wandb.init(project="test", entity="meridk")

    # 'resnet18/50/152' 'swin_small/tiny/base' 'vit_small/tiny/base'
    run_experiment(model_name='resnet18', pretrained=True)

    # save the model to wandb
    wandb.save(os.path.join(wandb.run.dir, 'model.pth'))

    # explicitly end wandb
    wandb.finish()
