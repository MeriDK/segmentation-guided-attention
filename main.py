import wandb
import os

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from utils import build_model, get_transformations, setup_criterion, setup_dataloader
from dataset import KidneyDataset
from trainer import Trainer
from loss import AttentionLoss
import random
import numpy as np
import os


# set up random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)


def run_experiment():

    # set up augmentation
    data_transforms = get_transformations()

    # all changeable parameters should be specified here
    wandb.config.update({
        'batch_size': 128,
        'num_epochs': 30,
        'threshold': 0.5,
        'loss': 'BCEAttention',     # 'BCE' 'AttentionLoss' 'BCEAttention'
        'model_name': 'resnet18',   # 'swin'/'vit' + '_'  + 'small', 'tiny', 'base' || 'resnet' + '18'/'50'
        'lr': 0.001,
        'weight_decay': 0.1,
        'gamma': 0.85,
        'freeze': False,
        'freeze_until': 'layer3',
        'device': 'cuda',
        'pretrained': True,
        'data_transforms': data_transforms,
        'data_path': '/home/mrizhko/hn_miccai/data/',
        'data_type': '_seg_only',    # '_seg' / '_seg_only' / '_seg_n_seg_only'
        'save_weights': False,
        'weights_path': ''  # '' / '/home/mrizhko/hn_miccai/weights/model.pth'
    })

    # setup model
    model = build_model(wandb.config)
    model = model.to(wandb.config['device'])

    # create train dataset and dataloaders
    train_loader, valid_loader = setup_dataloader(wandb.config, data_transforms)

    # set up loss function
    criterion = setup_criterion(wandb.config, model)

    # set up optimizer and scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=wandb.config['lr'],
                                 weight_decay=wandb.config['weight_decay'])
    scheduler = ExponentialLR(optimizer, gamma=wandb.config['gamma'])

    # create and run trainer
    Trainer(config=wandb.config, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler).run(
        train_loader, valid_loader
    )


if __name__ == '__main__':

    # init wandb
    wandb.init(project='hn-thesis', entity='meridk')

    # run experiment
    run_experiment()

    # save the model to wandb
    if wandb.config['save_weights']:
        wandb.save(os.path.join(wandb.run.dir, 'model.pth'))

    # explicitly end wandb
    wandb.finish()
