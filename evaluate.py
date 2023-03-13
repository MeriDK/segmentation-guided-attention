import wandb
import os

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM

from utils import build_model, get_transformations, setup_criterion
from dataset import KidneyDataset
from trainer import Trainer
from loss import AttentionLoss
import random
import numpy as np


# set up random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)


def run_evaluation(run_path):

    # set up augmentation
    data_transforms = get_transformations()

    # update data_transforms because it's saved in the wrong format
    wandb.config['data_transforms']['train'] = data_transforms['train']
    wandb.config['data_transforms']['valid'] = data_transforms['valid']
    wandb.config['data_path'] = '/home/mrizhko/hn_miccai/data/'

    # restore model
    model_path = wandb.restore('model.pth', run_path=run_path)

    # setup model
    model = build_model(wandb.config)

    model.load_state_dict(torch.load(model_path.name))
    model = model.to(wandb.config['device'])

    # create train and valid datasets
    train_dataset = KidneyDataset('SickKids_train_seg.csv', wandb.config['data_path'], data_transforms['valid'])
    valid_dataset = KidneyDataset('SickKids_valid_seg.csv', wandb.config['data_path'], data_transforms['valid'])

    # create other datasets
    test = KidneyDataset('SickKids_test_seg.csv', wandb.config['data_path'], data_transforms['valid'])
    stanford = KidneyDataset('Stanford_seg.csv', wandb.config['data_path'], data_transforms['valid'])
    chop = KidneyDataset('CHOP_seg.csv', wandb.config['data_path'], data_transforms['valid'])
    ui = KidneyDataset('UI_seg.csv', wandb.config['data_path'], data_transforms['valid'])
    silent = KidneyDataset('Prospective_seg.csv', wandb.config['data_path'], data_transforms['valid'])
    prenatal = KidneyDataset('Prenatal_seg.csv', wandb.config['data_path'], data_transforms['valid'])

    # pass datasets to pytorch loaders
    train_loader = DataLoader(train_dataset, batch_size=wandb.config['batch_size'])
    valid_loader = DataLoader(valid_dataset, batch_size=wandb.config['batch_size'])

    # pass other datasets to pytorch loaders
    test_loader = DataLoader(test, batch_size=wandb.config['batch_size'])
    stanford_loader = DataLoader(stanford, batch_size=wandb.config['batch_size'])
    chop_loader = DataLoader(chop, batch_size=wandb.config['batch_size'])
    ui_loader = DataLoader(ui, batch_size=wandb.config['batch_size'])
    silent_loader = DataLoader(silent, batch_size=wandb.config['batch_size'])
    prenatal_loader = DataLoader(prenatal, batch_size=wandb.config['batch_size'])

    # set up loss function
    criterion = setup_criterion(wandb.config, model)

    # create and run trainer
    Trainer(config=wandb.config, model=model, criterion=criterion, optimizer=None, scheduler=None).evaluate({
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader,
        'silent': silent_loader,
        'stanford': stanford_loader,
        'chop': chop_loader,
        'ui': ui_loader,
        'prenatal': prenatal_loader
    })


if __name__ == '__main__':

    run_path = 'hn_miccai_final_eval/runs/pq9gvq01'
    print(f'Evaluating run {run_path}')

    # restore wandb config
    config_path = wandb.restore('config.yaml', run_path=run_path)

    # init wandb using config above
    wandb.init(project='hn_miccai_final_eval', entity='meridk', config=config_path.name)

    # delete config file
    os.remove(config_path.name)

    # evaluate trained model on other datasets
    run_evaluation(run_path=run_path)

    # explicitly end wandb
    wandb.finish()
