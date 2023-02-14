import timm
import cv2
import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from collections import OrderedDict
from PIL import Image
import os
from torchvision import transforms, models
import matplotlib.pyplot as plt
from pathlib import Path

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataset import process_image


def build_model(config):
    # set up model name
    if config['model_name'].startswith('swin'):
        model_name = f"{config['model_name']}_patch4_window7_224"
    elif config['model_name'].startswith('vit'):
        model_name = f"{config['model_name']}_patch16_224"
    elif config['model_name'].startswith('resnet'):
        model_name = config['model_name']
    else:
        raise NotImplementedError('Unknown model')

    # create model
    model = timm.create_model(model_name, pretrained=config['pretrained'], num_classes=1)

    # freeze model
    if config['freeze']:
        submodules = [n for n, _ in model.named_children()]
        timm.freeze(model, submodules[:submodules.index(config['freeze_until']) + 1])

    return model


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)

    return result
