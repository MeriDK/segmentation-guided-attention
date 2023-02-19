import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
import wandb


class AttentionLoss(nn.Module):
    def __init__(self, cam):
        super(AttentionLoss, self).__init__()

        self.cam = cam

    def forward(self, X, y, mask):
        # calculate predictions
        y = y.sigmoid().data.gt(0.5)

        # create attention maps
        grayscale_cam = self.cam(X, y)

        # calculate loss
        loss = torch.sum((1 - mask) * grayscale_cam) / X.shape[0] / X.shape[2] / X.shape[3]

        return loss
