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
        n, h, w = X.shape[0], X.shape[2], X.shape[3]
        loss = torch.sum(torch.where(
            torch.sum(torch.sum(mask, dim=1), dim=1) == 0,
            0,
            torch.sum(torch.sum((grayscale_cam - mask) ** 2, dim=1), dim=1)
        )) / n / h / w

        return loss
