from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np


class KidneyDataset(Dataset):
    def __init__(self, csv_path, data_path, transform):
        self.data_path = data_path
        self.transform = transform

        # read csv data
        self.df = pd.read_csv(self.data_path + csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get surgery, path and maks info
        surgery = self.df['surgery'][idx]
        img_path = self.df['sag'][idx]
        mask_path = self.df['sag_kseg'][idx]

        # read image, convert it to rgb and convert it into np array
        img = process_image(self.data_path + img_path)
        img = np.asarray(img)

        # read mask if it exists otherwise create mask with all zeros
        if type(mask_path) is str:
            mask = process_mask(self.data_path + mask_path)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]))

        # transform image and mask
        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        # return img (X), surgery (y) and mask
        return img, surgery, mask


def process_image(img_path):
    # open image
    img = Image.open(img_path)

    # if image is grey or rgba change it to rgb
    if img.mode == 'L':
        arr = np.asarray(img)
        arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
        img = Image.fromarray(arr)
    elif img.mode == 'RGBA':
        arr = np.asarray(img)
        arr = arr[:, :, :3]
        img = Image.fromarray(arr)

    return img


def process_mask(mask_path):
    # open mask image
    img = Image.open(mask_path)

    # transform mask to numpy array
    array_img = np.asarray(img)

    # select rgb channels
    r = array_img[:, :, 0].reshape(-1)
    g = array_img[:, :, 1].reshape(-1)
    b = array_img[:, :, 2].reshape(-1)

    # check which pixels are red rgb = (236, 28, 36)
    mask = (np.where(r == 236, 1, 0) * np.where(g == 28, 1, 0) * np.where(b == 36, 1, 0))

    # reshape mask to 2d array
    mask = mask.reshape((array_img.shape[0], array_img.shape[1]))

    return mask
