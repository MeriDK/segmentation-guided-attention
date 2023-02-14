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
        # get surgery and path info
        surgery = self.df['surgery'][idx]
        img_path = self.df['sag'][idx]

        # read image and convert it to rgb
        img = process_image(self.data_path + img_path)

        # transform image
        img = self.transform(img)

        # return img (X) and surgery (y)
        return img, surgery


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
