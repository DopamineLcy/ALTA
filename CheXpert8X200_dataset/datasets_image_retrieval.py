import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import PIL
import torchvision.transforms as transforms


chexpert_root = 'path_to_CheXpert'

root = 'CheXpert8X200_dataset/image-retrieval'
candidate_file_path = os.path.join(root, 'candidate.csv')
query_file_path = os.path.join(root, 'query.csv')


def build_transform(reshape_size, crop_size, mean=None, std=None):
    t = []
    t.append(
        transforms.Resize(reshape_size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(crop_size))
    t.append(transforms.Grayscale(num_output_channels=3))
    t.append(transforms.ToTensor())
    if mean is not None:
        t.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(t)


class QueryRetrievalDataset(Dataset):
    def __init__(self, query_file, reshape_size, crop_size, mean=None, std=None):
        self.query_data = pd.read_csv(query_file)
        self.transform = build_transform(reshape_size, crop_size, mean, std)

    def __len__(self):
        return len(self.query_data)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img

    def __getitem__(self, index):
        query_row = self.query_data.iloc[index]
        path = os.path.join(chexpert_root,query_row['Path'])
        label = query_row['Variable']
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class CandidateRetrievalDataset(Dataset):
    def __init__(self, csv_file, reshape_size, crop_size, mean=None, std=None):
        self.data = pd.read_csv(csv_file)
        self.transform = build_transform(reshape_size, crop_size, mean, std)

    def __len__(self):
        return len(self.data)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img

    def __getitem__(self, idx):
        img_path = os.path.join(chexpert_root, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(self.data.iloc[idx, 5:].values.astype('float32'), dtype=torch.float32)

        return image, labels
