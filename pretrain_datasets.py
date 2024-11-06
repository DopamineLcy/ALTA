from copy import deepcopy
import os
from typing import List, Tuple, Any
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import tokenizers
from transformers import BertConfig, BertTokenizer
import random
import re


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class CXRBertTokenizer(BertTokenizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class ALTADataset(Dataset):
    def __init__(self, data_root, is_train, args, max_caption_length: int = 128):
        self.is_train = is_train
        self.max_caption_length = max_caption_length
        self.data_root = data_root
        self.transform_big = self._build_transform("big")
        self.transform_small = self._build_transform("small")
        df = self.read_csv()
        self.images_list = df["image_path"]
        self.auxview_images_list = df["auxview_image_path"]
        self.last_images_list = df["last_image_path"]
        self.last_auxview_images_list = df["last_auxview_image_path"]
        self.reports_list = df["report"]
        self.tokenizer = CXRBertTokenizer.from_pretrained("./BiomedVLP-CXR-BERT-specialized")
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def _build_transform(self, size="big"):
        if size == "big":
            resize_size = 512
            crop_size = 448
        elif size == "small":
            resize_size = 256
            crop_size = 224
        else:
            raise NotImplementedError("transform size must be big or small")

        mean_std = {"mean": [0.4785], "std": [0.2834]}
        if self.is_train:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        resize_size, interpolation=InterpolationMode.BICUBIC, antialias=True
                    ),  # to maintain same ratio w.r.t. 224 images
                    transforms.CenterCrop((crop_size, crop_size)),
                    transforms.RandomAffine(degrees=20, shear=10, translate=(0.1, 0.1), scale=(0.95, 1.05)),
                    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(**mean_std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        resize_size, interpolation=InterpolationMode.BICUBIC, antialias=True
                    ),  # to maintain same ratio w.r.t. 224 images
                    transforms.CenterCrop((crop_size, crop_size)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(**mean_std),
                ]
            )

        return transform

    def __len__(self):
        return len(self.images_list)

    def _shuffle_sentences(self, text):
        # spliting sentences
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

        # shuffling sentences
        random.shuffle(sentences)

        # re-combining sentences
        new_text = " ".join(sentences)
        return new_text

    # transformers/data/data_collator.py
    def _random_mask(self, inputs: Any, mask_ratio: int = 0.5) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mask_ratio)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels

    def __getitem__(self, index):
        img = pil_loader(self.images_list[index])
        seed = np.random.randint(999999) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        img = self.transform_big(img)

        seed_a = None

        if self.auxview_images_list[index] != "-1":
            seed_a = np.random.randint(999999) # make a seed with numpy generator 
            random.seed(seed_a) # apply this seed to img tranfsorms
            torch.manual_seed(seed_a) # needed for torchvision 0.7
            a_img = pil_loader(self.auxview_images_list[index])
            a_img = self.transform_small(a_img)
        else:
            a_img = torch.zeros((3, 224, 224))
        if self.last_images_list[index] != "-1":
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            l_img = pil_loader(self.last_images_list[index])
            l_img = self.transform_small(l_img)
        else:
            l_img = torch.zeros((3, 224, 224))
        if self.last_auxview_images_list[index] != "-1":
            if seed_a is not None:
                random.seed(seed_a) # apply this seed to img tranfsorms
                torch.manual_seed(seed_a) # needed for torchvision 0.7
            la_img = pil_loader(self.last_auxview_images_list[index])
            la_img = self.transform_small(la_img)
        else:
            la_img = torch.zeros((3, 224, 224))

        report = self.reports_list[index]
        report = self._shuffle_sentences(report)

        encoded = self.tokenizer(
            report,
            padding="max_length",
            max_length=self.max_caption_length,
            truncation=True,
        )
        ids = torch.tensor(encoded["input_ids"]).unsqueeze(0)
        inputs = ids.clone()
        attention_mask = torch.tensor(encoded["attention_mask"]).unsqueeze(0)
        inputs, labels = self._random_mask(inputs)

        return img, a_img, l_img, la_img, inputs, labels, ids, attention_mask

    def collate_fn(self, instances: List[Tuple]):
        img_list, a_img_list, l_img_list, la_img_list, inputs_list, labels_list, ids_list, attention_mask_list = [], [], [], [], [], [], [], []
        # flattern
        for img, a_img, l_img, la_img, inputs, labels, ids, attention_mask in instances:
            img_list.append(img)
            a_img_list.append(a_img)
            l_img_list.append(l_img)
            la_img_list.append(la_img)
            inputs_list.append(inputs)
            labels_list.append(labels)
            ids_list.append(ids)
            attention_mask_list.append(attention_mask)

        # stack
        img_stack = torch.stack(img_list, dim=0)
        a_img_stack = torch.stack(a_img_list, dim=0)
        l_img_stack = torch.stack(l_img_list, dim=0)
        la_img_stack = torch.stack(la_img_list, dim=0)
        inputs_stack = torch.stack(inputs_list, dim=0).squeeze()
        labels_stack = torch.stack(labels_list, dim=0).squeeze()
        ids_stack = torch.stack(ids_list, dim=0).squeeze()
        attention_mask_stack = torch.stack(attention_mask_list, dim=0).squeeze()

        # sort and add to dictionary
        return_dict = {
            "img": img_stack,
            "a_img": a_img_stack,
            "l_img": l_img_stack,
            "la_img": la_img_stack,
            "inputs": inputs_stack,
            "labels": labels_stack,
            "ids": ids_stack,
            "attention_mask": attention_mask_stack,
        }

        return return_dict

    def read_csv(self):
        split = "train" if self.is_train else "valid"
        df = pd.read_csv(os.path.join(self.data_root, "%s.csv" % (split)))
        columns_to_keep = [
            "image_path",
            "auxview_image_path",
            "last_image_path",
            "last_auxview_image_path",
            "report",
        ]
        return df[columns_to_keep]
