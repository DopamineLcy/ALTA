import os
import PIL
import re
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader,VisionDataset,IMG_EXTENSIONS
from typing import Any, Callable, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
from transformers import BertConfig, BertTokenizer
from nltk.tokenize import RegexpTokenizer
np.random.seed(0)


class CXRBertTokenizer(BertTokenizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class MultiLabelDatasetFolder(VisionDataset):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            tokenizer = None,
    ) -> None:
        super(MultiLabelDatasetFolder, self).__init__(root, transform=transform)
        df = pd.read_csv(root)
        self.loader = loader
        self.extensions = extensions
        self.tokenizer = tokenizer
        self.samples = df
        self.max_caption_length = 256
        self.chexpert_root = 'path_to_CheXpert'

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, original_text, target = os.path.join(self.chexpert_root, self.samples.iloc[index]['Path']), self.samples.iloc[index]['Report Impression'], self.samples.iloc[index][['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']]

        original_text = self.process_text(original_text)
        text = self.tokenizer(original_text, padding='max_length', max_length=self.max_caption_length, truncation=False)
        text = self._from_tokens_to_tensor(text)

        cap_lens = (text["attention_mask"] == 0).to(torch.int32).argmax(dim=0).detach()
        cap_lens = cap_lens.masked_fill_(cap_lens == 0, self.max_caption_length) - 1

        text["caption_ids"] = text["input_ids"]
        text["cap_lens"] = cap_lens

        target = torch.FloatTensor(target)
        
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return path, target, sample, text

    def _from_tokens_to_tensor(self, tokens):
        # 将 tokenized sentences 转换为 tensor
        input_ids = torch.tensor(tokens["input_ids"])
        token_type_ids = torch.tensor(tokens["token_type_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])

        batch_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        return batch_dict
    
    def process_text(self, text):

        if type(text) == str:
            text = [text]

        for t in text:
            # use space instead of newline
            t = t.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(t)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            all_sents = []

            for t in captions:
                t = t.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(t.lower())

                if len(tokens) <= 1:
                    continue

                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                all_sents.append(" ".join(included_tokens))

            t = " ".join(all_sents)

        return t

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

    def __len__(self) -> int:
        return len(self.samples)


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


def build_dataset(reshape_size, crop_size, mean, std, tokenizer):
    transform = build_transform(reshape_size, crop_size, mean, std)

    print(transform)
    
    root = 'CheXpert5X200_dataset/chexpert_5x200.csv'
    
    dataset = MultiLabelDatasetFolder(root, default_loader, IMG_EXTENSIONS, transform=transform, tokenizer=tokenizer)
    print(dataset)

    return dataset