import os
import PIL
from torchvision import transforms
from torchvision.datasets.folder import default_loader,VisionDataset,IMG_EXTENSIONS
from typing import Any, Callable, Optional, Tuple
import codecs
import torch
import numpy as np
import pydicom
import cv2
from PIL import Image
from transformers import BertTokenizer
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
        self.RSNA_root = './RSNA_dataset'
        samples = self.read_samples(self.root)
        self.loader = loader
        self.extensions = extensions
        self.samples = samples


        pos_query = [
            'There is pneumonia',
            ]
        neg_query = [
            'There is no pneumonia',
        ]

        if tokenizer is None:
            self.tokenizer = CXRBertTokenizer.from_pretrained("BiomedVLP-CXR-BERT-specialized")
        else:
            self.tokenizer = tokenizer
        
        self.max_caption_length = 10

        pos_tokens = []
        for query in pos_query:
            pos_tokens.append(self.tokenizer(query, padding='max_length', max_length=self.max_caption_length, truncation=False))


        neg_tokens = []
        for query in neg_query:
            neg_tokens.append(self.tokenizer(query, padding='max_length', max_length=self.max_caption_length, truncation=False))

        self.pos_batch_dict = self._from_tokens_to_tensor(pos_tokens)
        self.neg_batch_dict = self._from_tokens_to_tensor(neg_tokens)


    def _from_tokens_to_tensor(self, tokens):
        input_ids = [torch.tensor(item["input_ids"]) for item in tokens]
        token_type_ids = [torch.tensor(item["token_type_ids"]) for item in tokens]
        attention_mask = [torch.tensor(item["attention_mask"]) for item in tokens]

        input_ids_tensor = torch.stack(input_ids, dim=0)
        token_type_ids_tensor = torch.stack(token_type_ids, dim=0)
        attention_mask_tensor = torch.stack(attention_mask, dim=0)

        cap_lens = (attention_mask_tensor == 0).to(torch.int32).argmax(dim=1).detach()
        cap_lens = cap_lens.masked_fill_(cap_lens == 0, 128) - 1
        
        batch_dict = {
            "caption_ids": input_ids_tensor,
            "input_ids": input_ids_tensor,
            "token_type_ids": token_type_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "cap_lens": cap_lens
        }
        return batch_dict

    def read_samples(self, root):
        data = codecs.open(root,"r","utf-8","ignore")
        outputList = []
        for line in data:
            outputList.append(line.strip())
        path_list = []
        for output in outputList:
            path_list.append((os.path.join(self.RSNA_root, output.split( )[0]), [ int(i) for i in output.split(' ')[1:]]))
        return path_list

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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        target = torch.FloatTensor(target)

        dcm = pydicom.read_file(path)
        x = dcm.pixel_array

        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            x = cv2.bitwise_not(x)
        
        sample = Image.fromarray(x).convert("RGB")
                
        if self.transform is not None:
            sample = self.transform(sample)

        return path, target, sample, self.pos_batch_dict, self.neg_batch_dict

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


def build_dataset(split, reshape_size, crop_size, mean, std, tokenizer=None):
    transform = build_transform(reshape_size, crop_size, mean, std)

    print(transform)

    filename = 'test_list.txt' if split == 'test' else 'val_list.txt'
    
    root = os.path.join('RSNA_dataset', filename)
    
    dataset = MultiLabelDatasetFolder(root, default_loader, IMG_EXTENSIONS, transform=transform, tokenizer=tokenizer)
    print(dataset)

    return dataset