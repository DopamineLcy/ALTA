from typing import Any
import re
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from CheXpert5X200_dataset.datasets_zeroshot import build_dataset
import torch
import numpy as np
from model import alta
from transformers import BertTokenizer
from nltk.tokenize import RegexpTokenizer
import random
np.random.seed(0)


CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}


def generate_chexpert_class_prompts(n: int = 5):
    """Generate text prompts for each CheXpert classification task

    Parameters
    ----------
    n:  int
        number of prompts per class

    Returns
    -------
    class prompts : dict
        dictionary of class to prompts
    """

    prompts = {}
    for k, v in CHEXPERT_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}".strip())

        # prompts[k] = random.sample(cls_prompts, n)
        if n is not None and n < len(cls_prompts):
            prompts[k] = random.sample(cls_prompts, n)
        else:
            prompts[k] = cls_prompts
    return prompts


class CXRBertTokenizer(BertTokenizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


def process_text(text, device, imported_tokenizer):

        if type(text) == str:
            text = [text]

        processed_text_tensors = []
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

            text_tensors = imported_tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            processed_text_tensors.append(text_tensors)

        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack(
            [x["attention_mask"] for x in processed_text_tensors]
        )
        token_type_ids = torch.stack(
            [x["token_type_ids"] for x in processed_text_tensors]
        )

        if len(text) == 1:
            caption_ids = caption_ids.squeeze(0).to(device)
            attention_mask = attention_mask.squeeze(0).to(device)
            token_type_ids = token_type_ids.squeeze(0).to(device)
        else:
            caption_ids = caption_ids.squeeze().to(device)
            attention_mask = attention_mask.squeeze().to(device)
            token_type_ids = token_type_ids.squeeze().to(device)

        cap_lens = []
        for txt in text:
            cap_lens.append(len([w for w in txt if not w.startswith("[")]))

        return {
            "input_ids": caption_ids,
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cap_lens": cap_lens,
        }

def compute_AUROCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    return roc_auc_score(gt_np, pred_np)

JOINT_FEATURE_SIZE = 128

# create model
model = alta()
checkpoint = torch.load('ALTA_weights/ALTA.pth', map_location='cpu')

try:
    checkpoint_model = checkpoint['model']
except:
    checkpoint_model = checkpoint
msg = model.load_state_dict(checkpoint_model, strict=False)
print(msg)
reshape_size = 256
crop_size = 224
mean=[0.4785]
std=[0.2834]
tokenizer = CXRBertTokenizer.from_pretrained("BiomedVLP-CXR-BERT-specialized")


cls_prompts = generate_chexpert_class_prompts(n=10000)


processed_txt = {}
for k, v in cls_prompts.items():
    processed_txt[k] = process_text(v, 'cuda', tokenizer)


dataset_val = build_dataset(reshape_size, crop_size, mean, std, tokenizer)

sampler_val = torch.utils.data.SequentialSampler(dataset_val) 

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, sampler=sampler_val,
    batch_size=16,
    num_workers=10,
    pin_memory=True,
    drop_last=False
)

model.eval()
model.cuda()

pred = torch.FloatTensor().cuda()
gt = torch.FloatTensor().cuda()

with torch.no_grad():
    for batch in tqdm(data_loader_val):
        path, target, sample = batch
        target = target.squeeze().cuda()
        class_similarities = torch.FloatTensor().cuda()
        
        img_feature = model.forward_img_feature(sample.cuda(), 1)
        
        for cls_name, cls_txt in processed_txt.items():
            if len(cls_txt['input_ids'].shape) == 2:
                cls_txt['input_ids'] = cls_txt['input_ids'].unsqueeze(0)
                cls_txt['attention_mask'] = cls_txt['attention_mask'].unsqueeze(0)

            text_feature = model.forward_txt_feature(cls_txt).mean(dim=1)
            text_feature = F.normalize(text_feature, dim=-1, p=2)
            cos_sim = (img_feature * text_feature).sum(1)
            class_similarities = torch.cat([class_similarities, cos_sim.unsqueeze(1)], dim=1)

        pred = torch.cat([pred, class_similarities], dim=0)
        gt = torch.cat([gt, target], dim=0)

pred = pred.argmax(dim=1)
gt = gt.argmax(dim=1)

acc = len(gt[gt == pred]) / len(gt)
print('ACC:', acc)
