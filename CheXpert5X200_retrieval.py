import torch
import torch.nn.functional as F
from tqdm import tqdm
from CheXpert5X200_dataset.datasets_retrieval import build_dataset
import torch
import numpy as np
from typing import Any
from transformers import BertConfig, BertTokenizer
from model import alta
np.random.seed(0)


class CXRBertTokenizer(BertTokenizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


JOINT_FEATURE_SIZE = 128

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
tokenizer = CXRBertTokenizer.from_pretrained("./BiomedVLP-CXR-BERT-specialized")


# %%
dataset_test = build_dataset(reshape_size, crop_size, mean, std, tokenizer)


sampler_test = torch.utils.data.SequentialSampler(dataset_test) 

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, sampler=sampler_test,
    batch_size=16,
    num_workers=10,
    pin_memory=True,
    drop_last=False
)

model.eval()
model.cuda()

pred = torch.FloatTensor().cuda()
gt = torch.FloatTensor().cuda()
cnt = 0

Q = torch.FloatTensor()
Q_local = torch.FloatTensor()
C = torch.FloatTensor()
C_local = torch.FloatTensor()
cap_lens = torch.FloatTensor()

with torch.no_grad():
    for batch in tqdm(data_loader_test):
        path, target, sample, text = batch
        target = target.squeeze().cuda()
        class_similarities = torch.FloatTensor().cuda()
        for k in text.keys():
            text[k] = text[k].cuda()

        img_feature = model.forward_img_feature(sample.cuda(), 1)
        text_feature = model.forward_txt_feature_single(text)

        Q = torch.cat((Q, img_feature.cpu()), 0)
        C = torch.cat((C, text_feature.cpu()), 0)
        
        cnt+=1

sim = Q @ C.T

label = torch.tensor([[0],[1],[2],[3],[4]]).repeat(1, 200).reshape(-1)

topk_values, topk_indices = torch.topk(sim, k=5, dim=1)
print("P@5:", (label[topk_indices] == label.unsqueeze(1)).to(torch.float32).mean().item())

topk_values, topk_indices = torch.topk(sim, k=10, dim=1)
print("P@10:", (label[topk_indices] == label.unsqueeze(1)).to(torch.float32).mean().item())

topk_values, topk_indices = torch.topk(sim, k=100, dim=1)
print("P@100:", (label[topk_indices] == label.unsqueeze(1)).to(torch.float32).mean().item())
