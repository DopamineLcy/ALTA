import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from model import alta
from CheXpert8X200_dataset.datasets_image_retrieval import QueryRetrievalDataset, CandidateRetrievalDataset


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

model.eval()
model.cuda()

# create dataset
root = 'CheXpert8X200_dataset/image-retrieval'
candidate_file_path = os.path.join(root, 'candidate.csv')
query_file_path = os.path.join(root, 'query.csv')
dict = {
    'Atelectasis': 8,
    'Cardiomegaly': 2,
    'Edema': 5,
    'Fracture': 12,
    'No Finding': 0,
    'Pleural Effusion': 10,
    'Pneumonia': 7,
    'Pneumothorax': 9
}


q_dataset = QueryRetrievalDataset(query_file_path, reshape_size, crop_size, mean=mean, std=std)

batch_size = 4

Q = torch.FloatTensor()
Q_label = torch.FloatTensor()

q_data_loader = DataLoader(q_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

with torch.no_grad():
    for images, labels in tqdm(q_data_loader):
        images = images.cuda()
        img_feature = model.forward_img_feature(images, 1)
        img_feature = F.normalize(img_feature, dim=-1)

        labels = torch.tensor([dict[i] for i in labels])
        Q = torch.cat((Q, img_feature.cpu()), 0)
        Q_label = torch.cat((Q_label, labels), 0)


c_dataset = CandidateRetrievalDataset(candidate_file_path, reshape_size, crop_size, mean=mean, std=std)

batch_size = 16
c_data_loader = DataLoader(c_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

C = torch.FloatTensor()
C_label = torch.FloatTensor()

with torch.no_grad():
    for images, labels in tqdm(c_data_loader):
        images = images.cuda()
        img_feature = model.forward_img_feature(images, 1)
        img_feature = F.normalize(img_feature, dim=-1)
        C = torch.cat((C, img_feature.cpu()), 0)

        indices = torch.nonzero(labels == 1, as_tuple=False)
        C_label = torch.cat((C_label, indices[:, 1]), 0)

C.shape

sim = Q @ C.T

topk_values, topk_indices = torch.topk(sim, k=5, dim=1)

print("P@5:", (C_label[topk_indices] == Q_label.unsqueeze(1)).to(torch.float32).mean().item())

topk_values, topk_indices = torch.topk(sim, k=10, dim=1)

print("P@10:", (C_label[topk_indices] == Q_label.unsqueeze(1)).to(torch.float32).mean().item())

topk_values, topk_indices = torch.topk(sim, k=50, dim=1)

print("P@50:", (C_label[topk_indices] == Q_label.unsqueeze(1)).to(torch.float32).mean().item())

