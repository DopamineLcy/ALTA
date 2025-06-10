import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from RSNA_dataset.datasets_RSNA_zeroshot import build_dataset
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from model import alta
np.random.seed(0)


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

dataset_val = build_dataset('test', reshape_size, crop_size, mean, std)



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


gt = torch.FloatTensor().cuda()
pred = torch.FloatTensor().cuda()
pred_soft = torch.FloatTensor().cuda()

with torch.no_grad():
    for batch in tqdm(data_loader_val):
        path, target, sample, pos_batch_dict, neg_batch_dict = batch
        target = target.squeeze().cuda()
        img_feature = model.forward_img_feature(sample.cuda(), 1)
        img_feature = F.normalize(img_feature, dim=-1, p=2)
        
        input_ids=pos_batch_dict['input_ids'].cuda()
        attention_mask=pos_batch_dict['attention_mask'].cuda()
        token_type_ids = pos_batch_dict['token_type_ids'].cuda()
        B, C, N = input_ids.shape

        input_ids, attention_mask, token_type_ids = input_ids.reshape(-1, input_ids.shape[2]), attention_mask.reshape(-1, input_ids.shape[2]), token_type_ids.reshape(-1, input_ids.shape[2])
       
        pos_text_feature = model.forward_txt_feature(pos_batch_dict).mean(dim=1)
        pos_text_feature = F.normalize(pos_text_feature, dim=-1, p=2)

        input_ids=neg_batch_dict['input_ids'].cuda()
        attention_mask=neg_batch_dict['attention_mask'].cuda()
        token_type_ids = neg_batch_dict['token_type_ids'].cuda()
        B, C, N = input_ids.shape
        input_ids, attention_mask, token_type_ids = input_ids.reshape(-1, input_ids.shape[2]), attention_mask.reshape(-1, input_ids.shape[2]), token_type_ids.reshape(-1, input_ids.shape[2])

        neg_text_feature = model.forward_txt_feature(neg_batch_dict).mean(dim=1)
        neg_text_feature = F.normalize(neg_text_feature, dim=-1, p=2)

        pos_cos_sim = (pos_text_feature * img_feature).sum(1)
        neg_cos_sim = (neg_text_feature * img_feature).sum(1)
        
        predict = pos_cos_sim > neg_cos_sim

        predict_soft = torch.softmax(torch.cat([pos_cos_sim.unsqueeze(-1), neg_cos_sim.unsqueeze(-1)],dim=-1), dim=-1)[:, 0]

        gt = torch.cat((gt, target.to(torch.int)), 0)
        pred = torch.cat((pred, predict.to(torch.int)), 0)
        pred_soft = torch.cat((pred_soft, predict_soft), 0)



auroc = compute_AUROCs(gt, pred_soft)


def compute_acc_np(gt_torch, pred_torch, threshold):
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
  
    gt = gt_torch
    pred = (pred_torch > threshold).astype('bool')

    acc = np.mean(gt == pred)
    tp = np.sum(gt & pred)
    fp = np.sum(pred & ~gt)
    fn = np.sum(gt & ~pred)
    tn = np.sum(~gt & ~pred)
    recall = tp / (tp + fn)
    prec = tp / (tp + fp)
    f1 = 2 * prec * recall / (prec + recall)

    npv = tn / (tn + fn)
    return acc, f1, recall, prec, npv

pred_np = pred_soft.cpu().numpy()
gt_np = gt.cpu().numpy().astype('bool')

collect_acc = []
collect_F1 = []
collect_PPV = []
collect_NPV = []


num_folds = 10
stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)


for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X=pred_np, y=gt_np)):
    train_pred, test_pred = pred_np[train_idx], pred_np[test_idx]
    train_gt, test_gt = gt_np[train_idx], gt_np[test_idx]

    best_theshold = 0
    best_acc = -1

    inter = 0.005

    for i in np.arange(0, 1+inter, inter):
        _, current_acc, _, _, _ = compute_acc_np(train_gt, train_pred, i)
        
        if current_acc > best_acc:
            best_theshold = i
            best_acc = current_acc

    this_fold_acc, this_fold_F1, this_fold_recall, this_fold_prec, this_fold_npv = compute_acc_np(test_gt, test_pred, best_theshold)
    collect_acc.append(this_fold_acc)
    collect_F1.append(this_fold_F1)

print('AUC:', round(auroc, 3))
print('ACC:', round(np.mean(collect_acc), 3))
print('F1:', round(np.mean(collect_F1), 3))
