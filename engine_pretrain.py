# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import InterpolationMode

import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    mask_ratio = args.mask_ratio
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        with torch.cuda.amp.autocast():
            loss, _, _ = model(batch, mask_ratio=mask_ratio)
        
        if isinstance(loss, tuple) and len(loss) == 4:
            loss_value1 = loss[0].item()
            loss_value2 = loss[1].item()
            loss_value3 = loss[2].item()
            loss_value4 = loss[3].item()
            loss = loss[0] + loss[1] + loss[2] + loss[3]
            loss = loss / accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss1=loss_value1)
            metric_logger.update(loss2=loss_value2)
            metric_logger.update(loss3=loss_value3)
            metric_logger.update(loss4=loss_value4)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce1 = misc.all_reduce_mean(loss_value1)
            loss_value_reduce2 = misc.all_reduce_mean(loss_value2)
            loss_value_reduce3 = misc.all_reduce_mean(loss_value3)
            loss_value_reduce4 = misc.all_reduce_mean(loss_value4)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train_loss1', loss_value_reduce1, epoch_1000x)
                log_writer.add_scalar('train_loss2', loss_value_reduce2, epoch_1000x)
                log_writer.add_scalar('train_loss3', loss_value_reduce3, epoch_1000x)
                log_writer.add_scalar('train_loss4', loss_value_reduce4, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

        elif isinstance(loss, tuple) and len(loss) == 3:
            loss_value1 = loss[0].item()
            loss_value2 = loss[1].item()
            loss_value3 = loss[2].item()
            loss = loss[0] + loss[1] + loss[2]
            loss = loss / accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss1=loss_value1)
            metric_logger.update(loss2=loss_value2)
            metric_logger.update(loss3=loss_value3)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce1 = misc.all_reduce_mean(loss_value1)
            loss_value_reduce2 = misc.all_reduce_mean(loss_value2)
            loss_value_reduce3 = misc.all_reduce_mean(loss_value3)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train_loss1', loss_value_reduce1, epoch_1000x)
                log_writer.add_scalar('train_loss2', loss_value_reduce2, epoch_1000x)
                log_writer.add_scalar('train_loss3', loss_value_reduce3, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

        elif isinstance(loss, tuple) and len(loss) == 2:
            loss_value1 = loss[0].item()
            loss_value2 = loss[1].item()
            loss = loss[0] + loss[1]
            loss = loss / accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss1=loss_value1)
            metric_logger.update(loss2=loss_value2)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce1 = misc.all_reduce_mean(loss_value1)
            loss_value_reduce2 = misc.all_reduce_mean(loss_value2)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train_loss1', loss_value_reduce1, epoch_1000x)
                log_writer.add_scalar('train_loss2', loss_value_reduce2, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
        else:
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss = loss / accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
        
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            model.module.logit_scale.clamp_(0, math.log(100))
            if hasattr(model.module, 'logit_scale_local1'):
                model.module.logit_scale_local1.clamp_(0, math.log(100))
                model.module.logit_scale_local2.clamp_(0, math.log(100))
                model.module.logit_scale_local3.clamp_(0, math.log(100))

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('logit_scale/global', model.module.logit_scale.item(), epoch_1000x)
                if hasattr(model.module, 'logit_scale_local1'):
                    log_writer.add_scalar('logit_scale/local1', model.module.logit_scale_local1.item(), epoch_1000x)
                    log_writer.add_scalar('logit_scale/local2', model.module.logit_scale_local2.item(), epoch_1000x)
                    log_writer.add_scalar('logit_scale/local3', model.module.logit_scale_local3.item(), epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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

@torch.no_grad()
def valid_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    mask_ratio = args.mask_ratio
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        with torch.cuda.amp.autocast():
            loss, _, _ = model(batch, mask_ratio=mask_ratio)
        if isinstance(loss, tuple) and len(loss) == 4:
            loss_value = loss[0].item() + loss[1].item() + loss[2].item() + loss[3].item()
            loss = loss[0] + loss[1] + loss[2] + loss[3]
            loss = loss / accum_iter
            # loss_scaler(loss, optimizer, parameters=model.parameters(),
            #             update_grad=(data_iter_step + 1) % accum_iter == 0)
            
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

        elif isinstance(loss, tuple) and len(loss) == 3:
            loss_value = loss[0].item() + loss[1].item() + loss[2].item()
            loss = loss[0] + loss[1] + loss[2]
            loss = loss / accum_iter
            # loss_scaler(loss, optimizer, parameters=model.parameters(),
            #             update_grad=(data_iter_step + 1) % accum_iter == 0)
            
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

        elif isinstance(loss, tuple) and len(loss) == 2:
            loss_value = loss[0].item() + loss[1].item()
            loss = loss[0] + loss[1]
            loss = loss / accum_iter
            # loss_scaler(loss, optimizer, parameters=model.parameters(),
            #             update_grad=(data_iter_step + 1) % accum_iter == 0)
            
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
        else:
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss = loss / accum_iter
            # loss_scaler(loss, optimizer, parameters=model.parameters(),
            #             update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def compute_acc(gt, pred):
    gt = gt.cpu().numpy().astype('bool')
    pred = pred.cpu().numpy().astype('bool')

    acc = np.mean(gt == pred).astype('float32')
    tp = np.sum(gt & pred).astype('float32')
    fp = np.sum(pred & ~gt).astype('float32')
    fn = np.sum(gt & ~pred).astype('float32')
    recall = tp / (tp + fn)
    prec = tp / (tp + fp)
    f1 = 2 * prec * recall / (prec + recall)
    
    return acc, f1, recall, prec

@torch.no_grad()
def zeroshot_valid_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    pred_soft = torch.FloatTensor().cuda()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        with torch.cuda.amp.autocast():
            _, target, sample, pos_batch_dict, neg_batch_dict = batch
            target = target.squeeze().cuda()
            img_feature = model.module.forward_img_feature(sample, 1)

            pos_text_feature = model.module.forward_txt_feature(pos_batch_dict).mean(dim=1)
            pos_text_feature = F.normalize(pos_text_feature, dim=-1, p=2)

            neg_text_feature = model.module.forward_txt_feature(neg_batch_dict).mean(dim=1)
            neg_text_feature = F.normalize(neg_text_feature, dim=-1, p=2)

            pos_cos_sim = (pos_text_feature * img_feature).mean(dim=1)
            neg_cos_sim = (neg_text_feature * img_feature).mean(dim=1)
            
            predict_soft = torch.softmax(torch.cat([pos_cos_sim.unsqueeze(-1), neg_cos_sim.unsqueeze(-1)],dim=-1), dim=-1)[:, 0]
            predict = pos_cos_sim > neg_cos_sim
            
            gt = torch.cat((gt, target.to(torch.int)), 0)
            pred = torch.cat((pred, predict.to(torch.int)), 0)
            pred_soft = torch.cat((pred_soft, predict_soft), 0)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            
    acc, f1, recall, prec = compute_acc(gt, pred)
    auroc = compute_AUROCs(gt, pred_soft)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auroc, acc, f1, recall, prec