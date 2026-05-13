# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config
from utils.domain import DomainPrompter

class FullModel(nn.Module):

  def __init__(self, model, sem_loss, bd_loss, gate_loss=None):
    super(FullModel, self).__init__()
    self.model = model
    self.sem_loss = sem_loss
    self.bd_loss = bd_loss
    self.gate_loss = gate_loss





  def pixel_acc(self, pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

  def _single_semantic_loss(self, score, target, use_ohem=True):
    if use_ohem and hasattr(self.sem_loss, '_ohem_forward'):
      return self.sem_loss._ohem_forward(score, target)
    if hasattr(self.sem_loss, '_ce_forward'):
      return self.sem_loss._ce_forward(score, target)
    return self.sem_loss._forward(score, target)

  def precision_recall_f1(self, pred, label, num_classes, ignore_index=-1):

      _, preds = torch.max(pred, dim=1)  # [B, H, W]
      precision_list = []
      recall_list = []
      f1_list = []

      for cls in range(num_classes):
          pred_cls = (preds == cls)
          label_cls = (label == cls)
          valid = (label != ignore_index)

          # 仅统计 valid 区域
          tp = torch.sum((pred_cls & label_cls) & valid)
          fp = torch.sum((pred_cls & (~label_cls)) & valid)
          fn = torch.sum(((~pred_cls) & label_cls) & valid)

          precision = tp.float() / (tp.float() + fp.float() + 1e-10)
          recall = tp.float() / (tp.float() + fn.float() + 1e-10)
          f1 = 2 * precision * recall / (precision + recall + 1e-10)

          precision_list.append(precision)
          recall_list.append(recall)
          f1_list.append(f1)

      return torch.stack(precision_list), torch.stack(recall_list), torch.stack(f1_list)

  # def forward(self, inputs, labels, bd_gt,domains=None,clss=None,is_val = True, *args, **kwargs):
  def forward(self, inputs, labels,bd_gt,is_val = True, *args, **kwargs):


    outputs  = self.model(inputs, *args, **kwargs)

    h, w = labels.size(1), labels.size(2)
    ph, pw = outputs[0].size(2), outputs[0].size(3)
    if ph != h or pw != w:
        for i in range(len(outputs)):
            outputs[i] = F.interpolate(outputs[i], size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)


    acc  = self.pixel_acc(outputs[-2], labels)
    balance_weights = config.LOSS.BALANCE_WEIGHTS
    seg_weight = balance_weights[-1] if len(balance_weights) > 1 else 1.0
    loss_w = config.LOSS.WAVE_WEIGHTS * self._single_semantic_loss(outputs[0], labels, use_ohem=False)
    loss_s = seg_weight * self._single_semantic_loss(outputs[-2], labels, use_ohem=True)
    loss_b = self.bd_loss(outputs[-1], bd_gt)
    gates = [self.model.last_aux.get('gate16'), self.model.last_aux.get('gate32')]
    if self.gate_loss is None:
        loss_g = loss_s * 0.0
    else:
        loss_g = config.LOSS.GATE_WEIGHTS * self.gate_loss(gates, labels)

    filler = torch.ones_like(labels) * config.TRAIN.IGNORE_LABEL
    bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:])>0.8, labels, filler)
    loss_sb = self.sem_loss(outputs[-2], bd_label)



    loss = loss_w + loss_s + loss_b + loss_sb + loss_g
    # loss = loss_s
    if is_val:
        # precision, recall, f1 = self.precision_recall_f1(outputs[-2], labels, num_classes=2)
        precision, recall, f1 = self.precision_recall_f1(outputs[-2], labels, config.DATASET.NUM_CLASSES)
        return torch.unsqueeze(loss,0), outputs[:-1], acc, precision, recall, f1
    else:
        # return torch.unsqueeze(loss, 0), outputs[:-1], acc, [loss_s, loss_b, loss_d]
        return torch.unsqueeze(loss, 0), outputs[:-1], acc, [loss_w,loss_s,loss_b,loss_sb,loss_g]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int64)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr