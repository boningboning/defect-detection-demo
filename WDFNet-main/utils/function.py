# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate


def compute_pixel_fpr_fnr_dice(confusion_matrix, positive_class=1):
    num_classes = confusion_matrix.shape[0]
    eps = 1e-10
    total = confusion_matrix.sum()

    if num_classes == 2:
        cls = positive_class
        tp = confusion_matrix[cls, cls]
        fp = confusion_matrix[:, cls].sum() - tp
        fn = confusion_matrix[cls, :].sum() - tp
        tn = total - tp - fp - fn
        metric_mode = 'positive_class_{}'.format(cls)
    else:
        fpr_list = []
        fnr_list = []
        dice_list = []
        for cls in range(num_classes):
            tp = confusion_matrix[cls, cls]
            fp = confusion_matrix[:, cls].sum() - tp
            fn = confusion_matrix[cls, :].sum() - tp
            tn = total - tp - fp - fn
            fpr_list.append(fp / np.maximum(eps, fp + tn))
            fnr_list.append(fn / np.maximum(eps, fn + tp))
            dice_list.append((2 * tp) / np.maximum(eps, 2 * tp + fp + fn))
        return float(np.mean(fpr_list)), float(np.mean(fnr_list)), float(np.mean(dice_list)), 'macro_ovr'

    pixel_fpr = fp / np.maximum(eps, fp + tn)
    pixel_fnr = fn / np.maximum(eps, fn + tp)
    dice = (2 * tp) / np.maximum(eps, 2 * tp + fp + fn)
    logging.info('Binary confusion for %s: TP=%.0f, FP=%.0f, TN=%.0f, FN=%.0f', metric_mode, tp, fp, tn, fn)
    return float(pixel_fpr), float(pixel_fnr), float(dice), metric_mode


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    avg_wave_loss = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    avg_sb_loss = AverageMeter()
    avg_gate_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        # images, labels, bd_gts, _, _,domains,clss = batch
        images, labels, bd_gts, _, _ = batch

        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()
        losses, _, acc, loss_list = model(images, labels,bd_gts,is_val = False)
        loss = losses.mean()
        acc  = acc.mean()

        # before_update = torch.norm(torch.cat([p.flatten() for p in model.parameters()])).item()
        model.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_wave_loss.update(loss_list[0].mean().item())
        avg_sem_loss.update(loss_list[1].mean().item())
        avg_bce_loss.update(loss_list[2].mean().item())
        avg_sb_loss.update(loss_list[3].mean().item())
        avg_gate_loss.update(loss_list[4].mean().item())



        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Wave loss: {:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}, SB loss: {:.6f}, Gate loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average(), avg_wave_loss.average(), avg_sem_loss.average(), avg_bce_loss.average(), avg_sb_loss.average(), avg_gate_loss.average())
            logging.info(msg)
        # if i_iter % config.PRINT_FREQ == 0:
        #     msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
        #           'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f},Domain loss: {:.6f}, SB loss: {:.6f}' .format(
        #               epoch, num_epoch, i_iter, epoch_iters,
        #               batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
        #               ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(),avg_domain_loss.average(),ave_loss.average()-avg_sem_loss.average()-avg_bce_loss.average()-avg_domain_loss.average())
        #     logging.info(msg)


    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    acc_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()
    dice_meter = AverageMeter()

    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, _ = batch
            # image, label, _, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            bd_gts = bd_gts.float().cuda()


            # losses, pred, _, _ = model(image, label, bd_gts, is_val = True)
            # losses, pred, acc, precision, recall, f1 = model(image, label, bd_gts, is_val = True)
            losses, pred, acc, precision, recall, f1 = model(image, label,bd_gts, is_val = True)

            acc = acc.mean()
            precision = precision.mean()
            recall = recall.mean()
            f1 = f1.mean()


            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()

            acc_meter.update(acc.item())
            precision_meter.update(precision.item())
            recall_meter.update(recall.item())
            f1_meter.update(f1.item())
            ave_loss.update(loss.item())

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        
        logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    if config.DATASET.NUM_CLASSES == 2:
        dice = 2 * IoU_array[1] / np.maximum(1e-10, 1 + IoU_array[1])
    else:
        dice_array = 2 * IoU_array / np.maximum(1e-10, 1 + IoU_array)
        dice = dice_array.mean()
    dice_meter.update(float(dice))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_acc', acc_meter.average(), global_steps)
    writer.add_scalar('valid_precision', precision_meter.average(), global_steps)
    writer.add_scalar('valid_recall', recall_meter.average(), global_steps)
    writer.add_scalar('valid_f1', f1_meter.average(), global_steps)
    writer.add_scalar('valid_dice', dice_meter.average(), global_steps)
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array, dice_meter.average(), acc_meter.average(), precision_meter.average(), recall_meter.average(), f1_meter.average()


def testval(config, test_dataset, testloader, model,
            sv_dir='./', sv_pred=True):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            pred = test_dataset.single_scale_inference(config, model, image.cuda())

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    pixel_fpr, pixel_fnr, dice, metric_mode = compute_pixel_fpr_fnr_dice(confusion_matrix)

    return mean_IoU, IoU_array, pixel_acc, mean_acc, pixel_fpr, pixel_fnr, dice, metric_mode


def test(config, test_dataset, testloader, model,
         sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                
            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
