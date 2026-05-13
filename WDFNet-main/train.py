# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss, GateLoss
from utils.function import train, validate
from utils.utils import create_logger, FullModel


def plot_training_curves(records, save_path):
    if not records:
        return
    epochs = [item['epoch'] for item in records]
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)
    ax2 = ax1.twinx()
    lines = []
    lines += ax1.plot(epochs, [item['mean_iou'] for item in records], label='MeanIoU', linewidth=2.0)
    lines += ax1.plot(epochs, [item['best_miou'] for item in records], label='Best_mIoU', linewidth=2.0)
    lines += ax1.plot(epochs, [item['dice'] for item in records], label='Dice', linewidth=2.0)
    lines += ax1.plot(epochs, [item['acc'] for item in records], label='Acc', linewidth=2.0)
    lines += ax1.plot(epochs, [item['pre'] for item in records], label='Pre', linewidth=2.0)
    lines += ax1.plot(epochs, [item['recall'] for item in records], label='Recall', linewidth=2.0)
    lines += ax1.plot(epochs, [item['f1'] for item in records], label='F1', linewidth=2.0)
    lines += ax2.plot(epochs, [item['loss'] for item in records], label='Loss', color='#d62728', linewidth=2.0)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Metric')
    ax2.set_ylabel('Loss')
    x_end = max(epochs) + 1
    ax1.set_xlim(min(epochs), x_end)
    ax1.set_xticks(np.arange(min(epochs), x_end + 1, 100))
    ax1.set_ylim(0.4, 1.02)
    ax1.grid(True, linestyle='--', alpha=0.35)
    legend = ax1.legend(
        lines,
        [line.get_label() for line in lines],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.14),
        ncol=4,
        frameon=True,
        borderaxespad=0.0,
    )
    legend.set_zorder(100)
    fig.subplots_adjust(top=0.84, left=0.08, right=0.92, bottom=0.11)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="./configs/wdfnet_yepian.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    # if torch.cuda.device_count() != len(gpus):
    #     print("The gpu numbers do not match!")
    print(config.GPUS)
    #     return 0
    
    imgnet = 'imagenet-1k' in config.MODEL.PRETRAINED
    model = models.wdfnet.get_seg_model(config, imgnet_pretrained=True)
    # model = models.wdfnet.WDFNet()

    # batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(config.GPUS)
    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        scale_factor=config.TRAIN.SCALE_FACTOR)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)


    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)

    bd_criterion = BondaryLoss()
    gate_criterion = GateLoss(ignore_label=config.TRAIN.IGNORE_LABEL)
    
    model = FullModel(model, sem_criterion, bd_criterion, gate_criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')

    #训练数据经过模型训练的次数
    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
        
    best_mIoU = 0
    last_epoch = 0
    flag_rm = config.TRAIN.RESUME
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    curve_records = []
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    
    for epoch in range(last_epoch, real_end):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        train(config, epoch, config.TRAIN.END_EPOCH, 
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict)

        if flag_rm == 1 or (epoch % 5 == 0 and epoch < real_end - 100) or (epoch >= real_end - 100):
            valid_loss, mean_IoU, IoU_array, dice, acc, pre, recall, f1 = validate(config,
                        testloader, model, writer_dict)
        if flag_rm == 1:
            flag_rm = 0

        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))
        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best.pt'))
        msg = 'Loss: {:.3f}, MeanIoU: {: 4.4f}, Best_mIoU: {: 4.4f}, Dice: {: 4.4f}, Acc: {: 4.4f}, Pre: {: 4.4f}, Recall: {: 4.4f}, F1: {: 4.4f}'.format(
                    valid_loss, mean_IoU, best_mIoU, dice, acc, pre, recall, f1)
        logging.info(msg)
        logging.info(IoU_array)
        curve_records.append({
            'epoch': epoch,
            'loss': valid_loss,
            'mean_iou': mean_IoU,
            'best_miou': best_mIoU,
            'dice': dice,
            'acc': acc,
            'pre': pre,
            'recall': recall,
            'f1': f1,
        })



    torch.save(model.module.state_dict(),
            os.path.join(final_output_dir, 'final_state.pt'))
    plot_training_curves(
            curve_records,
            os.path.join(final_output_dir, '{}_metrics_loss_curve.png'.format(os.path.basename(args.cfg).split('.')[0])))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % np.int64((end-start)/3600))
    logger.info('Done')

if __name__ == '__main__':
    main()
