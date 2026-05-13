# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import logging
import timeit

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.function import testval, test
from utils.utils import create_logger

# 新增库
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch.utils.benchmark as benchmark


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="./configs/wdfnet_yepian.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = models.wdfnet.get_seg_model(config, imgnet_pretrained=True)

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pt')

    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.cuda()
    model.eval()

    # >>> 新增 参数量 & FLOPs 计算 (fvcore)
    dummy_input = torch.randn(1, 3, config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0]).cuda()
    logger.info("=== 模型统计 ===")
    logger.info(parameter_count_table(model))
    flops = FlopCountAnalysis(model, dummy_input)
    logger.info(f"FLOPs: {flops.total() / 1e9:.2f} G")
    # <<< 新增

    # >>> 新增 FPS 测试 (benchmark)
    t = benchmark.Timer(
        stmt="model(dummy_input)",
        globals={"model": model, "dummy_input": dummy_input}
    )
    res = t.timeit(100)
    fps = 1.0 / res.mean  # 单次平均耗时的倒数
    logger.info(f"=== FPS 测试结果 ===")
    logger.info(f"输入尺寸: {dummy_input.shape}, 推理次数: 100, FPS: {fps:.2f}")
    # <<< 新增

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
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
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    logger.info('Test batch size: {}'.format(config.TEST.BATCH_SIZE_PER_GPU))

    start = timeit.default_timer()

    if ('test' in config.DATASET.TEST_SET) and ('city' in config.DATASET.DATASET):
        test(config,
             test_dataset,
             testloader,
             model,
             sv_dir=final_output_dir)

    else:
        mean_IoU, IoU_array, pixel_acc, mean_acc, pixel_fpr, pixel_fnr, dice, metric_mode = testval(config,
                                                                                                      test_dataset,
                                                                                                      testloader,
                                                                                                      model,
                                                                                                      sv_dir=final_output_dir)

        msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, Mean_Acc: {: 4.4f}, Pixel_FPR: {: 4.4f}, Pixel_FNR: {: 4.4f}, Dice: {: 4.4f}, Metric_Mode: {}, Class IoU: '.format(
            mean_IoU, pixel_acc, mean_acc, pixel_fpr, pixel_fnr, dice, metric_mode)
        logging.info(msg)
        logging.info(IoU_array)
        logging.info('Pixel_FPR = FP / (FP + TN), Pixel_FNR = FN / (FN + TP), Dice = 2TP / (2TP + FP + FN)')

    end = timeit.default_timer()
    logger.info('Mins: %d' % int((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()
