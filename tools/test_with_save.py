import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import colorsys  # 新增

import _init_paths
import models
import datasets
from config import config
from config import update_config
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Test with saving predictions')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('opts', help="Modify config options", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    return args


def get_colors(num_colors):
    """为每个类别生成不同的颜色"""
    colors = []
    for i in range(num_colors):
        hue = i / max(num_colors, 1)  # 防止除零
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    # 背景类设为黑色
    if len(colors) > 0:
        colors[0] = (0, 0, 0)
    return colors


def main():
    args = parse_args()
    logger, final_output_dir, _ = create_logger(config, args.cfg, 'test_save')

    # 创建保存目录
    save_dir = final_output_dir
    vis_dir = os.path.join(save_dir, 'vis')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    logger.info(f'Predictions will be saved to {save_dir}')

    # build model
    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)

    # load model
    model_state_file = config.TEST.MODEL_FILE
    logger.info(f'=> loading model from {model_state_file}')
    pretrained_dict = torch.load(model_state_file, map_location='cuda')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']

    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # data parallel
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    model.eval()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    logger.info(f'Start inference on {len(testloader)} images...')

    # 生成颜色映射
    num_classes = config.DATASET.NUM_CLASSES
    colors = get_colors(num_classes)
    logger.info(f'Using {len(colors)} colors for visualization')

    # 推理并保存
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(testloader)):
            image = batch[0].cuda()

            # 推理
            output = model(image)
            if isinstance(output, tuple):
                output = output[0]

            # 获取预测结果
            pred = torch.argmax(output, 1)
            pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)

            # 保存灰度预测图（类别ID）
            #pred_img = Image.fromarray(pred)
            #pred_img.save(os.path.join(save_dir, f'{idx:06d}.png'))

            # 生成彩色可视化图
            h, w = pred.shape
            color_pred = np.zeros((h, w, 3), dtype=np.uint8)

            # 为每个类别着色
            for class_id in range(num_classes):
                mask = pred == class_id
                if mask.any():
                    color_pred[mask] = colors[class_id]

            # 保存彩色图
            vis_img = Image.fromarray(color_pred)
            vis_img.save(os.path.join(vis_dir, f'{idx:06d}_vis.png'))

    logger.info(f'Done! Predictions saved to {save_dir}')
    logger.info(f'Grayscale: {save_dir}')
    logger.info(f'Color visualizations: {vis_dir}')


if __name__ == '__main__':
    main()