# ------------------------------------------------------------------------------
# 简化版训练集评估脚本 (修复版)
# ------------------------------------------------------------------------------

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入项目模块
import _init_paths  # 初始化项目路径
import models  # 模型定义
import datasets  # 数据集定义
from config import config, update_config  # 配置文件
from utils.utils import create_logger  # 日志工具


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='在训练集上评估模型')
    parser.add_argument('--cfg', help='配置文件路径', required=True, type=str)
    parser.add_argument('opts', help="命令行配置参数", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)  # 更新配置
    return args


def main():
    """主函数"""
    args = parse_args()  # 解析参数

    # 创建日志
    logger, _, _ = create_logger(config, args.cfg, 'eval_train')
    logger.info(f"配置参数: {config}")

    # 设置GPU
    gpus = list(config.GPUS)
    torch.backends.cudnn.benchmark = True  # 启用cudnn加速

    # 1. 构建模型
    logger.info("===> 构建模型中...")
    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    model.eval()  # 设置为评估模式

    # 2. 加载训练好的权重
    logger.info("===> 加载模型权重...")
    checkpoint_path = r'F:\download\checkpoint.pth.tar'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        # 提取模型权重
        new_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # 去掉'model.'前缀

        model.module.load_state_dict(new_state_dict)
        logger.info(f"成功加载权重: {checkpoint_path}")
    else:
        logger.error(f"找不到权重文件: {checkpoint_path}")
        return

    # 3. 创建训练集数据加载器（不使用数据增强）
    logger.info("===> 准备训练集数据...")
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])

    # 创建训练集（禁用所有增强）
    train_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET,  # 使用训练集路径
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,  # 禁用多尺度
        flip=False,  # 禁用翻转
        ignore_label=config.TRAIN.IGNORE_LABEL,
        crop_size=test_size,  # 使用测试尺寸
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # 评估时不打乱
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # 4. 初始化评估指标
    num_classes = config.DATASET.NUM_CLASSES
    intersection = np.zeros(num_classes)  # 交集
    union = np.zeros(num_classes)  # 并集
    total_pixels = 0  # 总像素数
    correct_pixels = 0  # 正确像素数

    # 5. 开始评估
    logger.info("===== 开始评估训练集 =====")
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for batch_idx, batch_data in enumerate(train_loader):
            # 处理数据格式：数据集可能返回多于2个值
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) >= 2:
                    images = batch_data[0]  # 第一个是图像
                    labels = batch_data[1]  # 第二个是标签
                else:
                    logger.error(f"数据格式错误: 只有{len(batch_data)}个元素")
                    continue
            else:
                logger.error(f"数据格式错误: 不是list或tuple")
                continue

            # 移动数据到GPU
            images = images.cuda()
            labels = labels.cuda()

            # 前向传播
            outputs = model(images)

            # 处理模型输出 - 修复点
            # HRNet-OCR可能返回多个输出，通常第一个是主输出
            if isinstance(outputs, (list, tuple)):
                # 如果是列表，取第一个元素作为主输出
                main_output = outputs[0]
                logger.debug(f"输出类型: list/tuple, 长度: {len(outputs)}")
            else:
                main_output = outputs
                logger.debug(f"输出类型: {type(outputs)}")

            # 检查main_output是否有shape属性
            if not hasattr(main_output, 'shape'):
                logger.error(f"输出没有shape属性: {type(main_output)}")
                continue

            # 调整标签尺寸以匹配输出（如果需要）
            if main_output.shape[2:] != labels.shape[1:]:
                labels = F.interpolate(
                    labels.float().unsqueeze(1),
                    size=main_output.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()

            # 获取预测结果（每个像素的类别）
            preds = torch.argmax(main_output, dim=1)

            # 转换到numpy用于计算
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # 计算像素准确率
            correct_pixels += (preds_np == labels_np).sum()
            total_pixels += labels_np.size

            # 计算每个类别的IoU
            for cls in range(num_classes):
                pred_cls = (preds_np == cls)
                label_cls = (labels_np == cls)
                intersection[cls] += np.logical_and(pred_cls, label_cls).sum()
                union[cls] += np.logical_or(pred_cls, label_cls).sum()

            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"处理进度: {batch_idx + 1}/{len(train_loader)}")

    # 6. 计算并打印最终结果
    logger.info("\n" + "=" * 50)
    logger.info("评估结果:")
    logger.info("=" * 50)

    # 计算像素准确率
    if total_pixels > 0:
        pixel_acc = correct_pixels / total_pixels
        logger.info(f"像素准确率: {pixel_acc:.4f}")
    else:
        logger.warning("总像素数为0，无法计算准确率")

    # 计算每个类别的IoU
    ious = intersection / (union + 1e-10)
    logger.info(f"平均IoU: {np.mean(ious):.4f}")

    # 打印每个类别的IoU
    logger.info("各类别IoU:")
    for i, iou in enumerate(ious):
        logger.info(f"  类别{i}: {iou:.4f}")

    # 打印所有IoU值（便于复制）
    iou_str = ' '.join([f'{iou:.4f}' for iou in ious])
    logger.info(f"IoU值序列: {iou_str}")


if __name__ == '__main__':
    main()