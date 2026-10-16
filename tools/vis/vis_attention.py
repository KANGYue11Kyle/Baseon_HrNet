"""
可视化HRNet预测结果 - 输出元素多且准确率高的图片
- 输入：HRNet checkpoint文件
- 输出：前景占比高且准确率高的图片的可视化结果
"""

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

# ========== 请在这里填写你的路径 ==========
# 1. HRNet项目根目录
HRNET_ROOT = r'F:\download\HRNet-Semantic-Segmentation-HRNet-OCR'

# 2. 配置文件路径
CONFIG_FILE = r'F:\download\HRNet-Semantic-Segmentation-HRNet-OCR\experiments\surgery\seg_hrnet_ocr_w48_surgery.yaml'

# 3. 模型checkpoint路径
CHECKPOINT_FILE = r'C:\Users\DELL\Desktop\目前最佳模型80轮\checkpoint.pth.tar'

# 4. 输出目录
OUTPUT_DIR = r'F:\download\HRNet-Semantic-Segmentation-HRNet-OCR\visual_best_hrnet'

# 5. 测试集图片目录
TEST_IMG_DIR = r'F:\download\HRNet-Semantic-Segmentation-HRNet-OCR\data\surgery\images\test'

# 6. 测试集标注目录
TEST_GT_DIR = r'F:\download\HRNet-Semantic-Segmentation-HRNet-OCR\data\surgery\labels\test'

# 7. 类别数量
NUM_CLASSES = 10

# 8. 类别名称
CLASS_NAMES = [
    'background',           # 0: 背景
    'tool1_manipulator',    # 1: 器械1操作端
    'tool1_shaft',          # 2: 器械1杆身
    'tool2_manipulator',    # 3: 器械2操作端
    'tool2_shaft',          # 4: 器械2杆身
    'tool3_manipulator',    # 5: 器械3操作端
    'tool3_shaft',          # 6: 器械3杆身
    'tool4_manipulator',    # 7: 器械4操作端
    'tool4_shaft',          # 8: 器械4杆身
    'uterus',               # 9: 子宫
]

# ========== 参数设置 ==========
NUM_BEST_IMAGES = 10
NUM_SHOW = 15
USE_COLORFUL = True
WEIGHT_ACC = 0.6
WEIGHT_FOREGROUND = 0.4

# ========== 添加路径 ==========
sys.path.insert(0, HRNET_ROOT)
sys.path.insert(0, os.path.join(HRNET_ROOT, 'lib'))
sys.path.insert(0, os.path.join(HRNET_ROOT, 'tools'))

# 导入HRNet相关模块
import config as cfg_module
from config import update_config

# 从子模块导入模型
from models.seg_hrnet_ocr import get_seg_model

import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description='可视化HRNet预测结果')
    parser.add_argument('--cfg', type=str, default=CONFIG_FILE, help='config文件路径')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_FILE, help='模型checkpoint路径')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--num_best', type=int, default=NUM_BEST_IMAGES, help='输出图片数量')
    parser.add_argument('--num_show', type=int, default=NUM_SHOW, help='综合排序时展示的图片数量')
    parser.add_argument('--weight_acc', type=float, default=WEIGHT_ACC, help='准确率权重')
    parser.add_argument('--weight_foreground', type=float, default=WEIGHT_FOREGROUND, help='前景占比权重')
    parser.add_argument('--colorful', action='store_true', default=USE_COLORFUL, help='使用彩色调色板')
    parser.add_argument('--test_dir', type=str, default=TEST_IMG_DIR, help='测试集目录')
    parser.add_argument('--gt_dir', type=str, default=TEST_GT_DIR, help='标注目录')
    return parser.parse_args()


def create_colorful_palette(num_classes):
    """创建彩色调色板"""
    base_palette = [
        [0, 0, 0],           # 0: background - 黑色
        [255, 0, 0],         # 1: 红色
        [200, 100, 0],       # 2: 橙色
        [0, 255, 0],         # 3: 绿色
        [0, 200, 100],       # 4: 青绿
        [0, 0, 255],         # 5: 蓝色
        [100, 0, 200],       # 6: 紫色
        [255, 255, 0],       # 7: 黄色
        [255, 0, 255],       # 8: 洋红
        [0, 255, 255],       # 9: 青色
    ]
    return base_palette[:num_classes]


def apply_palette(mask, palette):
    """应用调色板"""
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        colored[mask == class_id] = color
    return colored


def get_test_images(test_dir):
    """获取测试集图片"""
    if os.path.exists(test_dir) and os.path.isdir(test_dir):
        images = list(Path(test_dir).glob('*.jpg')) + list(Path(test_dir).glob('*.JPG')) + \
                 list(Path(test_dir).glob('*.png')) + list(Path(test_dir).glob('*.PNG'))
        if len(images) > 0:
            return images
        print(f"警告: 目录 {test_dir} 中没有图片文件")

    if os.path.isfile(test_dir) and (test_dir.endswith('.jpg') or test_dir.endswith('.png')):
        return [Path(test_dir)]

    alt_paths = [
        os.path.join(HRNET_ROOT, 'data/surgery/images/testing'),
        os.path.join(HRNET_ROOT, 'data/surgery/images/validation'),
        'data/surgery/images/testing',
    ]

    for alt in alt_paths:
        if os.path.exists(alt) and os.path.isdir(alt):
            images = list(Path(alt).glob('*.jpg')) + list(Path(alt).glob('*.JPG'))
            if len(images) > 0:
                print(f"使用替代路径: {alt}")
                return images
    return []


def load_config(config_path):
    """加载配置文件"""
    print(f"加载配置文件: {config_path}")

    # 创建一个新的配置对象
    cfg = cfg_module.config

    # 更新配置
    cfg.merge_from_file(config_path)
    cfg.freeze()

    return cfg


def load_hrnet_model(checkpoint_path, cfg):
    """加载HRNet模型"""
    print("创建模型...")

    # 获取模型配置
    model = get_seg_model(cfg)
    print(f"模型创建成功，输出类别数: {cfg.MODEL.NUM_OUTPUTS}")

    if os.path.exists(checkpoint_path):
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0')

        if isinstance(checkpoint, dict):
            print(f"checkpoint keys: {list(checkpoint.keys())}")

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 移除module前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"缺失的键 (前10个): {missing_keys[:10]}")
        if unexpected_keys:
            print(f"多余的键 (前10个): {unexpected_keys[:10]}")

        print(f"模型参数加载完成")
    else:
        print(f"错误: checkpoint文件不存在: {checkpoint_path}")
        sys.exit(1)

    model = model.cuda()
    model.eval()
    return model


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    """获取仿射变换矩阵"""
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = np.array([np.cos(rot_rad), np.sin(rot_rad)], dtype=np.float32)
    dst_dir = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = dst_dir
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    """获取第三个点"""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def inference_hrnet(model, img_path, cfg):
    """HRNet推理 - 使用正确的预处理"""
    # 读取图片
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"无法读取图片: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]

    # 获取输入尺寸
    input_h, input_w = cfg.TRAIN.IMAGE_SIZE

    # 计算中心点和缩放
    c = np.array([w // 2, h // 2], dtype=np.float32)
    s = max(h, w) * 1.0

    # 获取仿射变换矩阵
    trans = get_affine_transform(c, s, 0, [input_w, input_h])

    # 应用仿射变换
    image_resized = cv2.warpAffine(image, trans, (input_w, input_h), flags=cv2.INTER_LINEAR)

    # 归一化
    image_resized = image_resized.astype(np.float32) / 255.0

    # 标准化（ImageNet均值和标准差）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_resized = (image_resized - mean) / std

    # 转换为tensor
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        image_tensor = image_tensor.cuda()
        outputs = model(image_tensor)

        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
        else:
            output = outputs

        # 上采样回原始尺寸
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
        pred = torch.argmax(output, dim=1)
        pred_mask = pred.cpu().numpy().squeeze().astype(np.uint8)

    return pred_mask


def load_gt_mask(img_path, gt_dir):
    """加载对应的真实掩码"""
    if not os.path.exists(gt_dir) or not os.path.isdir(gt_dir):
        if os.path.isfile(gt_dir):
            gt_mask = cv2.imread(gt_dir, cv2.IMREAD_GRAYSCALE)
            return gt_mask
        return None

    # 尝试多个可能的路径
    possible_paths = [
        Path(gt_dir) / f"{img_path.stem}.png",
        Path(gt_dir) / f"{img_path.stem}.jpg",
        Path(gt_dir) / f"{img_path.stem}.PNG",
        Path(gt_dir) / f"{img_path.stem}.JPG",
        Path(str(img_path).replace('images', 'annotations').replace('.jpg', '.png')),
        Path(str(img_path).replace('images', 'labels').replace('.jpg', '.png')),
    ]

    for mask_path in possible_paths:
        if mask_path.exists():
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                print(f"找到标注文件: {mask_path}")
                return gt_mask

    print(f"未找到标注文件: {img_path.stem}")
    return None


def compute_pixel_accuracy(pred_mask, gt_mask):
    """计算像素准确率"""
    if gt_mask is None:
        return 0.0
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    correct = np.sum(pred_mask == gt_mask)
    total = pred_mask.size
    return correct / total if total > 0 else 0.0


def compute_foreground_ratio(mask):
    """计算前景占比"""
    foreground = np.sum(mask > 0)
    total = mask.size
    return foreground / total if total > 0 else 0.0


def compute_composite_score(acc, fg_ratio, w_acc, w_fg):
    """计算综合评分"""
    return w_acc * acc + w_fg * fg_ratio


def visualize_prediction(img_info, save_path, palette_rgb):
    """可视化预测结果"""
    img_path = img_info['img_path']
    pred_mask = img_info['pred_mask']
    gt_mask = img_info['gt_mask']
    acc = img_info['acc']
    fg_ratio = img_info['fg_ratio']

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"警告: 无法读取图片 {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    colored_pred = apply_palette(pred_mask, palette_rgb)
    colored_gt = apply_palette(gt_mask, palette_rgb)
    overlay = cv2.addWeighted(img, 0.6, colored_pred, 0.4, 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Input Image', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(colored_gt)
    axes[0, 1].set_title(f'Ground Truth (前景占比: {fg_ratio * 100:.1f}%)', fontsize=12)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(colored_pred)
    axes[1, 0].set_title(f'HRNet Prediction (准确率: {acc * 100:.1f}%)', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay', fontsize=12)
    axes[1, 1].axis('off')

    plt.suptitle(f'{img_path.name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    print("=" * 60)
    print("HRNet 可视化工具")
    print("=" * 60)
    print(f"配置文件: {args.cfg}")
    print(f"模型权重: {args.checkpoint}")
    print(f"输出目录: {args.output_dir}")
    print(f"测试集目录: {args.test_dir}")
    print(f"标注目录: {args.gt_dir}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建调色板
    palette_rgb = create_colorful_palette(NUM_CLASSES)
    print("使用彩色调色板")

    # 获取测试图片
    test_images = get_test_images(args.test_dir)
    print(f"找到 {len(test_images)} 张测试图片")

    if len(test_images) == 0:
        print("错误: 没有找到任何测试图片！")
        print(f"请检查目录: {args.test_dir}")
        sys.exit(1)

    # 加载配置
    print("\n加载配置...")
    try:
        cfg = load_config(args.cfg)
        print("配置加载成功")
        print(f"输入图像尺寸: {cfg.TRAIN.IMAGE_SIZE}")
        print(f"类别数量: {cfg.DATASET.NUM_CLASSES}")
    except Exception as e:
        print(f"加载配置出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 加载模型
    print("加载模型中...")
    model = load_hrnet_model(args.checkpoint, cfg)
    print("模型加载完成！")

    # 计算每张图片的指标
    print("\n计算每张图片的指标...")
    image_scores = []

    for img_path in tqdm(test_images, desc="评估中"):
        try:
            # 加载真实掩码
            gt_mask = load_gt_mask(img_path, args.gt_dir)
            if gt_mask is None:
                print(f"  跳过 {img_path.name}: 没有找到标注文件")
                continue

            # 推理
            pred_mask = inference_hrnet(model, img_path, cfg)

            # 计算指标
            acc = compute_pixel_accuracy(pred_mask, gt_mask)
            fg_ratio = compute_foreground_ratio(gt_mask)
            composite_score = compute_composite_score(acc, fg_ratio, args.weight_acc, args.weight_foreground)

            print(f"  {img_path.name}: acc={acc:.3f}, fg_ratio={fg_ratio:.3f}, score={composite_score:.3f}")

            image_scores.append({
                'img_path': img_path,
                'pred_mask': pred_mask,
                'gt_mask': gt_mask,
                'acc': acc,
                'fg_ratio': fg_ratio,
                'composite_score': composite_score
            })
        except Exception as e:
            print(f"处理 {img_path.name} 时出错: {e}")
            import traceback
            traceback.print_exc()

    if len(image_scores) == 0:
        print("错误: 没有成功处理任何图片！")
        sys.exit(1)

    # 排序并输出最佳图片
    image_scores.sort(key=lambda x: x['composite_score'], reverse=True)
    best_images = image_scores[:args.num_best]

    print(f"\n综合评分最高的 {args.num_best} 张图片:")
    print(f"{'排名':<4} {'文件名':<35} {'准确率':<10} {'前景占比':<10} {'综合评分':<10}")
    print("-" * 75)
    for i, img in enumerate(best_images):
        print(f"{i + 1:<4} {img['img_path'].name:<35} {img['acc'] * 100:<10.2f}% "
              f"{img['fg_ratio'] * 100:<10.2f}% {img['composite_score'] * 100:<10.2f}%")

    # 生成可视化结果
    print("\n生成可视化结果...")
    for i, img_info in enumerate(best_images):
        acc = img_info['acc']
        fg = img_info['fg_ratio']
        save_path = os.path.join(args.output_dir,
                                 f'{i + 1:02d}_{img_info["img_path"].stem}_acc_{acc * 100:.1f}_fg_{fg * 100:.1f}.jpg')
        visualize_prediction(img_info, save_path, palette_rgb)
        print(f"  已保存: {os.path.basename(save_path)}")

    # 统计摘要
    avg_acc = sum([s['acc'] for s in image_scores]) / len(image_scores)
    avg_fg = sum([s['fg_ratio'] for s in image_scores]) / len(image_scores)

    print(f"\n📊 统计摘要:")
    print(f"  总图片数: {len(image_scores)}")
    print(f"  平均准确率: {avg_acc * 100:.2f}%")
    print(f"  平均前景占比: {avg_fg * 100:.2f}%")
    print(f"  最高准确率: {image_scores[0]['acc'] * 100:.2f}%")
    print(f"  最高前景占比: {max(image_scores, key=lambda x: x['fg_ratio'])['fg_ratio'] * 100:.2f}%")

    print(f"\n✅ 完成！结果保存至: {args.output_dir}")


if __name__ == '__main__':
    main()