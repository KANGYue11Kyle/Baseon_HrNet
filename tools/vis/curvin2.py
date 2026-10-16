"""
HRNet+OCR 训练曲线可视化脚本
从训练日志中提取mIoU指标并绘制曲线
"""

import re
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ========== 配置参数 ==========
# 日志文件路径
log_file = r'F:\cuhk\5910\汇报\hrnet\hrnet.txt'  # 修改为你的日志路径

# 输出图片保存路径
output_dir = r'F:\download\HRNet-Semantic-Segmentation-HRNet-OCR\training_curves_hrnet'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# ========== 可调整参数 ==========
# mIoU 峰值附近放大显示范围（可选）
ZOOM_MIOU_MIN = 10  # 放大显示的 mIoU 最小值 (%)
ZOOM_MIOU_MAX = 25  # 放大显示的 mIoU 最大值 (%)
SHOW_ZOOM = True  # 是否显示放大图

# 平滑窗口大小（移动平均）
SMOOTH_WINDOW = 3  # 平滑窗口，1表示不平滑

# 图片尺寸
FIG_SIZE = (12, 6)

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================


def smooth_data(data, window=SMOOTH_WINDOW):
    """移动平均平滑"""
    if window <= 1:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    return smoothed


def parse_hrnet_log(log_path):
    """解析HRNet+OCR训练日志"""

    # 存储数据
    train_losses = []
    train_epochs = []
    train_iter_nums = []

    val_epochs = []
    val_mious = []
    val_losses = []
    val_class_ious = []  # 存储每轮的各类别IoU

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        # 匹配训练指标 (Epoch: [x/60] Iter:[y/255], Loss: z)
        train_match = re.search(r'Epoch:\s*\[(\d+)/\d+\]\s*Iter:\[(\d+)/\d+\].*?Loss:\s*([\d\.]+)', line)
        if train_match:
            epoch = int(train_match.group(1))
            iter_num = int(train_match.group(2))
            loss = float(train_match.group(3))
            train_epochs.append(epoch)
            train_iter_nums.append(iter_num)
            train_losses.append(loss)

        # 匹配验证结果（MeanIU 和 Loss）
        val_match = re.search(r'Loss:\s*([\d\.]+),\s*MeanIU:\s*([\d\.]+),\s*Best_mIoU:\s*([\d\.]+)', line)
        if val_match:
            val_losses.append(float(val_match.group(1)))
            val_mious.append(float(val_match.group(2)) * 100)  # 转换为百分比
            # 获取当前epoch（从最近的train记录）
            if train_epochs:
                val_epochs.append(train_epochs[-1])
            else:
                val_epochs.append(len(val_epochs))

        # 匹配各类别IoU数组（格式如 [0.4, 0.0, 0.0, ...]）
        class_match = re.search(r'\]\s*([\d\.\s]+)\]\s*([\d\.]+)', line)
        if class_match and 'MeanIU' not in line:
            # 提取数组
            array_str = re.findall(r'[\d\.]+', line)
            if len(array_str) >= 10:  # 至少有10个类别
                ious = [float(x) for x in array_str[:10]]
                val_class_ious.append(ious)

    return {
        'train': {
            'epochs': train_epochs,
            'iter_nums': train_iter_nums,
            'losses': train_losses
        },
        'val': {
            'epochs': val_epochs,
            'losses': val_losses,
            'mious': val_mious,
            'class_ious': val_class_ious
        }
    }


def plot_miou_curves(data, output_dir, zoom_range=None):
    """绘制mIoU曲线"""

    if not data['val']['mious']:
        print("未找到mIoU数据，无法绘制曲线")
        return

    epochs = data['val']['epochs']
    mious = data['val']['mious']

    # 1. 完整mIoU曲线图
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 绘制原始曲线
    ax.plot(epochs, mious, 'b-o', linewidth=1.5, markersize=5, label='Validation mIoU')

    # 绘制平滑曲线（可选）
    if SMOOTH_WINDOW > 1:
        smooth_mious = smooth_data(mious)
        ax.plot(epochs, smooth_mious, 'r--', linewidth=1.5, alpha=0.7,
                label=f'Smoothed mIoU (window={SMOOTH_WINDOW})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('mIoU (%)', fontsize=12)
    ax.set_title('Validation mIoU Curve', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # 标注最佳 mIoU
    best_idx = np.argmax(mious)
    best_epoch = epochs[best_idx]
    best_miou = mious[best_idx]
    ax.plot(best_epoch, best_miou, 'g*', markersize=15, label=f'Best: {best_miou:.2f}%')
    ax.annotate(f'Best: {best_miou:.2f}% @ Epoch {best_epoch}',
                xy=(best_epoch, best_miou),
                xytext=(best_epoch + 2, best_miou + 2),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green'))

    # 标注最新mIoU
    latest_epoch = epochs[-1]
    latest_miou = mious[-1]
    ax.plot(latest_epoch, latest_miou, 'r*', markersize=12, label=f'Latest: {latest_miou:.2f}%')
    ax.annotate(f'Latest: {latest_miou:.2f}% @ Epoch {latest_epoch}',
                xy=(latest_epoch, latest_miou),
                xytext=(latest_epoch - 8, latest_miou - 3),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'miou_curve_full.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存完整mIoU曲线: {output_dir}/miou_curve_full.png")

    # 2. mIoU 峰值附近放大图（可选）
    if zoom_range:
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        # 筛选范围内的点
        zoom_epochs = []
        zoom_mious = []
        for i, miou in enumerate(mious):
            if zoom_range[0] <= miou <= zoom_range[1]:
                zoom_epochs.append(epochs[i])
                zoom_mious.append(miou)

        if zoom_epochs:  # 只有当有数据在范围内时才绘制
            # 绘制完整曲线（半透明）
            ax.plot(epochs, mious, 'b-', linewidth=1, alpha=0.3, label='All mIoU')

            # 绘制放大区域
            ax.plot(zoom_epochs, zoom_mious, 'r-o', linewidth=2, markersize=6,
                    label=f'Zoom ({zoom_range[0]}-{zoom_range[1]}%)')

            # 标注峰值
            best_idx = np.argmax(mious)
            best_epoch = epochs[best_idx]
            best_miou = mious[best_idx]
            ax.plot(best_epoch, best_miou, 'g*', markersize=15,
                    label=f'Peak: {best_miou:.2f}% @ Epoch {best_epoch}')

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('mIoU (%)', fontsize=12)
            ax.set_title(f'mIoU Curve - Zoomed View ({zoom_range[0]}-{zoom_range[1]}%)', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            plt.tight_layout()
            plt.savefig(Path(output_dir) / 'miou_curve_zoom.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"已保存mIoU放大图: {output_dir}/miou_curve_zoom.png")
        else:
            print(f"警告: 在范围 {zoom_range} 内没有找到mIoU数据")

    # 打印最佳和最新mIoU信息
    print(f"\n最佳 mIoU: {best_miou:.2f}% @ Epoch {best_epoch}")
    print(f"最新 mIoU: {latest_miou:.2f}% @ Epoch {latest_epoch}")


def print_summary(data):
    """打印统计摘要"""
    print("\n" + "=" * 60)
    print("HRNet+OCR 训练统计摘要")
    print("=" * 60)

    if data['val']['mious']:
        epochs = data['val']['epochs']
        mious = data['val']['mious']

        # 统计信息
        print(f"\n验证次数: {len(mious)}")
        print(f"初始 mIoU: {mious[0]:.2f}% @ Epoch {epochs[0]}")
        print(f"最终 mIoU: {mious[-1]:.2f}% @ Epoch {epochs[-1]}")

        # 最佳 mIoU
        best_idx = np.argmax(mious)
        print(f"最佳 mIoU: {mious[best_idx]:.2f}% @ Epoch {epochs[best_idx]}")

        # 改进幅度
        improvement = mious[-1] - mious[0]
        print(f"总改进幅度: {improvement:+.2f}%")

        # 最新的几个epoch的mIoU
        print("\n最近5次验证结果:")
        recent_n = min(5, len(epochs))
        for i in range(-recent_n, 0):
            print(f"  Epoch {epochs[i]}: {mious[i]:.2f}%")


# ========== 主程序 ==========
if __name__ == '__main__':
    print(f"正在解析日志: {log_file}")

    if not Path(log_file).exists():
        print(f"错误: 日志文件不存在 - {log_file}")
        print("请修改 log_file 变量为正确的日志文件路径")
        exit(1)

    data = parse_hrnet_log(log_file)

    print(f"找到 {len(data['train']['losses'])} 条训练记录")
    print(f"找到 {len(data['val']['mious'])} 条验证记录")

    # 打印验证记录
    if data['val']['mious']:
        print("\nmIoU验证记录:")
        for epoch, miou in zip(data['val']['epochs'], data['val']['mious']):
            print(f"  Epoch {epoch}: mIoU = {miou:.2f}%")

    # 绘制mIoU曲线
    zoom_range = (ZOOM_MIOU_MIN, ZOOM_MIOU_MAX) if SHOW_ZOOM else None
    plot_miou_curves(data, output_dir, zoom_range)

    # 打印统计摘要
    print_summary(data)

    print(f"\n所有图片已保存到: {output_dir}")