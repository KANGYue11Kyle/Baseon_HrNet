"""
HRNet+OCR 训练曲线可视化脚本
从训练日志中提取 Loss、mIoU、各类别IoU等指标并绘制曲线
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
FIG_SIZE = (14, 10)

# 类别名称（根据你的输出顺序）
class_names = [
    'background', 'tool1_manipulator', 'tool1_shaft',
    'tool2_manipulator', 'tool2_shaft', 'tool3_manipulator',
    'tool3_shaft', 'tool4_manipulator', 'tool4_shaft', 'uterus'
]

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


def plot_curves(data, output_dir, zoom_range=None):
    """绘制曲线"""

    has_val = len(data['val']['epochs']) > 0

    if has_val:
        fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 训练 Loss 曲线
    ax1 = axes[0, 0] if has_val else axes[0]
    if data['train']['losses']:
        # 按iteration绘制
        losses = smooth_data(data['train']['losses'])
        ax1.plot(data['train']['iter_nums'], losses, 'b-', linewidth=1.5, alpha=0.7, label='Training Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # 2. 训练 Loss 按Epoch绘制
    ax2 = axes[0, 1] if has_val else axes[1]
    if data['train']['losses']:
        # 按epoch绘制（每个epoch取平均）
        epoch_losses = {}
        for epoch, loss in zip(data['train']['epochs'], data['train']['losses']):
            if epoch not in epoch_losses:
                epoch_losses[epoch] = []
            epoch_losses[epoch].append(loss)

        epochs = sorted(epoch_losses.keys())
        avg_losses = [np.mean(epoch_losses[e]) for e in epochs]

        ax2.plot(epochs, avg_losses, 'g-o', linewidth=1.5, markersize=3, label='Avg Loss per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss per Epoch')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    if has_val:
        # 3. mIoU 曲线
        ax3 = axes[1, 0]
        if data['val']['mious']:
            epochs = data['val']['epochs']
            mious = data['val']['mious']
            ax3.plot(epochs, mious, 'r-o', linewidth=1.5, markersize=5, label='Validation mIoU')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('mIoU (%)')
            ax3.set_title('Validation mIoU Curve')
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            # 标注最佳 mIoU
            best_idx = np.argmax(mious)
            best_epoch = epochs[best_idx]
            best_miou = mious[best_idx]
            ax3.plot(best_epoch, best_miou, 'r*', markersize=12, label=f'Best: {best_miou:.2f}%')
            ax3.annotate(f'Best: {best_miou:.2f}% @ Epoch {best_epoch}',
                         xy=(best_epoch, best_miou), xytext=(best_epoch + 2, best_miou + 2),
                         arrowprops=dict(arrowstyle='->', color='red'))

        # 4. 验证 Loss 曲线
        ax4 = axes[1, 1]
        if data['val']['losses']:
            val_losses = smooth_data(data['val']['losses'])
            ax4.plot(data['val']['epochs'], val_losses, 'orange', linewidth=1.5, marker='s', markersize=4,
                     label='Validation Loss')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Validation Loss Curve')
            ax4.grid(True, alpha=0.3)
            ax4.legend()

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'training_curves_full.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir}/training_curves_full.png")

    # ========== 图2：mIoU 峰值附近放大图 ==========
    if has_val and zoom_range and data['val']['mious']:
        fig, ax = plt.subplots(figsize=(12, 6))

        epochs = data['val']['epochs']
        mious = data['val']['mious']

        # 筛选范围内的点
        zoom_epochs = []
        zoom_mious = []
        for i, miou in enumerate(mious):
            if zoom_range[0] <= miou <= zoom_range[1]:
                zoom_epochs.append(epochs[i])
                zoom_mious.append(miou)

        ax.plot(epochs, mious, 'b-', linewidth=1.5, alpha=0.5, label='All mIoU')
        ax.plot(zoom_epochs, zoom_mious, 'r-o', linewidth=2, markersize=6,
                label=f'Zoom ({zoom_range[0]}-{zoom_range[1]}%)')

        # 标注峰值
        best_idx = np.argmax(mious)
        best_epoch = epochs[best_idx]
        best_miou = mious[best_idx]
        ax.plot(best_epoch, best_miou, 'g*', markersize=15, label=f'Peak: {best_miou:.2f}% @ Epoch {best_epoch}')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('mIoU (%)')
        ax.set_title(f'mIoU Curve - Zoomed View ({zoom_range[0]}-{zoom_range[1]}%)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'miou_zoom.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_dir}/miou_zoom.png")

    # ========== 图3：各类别 IoU 变化曲线 ==========
    if data['val']['class_ious']:
        fig, ax = plt.subplots(figsize=(12, 8))

        epochs = data['val']['epochs']
        class_ious_by_epoch = {}

        for i, ious in enumerate(data['val']['class_ious']):
            epoch = epochs[i] if i < len(epochs) else i
            for c_idx, iou in enumerate(ious):
                if c_idx not in class_ious_by_epoch:
                    class_ious_by_epoch[c_idx] = {'epochs': [], 'ious': []}
                class_ious_by_epoch[c_idx]['epochs'].append(epoch)
                class_ious_by_epoch[c_idx]['ious'].append(iou * 100)  # 转换为百分比

        # 只绘制非零的类别
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for c_idx, data in class_ious_by_epoch.items():
            if c_idx < len(class_names) and max(data['ious']) > 5:  # 忽略最大IoU<5%的类别
                ax.plot(data['epochs'], data['ious'], '-', linewidth=1.5,
                        color=colors[c_idx % len(colors)], label=class_names[c_idx])

        ax.set_xlabel('Epoch')
        ax.set_ylabel('IoU (%)')
        ax.set_title('Per-Class IoU Over Training')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'per_class_iou_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_dir}/per_class_iou_curves.png")

    # ========== 图4：最新各类别 IoU 柱状图 ==========
    if data['val']['class_ious']:
        fig, ax = plt.subplots(figsize=(12, 6))

        latest_ious = data['val']['class_ious'][-1]
        latest_ious_percent = [iou * 100 for iou in latest_ious]

        x = np.arange(len(class_names))
        bars = ax.bar(x, latest_ious_percent, color='steelblue')

        ax.set_xlabel('Class')
        ax.set_ylabel('IoU (%)')
        ax.set_title(f'Per-Class IoU (Latest Validation - Epoch {data["val"]["epochs"][-1]})')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, iou in zip(bars, latest_ious_percent):
            if iou > 0:
                ax.annotate(f'{iou:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, iou),
                            xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'latest_per_class_iou.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_dir}/latest_per_class_iou.png")

    # ========== 图5：mIoU 和 Loss 联合图 ==========
    if has_val and data['val']['mious'] and data['val']['losses']:
        fig, ax1 = plt.subplots(figsize=(12, 6))

        epochs = data['val']['epochs']
        mious = data['val']['mious']

        ax1.plot(epochs, mious, 'b-o', linewidth=1.5, markersize=5, label='mIoU (%)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('mIoU (%)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        val_losses = smooth_data(data['val']['losses'])
        ax2.plot(epochs, val_losses, 'r-s', linewidth=1.5, markersize=4, label='Validation Loss')
        ax2.set_ylabel('Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title('mIoU and Validation Loss vs Epoch')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'miou_and_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_dir}/miou_and_loss.png")


def print_summary(data):
    """打印统计摘要"""
    print("\n" + "=" * 60)
    print("HRNet+OCR 训练统计摘要")
    print("=" * 60)

    if data['train']['losses']:
        print(f"\n训练记录数: {len(data['train']['losses'])}")
        print(f"最终训练 Loss: {data['train']['losses'][-1]:.4f}")

        # 最佳训练 Loss
        best_idx = np.argmin(data['train']['losses'])
        print(f"最佳训练 Loss: {data['train']['losses'][best_idx]:.4f}")

    if data['val']['mious']:
        best_idx = np.argmax(data['val']['mious'])
        print(f"\n最佳 mIoU: {data['val']['mious'][best_idx]:.2f}% @ Epoch {data['val']['epochs'][best_idx]}")
        print(f"对应 Validation Loss: {data['val']['losses'][best_idx]:.4f}")

        # 最新验证结果
        print(f"\n最新验证结果 @ Epoch {data['val']['epochs'][-1]}:")
        print(f"  mIoU: {data['val']['mious'][-1]:.2f}%")
        print(f"  Loss: {data['val']['losses'][-1]:.4f}")

    if data['val']['class_ious']:
        print("\n最新各类别 IoU:")
        latest_ious = data['val']['class_ious'][-1]
        for i, (name, iou) in enumerate(zip(class_names, latest_ious)):
            print(f"  {name}: {iou * 100:.2f}%")


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

    # 打印各轮验证的mIoU
    if data['val']['mious']:
        print("\n验证结果记录:")
        for epoch, miou in zip(data['val']['epochs'], data['val']['mious']):
            print(f"  Epoch {epoch}: mIoU = {miou:.2f}%")

    # 绘制曲线
    zoom_range = (ZOOM_MIOU_MIN, ZOOM_MIOU_MAX) if SHOW_ZOOM else None
    plot_curves(data, output_dir, zoom_range)

    # 打印统计摘要
    print_summary(data)

    print(f"\n所有图片已保存到: {output_dir}")