# Baseon_HrNet
# Surgical Image Segmentation Project Based on HRNet

A surgical image semantic segmentation project implemented based on [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation) (High-Resolution Network). It adopts the HRNet + OCR architecture and supports model training, validation, automatic checkpoint selection, and visualization.

## 📁 Repository File Description

```
.
├── models/                                        # HRNet-related model definition code
├── tools/                                         # Training, validation, and visualization utility scripts
├── best.pth                                       # Optimal checkpoint weight file automatically selected during training
├── hrnet.txt                                      # Text record of the results of this training run
├── seg_hrnet_ocr_w18_surgery_one.yaml             # Model training configuration file
├── vis1.png                                       # Visualization result image produced by model inference
└── vis2.png                                       # Visualization result image produced by model inference
```

## 🔍 Module Description

| File / Directory | Description |
| --- | --- |
| `models/` | Stores HRNet-related model definition code |
| `tools/` | Contains training, validation, and visualization utility scripts |
| `best.pth` | Optimal checkpoint binary weight file automatically selected during training |
| `hrnet.txt` | Text record of the results of this training run |
| `seg_hrnet_ocr_w18_surgery_one.yaml` | Model training configuration file |
| `vis1.png` / `vis2.png` | Visualization result images produced by model inference |

## 🧠 Model Architecture

- **Backbone**: HRNet-W18
- **Segmentation Head**: OCR (Object-Contextual Representations)
- **Task**: Surgical image semantic segmentation

## 🚀 Quick Start

### Environment Dependencies

> TODO: Add Python, PyTorch, CUDA versions and dependency installation commands

```bash
# Example
pip install -r requirements.txt
```

### Model Training

Launch training using the configuration file `seg_hrnet_ocr_w18_surgery_one.yaml`:

```bash
# Example (adjust the specific command according to the scripts under tools/)
python tools/train.py --cfg seg_hrnet_ocr_w18_surgery_one.yaml
```

During training, the optimal checkpoint is automatically selected and saved as `best.pth`.

### Model Validation / Inference

> TODO: Add validation and inference commands

```bash
# Example
python tools/val.py --cfg seg_hrnet_ocr_w18_surgery_one.yaml --weight best.pth
```

### Result Visualization

> TODO: Add instructions for invoking the visualization script

```bash
# Example
python tools/vis.py --cfg seg_hrnet_ocr_w18_surgery_one.yaml --weight best.pth
```

Visualization results are shown in `vis1.png` and `vis2.png`.

## 📊 Training Results

Training results are recorded in `hrnet.txt`.

> TODO: Key metrics (mIoU, Accuracy, etc.) and demonstrations of the results in `vis1.png` / `vis2.png` can be added here

| Visualization Results |
| --- |
| ![vis1](vis1.png) |
| ![vis2](vis2.png) |

## 🏆 Optimal Weights

The optimal checkpoint automatically selected during training has been saved as **`best.pth`**.


## 🙏 Acknowledgements

This project is implemented based on [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation). We thank the original authors for their open-source work.
