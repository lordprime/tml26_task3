# Adversarial Robustness – Assignment 3

This repository contains our solution for the Adversarial Robustness task. We train a robust ResNet50 model using a Phased Adversarial Training approach.

## Team Information
- Abhishek Reddy Ramesh Reddy (7070144)
- Khushi Ankush Bhawkar (7072749)

## How to Reproduce Best Leaderboard Result

### Prerequisites
- Python 3.8+
- PyTorch >= 1.8 with CUDA support
- torchvision
- numpy

Install the required packages:
```bash
pip install torch torchvision numpy
```

### Training

To train the model and reproduce the best result locally, run:

```bash
python task_template.py
```

This script will run the full 200-epoch training process. It evaluates the model on a validation split and tracks the best combined score (50% clean accuracy + 50% robust accuracy).
Once training and SWA finalization are complete, it will save the state dictionary of the best model to:
- `model.pt`

**Note:** The script will automatically load the dataset from `train.npz` in the current directory.

### Method Summary
Our method combines several adversarial training techniques:
- **Architecture**: ResNet50 (modified to output 9 classes).
- **Phased Training Strategy**:
  - **Phase 1 (Epochs 0-9)**: Standard PGD Adversarial Training (PGD-AT) warmup for stability.
  - **Phase 2 (Epochs 10-159)**: TRADES loss combined with Adversarial Weight Perturbation (AWP) for high performance.
  - **Phase 3 (Epochs 160-199)**: TRADES + AWP combined with Stochastic Weight Averaging (SWA) for better generalization.
- **Augmentations**: Random Crop, Random Horizontal Flip, Cutout (size 16), and Label Smoothing (0.1).

### Key Hyperparameters
| Parameter | Value |
|---|---|
| Architecture | ResNet50 |
| Epochs | 200 |
| Batch Size | 128 |
| Initial LR | 0.1 (Decays at epoch 100 and 150) |
| Optimizer | SGD (Momentum 0.9, Weight Decay 5e-4) |
| ε (L∞) | 8/255 |
| PGD steps (train/eval) | 10/20 |
| TRADES β | 6.0 |
| AWP γ | 0.005 (Warmup 10 epochs) |
| SWA Start Epoch | 160 |
| SWA LR | 0.001 |

### Submission

To submit the trained model to the leaderboard, use the provided submission script:
```bash
python submission.py
```
Make sure `MODEL_NAME = "resnet50"` is set in `submission.py` and your API token is updated.

