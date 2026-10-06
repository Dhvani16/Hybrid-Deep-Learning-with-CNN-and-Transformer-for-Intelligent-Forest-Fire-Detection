# Forest Fire Detection — Hybrid CNN + Transformer

A hybrid deep learning framework that combines CNN local feature extraction with Transformer global attention for accurate forest fire detection.

## Project Structure
```
forest_fire_detection/
├── src/
│   ├── dataset.py        # Data loading and augmentation pipeline
│   ├── model.py          # CNN baseline (EfficientNet-B4)
│   ├── hybrid_model.py   # Hybrid CNN + Transformer architecture
│   ├── train.py          # Training loop with two-phase fine-tuning
│   ├── evaluate.py       # Evaluation metrics
│   └── visualize.py      # Plots: Grad-CAM, attention maps, ROC, confusion matrix
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_hybrid_training.ipynb
│   └── 04_evaluation_visualization.ipynb
├── data/
│   └── processed/train|val|test/fire|no_fire/
├── outputs/
│   ├── checkpoints/
│   ├── plots/
│   └── results/
├── research/
│   ├── notes.md            # Literature review notes
│   └── problem_statement.md
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download one or more of these datasets and place images under `data/processed/`:
- **Kaggle Wildfire**: `kaggle datasets download -d elmadafri/the-wildfire-dataset`
- **D-Fire**: `git clone https://github.com/gaiasd/DFireDataset`

After downloading, run the split script in `notebooks/01_data_exploration.ipynb`.

## Training

### CNN Baseline
```python
from src.dataset import get_dataloaders
from src.model import CNNBaseline
from src.train import train_model

dataloaders = get_dataloaders('data/processed', batch_size=32)
model = CNNBaseline(backbone='efficientnet_b4', pretrained=True)

model.freeze_backbone()
model, _ = train_model(model, dataloaders, num_epochs=5, lr=1e-3,
                        save_dir='outputs/checkpoints/baseline')
model.unfreeze_backbone()
model, history = train_model(model, dataloaders, num_epochs=20, lr=1e-4,
                              save_dir='outputs/checkpoints/baseline')
```

### Hybrid Model
```python
from src.hybrid_model import HybridCNNTransformer

model = HybridCNNTransformer(pretrained=True)
model.freeze_backbone()
model, _ = train_model(model, dataloaders, num_epochs=8, lr=5e-4,
                        save_dir='outputs/checkpoints/hybrid')
model.unfreeze_backbone()
model, history = train_model(model, dataloaders, num_epochs=20, lr=1e-4,
                              backbone_lr_multiplier=0.1,
                              save_dir='outputs/checkpoints/hybrid')
```

## Evaluation

```python
from src.evaluate import evaluate_model, print_results_table

metrics = evaluate_model(model, dataloaders['test'])
```

## Results

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| CNN (EfficientNet-B4) | — | — | — |
| ViT-B/16 | — | — | — |
| **Hybrid CNN+Transformer (Ours)** | — | — | — |

*Fill in after training.*

## Technologies
Python · PyTorch · timm · albumentations · OpenCV · QGIS · Scikit-learn
