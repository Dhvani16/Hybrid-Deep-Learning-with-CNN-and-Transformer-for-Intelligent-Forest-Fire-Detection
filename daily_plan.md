# Day-by-Day Execution Plan
## Hybrid Deep Learning with CNN and Transformer for Intelligent Forest Fire Detection
### Duration: 4 Weeks (Sep 8 – Oct 5, 2026)

---

# WEEK 1 — Foundation, Data & Setup
> Goal: Have a clean, augmented dataset ready to feed into a model by end of week.

---

## Day 1 (Sep 8) — Literature Review Part 1

**Goal:** Understand the problem space and existing solutions.

### What to Do:
1. Open Google Scholar / IEEE Xplore / arXiv
2. Search these exact queries and download 5–7 papers:
   - `"forest fire detection deep learning CNN"` — find 2–3 papers
   - `"vision transformer image classification fire"` — find 2 papers
   - `"hybrid CNN transformer image classification"` — find 2 papers
3. For each paper, note in a text file:
   - What dataset did they use?
   - What accuracy/F1 did they achieve?
   - What was their main limitation?
4. Recommended papers to find:
   - "FireNet" or similar CNN-based fire detection papers
   - "ViT: An Image is Worth 16x16 Words" (Dosovitskiy et al.) — foundational ViT paper
   - Any paper combining CNN + Transformer for classification
5. Create a folder: `research/papers/` and save PDFs there

### Output:
- A `research/notes.md` file with a table: Paper Title | Method | Dataset | Accuracy | Limitation

---

## Day 2 (Sep 9) — Literature Review Part 2 + Research Gap

**Goal:** Finish reading, identify your research gap, finalize problem statement.

### What to Do:
1. Read remaining papers from Day 1 (skim abstract, intro, results sections — don't read every word)
2. Complete your notes table from Day 1
3. Write a short paragraph (5–8 sentences) answering:
   - What do CNN-only methods miss? (Answer: global context, long-range dependencies)
   - What do Transformer-only methods miss? (Answer: fine-grained local texture, need lots of data)
   - How does your hybrid solve both? (Answer: CNN extracts local features → Transformer captures global context)
4. This paragraph becomes your **Introduction / Motivation** section in the report later — save it
5. List the 3 public datasets you will use (confirm they are downloadable):
   - Kaggle Forest Fire Dataset: https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset
   - D-Fire Dataset: https://github.com/gaiasd/DFireDataset
   - FLAME Dataset (optional): search "FLAME dataset fire detection"

### Output:
- Completed `research/notes.md`
- A `research/problem_statement.md` with your motivation paragraph and dataset list

---

## Day 3 (Sep 10) — Dataset Download + Folder Structure

**Goal:** Have all raw data downloaded and organized.

### What to Do:

**1. Set up your project folder structure:**
```
forest_fire_detection/
├── data/
│   ├── raw/
│   │   ├── fire/
│   │   └── no_fire/
│   ├── processed/
│   │   ├── train/
│   │   │   ├── fire/
│   │   │   └── no_fire/
│   │   ├── val/
│   │   │   ├── fire/
│   │   │   └── no_fire/
│   │   └── test/
│   │       ├── fire/
│   │       └── no_fire/
├── models/
├── notebooks/
├── outputs/
│   ├── checkpoints/
│   ├── plots/
│   └── results/
└── src/
```

**2. Install Kaggle API and download dataset:**
```bash
pip install kaggle
# Place your kaggle.json API key in ~/.kaggle/
kaggle datasets download -d elmadafri/the-wildfire-dataset
unzip the-wildfire-dataset.zip -d data/raw/
```

**3. Also download D-Fire from GitHub:**
```bash
git clone https://github.com/gaiasd/DFireDataset
```

**4. Check your data:**
```python
import os
fire_count = len(os.listdir('data/raw/fire'))
no_fire_count = len(os.listdir('data/raw/no_fire'))
print(f"Fire images: {fire_count}, No-fire images: {no_fire_count}")
```
- You want at least 1000 images per class; ideally 2000+
- Note the class imbalance ratio — you'll handle it on Day 5

### Output:
- Raw images in `data/raw/fire/` and `data/raw/no_fire/`
- Project folder structure created

---

## Day 4 (Sep 11) — QGIS Preprocessing + Data Splitting

**Goal:** Do one round of GIS work, then split dataset into train/val/test.

### Part A — QGIS (2–3 hours)
1. Install QGIS: https://qgis.org/en/site/forusers/download.html
2. Open QGIS → New Project
3. Add a base layer: `Layer → Add Layer → Add XYZ Tile Layer` → use OpenStreetMap
4. Get fire GPS coordinates from NASA FIRMS:
   - Go to https://firms.modaps.eosdis.nasa.gov/
   - Download CSV of fire detections for your region of interest (e.g., last 7 days, global)
   - In QGIS: `Layer → Add Delimited Text Layer` → load the CSV → set X = longitude, Y = latitude
5. You will see fire hotspot points plotted on the map
6. Take a screenshot of the map — this becomes **Figure 1** in your report (spatial distribution of fire data)
7. Export the map as PNG: `Project → Import/Export → Export Map to Image`

### Part B — Train/Val/Test Split (1–2 hours)
```python
import os, shutil, random

def split_dataset(src_dir, dest_dir, split=(0.70, 0.15, 0.15)):
    classes = ['fire', 'no_fire']
    for cls in classes:
        images = os.listdir(os.path.join(src_dir, cls))
        random.seed(42)
        random.shuffle(images)
        n = len(images)
        train_end = int(n * split[0])
        val_end = train_end + int(n * split[1])
        splits = {
            'train': images[:train_end],
            'val':   images[train_end:val_end],
            'test':  images[val_end:]
        }
        for split_name, files in splits.items():
            out = os.path.join(dest_dir, split_name, cls)
            os.makedirs(out, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(src_dir, cls, f), os.path.join(out, f))
    print("Split complete.")

split_dataset('data/raw', 'data/processed')
```
Run this script and verify counts:
```python
for split in ['train', 'val', 'test']:
    for cls in ['fire', 'no_fire']:
        count = len(os.listdir(f'data/processed/{split}/{cls}'))
        print(f"{split}/{cls}: {count} images")
```

### Output:
- QGIS screenshot saved to `outputs/plots/qgis_fire_map.png`
- `data/processed/` fully populated with train/val/test splits

---

## Day 5 (Sep 12) — Environment Setup + Data Augmentation Pipeline

**Goal:** Get PyTorch environment running and write the full data loading + augmentation code.

### Part A — Environment Setup
```bash
# If using Google Colab — skip this, Colab has everything
# If using local machine:
pip install torch torchvision timm albumentations opencv-python scikit-learn matplotlib seaborn tqdm
```
- Verify GPU:
```python
import torch
print(torch.cuda.is_available())       # Should print True
print(torch.cuda.get_device_name(0))   # Should print your GPU name
```

### Part B — Write dataset.py
Create `src/dataset.py`:
```python
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class FireDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224):
        self.root = os.path.join(root_dir, split)
        self.classes = ['no_fire', 'fire']   # 0 = no_fire, 1 = fire
        self.samples = []
        for label, cls in enumerate(self.classes):
            folder = os.path.join(self.root, cls)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(folder, fname), label))

        if split == 'train':
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.2),   # simulates smoke/haze
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert('RGB'))
        img = self.transform(image=img)['image']
        return img, label

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = FireDataset(data_dir, split=split, img_size=img_size)
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split == 'train'),
                                    num_workers=2, pin_memory=True)
    return loaders
```
- Test it: run `get_dataloaders('data/processed')` and iterate one batch to confirm shapes

### Output:
- `src/dataset.py` working and tested
- One sample batch of shape `[32, 3, 224, 224]` confirmed

---

## Day 6 (Sep 13) — Visualize Augmented Samples + Class Balance Check

**Goal:** Verify your data pipeline visually and handle class imbalance.

### What to Do:

**1. Visualize augmented samples:**
```python
import matplotlib.pyplot as plt
from src.dataset import FireDataset
import numpy as np

ds = FireDataset('data/processed', split='train')
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    img, label = ds[i * 50]
    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) +
               np.array([0.485, 0.456, 0.406])).clip(0, 1)
    ax.imshow(img_np)
    ax.set_title(['No Fire', 'Fire'][label])
    ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/plots/augmented_samples.png')
plt.show()
```

**2. Check class balance:**
```python
from collections import Counter
labels = [s[1] for s in ds.samples]
print(Counter(labels))   # {0: XXXX, 1: XXXX}
```

**3. Handle imbalance (if ratio > 2:1):**
- Option A (recommended): Use weighted loss in training
```python
import torch
class_counts = [no_fire_count, fire_count]
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
# pass to: nn.CrossEntropyLoss(weight=weights.to(device))
```
- Option B: Use `WeightedRandomSampler` in DataLoader to oversample minority class

**4. Save a sample grid image** — this goes in your report as a data visualization figure

### Output:
- `outputs/plots/augmented_samples.png` saved
- Class weights tensor computed and noted for use in training

---

## Day 7 (Sep 14) — Week 1 Review + CNN Backbone Code

**Goal:** Review Week 1 progress, start writing the model code.

### Morning: Review Checklist
- [ ] 10+ papers read, notes saved
- [ ] Dataset downloaded and split (train/val/test)
- [ ] QGIS map screenshot saved
- [ ] `src/dataset.py` working
- [ ] Augmented samples visualized
- [ ] Class imbalance handled

### Afternoon: Write CNN Baseline Model
Create `src/model.py` — start with just the CNN baseline:
```python
import torch
import torch.nn as nn
import timm

class CNNBaseline(nn.Module):
    def __init__(self, num_classes=2, backbone='efficientnet_b4', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
```
- Test: `model = CNNBaseline(); out = model(torch.randn(2, 3, 224, 224)); print(out.shape)` → should be `[2, 2]`

### Output:
- `src/model.py` with `CNNBaseline` class tested and working

---

# WEEK 2 — Model Training & Hybrid Architecture
> Goal: Have both the baseline CNN and the hybrid model trained and saved by end of week.

---

## Day 8 (Sep 15) — Write Training Loop

**Goal:** Complete the full training script so you can train any model.

### Write `src/train.py`:
```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

def train_model(model, dataloaders, num_epochs=30, lr=1e-4,
                class_weights=None, save_dir='outputs/checkpoints', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, correct, total = 0.0, 0, 0

            for inputs, labels in tqdm(dataloaders[phase],
                                       desc=f"Epoch {epoch+1}/{num_epochs} [{phase}]"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            print(f"  {phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save(model.state_dict(),
                           os.path.join(save_dir, 'best_model.pth'))
                print(f"  *** Best model saved (val_acc={best_val_acc:.4f}) ***")

        scheduler.step()

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    return model, history
```

### Output:
- `src/train.py` complete and importable

---

## Day 9 (Sep 16) — Train CNN Baseline

**Goal:** Train EfficientNet-B4 baseline; log results.

### Steps:
1. Open `notebooks/01_baseline_training.ipynb` (or run as a script)
2. Run this notebook:
```python
from src.dataset import get_dataloaders
from src.model import CNNBaseline
from src.train import train_model
import torch

dataloaders = get_dataloaders('data/processed', batch_size=32)
model = CNNBaseline(backbone='efficientnet_b4', pretrained=True)

# Two-phase training:
# Phase 1: Freeze backbone, train head only (5 epochs)
for param in model.backbone.parameters():
    param.requires_grad = False
model, history1 = train_model(model, dataloaders, num_epochs=5, lr=1e-3,
                               save_dir='outputs/checkpoints/baseline')

# Phase 2: Unfreeze all, fine-tune (20 epochs)
for param in model.backbone.parameters():
    param.requires_grad = True
model, history2 = train_model(model, dataloaders, num_epochs=20, lr=1e-4,
                               save_dir='outputs/checkpoints/baseline')
```
3. This will take 1–3 hours depending on dataset size and GPU
4. While it trains, start drafting the Methodology section of your report

### Expected Results:
- Baseline CNN accuracy: ~88–93% on validation set
- Save the history dict for plotting later

### Output:
- `outputs/checkpoints/baseline/best_model.pth` saved
- Training history saved: `torch.save(history, 'outputs/results/baseline_history.pth')`

---

## Day 10 (Sep 17) — Design & Code the Hybrid Architecture

**Goal:** Build the CNN + Transformer hybrid model.

### Understanding the Architecture:
- CNN backbone extracts feature maps of shape `[B, C, H, W]` (e.g., `[32, 1792, 7, 7]` for EfficientNet-B4)
- We reshape this to `[B, num_patches, C]` where `num_patches = H * W = 49`
- Transformer encoder processes these 49 patch tokens with self-attention
- CLS token aggregates global information
- Final FC layer outputs class scores

### Write `src/hybrid_model.py`:
```python
import torch
import torch.nn as nn
import timm
import math

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)    # prepend CLS token
        x = self.encoder(x)
        return self.norm(x[:, 0])          # return only CLS token output

class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=2, backbone='efficientnet_b4',
                 num_heads=8, num_layers=4, dropout=0.3, pretrained=True):
        super().__init__()
        # CNN backbone — remove head and pooling to get feature maps
        self.cnn = timm.create_model(backbone, pretrained=pretrained,
                                     num_classes=0, global_pool='')
        feat_dim = self.cnn.num_features    # e.g., 1792 for EfficientNet-B4

        # Project CNN features to Transformer embedding dim
        embed_dim = 512
        self.proj = nn.Conv2d(feat_dim, embed_dim, kernel_size=1)

        # Positional embedding: 7*7 spatial positions + 1 CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # Extract CNN feature map: [B, feat_dim, H, W]
        feat = self.cnn(x)
        # Project to embed_dim: [B, embed_dim, H, W]
        feat = self.proj(feat)
        B, C, H, W = feat.shape
        # Flatten spatial dims to sequence: [B, H*W, embed_dim]
        tokens = feat.flatten(2).permute(0, 2, 1)
        # Add positional embedding (skip CLS position, added inside transformer)
        tokens = tokens + self.pos_embed[:, 1:tokens.size(1)+1, :]
        # Transformer: returns CLS token [B, embed_dim]
        cls_out = self.transformer(tokens)
        return self.classifier(cls_out)
```

- Test: `model = HybridCNNTransformer(); out = model(torch.randn(2, 3, 224, 224)); print(out.shape)` → `[2, 2]`

### Output:
- `src/hybrid_model.py` complete and tested

---

## Day 11 (Sep 18) — Train Hybrid Model (Phase 1)

**Goal:** Start training the hybrid model; complete phase 1 (frozen backbone).

### Steps:
```python
from src.dataset import get_dataloaders
from src.hybrid_model import HybridCNNTransformer
from src.train import train_model

dataloaders = get_dataloaders('data/processed', batch_size=16)  # smaller batch for hybrid
model = HybridCNNTransformer(pretrained=True)

# Phase 1: Freeze CNN backbone, train only Transformer + projection + head
for param in model.cnn.parameters():
    param.requires_grad = False

model, history_p1 = train_model(model, dataloaders, num_epochs=8, lr=5e-4,
                                  save_dir='outputs/checkpoints/hybrid')
```
- This phase should take 1–2 hours
- Monitor: val_acc should increase steadily; if it stagnates after epoch 3, reduce lr to 1e-4
- While training: write the Dataset section of your report

### Output:
- Phase 1 training complete, checkpoint saved

---

## Day 12 (Sep 19) — Train Hybrid Model (Phase 2 — Fine-tuning)

**Goal:** Fine-tune entire hybrid model end-to-end.

### Steps:
```python
# Unfreeze CNN backbone
for param in model.cnn.parameters():
    param.requires_grad = True

# Use a lower LR for fine-tuning (backbone gets 10x lower LR)
from torch.optim import AdamW
optimizer = AdamW([
    {'params': model.cnn.parameters(), 'lr': 1e-5},        # backbone: low LR
    {'params': list(model.proj.parameters()) +
               list(model.transformer.parameters()) +
               list(model.classifier.parameters()), 'lr': 1e-4}  # new layers: normal LR
], weight_decay=1e-2)
```
- Train for 20 more epochs using this custom optimizer (pass to train_model or modify it)
- Monitor val_acc and loss curves — if overfitting (val_loss increases while train_loss drops), increase dropout to 0.5
- Expected hybrid accuracy: ~91–95% (should beat baseline by 1–3%)

### Output:
- `outputs/checkpoints/hybrid/best_model.pth` — your best hybrid model saved
- Training history saved for both phases

---

## Day 13 (Sep 20) — Also Train ViT-Only Baseline (for full comparison)

**Goal:** Train a pure Transformer baseline so you have 3 models to compare.

### Steps:
```python
import timm

# ViT-B/16 pretrained on ImageNet
vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

# Fine-tune head only first
for name, param in vit_model.named_parameters():
    param.requires_grad = ('head' in name)

model_vit, hist_vit1 = train_model(vit_model, dataloaders, num_epochs=5, lr=1e-3,
                                     save_dir='outputs/checkpoints/vit')

# Then fine-tune all
for param in vit_model.parameters():
    param.requires_grad = True
model_vit, hist_vit2 = train_model(vit_model, dataloaders, num_epochs=15, lr=1e-5,
                                     save_dir='outputs/checkpoints/vit')
```
- Note: ViT-only often underperforms CNN on smaller datasets — that's expected and proves your point

### Output:
- ViT baseline trained and saved
- You now have 3 models: CNN-only, ViT-only, Hybrid

---

## Day 14 (Sep 21) — Week 2 Review + Plot Training Curves

**Goal:** Verify all models trained successfully; visualize training curves.

### Plot training curves for all models:
```python
import matplotlib.pyplot as plt

def plot_history(history, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title(f'{title} — Loss')
    ax1.legend(); ax1.set_xlabel('Epoch')

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title(f'{title} — Accuracy')
    ax2.legend(); ax2.set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

plot_history(history_cnn, 'CNN Baseline', 'outputs/plots/cnn_curves.png')
plot_history(history_hybrid, 'Hybrid CNN+Transformer', 'outputs/plots/hybrid_curves.png')
plot_history(history_vit, 'ViT Baseline', 'outputs/plots/vit_curves.png')
```

### Checklist:
- [ ] CNN baseline trained and saved
- [ ] Hybrid model trained and saved
- [ ] ViT baseline trained and saved
- [ ] Training curves plotted and saved

---

# WEEK 3 — Evaluation, Visualization & GIS
> Goal: Generate all results, figures, and maps needed for the report.

---

## Day 15 (Sep 22) — Full Evaluation on Test Set

**Goal:** Compute all evaluation metrics for all 3 models on the held-out test set.

### Write `src/evaluate.py`:
```python
import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)

def evaluate_model(model, dataloader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    metrics = {
        'accuracy':  accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall':    recall_score(all_labels, all_preds),
        'f1':        f1_score(all_labels, all_preds),
        'auc_roc':   roc_auc_score(all_labels, all_probs),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }
    print(classification_report(all_labels, all_preds,
                                 target_names=['No Fire', 'Fire']))
    return metrics
```

### Run for all 3 models:
```python
cnn_metrics    = evaluate_model(cnn_model,    dataloaders['test'])
hybrid_metrics = evaluate_model(hybrid_model, dataloaders['test'])
vit_metrics    = evaluate_model(vit_model,    dataloaders['test'])
```

### Build results comparison table:
```python
import pandas as pd
results = pd.DataFrame({
    'Model':     ['CNN (EfficientNet-B4)', 'ViT-B/16', 'Hybrid CNN+Transformer (Ours)'],
    'Accuracy':  [cnn_metrics['accuracy'], vit_metrics['accuracy'], hybrid_metrics['accuracy']],
    'Precision': [cnn_metrics['precision'], vit_metrics['precision'], hybrid_metrics['precision']],
    'Recall':    [cnn_metrics['recall'], vit_metrics['recall'], hybrid_metrics['recall']],
    'F1-Score':  [cnn_metrics['f1'], vit_metrics['f1'], hybrid_metrics['f1']],
    'AUC-ROC':  [cnn_metrics['auc_roc'], vit_metrics['auc_roc'], hybrid_metrics['auc_roc']],
})
results = results.round(4)
print(results.to_string(index=False))
results.to_csv('outputs/results/comparison_table.csv', index=False)
```

### Output:
- `outputs/results/comparison_table.csv` — this becomes **Table 1** in your report

---

## Day 16 (Sep 23) — Confusion Matrix + ROC Curve Plots

**Goal:** Generate publication-quality evaluation figures.

### Confusion Matrix for Hybrid Model:
```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = hybrid_metrics['confusion_matrix']
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Fire', 'Fire'],
            yticklabels=['No Fire', 'Fire'])
plt.title('Hybrid CNN+Transformer — Confusion Matrix')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('outputs/plots/hybrid_confusion_matrix.png', dpi=150)
```

### ROC Curves (all 3 models on one plot):
```python
from sklearn.metrics import roc_curve

plt.figure(figsize=(7, 5))
for name, metrics in [('CNN', cnn_metrics), ('ViT', vit_metrics), ('Hybrid (Ours)', hybrid_metrics)]:
    fpr, tpr, _ = roc_curve(metrics['labels'], metrics['probs'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={metrics['auc_roc']:.3f})")

plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curves — Model Comparison')
plt.legend(); plt.tight_layout()
plt.savefig('outputs/plots/roc_curves.png', dpi=150)
```

### Output:
- `outputs/plots/hybrid_confusion_matrix.png`
- `outputs/plots/roc_curves.png`
- Both figures go directly into your Results section

---

## Day 17 (Sep 24) — Grad-CAM Visualization

**Goal:** Show what the CNN part of your hybrid model is looking at.

### Install and use `grad-cam`:
```bash
pip install grad-cam
```

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.dataset import FireDataset

# Target the last conv layer of EfficientNet backbone
target_layers = [hybrid_model.cnn.blocks[-1][-1]]

cam = GradCAM(model=hybrid_model, target_layers=target_layers)

dataset = FireDataset('data/processed', split='test')
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Show 6 fire + 6 no-fire examples
samples_fire    = [s for s in dataset.samples if s[1] == 1][:6]
samples_no_fire = [s for s in dataset.samples if s[1] == 0][:6]

for idx, (path, label) in enumerate(samples_fire + samples_no_fire):
    img_orig = np.array(cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (224, 224)))
    img_norm = dataset.transform(image=img_orig)['image'].unsqueeze(0)
    grayscale_cam = cam(input_tensor=img_norm)[0]
    cam_image = show_cam_on_image(img_orig / 255.0, grayscale_cam, use_rgb=True)
    row, col = divmod(idx, 4)
    axes[row][col].imshow(cam_image)
    axes[row][col].set_title(['No Fire', 'Fire'][label])
    axes[row][col].axis('off')

plt.suptitle('Grad-CAM Visualizations — Hybrid CNN+Transformer', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/plots/gradcam.png', dpi=150)
plt.show()
```

### Output:
- `outputs/plots/gradcam.png` — becomes **Figure 3** in your report
- Verify the model highlights fire/smoke regions (red/orange areas), not background

---

## Day 18 (Sep 25) — Transformer Attention Map Visualization

**Goal:** Visualize what the Transformer is attending to globally.

### Extract attention weights from the Transformer:
```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def get_attention_maps(model, image_tensor, layer_idx=0):
    """Extract attention maps from a specific Transformer layer."""
    attention_maps = {}
    def hook_fn(module, input, output):
        # output shape: [B, num_heads, seq_len, seq_len]
        attention_maps['attn'] = output[1]  # attention weights

    # Register hook on the self-attention of the specified layer
    handle = model.transformer.encoder.layers[layer_idx].self_attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(image_tensor)
    handle.remove()
    return attention_maps.get('attn')

# Visualize for a sample fire image
sample_path = [s[0] for s in dataset.samples if s[1] == 1][0]
img_orig = np.array(cv2.resize(cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB), (224, 224)))
img_tensor = dataset.transform(image=img_orig)['image'].unsqueeze(0)

attn = get_attention_maps(hybrid_model, img_tensor, layer_idx=-1)
# attn shape: [1, num_heads, seq_len, seq_len]
# CLS token attention: attn[0, :, 0, 1:] → attention from CLS to each patch
cls_attn = attn[0, :, 0, 1:].mean(0)  # average over heads: [49]
cls_attn = cls_attn.reshape(7, 7).numpy()
cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())

# Upsample to 224x224
attn_map = cv2.resize(cls_attn, (224, 224))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_orig); axes[0].set_title('Original Image'); axes[0].axis('off')
axes[1].imshow(attn_map, cmap='hot'); axes[1].set_title('Attention Map'); axes[1].axis('off')
overlay = img_orig / 255.0 * 0.6 + plt.cm.hot(attn_map)[:, :, :3] * 0.4
axes[2].imshow(overlay); axes[2].set_title('Overlay'); axes[2].axis('off')
plt.suptitle('Transformer Self-Attention — CLS Token')
plt.tight_layout()
plt.savefig('outputs/plots/attention_maps.png', dpi=150)
```

### Output:
- `outputs/plots/attention_maps.png` — becomes **Figure 4** in your report

---

## Day 19 (Sep 26) — QGIS Fire Probability Heatmap

**Goal:** Use your model's predictions to create a spatial fire risk map in QGIS.

### Steps:

**1. Get a set of geo-tagged images or use NASA FIRMS coordinates:**
- Download fire hotspot CSV from FIRMS for a specific country/region
- Each row has: latitude, longitude, fire_radiative_power (FRP), confidence

**2. Create a prediction CSV:**
```python
import pandas as pd
import numpy as np

# Simulated: in practice use actual geo-tagged images with your model
# For report purposes: use FIRMS confidence values as proxy for fire probability
firms_df = pd.read_csv('data/MODIS_fire_data.csv')  # download from FIRMS
firms_df['fire_probability'] = firms_df['confidence'] / 100.0
firms_df[['latitude', 'longitude', 'fire_probability']].to_csv(
    'outputs/results/fire_predictions_geo.csv', index=False
)
```

**3. In QGIS:**
- `Layer → Add Delimited Text Layer` → load `fire_predictions_geo.csv`
- Set X = longitude, Y = latitude
- Right-click layer → Properties → Symbology → Graduated → set field to `fire_probability`
- Choose a color ramp: yellow → orange → red
- `Raster → Heatmap (Kernel Density Estimation)` → generate a density raster
- Overlay on OpenStreetMap basemap
- Export: `Project → Import/Export → Export Map to Image`
- Save as `outputs/plots/qgis_fire_probability_map.png`

### Output:
- `outputs/plots/qgis_fire_probability_map.png` — **Figure 5** in your report

---

## Day 20 (Sep 27) — Robustness & Failure Case Analysis

**Goal:** Test your model on hard cases and document what it gets wrong.

### Steps:

**1. Find failure cases:**
```python
# Get all test set wrong predictions
wrong_preds = [(path, true_label, pred)
               for (path, true_label), pred in zip(dataset.samples, hybrid_metrics['preds'])
               if true_label != pred]

print(f"Total errors: {len(wrong_preds)}")
print(f"False positives (predicted fire, was no_fire): {sum(1 for _,t,p in wrong_preds if t==0 and p==1)}")
print(f"False negatives (missed fire): {sum(1 for _,t,p in wrong_preds if t==1 and p==0)}")
```

**2. Visualize the worst mistakes:**
```python
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, (path, true, pred) in enumerate(wrong_preds[:10]):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    ax = axes[i // 5][i % 5]
    ax.imshow(img)
    ax.set_title(f"True: {['No Fire','Fire'][true]}\nPred: {['No Fire','Fire'][pred]}", color='red')
    ax.axis('off')
plt.suptitle('Failure Cases')
plt.tight_layout()
plt.savefig('outputs/plots/failure_cases.png', dpi=150)
```

**3. Categorize failures (manually look at images):**
- Red/orange skies mistaken for fire (false positive)
- Thick smoke with no visible flame (false negative)
- Night images with city lights (false positive)
- Document these as model limitations in your Discussion section

### Output:
- `outputs/plots/failure_cases.png`
- Notes on failure patterns → goes into Discussion section

---

## Day 21 (Sep 28) — Week 3 Review + Bar Chart Comparison

**Goal:** Generate the final comparison bar chart; verify all figures are saved.

### Bar chart for results comparison:
```python
import matplotlib.pyplot as plt
import numpy as np

models = ['CNN\n(EfficientNet-B4)', 'ViT-B/16', 'Hybrid\n(Ours)']
metrics_data = {
    'Accuracy':  [cnn_metrics['accuracy'], vit_metrics['accuracy'], hybrid_metrics['accuracy']],
    'F1-Score':  [cnn_metrics['f1'], vit_metrics['f1'], hybrid_metrics['f1']],
    'Recall':    [cnn_metrics['recall'], vit_metrics['recall'], hybrid_metrics['recall']],
    'AUC-ROC':  [cnn_metrics['auc_roc'], vit_metrics['auc_roc'], hybrid_metrics['auc_roc']],
}

x = np.arange(len(models))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
for i, (metric, values) in enumerate(metrics_data.items()):
    ax.bar(x + i * width, values, width, label=metric)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models)
ax.set_ylim(0.7, 1.0)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/plots/model_comparison_bar.png', dpi=150)
```

### Final figure checklist:
- [ ] `augmented_samples.png`
- [ ] `qgis_fire_map.png`
- [ ] `cnn_curves.png`, `hybrid_curves.png`, `vit_curves.png`
- [ ] `hybrid_confusion_matrix.png`
- [ ] `roc_curves.png`
- [ ] `gradcam.png`
- [ ] `attention_maps.png`
- [ ] `qgis_fire_probability_map.png`
- [ ] `failure_cases.png`
- [ ] `model_comparison_bar.png`

---

# WEEK 4 — Report Writing & Submission
> Goal: Write, proofread, and submit the complete research report.

---

## Day 22 (Sep 29) — Write: Abstract + Introduction + Literature Review

**Goal:** Complete the first three sections of the report.

### Abstract (150–200 words):
Cover: problem, existing limitations, proposed method, dataset used, key results (accuracy/F1), conclusion

### Introduction (1–1.5 pages):
- Paragraph 1: Why forest fire detection matters (cite statistics: acres burned, economic/ecological damage)
- Paragraph 2: Limitations of existing automated systems (diverse backgrounds, smoke, false positives)
- Paragraph 3: How deep learning has advanced fire detection (cite 3–4 papers from your notes)
- Paragraph 4: Gap — CNN-only lacks global context; ViT-only needs huge data
- Paragraph 5: Your contribution — hybrid architecture combining both strengths
- Paragraph 6: Paper organization ("The rest of this paper is organized as follows...")

### Literature Review (1–1.5 pages):
- Organize into 3 subsections: CNN-based methods, Transformer-based methods, Hybrid approaches
- For each paper: summarize method, dataset, accuracy, limitation (use your Day 1–2 notes)
- End with a table: Paper | Method | Dataset | Accuracy | Limitation

---

## Day 23 (Sep 30) — Write: Dataset + Methodology Sections

**Goal:** Document your data and architecture in detail.

### Dataset Section (0.5–1 page):
- Name each dataset, number of images per class, source URL
- Describe preprocessing: resize, normalize, augmentation techniques (list them)
- Include the augmented samples figure
- Include the QGIS fire map figure
- State the train/val/test split ratio and counts

### Methodology Section (1.5–2 pages):
- Draw the full architecture diagram (use draw.io or PowerPoint):
  - Input image → CNN backbone → Feature map → Projection layer → Patch tokens → Positional embedding → Transformer encoder blocks → CLS token → FC head → Output
- Describe each component:
  - CNN backbone: which model, pretrained on what, output dimensions
  - Projection layer: purpose (dimension reduction from 1792 → 512)
  - Transformer encoder: how many heads, how many layers, what is self-attention doing
  - CLS token: how it aggregates global context
  - Classifier head: dropout + linear
- Describe training procedure:
  - Two-phase training (freeze → fine-tune)
  - Loss function, optimizer, learning rate schedule
  - All hyperparameters in a table

---

## Day 24 (Oct 1) — Write: Experiments + Results Section

**Goal:** Present your results clearly and objectively.

### Experiments Section (0.5 page):
- List all baseline models and why you chose them
- State evaluation metrics and why each matters
- Briefly describe experimental setup: GPU, batch size, epochs, framework

### Results Section (1–1.5 pages):
- **Table 1:** Main results comparison (paste from `comparison_table.csv`)
- **Figure 1:** ROC curves
- **Figure 2:** Confusion matrix for hybrid model
- **Figure 3:** Training curves (loss + accuracy)
- **Figure 4:** Grad-CAM visualizations
- **Figure 5:** Transformer attention maps
- **Figure 6:** QGIS fire probability map
- **Figure 7:** Model comparison bar chart

For each figure, write 2–3 sentences explaining what it shows and what conclusion to draw.

---

## Day 25 (Oct 2) — Write: Discussion + Conclusion + References

**Goal:** Complete the full report draft.

### Discussion Section (1 page):
- What do the results mean? Why does the hybrid outperform CNN-only and ViT-only?
- Analyze failure cases: what types of images fool your model?
- Compare your results to numbers from the literature (are you better or worse? why?)
- Limitations: small dataset, binary classification only, no real-time evaluation

### Conclusion Section (0.5 page):
- Restate problem and your approach in 2–3 sentences
- Summarize key findings (e.g., "The hybrid model achieved X% F1, outperforming the CNN baseline by Y%")
- State future work (3 bullet points):
  - Real-time video stream detection
  - Multi-modal fusion (RGB + thermal)
  - Edge deployment for UAV/drone systems

### References:
- Use IEEE citation format
- Cite all papers from literature review
- Cite datasets (Kaggle, FLAME, NASA FIRMS)
- Cite PyTorch, timm, QGIS, albumentations

---

## Day 26 (Oct 3) — Architecture Diagram + Figure Polish

**Goal:** Create a clean architecture diagram and ensure all figures are publication-quality.

### Architecture Diagram (use draw.io — free at draw.io):
1. Go to https://app.diagrams.net/
2. Draw boxes for each component with arrows:
   - Input (224×224 RGB) → EfficientNet-B4 Backbone → Feature Map (7×7×1792) → 1×1 Conv Projection → Patch Tokens (49×512) → + Positional Embedding → Transformer Encoder (4 layers) → CLS Token (512-d) → Dropout → Linear → Output (Fire / No Fire)
3. Label every box with the output dimensions
4. Export as PNG: `outputs/plots/architecture_diagram.png`

### Figure Polish Checklist:
- All figures have titles
- All axes are labeled
- Resolution is at least 150 DPI
- Colors are consistent and distinguishable
- Font sizes are readable when embedded in a PDF

---

## Day 27 (Oct 4) — Full Proofread + GitHub Cleanup

**Goal:** Finalize the report and make code shareable.

### Report Proofread:
- Read the entire report from start to finish
- Check: does every figure have a caption? Is every figure referenced in the text?
- Check: does every claim have a citation?
- Check: is the abstract consistent with the results?
- Run spell check
- Ensure consistent formatting: font, heading sizes, figure numbering

### GitHub Cleanup:
```
repo structure:
├── README.md             ← How to run your code
├── requirements.txt      ← pip freeze > requirements.txt
├── notebooks/
│   ├── 01_baseline_training.ipynb
│   ├── 02_hybrid_training.ipynb
│   └── 03_evaluation_visualization.ipynb
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── hybrid_model.py
│   ├── train.py
│   └── evaluate.py
└── outputs/
    ├── plots/            ← all figures
    └── results/          ← CSVs and metrics
```
- Write a clear README.md:
  - What the project does
  - How to install dependencies
  - How to download the dataset
  - How to run training
  - How to reproduce results

---

## Day 28 (Oct 5) — Final Submission

**Goal:** Submit everything.

### Final Checklist:
- [ ] Report PDF exported and named correctly (e.g., `YourName_ForestFireDetection_Report.pdf`)
- [ ] All figures embedded in the report (no broken image links)
- [ ] GitHub repository link included in the report
- [ ] `requirements.txt` is accurate (test with a fresh `pip install -r requirements.txt`)
- [ ] All model checkpoint files accessible (or provide download link)
- [ ] Report page count meets requirements
- [ ] References formatted in IEEE style
- [ ] Submitted before deadline

---

## Key Tips for Staying on Track

| Risk | Mitigation |
|------|-----------|
| Training takes too long | Use Google Colab Pro (T4 GPU) or reduce dataset to 2000 images per class |
| Model not converging | Reduce learning rate by 10x; check data normalization |
| Low accuracy (<85%) | Try ResNet-50 backbone instead; increase augmentation; check class imbalance |
| QGIS takes too long | One simple heatmap is sufficient; don't over-engineer the GIS part |
| Running out of time | Skip ViT-only baseline; compare only CNN vs Hybrid |
| Colab session disconnects | Save checkpoints every 5 epochs; use Google Drive mounting |
