import os
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(
    model: nn.Module,
    dataloaders: dict,
    num_epochs: int = 30,
    lr: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None,
    save_dir: str = 'outputs/checkpoints',
    device: Optional[torch.device] = None,
    backbone_lr_multiplier: float = 1.0,   # set < 1 when fine-tuning backbone at lower LR
) -> tuple[nn.Module, dict]:
    """Train a model and return the best-checkpoint model and history dict."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    model = model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    # Optionally use different LR for backbone vs. new layers
    if backbone_lr_multiplier != 1.0 and hasattr(model, 'cnn'):
        param_groups = [
            {'params': model.cnn.parameters(), 'lr': lr * backbone_lr_multiplier},
            {'params': [p for n, p in model.named_parameters()
                        if not n.startswith('cnn')], 'lr': lr},
        ]
    else:
        param_groups = model.parameters()

    optimizer = AdamW(param_groups, lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_ckpt = os.path.join(save_dir, 'best_model.pth')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for phase in ('train', 'val'):
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=phase, leave=False):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc  = correct / total
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            print(f"  {phase:5s}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save(model.state_dict(), best_ckpt)
                print(f"  *** Checkpoint saved  (val_acc={best_val_acc:.4f}) ***")

        scheduler.step()

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    return model, history


def compute_class_weights(dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    from collections import Counter
    counts = Counter(label for _, label in dataset.samples)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / counts[i] for i in range(len(counts))], dtype=torch.float
    )
    return weights / weights.sum() * len(counts)   # normalize so weights sum to num_classes
