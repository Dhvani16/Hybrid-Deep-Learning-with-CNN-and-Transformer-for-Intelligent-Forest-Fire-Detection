from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    class_names: list = None,
) -> dict:
    """Run inference on dataloader and return a dict of all evaluation metrics."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if class_names is None:
        class_names = ['No Fire', 'Fire']

    model.eval().to(device)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    metrics = {
        'accuracy':         accuracy_score(all_labels, all_preds),
        'precision':        precision_score(all_labels, all_preds, zero_division=0),
        'recall':           recall_score(all_labels, all_preds, zero_division=0),
        'f1':               f1_score(all_labels, all_preds, zero_division=0),
        'auc_roc':          roc_auc_score(all_labels, all_probs),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'fpr_tpr':          roc_curve(all_labels, all_probs),
        'preds':            all_preds,
        'labels':           all_labels,
        'probs':            all_probs,
    }

    print("\n" + "=" * 50)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(f"  AUC-ROC : {metrics['auc_roc']:.4f}")
    print("=" * 50)
    return metrics


def print_results_table(results: dict) -> None:
    """Pretty-print a comparison table. results = {model_name: metrics_dict}"""
    import pandas as pd
    rows = []
    for name, m in results.items():
        rows.append({
            'Model':     name,
            'Accuracy':  round(m['accuracy'],  4),
            'Precision': round(m['precision'], 4),
            'Recall':    round(m['recall'],    4),
            'F1-Score':  round(m['f1'],        4),
            'AUC-ROC':  round(m['auc_roc'],   4),
        })
    df = pd.DataFrame(rows).set_index('Model')
    print(df.to_string())
    return df
