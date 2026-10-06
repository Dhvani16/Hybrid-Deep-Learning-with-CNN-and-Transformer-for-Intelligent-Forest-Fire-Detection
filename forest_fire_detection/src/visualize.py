import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import cv2
from sklearn.metrics import roc_curve

from src.dataset import IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def denormalize(tensor):
    """Convert a normalized CHW tensor back to a HWC uint8 numpy array."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return (img.clip(0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, title: str, save_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'],   label='Val')
    ax1.set_title(f'{title} — Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'],   label='Val')
    ax2.set_title(f'{title} — Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm_array, class_names: list, title: str, save_path: str
) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_array, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5,
    )
    plt.title(title)
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(results: dict, save_path: str) -> None:
    """results = {model_name: metrics_dict}"""
    plt.figure(figsize=(7, 5))
    for name, m in results.items():
        fpr, tpr, _ = m['fpr_tpr']
        plt.plot(fpr, tpr, label=f"{name}  (AUC={m['auc_roc']:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — Model Comparison')
    plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison_bar(results: dict, save_path: str) -> None:
    model_names = list(results.keys())
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

    x = np.arange(len(model_names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        values = [results[n][key] for n in model_names]
        bars = ax.bar(x + i * width, values, width, label=label)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel('Score'); ax.set_title('Model Performance Comparison')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def plot_gradcam(model, dataset, target_layer, device, num_samples: int = 8,
                 save_path: str = 'outputs/plots/gradcam.png') -> None:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("Install grad-cam: pip install grad-cam")
        return

    cam = GradCAM(model=model, target_layers=[target_layer])
    model.eval()

    fire_samples    = [s for s in dataset.samples if s[1] == 1][:num_samples // 2]
    no_fire_samples = [s for s in dataset.samples if s[1] == 0][:num_samples // 2]
    selected = fire_samples + no_fire_samples

    cols = 4
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flat

    for ax, (path, label) in zip(axes, selected):
        img_orig = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img_orig = cv2.resize(img_orig, (224, 224))
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        tfm = A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()
        ])
        tensor = tfm(image=img_orig)['image'].unsqueeze(0).to(device)
        grayscale = cam(input_tensor=tensor)[0]
        overlay = show_cam_on_image(img_orig / 255.0, grayscale, use_rgb=True)
        ax.imshow(overlay)
        ax.set_title(['No Fire', 'Fire'][label], fontsize=10)
        ax.axis('off')

    for ax in axes:
        ax.axis('off')

    plt.suptitle('Grad-CAM — Hybrid CNN+Transformer', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Transformer attention map
# ---------------------------------------------------------------------------

def plot_attention_map(model, dataset, device, spatial_size: int = 7,
                       save_path: str = 'outputs/plots/attention_maps.png') -> None:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    fire_samples = [s for s in dataset.samples if s[1] == 1][:3]
    tfm = A.Compose([A.Resize(224, 224),
                     A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])

    fig, axes = plt.subplots(len(fire_samples), 3, figsize=(12, 4 * len(fire_samples)))

    for row, (path, _) in enumerate(fire_samples):
        img_orig = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img_orig = cv2.resize(img_orig, (224, 224))
        tensor = tfm(image=img_orig)['image'].unsqueeze(0).to(device)

        attn = model.get_transformer_attention(tensor, layer_idx=-1)
        # attn: [1, heads, N+1, N+1] — CLS-to-patch attention
        cls_attn = attn[0, :, 0, 1:].mean(0).cpu().numpy()   # [N]
        cls_attn = cls_attn.reshape(spatial_size, spatial_size)
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        attn_up  = cv2.resize(cls_attn, (224, 224))

        overlay = img_orig / 255.0 * 0.55 + cm.hot(attn_up)[:, :, :3] * 0.45

        axes[row][0].imshow(img_orig); axes[row][0].set_title('Original')
        axes[row][1].imshow(attn_up, cmap='hot'); axes[row][1].set_title('Attention Map')
        axes[row][2].imshow(overlay); axes[row][2].set_title('Overlay')
        for ax in axes[row]:
            ax.axis('off')

    plt.suptitle('Transformer Self-Attention (CLS token) — Fire Samples', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def plot_failure_cases(dataset, preds: np.ndarray, save_path: str, max_show: int = 10) -> None:
    wrong = [(path, true, pred)
             for (path, true), pred in zip(dataset.samples, preds)
             if true != pred]

    fp = sum(1 for _, t, p in wrong if t == 0 and p == 1)
    fn = sum(1 for _, t, p in wrong if t == 1 and p == 0)
    print(f"Total errors: {len(wrong)}  |  False Positives: {fp}  |  False Negatives: {fn}")

    selected = wrong[:max_show]
    cols = 5
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flat

    for ax, (path, true, pred) in zip(axes, selected):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        ax.imshow(img)
        ax.set_title(
            f"True: {['No Fire','Fire'][true]}\nPred: {['No Fire','Fire'][pred]}",
            color='red', fontsize=9
        )
        ax.axis('off')
    for ax in axes:
        ax.axis('off')

    plt.suptitle('Failure Cases (Incorrect Predictions)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")
