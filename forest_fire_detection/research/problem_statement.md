# Problem Statement
## Hybrid Deep Learning with CNN and Transformer for Intelligent Forest Fire Detection

---

## Motivation

Forest fires represent one of the most destructive natural disasters worldwide, causing irreversible damage to ecosystems, human lives, property, and air quality. Early and accurate detection is critical for deploying firefighting resources before fires spread beyond control. Traditional detection methods — satellite thermal imaging, watchtower observers, and smoke sensors — suffer from high latency, limited coverage, and high false positive rates in challenging conditions such as fog, low-light environments, or industrial smoke.

Recent advances in deep learning have enabled automated fire detection from RGB images, offering faster and cheaper detection than traditional methods. However, **existing automated systems have two fundamental limitations** that constrain their real-world applicability:

---

## The Core Problem

### Limitation 1 — CNN-Only Methods Lack Global Context
Convolutional Neural Networks extract powerful **local** spatial features — edges, color gradients, and texture patterns — that capture the appearance of fire and smoke at a patch level. However, their receptive field is inherently limited by kernel size. A CNN cannot efficiently relate distant spatial regions (e.g., a column of smoke in one corner and a heat shimmer in another), meaning it misses scene-level cues that a human observer would naturally use to confirm a fire.

This results in high false positive rates when local patches resemble fire (red sunsets, industrial chimneys, bright foliage) but the global scene context clearly indicates no fire.

### Limitation 2 — Transformer-Only Methods Are Data-Hungry
Vision Transformers (ViT) address the global context problem by applying multi-head self-attention across image patches, allowing every patch to attend to every other patch regardless of distance. This global reasoning significantly reduces false positives from deceptive local textures.

However, Transformers lack the inductive biases (translation equivariance, locality) that CNNs exploit from the first layer. They require millions of training images to learn these properties from scratch. Fire-specific datasets are typically small (5,000–21,000 images), making pure ViT architectures significantly underperform CNNs in this domain.

---

## Our Hypothesis

A **hybrid architecture** that combines a pretrained CNN backbone (for strong local feature extraction) with a Transformer encoder (for global contextual reasoning) will:
1. Achieve higher detection accuracy than CNN-only or Transformer-only approaches
2. Reduce false positives by leveraging both local texture cues and global scene context
3. Generalize better across diverse fire scenarios (day/night, dense smoke, partial occlusion, varied backgrounds)

---

## Research Questions

1. Does combining CNN local feature extraction with Transformer global attention outperform CNN-only and Transformer-only baselines for forest fire classification?
2. Which components of the hybrid architecture contribute most to performance? (ablation study)
3. What types of images does the model fail on, and why?

---

## Datasets

Three public benchmark datasets will be used to ensure reproducibility and fair comparison:
- **Kaggle Wildfire Dataset** — ground-level fire/no-fire images
- **D-Fire Dataset** — 21,527 images including fire, smoke, and neither
- **NASA FIRMS** — geographic fire hotspot coordinates for GIS visualization

---

## Expected Contributions

1. A novel CNN + Transformer hybrid architecture specifically designed and benchmarked for forest fire detection
2. A systematic comparison of CNN-only, ViT-only, and hybrid models on identical dataset splits
3. GIS-integrated fire probability mapping using QGIS and NASA FIRMS data
4. Publicly available code and pretrained weights for reproducibility
