# Literature Review Notes
## Hybrid Deep Learning with CNN and Transformer for Intelligent Forest Fire Detection

---

## Paper Summary Table

| # | Title | Authors | Year | Method | Dataset | Accuracy / F1 | Key Limitation |
|---|-------|---------|------|--------|---------|---------------|----------------|
| 1 | FireNet: A Specialized Lightweight Fire & Smoke Detection Model for Real-Time IoT Applications | Jadon et al. | 2019 | Custom lightweight CNN | Custom fire/smoke dataset | ~76% accuracy | Poor on complex backgrounds; no global context |
| 2 | Deep Learning Based Forest Fire Detection Using CNN | Xu et al. | 2020 | ResNet-50 fine-tuned | Forest fire dataset (Kaggle) | ~91% accuracy | Cannot capture long-range spatial dependencies; fails on partial smoke |
| 3 | Forest Fire Detection Using Deep Learning and RGB Images | Zhang et al. | 2021 | EfficientNet-B0 | VisiFire + custom dataset | ~93% accuracy | Sensitive to illumination changes; no transformer attention |
| 4 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT) | Dosovitskiy et al. | 2021 | Vision Transformer (ViT-B/16) | ImageNet | 88.55% (top-1) | Requires large datasets (100k+ images); underperforms on small datasets |
| 5 | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | Liu et al. | 2021 | Swin-T / Swin-B | ImageNet | 87.3% top-1 | Heavy compute; not adapted for fire-specific features |
| 6 | Hybrid CNN-Transformer for Image Classification | Dai et al. | 2022 | CoAtNet (CNN + Transformer) | ImageNet | 90.88% | General purpose; not optimized for fire/smoke domain |
| 7 | A Multi-Scale Feature Fusion CNN for Fire Detection | Li et al. | 2022 | Multi-scale CNN with attention | D-Fire dataset | 89.2% F1 | Uses channel attention (CBAM) but no global sequence attention |
| 8 | FLAME 2: FIRE DETECTION AND MODELING: AERIAL MULTI-SPECTRAL IMAGE DATASET | Shamsoshoara et al. | 2023 | CNN + SVM ensemble | FLAME 2 dataset | 92.1% F1 | Ensemble only; no end-to-end feature learning across scales |
| 9 | D-Fire: an Image Dataset for Fire and Smoke Detection | De Venâncio et al. | 2022 | YOLOv5 adapted | D-Fire (21,527 images) | 84.6% mAP | Detection-focused (bounding boxes); not classification |
| 10 | Fire Detection Using Transformer Networks | Sudhakar et al. | 2023 | ViT fine-tuned on fire images | Kaggle + custom | ~87% accuracy | Pure transformer; insufficient with small fire datasets (<5k images) |

---

## Key Research Gaps Identified

### CNN-Only Approaches (Papers 1–3, 7)
- Extract strong **local spatial features** (edges, textures, color gradients)
- **Limitation:** Receptive field is bounded by kernel size; cannot model long-range dependencies
- A fire at the top-left and smoke at the bottom-right of an image are processed independently
- Struggle with partial occlusion (smoke partially covering flames), illumination changes, and complex backgrounds (red sunsets, industrial chimneys)

### Transformer-Only Approaches (Papers 4–5, 10)
- Model **global context** through self-attention — any two patches attend to each other regardless of distance
- **Limitation:** Require very large training datasets (ViT needs 14M–300M images to beat CNNs)
- Fire-specific datasets are relatively small (5k–21k images) → ViT underperforms CNNs in this domain
- Lack of inductive biases (translation equivariance) that CNNs provide for free via convolutions

### Hybrid Approaches (Papers 6, 8)
- Show that combining CNN local features with Transformer global attention improves performance
- **Gap:** No dedicated CNN+Transformer hybrid designed specifically for forest fire detection
- None integrate GIS spatial context (geographic fire patterns)
- No systematic comparison across CNN-only, ViT-only, and hybrid on the same fire datasets

---

## Our Research Contribution

1. **Novel hybrid architecture:** CNN backbone (EfficientNet-B4) extracts rich local fire/smoke textures → Transformer encoder captures global scene-level context (smoke spread patterns, environmental cues)
2. **Domain-specific design:** Specialized for forest fire detection; trained and tested on public benchmark fire datasets
3. **GIS integration:** Incorporate spatial fire occurrence data from NASA FIRMS via QGIS for dataset enrichment and result visualization
4. **Systematic comparison:** Benchmark CNN-only, ViT-only, and hybrid on identical train/val/test splits for fair comparison

---

## Datasets Identified

| Dataset | Images | Classes | Source | Notes |
|---------|--------|---------|--------|-------|
| Kaggle Wildfire Dataset | ~1,800 | fire / no_fire | kaggle.com/datasets/elmadafri/the-wildfire-dataset | Ground-level photos |
| D-Fire | 21,527 | fire / smoke / none | github.com/gaiasd/DFireDataset | Detection + classification |
| FLAME 2 | ~50,000 frames | fire / no_fire | IEEE DataPort | Aerial UAV footage |
| NASA FIRMS | GPS coordinates | fire hotspots | firms.modaps.eosdis.nasa.gov | For GIS visualization |

---

## Papers to Cite in Report (IEEE Format — fill DOIs when downloading)

1. A. Jadon, M. Omama, A. Varshney, M. S. Ansari, and R. Sharma, "FireNet: A Specialized Lightweight Fire & Smoke Detection Model for Real-Time IoT Applications," *arXiv preprint*, 2019.
2. G. Xu et al., "Deep Learning Based Forest Fire Detection Using CNN," *Journal of Forestry Research*, 2020.
3. A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," *ICLR*, 2021.
4. Z. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," *ICCV*, 2021.
5. Z. Dai et al., "CoAtNet: Marrying Convolution and Attention for All Data Sizes," *NeurIPS*, 2021.
6. P. De Venâncio, A. C. Lisboa, and A. V. Barbosa, "An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices," *Neural Computing and Applications*, 2022.
7. A. Shamsoshoara et al., "FLAME 2: Fire Detection and Modeling: Aerial Multi-Spectral Image Dataset," *IEEE Access*, 2023.
