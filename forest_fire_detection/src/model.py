import torch
import torch.nn as nn
import timm


class CNNBaseline(nn.Module):
    """EfficientNet-B4 (or any timm backbone) with a dropout + linear head."""

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = 'efficientnet_b4',
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool='avg'
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
