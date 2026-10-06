import torch
import torch.nn as nn
import timm


class _TransformerEncoder(nn.Module):
    """Lightweight Transformer encoder with a learnable CLS token."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-LN is more stable
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, embed_dim]
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)          # [B, 1, embed_dim]
        x = torch.cat([cls, tokens], dim=1)              # [B, N+1, embed_dim]
        x = self.encoder(x)
        return self.norm(x[:, 0])                        # CLS token output [B, embed_dim]


class HybridCNNTransformer(nn.Module):
    """Hybrid model: CNN backbone for local features + Transformer for global context.

    Architecture:
        Input (224×224×3)
        → CNN backbone (EfficientNet-B4)  →  feature map [B, C, H, W]
        → 1×1 Conv projection             →  [B, embed_dim, H, W]
        → flatten + positional embedding  →  tokens [B, H*W, embed_dim]
        → Transformer encoder             →  CLS token [B, embed_dim]
        → Dropout + Linear                →  logits [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = 'efficientnet_b4',
        pretrained: bool = True,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # CNN feature extractor (no head, no global pool)
        self.cnn = timm.create_model(backbone, pretrained=pretrained,
                                     num_classes=0, global_pool='')
        feat_dim = self.cnn.num_features          # e.g. 1792 for EfficientNet-B4

        # Project CNN feature maps to Transformer embedding dimension
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )

        # Positional embedding: max 196 patches (14×14) + 1 CLS (added inside encoder)
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = _TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract feature map
        feat = self.cnn(x)                              # [B, C, H, W]
        feat = self.proj(feat)                          # [B, embed_dim, H, W]
        B, C, H, W = feat.shape
        N = H * W

        # Flatten spatial dims → token sequence
        tokens = feat.flatten(2).permute(0, 2, 1)      # [B, N, embed_dim]
        tokens = tokens + self.pos_embed[:, :N, :]     # add positional embedding

        cls_out = self.transformer(tokens)              # [B, embed_dim]
        return self.classifier(cls_out)

    def freeze_backbone(self) -> None:
        for p in self.cnn.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.cnn.parameters():
            p.requires_grad = True

    def get_transformer_attention(self, x: torch.Tensor, layer_idx: int = -1):
        """Return attention weights from the specified Transformer layer for visualization.

        Avoids hooks entirely (hooks cause infinite recursion in PyTorch 2.x because
        _sa_block calls self_attn inside __call__, re-firing any registered hook).
        Instead, the forward pass is replayed step-by-step up to the target layer and
        F.multi_head_attention_forward is called directly with need_weights=True.
        """
        import torch.nn.functional as F

        with torch.no_grad():
            feat   = self.cnn(x)
            feat   = self.proj(feat)
            B, _, H, W = feat.shape
            tokens = feat.flatten(2).permute(0, 2, 1)
            tokens = tokens + self.pos_embed[:, :H * W, :]

            enc = self.transformer
            seq = torch.cat([enc.cls_token.expand(B, -1, -1), tokens], dim=1)

            layers = list(enc.encoder.layers)
            n      = len(layers)
            target = n + layer_idx if layer_idx < 0 else layer_idx

            for i in range(target):
                seq = layers[i](seq)

            layer = layers[target]
            mha   = layer.self_attn
            x_in  = layer.norm1(seq) if layer.norm_first else seq
            x_in  = x_in.transpose(0, 1)   # [B, N+1, D] → [N+1, B, D] (F expects seq-first)

            _, weights = F.multi_head_attention_forward(
                x_in, x_in, x_in,
                mha.embed_dim, mha.num_heads,
                mha.in_proj_weight, mha.in_proj_bias,
                mha.bias_k, mha.bias_v, mha.add_zero_attn,
                mha.dropout, mha.out_proj.weight, mha.out_proj.bias,
                training=False, need_weights=True, average_attn_weights=False,
            )
        return weights  # [B, num_heads, N+1, N+1]
