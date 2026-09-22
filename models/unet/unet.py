"""
models/unet/unet.py
====================
U-Net architecture for camouflage-aware binary segmentation.

Architecture (v2 -- post-diagnosis fixes):
  - 4-level encoder: each level = 2 x Conv2d -> BatchNorm -> ReLU -> Dropout2d
  - CBAM attention module after the bottleneck (channel + spatial attention)
  - 4-level decoder: ConvTranspose2d (upsample) + skip connection concat, then 2xConv
  - Output head: Conv2d(1) -> raw logits (sigmoid applied in loss / metrics)

DIAGNOSIS FIX (RC-1): init_features reduced from 64 -> 32 (config.UNET_INIT_FEATURES).
  - F=64: ~31M parameters  /  F=32: ~4M parameters
  - 31M params on ~3,053 training samples guarantees memorisation (train IoU 0.86,
    val IoU 0.127 gap of 0.73). 4M params is far more appropriate for this
    dataset size while retaining sufficient representational capacity.
  - Architecture at F=32: 32->64->128->256 (bottleneck 512) decoder 256->128->64->32

DIAGNOSIS FIX (RC-1 extension): Spatial dropout in encoder DoubleConv blocks.
  - nn.Dropout2d(p=DROPOUT_RATE) drops entire feature-map channels, providing
    stronger regularisation than per-pixel dropout for convolutional features.
  - Applied after each encoder DoubleConv only; decoder and bottleneck untouched.
  - Disabled at eval time (model.eval()) automatically.

DIAGNOSIS FIX (RC-6): CBAM attention at the bottleneck.
  - Convolutional Block Attention Module (channel + spatial attention) after
    the bottleneck DoubleConv. Applied only at the smallest feature map
    (16x16 at 256px input) where global context matters most.
  - Adds ~100K parameters -- negligible vs the 4M model.
  - Provides the global context needed to distinguish camouflaged texture
    from similar background texture (a purely local U-Net cannot).
  - Reference: Woo et al., "CBAM: Convolutional Block Attention Module",
    ECCV 2018.

Input  : (B, 3, H, W)  -- RGB frame, H and W must be divisible by 16
Output : (B, 1, H, W)  -- per-pixel raw logits (NOT probabilities)
                           Apply torch.sigmoid() outside the model for probabilities.

All parameters come from config.py:
  - UNET_IN_CHANNELS   (default 3)
  - UNET_OUT_CHANNELS  (default 1)
  - UNET_INIT_FEATURES (now 32, was 64)
  - DROPOUT_RATE       (now 0.2)

Reference architecture:
  Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
  Segmentation", MICCAI 2015.
"""

import torch
import torch.nn as nn
import sys
import os

# ---------------------------------------------------------------------------
# Allow imports from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


# ---------------------------------------------------------------------------
# Building block: double convolution with optional spatial dropout
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """
    Two consecutive Conv2d -> BatchNorm2d -> ReLU blocks, with an optional
    spatial dropout layer appended at the end.

    Conv(in_ch -> out_ch) -> BN -> ReLU -> Conv(out_ch -> out_ch) -> BN -> ReLU
                                         [ -> Dropout2d(p) ]  # if dropout_p > 0

    DIAGNOSIS FIX (RC-1): dropout_p=0.2 added to encoder blocks.
    nn.Dropout2d drops entire feature-map channels (zero-ing a random set of
    C channels across the whole spatial map), providing stronger regularisation
    than per-pixel dropout for convolutional features.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 dropout_p: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            # Dropout2d zeros entire channels, not individual pixels.
            # This forces the network to learn redundant representations
            # across channels, improving generalisation on small datasets.
            layers.append(nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Encoder block: DoubleConv -> record skip -> MaxPool
# ---------------------------------------------------------------------------

class _EncoderBlock(nn.Module):
    """
    One level of the contracting (encoder) path:
      1. DoubleConv (with spatial dropout) -- extracts features at current scale
      2. MaxPool2d(2) -- halves spatial dimensions

    Returns:
      skip : feature map BEFORE pooling (fed to corresponding decoder block)
      x    : pooled feature map (fed to the next encoder level)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 dropout_p: float = 0.0) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_channels, out_channels, dropout_p=dropout_p)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return skip, self.pool(skip)


# ---------------------------------------------------------------------------
# ASPP: Atrous Spatial Pyramid Pooling
# ---------------------------------------------------------------------------

class _ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.
    Captures multi-scale context by using parallel dilated convolutions.
    Dilations: [1, 6, 12, 18] by default.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Parallel branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling branch
        self.branch5_avg = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output 1x1 conv to fuse branches
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=config.DROPOUT_RATE) # spatial dropout on fused ASPP output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Branch 5: GAP -> 1x1 conv -> Upsample to match
        b5 = self.branch5_avg(x)
        b5 = self.branch5_conv(b5)
        b5 = nn.functional.interpolate(b5, size=size, mode='bilinear', align_corners=False)
        
        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return self.out_conv(out)


# ---------------------------------------------------------------------------
# CBAM: Convolutional Block Attention Module
# ---------------------------------------------------------------------------


class _ChannelAttention(nn.Module):
    """
    Channel attention: squeeze spatial dims -> MLP -> re-calibrate channels.
    Asks "which feature channels are most relevant for detecting camouflage?"
    """
    def __init__(self, in_channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Both avg and max pooling aggregated for richer statistics
        ca = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * ca  # broadcast multiply: (B, C, H, W) * (B, C, 1, 1)


class _SpatialAttention(nn.Module):
    """
    Spatial attention: aggregate channels -> conv -> re-calibrate positions.
    Asks "where in the image is the camouflaged animal likely to be?"
    """
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise avg and max pooling -> concatenate -> convolve
        avg_out = x.mean(dim=1, keepdim=True)               # (B, 1, H, W)
        max_out = x.amax(dim=1, keepdim=True)               # (B, 1, H, W)
        sa = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * sa  # broadcast multiply: (B, C, H, W) * (B, 1, H, W)


class _CBAM(nn.Module):
    """
    CBAM: Channel Attention then Spatial Attention, sequentially.

    DIAGNOSIS FIX (RC-6): Placed after the bottleneck DoubleConv.
    At the bottleneck (16x16 at 256px input), each spatial location covers
    a 256x256/16x16 = 256-pixel receptive field. Channel attention selects
    which feature types matter; spatial attention suppresses background blobs
    and focuses on the animal region.

    Adding CBAM only at the bottleneck:
      - Adds ~100K parameters (negligible on the 4M model).
      - Provides global context unavailable to purely local convolutions.
      - Does NOT add to decoder or encoder (avoids parameter bloat).

    Reference:
      Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
    """
    def __init__(self, in_channels: int, reduction: int = 8,
                 spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_attn  = _ChannelAttention(in_channels, reduction=reduction)
        self.spatial_attn  = _SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)   # re-calibrate channels
        x = self.spatial_attn(x)   # re-calibrate spatial positions
        return x


# ---------------------------------------------------------------------------
# Decoder block: ConvTranspose2d -> cat(skip) -> DoubleConv
# ---------------------------------------------------------------------------

class _DecoderBlock(nn.Module):
    """
    One level of the expanding (decoder) path:
      1. ConvTranspose2d -- doubles spatial dimensions (learnable upsample)
      2. Concatenate skip connection from the corresponding encoder level
      3. DoubleConv -- refines features after concatenation

    No dropout in decoder blocks -- only encoder blocks are regularised.
    The decoder needs stable gradients flowing back to the encoder.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = _DoubleConv(in_channels, out_channels, dropout_p=0.0)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch caused by odd input dimensions
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full U-Net (v2)
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    4-level U-Net for binary camouflage segmentation. (v2 -- post-diagnosis)

    Changes from v1:
      - init_features: 64 -> 32 (DIAGNOSIS FIX RC-1: reduces params from 31M to 4M)
      - Spatial Dropout2d in all encoder DoubleConv blocks (RC-1)
      - CBAM attention module inserted after the bottleneck (RC-6)

    Encoder feature channels  : F -> 2F -> 4F -> 8F    (F = init_features = 32)
    Bottleneck feature channels: 16F                    (= 512 at F=32)
    Decoder feature channels  : 8F -> 4F -> 2F -> F
    Output                    : 1 channel, raw logits (NO Sigmoid in forward)
                                sigmoid is applied inside TverskyLoss and
                                torch.sigmoid() inside DiceLoss / compute_metrics.

    Total parameters (F=32, ~4M vs old F=64 ~31M).
    """

    def __init__(
        self,
        in_channels:   int   = config.UNET_IN_CHANNELS,
        out_channels:  int   = config.UNET_OUT_CHANNELS,
        init_features: int   = config.UNET_INIT_FEATURES,
        dropout_p:     float = config.DROPOUT_RATE,
    ) -> None:
        super().__init__()
        f = init_features  # short alias (default = 32)

        # ---- Encoder (with spatial dropout for regularisation) ----
        self.enc1 = _EncoderBlock(in_channels, f,     dropout_p=dropout_p)  # 3   -> F
        self.enc2 = _EncoderBlock(f,           f * 2, dropout_p=dropout_p)  # F   -> 2F
        self.enc3 = _EncoderBlock(f * 2,       f * 4, dropout_p=dropout_p)  # 2F  -> 4F
        self.enc4 = _EncoderBlock(f * 4,       f * 8, dropout_p=dropout_p)  # 4F  -> 8F

        # ---- Bottleneck (ASPP multi-scale context) ----
        self.bottleneck = _ASPP(f * 8, f * 16)         # 8F  -> 16F

        # ---- CBAM Attention (after bottleneck, before decoder) ----
        # DIAGNOSIS FIX (RC-6): lightweight attention at the 16x16 feature map.
        # Channel attention selects relevant feature types; spatial attention
        # focuses on the animal location. Adds ~100K params total.
        self.cbam = _CBAM(in_channels=f * 16, reduction=8, spatial_kernel=7)

        # ---- Decoder (no dropout) ----
        self.dec4 = _DecoderBlock(f * 16, f * 8)   # 16F -> 8F
        self.dec3 = _DecoderBlock(f * 8,  f * 4)   # 8F  -> 4F
        self.dec2 = _DecoderBlock(f * 4,  f * 2)   # 4F  -> 2F
        self.dec1 = _DecoderBlock(f * 2,  f)        # 2F  -> F

        # ---- Output head ----
        # NOTE: No Sigmoid here. The model returns raw logits.
        # Sigmoid is applied inside TverskyLoss (loss) and
        # torch.sigmoid() inside DiceLoss / compute_metrics.
        self.output_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder (stores skip connections)
        skip1, x = self.enc1(x)   # skip1: (B, F,   H,    W   )
        skip2, x = self.enc2(x)   # skip2: (B, 2F,  H/2,  W/2 )
        skip3, x = self.enc3(x)   # skip3: (B, 4F,  H/4,  W/4 )
        skip4, x = self.enc4(x)   # skip4: (B, 8F,  H/8,  W/8 )

        # Bottleneck
        x = self.bottleneck(x)    # x:     (B, 16F, H/16, W/16)

        # CBAM Attention at bottleneck
        # DIAGNOSIS FIX (RC-6): global channel + spatial attention before decode.
        # This is where the model decides WHICH features and WHERE to attend.
        x = self.cbam(x)          # x:     (B, 16F, H/16, W/16) -- attended

        # Decoder (uses skip connections in reverse order)
        x = self.dec4(x, skip4)   # x:     (B, 8F,  H/8,  W/8 )
        x = self.dec3(x, skip3)   # x:     (B, 4F,  H/4,  W/4 )
        x = self.dec2(x, skip2)   # x:     (B, 2F,  H/2,  W/2 )
        x = self.dec1(x, skip1)   # x:     (B, F,   H,    W   )

        # Output -- raw logits, no sigmoid applied here
        x = self.output_conv(x)   # x:     (B, 1,   H,    W   )  raw logits
        return x


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    print("=== U-Net v2 self-test ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UNet().to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total_params:,}  (target: ~4M for F=32)")
    print(f"Trainable parameters: {trainable:,}")
    assert total_params < 10_000_000, (
        f"Model has {total_params:,} params -- check UNET_INIT_FEATURES is 32, not 64."
    )

    # Forward pass
    dummy = torch.randn(2, 3, config.INPUT_HEIGHT, config.INPUT_WIDTH).to(device)
    with torch.no_grad():
        out = model(dummy)

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    assert out.shape == (2, 1, config.INPUT_HEIGHT, config.INPUT_WIDTH), \
        f"Unexpected output shape: {out.shape}"

    # Output is raw logits -- NOT clamped to [0,1]
    probs = torch.sigmoid(out)
    assert probs.min() >= 0.0 and probs.max() <= 1.0, \
        "Sigmoid(logits) outside [0, 1] -- unexpected."

    # Verify train vs eval behaviour (Dropout2d is active only during train)
    model.train()
    with torch.no_grad():
        out_train = model(dummy)
    model.eval()
    with torch.no_grad():
        out_eval = model(dummy)
    # Outputs differ in train mode due to dropout (usually)
    print(f"Train output range : [{out_train.min():.3f}, {out_train.max():.3f}]")
    print(f"Eval  output range : [{out_eval.min():.3f},  {out_eval.max():.3f}]")

    print("PASS -- U-Net v2 forward pass OK.")
    print(f"  init_features : {config.UNET_INIT_FEATURES}  (was 64, now 32)")
    print(f"  dropout_rate  : {config.DROPOUT_RATE}")
    print(f"  CBAM          : bottleneck only (channel + spatial attention)")
