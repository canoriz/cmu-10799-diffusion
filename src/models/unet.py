"""
U-Net Architecture for Diffusion Models

In this file, you should implements a U-Net architecture suitable for DDPM.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep
    
    Encoder (Downsampling path)

    Middle
    
    Decoder (Upsampling path)
    
    Output: (batch_size, channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


def _compatible_heads(channels: int, requested: int) -> int:
    """Return the largest requested head count that divides ``channels``."""
    heads = min(int(requested), int(channels))
    while heads > 1 and channels % heads != 0:
        heads -= 1
    return max(1, heads)


class _ResidualAttentionBlock(nn.Module):
    """A residual block followed by optional resolution-gated attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float,
        use_scale_shift_norm: bool,
        attention_resolutions: Tuple[int, ...],
        num_heads: int,
    ):
        super().__init__()
        self.resblock = ResBlock(
            in_channels,
            out_channels,
            time_embed_dim,
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        self.attention_resolutions = set(int(r) for r in attention_resolutions)
        self.attention = AttentionBlock(
            out_channels,
            num_heads=_compatible_heads(out_channels, num_heads),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.resblock(x, time_emb)
        if h.shape[-2] in self.attention_resolutions and h.shape[-1] in self.attention_resolutions:
            h = self.attention(h)
        return h


class UNet(nn.Module):
    """
    TODO: design your own U-Net architecture for diffusion models.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks
    
    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3, 
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = tuple(int(r) for r in attention_resolutions)
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm

        # TODO: build your own unet architecture here
        # Pro tips: remember to take care of the time embeddings!

        if not channel_mult:
            raise ValueError("channel_mult must contain at least one resolution level")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if any(int(mult) < 1 for mult in channel_mult):
            raise ValueError("channel_mult values must be positive")
        if num_heads < 1:
            raise ValueError("num_heads must be positive")

        # Architecture values validated; begin constructing the U-Net modules.
        self.time_embed_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(self.time_embed_dim)
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        level_channels = [base_channels * int(mult) for mult in channel_mult]
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        skip_channels: List[int] = []
        in_ch = base_channels
        for level, out_ch in enumerate(level_channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(_ResidualAttentionBlock(
                    in_ch,
                    out_ch,
                    self.time_embed_dim,
                    dropout,
                    use_scale_shift_norm,
                    self.attention_resolutions,
                    num_heads,
                ))
                skip_channels.append(out_ch)
                in_ch = out_ch
            self.down_blocks.append(blocks)
            if level < len(level_channels) - 1:
                self.downsamplers.append(Downsample(out_ch))

        self.mid_block1 = ResBlock(
            in_ch, in_ch, self.time_embed_dim, dropout, use_scale_shift_norm
        )
        self.mid_attention = AttentionBlock(
            in_ch, num_heads=_compatible_heads(in_ch, num_heads)
        )
        self.mid_block2 = ResBlock(
            in_ch, in_ch, self.time_embed_dim, dropout, use_scale_shift_norm
        )

        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        skip_index = len(skip_channels) - 1
        current_ch = in_ch
        for level in reversed(range(len(level_channels))):
            blocks = nn.ModuleList()
            out_ch = level_channels[level]
            for _ in range(num_res_blocks):
                skip_ch = skip_channels[skip_index]
                skip_index -= 1
                blocks.append(_ResidualAttentionBlock(
                    current_ch + skip_ch,
                    out_ch,
                    self.time_embed_dim,
                    dropout,
                    use_scale_shift_norm,
                    self.attention_resolutions,
                    num_heads,
                ))
                current_ch = out_ch
            self.up_blocks.append(blocks)
            if level > 0:
                self.upsamplers.append(Upsample(out_ch))

        self.out_norm = GroupNorm32(32, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

        # Added implementation: construct the time-conditioned encoder,
        # bottleneck, decoder, skip connections, and output projection.
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement the forward pass of the unet

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
               This is typically the noisy image x_t
            t: Timestep tensor of shape (batch_size,)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """

        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B,C,H,W), got {tuple(x.shape)}")
        batch_size = x.shape[0]
        if t.ndim == 0:
            t = t.expand(batch_size)
        elif t.ndim != 1 or t.shape[0] not in (1, batch_size):
            raise ValueError(f"Expected t with shape (B,), got {tuple(t.shape)}")
        elif t.shape[0] == 1 and batch_size != 1:
            t = t.expand(batch_size)

        # Input types and shapes validated; begin the U-Net forward pass.
        time_emb = self.time_embed(t.to(device=x.device))
        h = self.input_conv(x)
        input_size = x.shape[-2:]
        skips: List[torch.Tensor] = []
        level_shapes: List[torch.Size] = []

        for level, blocks in enumerate(self.down_blocks):
            level_shapes.append(h.shape[-2:])
            for block in blocks:
                h = block(h, time_emb)
                skips.append(h)
            if level < len(self.downsamplers):
                h = self.downsamplers[level](h)

        h = self.mid_block1(h, time_emb)
        if h.shape[-2] in self.attention_resolutions and h.shape[-1] in self.attention_resolutions:
            h = self.mid_attention(h)
        h = self.mid_block2(h, time_emb)

        for decoder_index, blocks in enumerate(self.up_blocks):
            level = len(self.up_blocks) - 1 - decoder_index
            if decoder_index > 0:
                h = self.upsamplers[decoder_index - 1](h)
                target_size = level_shapes[level]
                if h.shape[-2:] != target_size:
                    h = F.interpolate(h, size=target_size, mode="nearest")
            for block in blocks:
                skip = skips.pop()
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = block(torch.cat([h, skip], dim=1), time_emb)

        if h.shape[-2:] != input_size:
            h = F.interpolate(h, size=input_size, mode="nearest")
        h = self.out_norm(h)
        h = F.silu(h)
        # Added implementation: run the complete encoder/bottleneck/decoder
        # pass and restore the input spatial resolution before projection.
        return self.out_conv(h)


def create_model_from_config(config: dict) -> UNet:
    """
    Factory function to create a UNet from a configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration
                Expected to have a 'model' key with the relevant parameters
    
    Returns:
        Instantiated UNet model
    """
    model_config = config['model']
    data_config = config['data']
    
    return UNet(
        in_channels=data_config['channels'],
        out_channels=data_config['channels'],
        base_channels=model_config['base_channels'],
        channel_mult=tuple(model_config['channel_mult']),
        num_res_blocks=model_config['num_res_blocks'],
        attention_resolutions=model_config['attention_resolutions'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        use_scale_shift_norm=model_config['use_scale_shift_norm'],
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
