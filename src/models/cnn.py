"""Custom VGG-inspired CNN from the project blueprint.

Baseline (64 x 64 x 1 input):
    block1  conv3x3(1->8,  s1, p1) + ReLU + maxpool2x2  -> 32 x 32 x 8
    block2  conv3x3(8->16, s1, p1) + ReLU + maxpool2x2  -> 16 x 16 x 16
    block3  conv3x3(16->32,s1, p1) + ReLU + maxpool2x2  ->  8 x  8 x 32
    block4  conv3x3(32->64,s1, p1) + ReLU + maxpool2x2  ->  4 x  4 x 64
    flatten (1024) -> linear(1024 -> 6) -> logits
Softmax is applied only at inference time (CrossEntropyLoss expects raw logits).
Trainable parameters at 64 px: 30,534.

The only optional deviation is `use_batchnorm` (a documented experiment, off by default).
For 100 / 128 px inputs the conv stack is unchanged and the linear layer grows to match the
flattened feature size (6x6x64 = 2304 or 8x8x64 = 4096 features).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from src.config import NUM_CLASSES

BLOCK_NAMES = ("block1", "block2", "block3", "block4")


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = 64
    in_channels: int = 1
    num_classes: int = NUM_CLASSES
    channels: tuple[int, ...] = field(default=(8, 16, 32, 64))
    use_batchnorm: bool = False

    def __post_init__(self):
        if len(self.channels) != 4:
            raise ValueError("The baseline architecture has exactly 4 convolution blocks.")
        if self.feature_size < 1:
            raise ValueError(f"input_size {self.input_size} is too small for 4 pooling stages")

    @property
    def feature_size(self) -> int:
        s = self.input_size
        for _ in range(4):
            s //= 2  # conv keeps size (padding 1); 2x2/2 max pooling floors
        return s

    @property
    def flat_features(self) -> int:
        return self.channels[-1] * self.feature_size ** 2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        d["channels"] = tuple(d.get("channels", (8, 16, 32, 64)))
        return cls(**d)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, batchnorm: bool):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=not batchnorm)
        self.bn = nn.BatchNorm2d(cout) if batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))

    def forward_with_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pooled output, post-ReLU activation before pooling) for Grad-CAM."""
        act = self.relu(self.bn(self.conv(x)))
        return self.pool(act), act


class SteelCNN(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        chans = (c.in_channels, *c.channels)
        self.block1 = ConvBlock(chans[0], chans[1], c.use_batchnorm)
        self.block2 = ConvBlock(chans[1], chans[2], c.use_batchnorm)
        self.block3 = ConvBlock(chans[2], chans[3], c.use_batchnorm)
        self.block4 = ConvBlock(chans[3], chans[4], c.use_batchnorm)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(c.flat_features, c.num_classes)

    def _check_input(self, x: torch.Tensor) -> None:
        c = self.config
        if x.ndim != 4 or x.shape[1] != c.in_channels or x.shape[2] != c.input_size or x.shape[3] != c.input_size:
            raise ValueError(
                f"Expected input of shape (N, {c.in_channels}, {c.input_size}, {c.input_size}), got {tuple(x.shape)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = self.block4(self.block3(self.block2(self.block1(x))))
        return self.classifier(self.flatten(x))

    def forward_with_activation(self, x: torch.Tensor, layer: str = "block4") -> tuple[torch.Tensor, torch.Tensor]:
        """Logits plus the post-ReLU activation of one block (thread-safe: no hooks)."""
        if layer not in BLOCK_NAMES:
            raise ValueError(f"layer must be one of {BLOCK_NAMES}")
        self._check_input(x)
        act = None
        for name in BLOCK_NAMES:
            block: ConvBlock = getattr(self, name)
            if name == layer:
                x, act = block.forward_with_activation(x)
            else:
                x = block(x)
        return self.classifier(self.flatten(x)), act

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)

    def layer_shapes(self) -> list[tuple[str, tuple[int, ...]]]:
        """Output shape (C, H, W) after each block - used to verify the blueprint shapes."""
        x = torch.zeros(1, self.config.in_channels, self.config.input_size, self.config.input_size)
        shapes = []
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for name in BLOCK_NAMES:
                x = getattr(self, name)(x)
                shapes.append((name, tuple(x.shape[1:])))
            flat = self.flatten(x)
            shapes.append(("flatten", tuple(flat.shape[1:])))
            shapes.append(("classifier", tuple(self.classifier(flat).shape[1:])))
        self.train(was_training)
        return shapes


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
