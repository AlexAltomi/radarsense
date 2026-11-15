"""The cnn-v1 model: a small per-frame CNN followed by a temporal convolution head.

Input:  float32 tensor of shape (batch, 32, 3, 32, 32), which is 32 frames of 3 channels of 32 by 32 values.
Output: 6 class scores (logits) per window, in the class order of dsp/radarsense_dsp/windows.py.

Every frame goes through the same encoder (three blocks of 3x3 convolution, batch norm, ReLU and 2x2 max-pool, then a
global average pool), which gives 64 values per frame. Two temporal convolutions over the 32 frames, a global average
pool over time and a linear layer then give the class scores.
"""
import torch
from torch import nn

MODEL_NAME = "cnn-v1"
N_CLASSES = 6
WINDOW = 32
FEATURES = 64


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class GestureCNN(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.encoder = nn.Sequential(_block(3, 16), _block(16, 32), _block(32, FEATURES), nn.AdaptiveAvgPool2d(1))
        self.temporal = nn.Sequential(
            nn.Conv1d(FEATURES, FEATURES, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(FEATURES, FEATURES, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Linear(FEATURES, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        per_frame = self.encoder(x.reshape(b * t, *x.shape[2:])).reshape(b, t, FEATURES)
        over_time = self.temporal(per_frame.transpose(1, 2))  # (batch, features, frames)
        return self.head(over_time.mean(dim=2))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print(f"{MODEL_NAME} has {parameter_count(GestureCNN())} parameters.")
