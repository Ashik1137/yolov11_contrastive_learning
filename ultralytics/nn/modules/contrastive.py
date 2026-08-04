import torch
import torch.nn.functional as F
from torch import nn


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning.

    Input:
        Feature map from YOLO backbone
        (B, C, H, W)

    Output:
        Normalized embedding
        (B, out_dim)
    """

    def __init__(self, in_channels: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projector = nn.Sequential(
            nn.Linear(in_channels, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # x : (B,C,H,W)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.projector(x)

        x = F.normalize(x, p=2, dim=1)

        return x
