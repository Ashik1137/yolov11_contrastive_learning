import torch
import torch.nn.functional as F
from torch import nn


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR InfoNCE Loss).
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """z1 : (B,128) z2 : (B,128).
        """
        batch_size = z1.size(0)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        representations = torch.cat([z1, z2], dim=0)

        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        positives = torch.cat([torch.diag(similarity_matrix, batch_size), torch.diag(similarity_matrix, -batch_size)])

        numerator = torch.exp(positives / self.temperature)

        denominator = torch.sum(torch.exp(similarity_matrix / self.temperature), dim=1)

        loss = -torch.log(numerator / denominator)

        return loss.mean()
