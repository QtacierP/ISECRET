import torch.nn as nn
import torch

class SimSiamLoss(nn.Module):
    # [-1, 1] 1 means the
    def __init__(self, args):
        super().__init__()
        self.loss = nn.CosineSimilarity(dim=-1)  # B N D

    def forward(self, source, target):
        return - torch.mean(self.loss(source, target.detach()))