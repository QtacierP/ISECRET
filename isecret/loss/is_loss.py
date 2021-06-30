# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F

class ISLoss(nn.Module):
    # Importance supervised loss (IS-loss)
    def __init__(self,  reduction='mean'):
        super(ISLoss, self).__init__()
        self.reduction = reduction

    def forward(self, source, target, weight):
        loss = self.calculate_loss(source, target, weight)
        return loss

    def calculate_loss(self, pred, target, log_weight):
        weight = torch.exp(-log_weight)
        mse = F.mse_loss(pred, target)
        if self.reduction == 'mean':
            return torch.mean(weight * mse + log_weight)
        return weight * mse + log_weight