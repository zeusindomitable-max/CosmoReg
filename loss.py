# cosmo_reg/loss.py
"""
CosmoReg — Cosmology-Inspired Adaptive Activation Regularizer
Author: Hari Hardiyan (@haritedjamantri)
Date: November 15, 2025
Country: Indonesia
License: MIT
"""

import torch
import torch.nn as nn

__version__ = "1.0.0"

class CosmoRegLoss(nn.Module):
    def __init__(self, M_scale: float = 1.0, lambda_0: float = 0.01, alpha: float = 0.05):
        super().__init__()
        self.M_sq = M_scale ** 2
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self.eps = 1e-8
        self.register_buffer("current_lambda", torch.tensor(lambda_0))

    def _phi(self, r_sq: torch.Tensor) -> torch.Tensor:
        log_arg = r_sq + self.eps
        entropic = -log_arg * torch.log(log_arg)
        quartic = (1.0 / self.M_sq) * (r_sq ** 2)
        return entropic + quartic

    def _adaptive_lambda(self, grad_norm_sq: torch.Tensor) -> torch.Tensor:
        exp_in = torch.clamp(self.alpha * grad_norm_sq, max=70.0)
        return self.lambda_0 * torch.exp(-exp_in)

    def forward(self, activations: torch.Tensor, task_grad_norm_sq: float):
        r_sq = torch.sum(activations ** 2, dim=-1)
        S_reg = torch.sum(self._phi(r_sq))
        lambda_t = self._adaptive_lambda(torch.tensor(task_grad_norm_sq, device=activations.device))
        self.current_lambda.copy_(lambda_t.detach())
        return lambda_t * S_reg
