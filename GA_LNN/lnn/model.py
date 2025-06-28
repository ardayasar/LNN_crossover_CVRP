import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
#  Full Liquid‑Time‑Constant (LTC) layer – Hasani et al. 2021
# -----------------------------------------------------------------------------
class LiquidLayer(nn.Module):
    """
    Implements the continuous‑time neural ODE cell used in Liquid Time‑constant
    Networks (LTC).  For each neuron j:

        hᵗⱼ = hᵗ⁻¹ⱼ + (dt / τⱼ) · (‑hᵗ⁻¹ⱼ + σ( (Wₓ xᵗ)ⱼ + (Wₕ hᵗ⁻¹)ⱼ + bⱼ ))

    where τⱼ > 0 is a learnable time constant and σ is tanh.
    We apply the update *token‑wise* (sequence dimension) but share parameters
    across tokens, which is a good fit for per‑customer feature vectors.

    Args
    ----
    in_dim      : feature size of the input xᵗ
    hidden_dim  : number of LTC neurons (d_model)
    tau_min/max : range for initial τ values
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 tau_min: float = 1.0,
                 tau_max: float = 2.0):
        super().__init__()
        self.in_dim      = in_dim
        self.hidden_dim  = hidden_dim

        # input → hidden and hidden → hidden linear maps (bias in separate param)
        self.W_x  = nn.Linear(in_dim,    hidden_dim, bias=False)
        self.W_h  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # initialise τ in (tau_min, tau_max) and store log‑τ so positivity holds
        init_tau = torch.empty(hidden_dim).uniform_(tau_min, tau_max)
        self.log_tau = nn.Parameter(init_tau.log())   # τ = exp(log_tau)

    def forward(self, x: torch.Tensor, steps: int = 1, dt: float = 1.0):
        """
        x : Tensor (..., seq_len, in_dim)  – typically (B,N,F) or (N,F)
        Returns a tensor of the same leading dims with `hidden_dim` features.
        The update is iterated `steps` times (Euler integration).
        """
        # ensure batch dimension
        single_sample = x.dim() == 2
        if single_sample:
            x = x.unsqueeze(0)                        # (1,N,F)

        B, N, _ = x.shape
        h = torch.zeros(B, N, self.hidden_dim, device=x.device, dtype=x.dtype)
        tau = torch.exp(self.log_tau)                 # positivity

        for _ in range(steps):
            z = torch.tanh(self.W_x(x) + self.W_h(h) + self.bias)  # (B,N,H)
            dh = (-h + z) / tau                        # broadcast τ
            h  = h + dt * dh

        return h.squeeze(0) if single_sample else h


# -----------------------------------------------------------------------------
#  LiquidCrossover
# -----------------------------------------------------------------------------
class LiquidCrossover(nn.Module):
    """
    Maps customer-feature tensor → scalar score per customer.

    `forward(x, mask)` works with:
      • x  : (N, F)         – single sample (legacy GA usage)
      • x  : (B, N, F)      – batch  (new trainer)
      • mask: (B, N) bool   – 1 where valid token (currently unused)
    """
    def __init__(self, in_dim: int, d_model: int = 48, liquid_steps: int = 1):
        super().__init__()
        self.in_proj  = nn.Linear(in_dim, d_model)
        self.liquid   = LiquidLayer(d_model, d_model)
        self.liquid2  = LiquidLayer(d_model, d_model)
        self.head     = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.steps    = liquid_steps

    def forward(
        self,
        feats: torch.Tensor,                 # (N,F) or (B,N,F)
        mask : Optional[torch.Tensor] = None # (B,N)  – ignored for now
    ) -> torch.Tensor:
        single_sample = feats.dim() == 2
        if single_sample:
            feats = feats.unsqueeze(0)       # → (1,N,F)

        x = F.relu(self.in_proj(feats))      # (B,N,d_model)
        x = self.liquid(x, steps=self.steps)
        x = self.liquid2(x, steps=self.steps)
        scores = self.head(x).squeeze(-1)

        return scores.squeeze(0) if single_sample else scores


# -----------------------------------------------------------------------------
#  Quick sanity test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    n, F = 21, 5
    dummy_feats = torch.randn(n, F)
    model = LiquidCrossover(in_dim=F)
    out   = model(dummy_feats)
    print("scores shape (single):", out.shape)

    batch_feats = torch.randn(4, n, F)
    out_b       = model(batch_feats)
    print("scores shape (batch): ", out_b.shape)