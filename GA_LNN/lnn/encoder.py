# GA_LNN/lnn/encoder.py
import numpy as np
import torch

def encode_parents(p1_perm, p2_perm, instance):
    """
    Build an (n, F) PyTorch tensor for LNN.
    Features per customer:
      - Normalized rank in parent1
      - Normalized rank in parent2
      - Demand / capacity
    """
    n   = len(p1_perm)
    cap = float(instance.vehicle_capacity)

    # Demands, normalized
    demands = np.array(instance.demands[1:]) / cap  # shape (n,)

    # Ranks: position → [0..1]
    rank1 = np.argsort(np.argsort(p1_perm)) / (n - 1 + 1e-8)
    rank2 = np.argsort(np.argsort(p2_perm)) / (n - 1 + 1e-8)

    # Stack → (n, 3)
    feats = np.column_stack([rank1, rank2, demands])
    return torch.tensor(feats, dtype=torch.float32)