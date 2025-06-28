# train_lnn.py – advanced supervised training for Liquid-Network Crossover (LNN)
"""
Run example
-----------
$ conda activate ga_lnn
$ python -m GA_LNN.train_lnn \
        --log_dir GA_LNN/data \
        --epochs 200 \
        --batch 128 \
        --lr 3e-4 \
        --wd 1e-4 \
        --alpha 0.2 \
        --aug
"""
from __future__ import annotations

import argparse, os, pickle, random, warnings
from pathlib import Path
from typing import List

import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from GA_LNN.lnn.encoder import encode_parents
from GA_LNN.lnn.model    import LiquidCrossover
from cvrp_loader         import load_all_instances, CVRPInstance

# -------------------------------------------------------------
#  Static config
# -------------------------------------------------------------
F_DIM   = 3                 # rank1, rank2, demand
D_MODEL = 48
ROOT     = Path(__file__).resolve().parent
SAVE_PATH = ROOT / "lnn" / "lnn_hyx.pt"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
#  Pre-load all CVRP instances (for fast lookup in Dataset)
# -------------------------------------------------------------
_INST: dict[str, CVRPInstance] = {inst.name: inst for inst in load_all_instances()}

# -------------------------------------------------------------
#  Helper – permutation → position tensor
# -------------------------------------------------------------
def _perm_to_pos(perm: List[int]) -> torch.Tensor:
    pos = torch.zeros(len(perm), dtype=torch.float32)
    for idx, cid in enumerate(perm):
        pos[cid - 1] = float(idx)
    return pos

# -------------------------------------------------------------
#  Differentiable Kendall-Tau loss
# -------------------------------------------------------------
def kendall_tau_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n   = pred.size(0)
    idx = torch.arange(n, device=pred.device)
    i, j = torch.meshgrid(idx, idx, indexing="ij")
    mask = i < j
    hinge = nn.functional.relu(1 - torch.sign(target[i] - target[j]) * (pred[i] - pred[j]))
    return hinge[mask].mean()

# -------------------------------------------------------------
#  Dataset (with optional PMX + parent-swap augmentation)
# -------------------------------------------------------------
class ParentChildDataset(Dataset):
    def __init__(self, pkl_files: List[Path], augment: bool = False):
        self.samples: list[tuple] = []
        for pf in pkl_files:
            with open(pf, "rb") as fh:
                try:
                    while True:
                        self.samples.extend(pickle.load(fh))
                except EOFError:
                    pass
        if augment:
            self._augment()
        random.shuffle(self.samples)

    @staticmethod
    def _pmx(p1: list[int], p2: list[int]) -> list[int]:
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        for i in range(a, b+1):
            gene = p2[i]
            if gene not in child:
                pos = i
                while gene in child:
                    gene = p2[pos]
                    pos  = p1.index(gene)
                child[pos] = p2[i]
        for i in range(size):
            if child[i] is None:
                child[i] = p2[i]
        return child

    def _augment(self):
        extra = []
        for p1, p2, child, inst in self.samples:
            extra.append((p2, p1, child, inst))                 # parent swap
            extra.append((p1, p2, self._pmx(p1, p2), inst))     # lightweight PMX
        self.samples.extend(extra)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        p1, p2, child, inst_name = self.samples[idx]
        inst   = _INST[inst_name]
        feats  = encode_parents(p1, p2, inst)   # (n, F_DIM)
        target = _perm_to_pos(child)            # (n,)
        return feats, target

# -------------------------------------------------------------
#  Pad-collate (variable length → uniform batch)
# -------------------------------------------------------------
def pad_collate(batch):
    feats_list, pos_list = zip(*batch)
    max_n = max(f.size(0) for f in feats_list)
    F     = feats_list[0].size(1)

    feats_pad = torch.zeros(len(batch), max_n, F)
    pos_pad   = torch.zeros(len(batch), max_n)
    mask      = torch.zeros(len(batch), max_n, dtype=torch.bool)

    for i, (f, p) in enumerate(zip(feats_list, pos_list)):
        n = f.size(0)
        feats_pad[i, :n] = f
        pos_pad[i, :n]   = p
        mask[i, :n]      = True
    return feats_pad, pos_pad, mask

# -------------------------------------------------------------
#  Forward / Train / Eval helpers
# -------------------------------------------------------------
def _forward(model, feats, mask, pos, alpha):
    preds = model(feats, mask)
    mse   = ((preds - pos) ** 2)[mask].mean()
    kt    = kendall_tau_loss(preds[mask], pos[mask])
    return mse + alpha * kt, mse, kt, preds

def train_epoch(model, loader, opt, device, alpha):
    model.train()
    tl = tm = tk = 0.0
    for feats, pos, mask in loader:
        feats, pos, mask = feats.to(device), pos.to(device), mask.to(device)
        loss, mse, kt, _ = _forward(model, feats, mask, pos, alpha)
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item(); tm += mse.item(); tk += kt.item()
    n = len(loader)
    return tl / n, tm / n, tk / n

@torch.no_grad()
def eval_epoch(model, loader, device, alpha):
    model.eval()
    vl = vm = vk = va = 0.0
    for feats, pos, mask in loader:
        feats, pos, mask = feats.to(device), pos.to(device), mask.to(device)
        loss, mse, kt, preds = _forward(model, feats, mask, pos, alpha)
        # simple position-match accuracy
        for b in range(feats.size(0)):
            n = mask[b].sum()
            if n == 0: continue
            order = torch.argsort(preds[b, :n], descending=True)
            inv   = torch.empty_like(order); inv[order] = torch.arange(n, device=order.device)
            va   += (inv == pos[b, :n]).float().mean().item()
        vl += loss.item(); vm += mse.item(); vk += kt.item()
    n = len(loader)
    return vl / n, vm / n, vk / n, va / len(loader.dataset)

# -------------------------------------------------------------
#  Main training driver
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="GA_LNN/data")
    parser.add_argument("--epochs",  type=int, default=200)
    parser.add_argument("--batch",   type=int, default=128)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--wd",      type=float, default=1e-4)
    parser.add_argument("--alpha",   type=float, default=0.2)
    parser.add_argument("--aug",     action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    # ---------- dataset ----------
    all_pkl = sorted(Path(args.log_dir).glob("lnn_log_*.pkl"))
    if not all_pkl:
        raise FileNotFoundError("No *.pkl logs found – run GA logger first.")
    random.shuffle(all_pkl)
    cut = int(0.9 * len(all_pkl))
    train_ds = ParentChildDataset(all_pkl[:cut], augment=args.aug)
    val_ds   = ParentChildDataset(all_pkl[cut:])

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=pad_collate)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=pad_collate)

    # ---------- model / optim / sched ----------
    model = LiquidCrossover(in_dim=F_DIM, d_model=D_MODEL).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.1)

    writer   = SummaryWriter(log_dir=Path(args.log_dir) / "runs")

    # ---------- CSV logger ----------
    csv_path  = Path(args.log_dir) / "training_curve.csv"
    csv_file  = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch","train_loss","train_mse","train_kt",
                         "val_loss","val_mse","val_kt","val_acc","lr"])

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mse, tr_kt   = train_epoch(model, train_ld, opt, device, args.alpha)
        val_loss, val_mse, val_kt, val_acc = eval_epoch(model, val_ld, device, args.alpha)
        sched.step()

        writer.add_scalars("Loss", {"train": tr_loss, "val": val_loss}, epoch)
        writer.add_scalars("MSE",  {"train": tr_mse,  "val": val_mse }, epoch)
        writer.add_scalars("KT",   {"train": tr_kt,   "val": val_kt  }, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", sched.get_last_lr()[0], epoch)

        csv_writer.writerow([epoch, tr_loss, tr_mse, tr_kt,
                             val_loss, val_mse, val_kt, val_acc,
                             sched.get_last_lr()[0]])

        print(f"Epoch {epoch:03d} │ val MSE={val_mse:6.4f} │ KT={val_kt:6.4f} │ Acc={val_acc:.3f}")

        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), SAVE_PATH)
            print("🔥 new best – model saved →", SAVE_PATH)

    writer.close()
    csv_file.close()
    print("✅ Training complete. Best val MSE:", best_val)

# -------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()