"""Train a lesion classifier.

Usage: uv run python -m vibe_skin.train --config configs/baseline.yaml
"""
import argparse
import json
import math
import random
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vibe_skin.config import load_config, resolve
from vibe_skin.data.dataset import LesionDataset
from vibe_skin.data.ham10000 import CLASSES, load_splits
from vibe_skin.data.transforms import eval_transforms, train_transforms
from vibe_skin.metrics import compute_metrics, plot_confusion, summary_line
from vibe_skin.models.factory import create_model, set_backbone_trainable


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(labels: np.ndarray, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    w = len(labels) / (len(CLASSES) * counts)
    if mode == "sqrt_inverse":
        w = np.sqrt(w)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.to(device, non_blocking=True))
        probs.append(logits.float().softmax(1).cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_phase(name, model, epochs, lr, warmup_epochs, cfg, loaders, criterion, device, state):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])
    steps = len(loaders["train"])
    scheduler = make_scheduler(optimizer, epochs * steps, warmup_epochs * steps)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, n = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(x).float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(y)
            n += len(y)

        y_val, p_val = predict(model, loaders["val"], device)
        m = compute_metrics(y_val, p_val, CLASSES)
        state["epoch"] += 1
        record = {"epoch": state["epoch"], "phase": name, "train_loss": total_loss / n,
                  "lr": scheduler.get_last_lr()[0], "seconds": time.time() - t0,
                  **{k: v for k, v in m.items() if not isinstance(v, (dict, list))}}
        state["history"].append(record)

        improved = m["balanced_accuracy"] > state["best"]
        if improved:
            state["best"] = m["balanced_accuracy"]
            torch.save(model.state_dict(), state["run_dir"] / "best.pt")
            state["best_metrics"] = m
        print(f"[{name} {epoch:2d}/{epochs}] loss {record['train_loss']:.4f} | {summary_line(m)} | "
              f"{record['seconds']:.0f}s{'  *' if improved else ''}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = load_config(cfg["data_config"])
    seed_everything(cfg["seed"])
    device = torch.device("cuda")

    run_dir = resolve("runs") / f"{datetime.now():%Y%m%d_%H%M}_{cfg['name']}"
    run_dir.mkdir(parents=True)
    shutil.copy(resolve(args.config), run_dir / "config.yaml")
    print(f"Run dir: {run_dir}")

    df = load_splits(data_cfg)
    train_df, val_df = df[df["split"] == "train"], df[df["split"] == "val"]
    size, bs, nw = cfg["img_size"], cfg["batch_size"], cfg["num_workers"]
    loaders = {
        "train": DataLoader(LesionDataset(train_df, train_transforms(size)), batch_size=bs, shuffle=True,
                            num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=True),
        "val": DataLoader(LesionDataset(val_df, eval_transforms(size)), batch_size=bs * 2,
                          num_workers=nw, pin_memory=True, persistent_workers=True),
    }

    weights = class_weights(train_df["label"].to_numpy(), cfg["class_weighting"])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    model = create_model(cfg["model"], len(CLASSES)).to(device)

    state = {"epoch": 0, "best": -1.0, "history": [], "run_dir": run_dir, "best_metrics": None}
    t0 = time.time()

    set_backbone_trainable(model, False)
    run_phase("head", model, cfg["head"]["epochs"], cfg["head"]["lr"], 0, cfg, loaders, criterion, device, state)
    set_backbone_trainable(model, True)
    run_phase("full", model, cfg["full"]["epochs"], cfg["full"]["lr"], cfg["full"]["warmup_epochs"],
              cfg, loaders, criterion, device, state)

    minutes = (time.time() - t0) / 60
    (run_dir / "history.json").write_text(json.dumps(state["history"], indent=2))
    best = state["best_metrics"]
    (run_dir / "metrics_val.json").write_text(json.dumps(best, indent=2))
    plot_confusion(best["confusion_matrix"], CLASSES, run_dir / "confusion_val.png", "Validation (best epoch)")
    print(f"Done in {minutes:.1f} min. Best val: {summary_line(best)}")


if __name__ == "__main__":
    main()
