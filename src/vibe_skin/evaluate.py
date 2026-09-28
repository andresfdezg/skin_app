"""Evaluate a trained run on a split (default: the official ISIC 2018 test set).

Usage: uv run python -m vibe_skin.evaluate runs/<run_dir> [--split test]
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from vibe_skin.config import load_config, resolve
from vibe_skin.data.dataset import LesionDataset
from vibe_skin.data.ham10000 import CLASSES, load_splits
from vibe_skin.data.transforms import eval_transforms
from vibe_skin.metrics import compute_metrics, plot_confusion, summary_line
from vibe_skin.models.factory import create_model
from vibe_skin.train import predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    run_dir = resolve(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    data_cfg = load_config(cfg["data_config"])
    device = torch.device("cuda")

    model = create_model({**cfg["model"], "pretrained": False}, len(CLASSES))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    model.to(device)

    df = load_splits(data_cfg)
    df = df[df["split"] == args.split].reset_index(drop=True)
    loader = DataLoader(LesionDataset(df, eval_transforms(cfg["img_size"])),
                        batch_size=128, num_workers=cfg["num_workers"], pin_memory=True)
    y, probs = predict(model, loader, device)

    m = compute_metrics(y, probs, CLASSES)
    (run_dir / f"metrics_{args.split}.json").write_text(json.dumps(m, indent=2))
    plot_confusion(m["confusion_matrix"], CLASSES, run_dir / f"confusion_{args.split}.png",
                   f"{args.split} ({len(df)} images)")
    preds = pd.DataFrame(probs, columns=CLASSES).assign(image_id=df["image_id"], dx=df["dx"],
                                                          pred=[CLASSES[i] for i in probs.argmax(1)])
    preds.to_csv(run_dir / f"predictions_{args.split}.csv", index=False)

    print(f"{args.split}: {summary_line(m)}")
    print(f"{'class':6s} {'n':>5s} {'sens':>6s} {'spec':>6s} {'auc':>6s}")
    for c, v in m["per_class"].items():
        print(f"{c:6s} {v['support']:5d} {v['sensitivity']:6.3f} {v['specificity']:6.3f} {v['auc']:6.3f}")


if __name__ == "__main__":
    main()
