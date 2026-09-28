"""Load a trained run and classify single images.

Usage: uv run python -m vibe_skin.predict runs/<run_dir> image.jpg [image2.jpg ...]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from vibe_skin.config import load_config, resolve
from vibe_skin.data.dataset import load_image
from vibe_skin.data.ham10000 import CLASS_NAMES, CLASSES
from vibe_skin.data.transforms import eval_transforms
from vibe_skin.metrics import MALIGNANT
from vibe_skin.models.factory import create_model


class Predictor:
    def __init__(self, run_dir: str | Path, device: str | None = None):
        self.run_dir = resolve(run_dir)
        cfg = load_config(self.run_dir / "config.yaml")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = create_model({**cfg["model"], "pretrained": False}, len(CLASSES))
        state = torch.load(self.run_dir / "best.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.transform = eval_transforms(cfg["img_size"])
        self.name = self.run_dir.name

    @torch.no_grad()
    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        x = self.transform(image=image)["image"].unsqueeze(0).to(self.device)
        return self.model(x).float().softmax(1)[0].cpu().numpy()

    def predict(self, image: np.ndarray) -> dict:
        probs = self.predict_proba(image)
        top = int(probs.argmax())
        return {
            "prediction": {
                "class": CLASSES[top],
                "name": CLASS_NAMES[CLASSES[top]],
                "probability": float(probs[top]),
            },
            "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
            "malignant_probability": float(sum(probs[CLASSES.index(c)] for c in MALIGNANT)),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    predictor = Predictor(args.run_dir)
    for path in args.images:
        r = predictor.predict(load_image(path))
        p = r["prediction"]
        print(f"{path}: {p['class']} ({p['probability']:.2f}) | malignant {r['malignant_probability']:.2f}")


if __name__ == "__main__":
    main()
