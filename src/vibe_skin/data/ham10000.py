"""Read HAM10000 metadata and build leak-free train/val splits."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from vibe_skin.config import resolve

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}


def _find_images(dirs: list[Path]) -> dict[str, Path]:
    return {p.stem: p for d in dirs for p in d.glob("*.jpg")}


def load_train_metadata(cfg: dict) -> pd.DataFrame:
    root = resolve(cfg["root"])
    df = pd.read_csv(root / cfg["metadata"])
    images = _find_images([root / d for d in cfg["image_dirs"]])
    df["path"] = df["image_id"].map(images)
    return df


def load_test_metadata(cfg: dict) -> pd.DataFrame:
    root = resolve(cfg["root"])
    df = pd.read_csv(root / cfg["test_metadata"])
    images = _find_images([root / cfg["test_image_dir"]])
    df["path"] = df["image_id"].map(images)
    return df


def make_splits(df: pd.DataFrame, val_fraction: float, seed: int) -> pd.Series:
    """Assign each image to train or val, keeping all images of a lesion together
    and roughly preserving class proportions."""
    n_splits = round(1 / val_fraction)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(df, df["dx"], groups=df["lesion_id"]))
    split = pd.Series("train", index=df.index)
    split.iloc[val_idx] = "val"
    return split


def load_splits(cfg: dict) -> pd.DataFrame:
    """Train/val/test table with columns image_id, lesion_id, dx, label, split, path."""
    train = load_train_metadata(cfg)
    splits = pd.read_csv(resolve(cfg["splits_file"]))
    train = train.merge(splits[["image_id", "split"]], on="image_id", how="inner")
    test = load_test_metadata(cfg).dropna(subset=["path"]).assign(split="test")
    df = pd.concat([train, test], ignore_index=True)
    df["label"] = df["dx"].map(CLASSES.index)
    return df
