import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class LesionDataset(Dataset):
    """Images and integer labels from a table with `path` and `label` columns."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].astype(str).tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        image = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"], self.labels[i]
