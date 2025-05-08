from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torchvision.io import decode_image
import torch


class EnglishLabel(str, Enum):
    DOG = "dog"
    HORSE = "horse"
    ELEPHANT = "elephant"
    BUTTERFLY = "butterfly"
    CHICKEN = "chicken"
    CAT = "cat"
    COW = "cow"
    SHEEP = "sheep"
    SPIDER = "spider"
    SQUIRREL = "squirrel"


class ItalianLabel(str, Enum):
    CANE = "cane"
    CAVALLO = "cavallo"
    ELEFANTE = "elefante"
    FARFALLA = "farfalla"
    GALLINA = "gallina"
    GATTO = "gatto"
    MUCCA = "mucca"
    PECORA = "pecora"
    RAGNO = "ragno"
    SCOIATTOLO = "scoiattolo"


def translate_labels(labels: list[EnglishLabel]) -> list[ItalianLabel]:
    translate = {
        EnglishLabel.DOG: ItalianLabel.CANE,
        EnglishLabel.HORSE: ItalianLabel.CAVALLO,
        EnglishLabel.ELEPHANT: ItalianLabel.ELEFANTE,
        EnglishLabel.BUTTERFLY: ItalianLabel.FARFALLA,
        EnglishLabel.CHICKEN: ItalianLabel.GALLINA,
        EnglishLabel.CAT: ItalianLabel.GATTO,
        EnglishLabel.COW: ItalianLabel.MUCCA,
        EnglishLabel.SHEEP: ItalianLabel.PECORA,
        EnglishLabel.SPIDER: ItalianLabel.RAGNO,
        EnglishLabel.SQUIRREL: ItalianLabel.SCOIATTOLO,
    }
    return [translate[label] for label in labels]


def build_dataframes(
    data_dir: str | Path, english_labels: list[EnglishLabel], test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    data_dir = Path(data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Invalid data directory: {data_dir}")

    italian_labels = translate_labels(english_labels)
    num_labels = len(italian_labels)

    data = []
    for label_idx, italian_label in enumerate(italian_labels):
        label_dir = data_dir / italian_label.value
        for fpath in label_dir.glob("*"):
            if fpath.is_file():
                data.append(
                    {
                        "italian_label": italian_label,
                        "english_label": english_labels[label_idx],
                        "label_idx": label_idx,
                        "path": str(fpath),
                    }
                )
    data_df = pd.DataFrame(data)

    train_df, test_df = train_test_split(data_df, test_size=test_size)
    return train_df, test_df, num_labels


class AnimalsDataset(Dataset):
    def __init__(
        self, data_df: pd.DataFrame, num_labels: int, transform: Any | None = None
    ):
        self.data_df = data_df
        self.num_labels = num_labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.data_df.iloc[idx]
        image = decode_image(entry["path"])
        if self.transform is not None:
            image = self.transform(image)

        labels = np.zeros(self.num_labels, dtype=int)
        labels[entry["label_idx"]] = 1
        return image, torch.tensor(labels)


def extract_episode(n_support: int, n_query: int, d):
    # data: N x C x H x W
    n_examples = d["data"].size(0)

    if n_query == -1:
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[: (n_support + n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d["data"][support_inds]
    xq = d["data"][query_inds]

    return {"class": d["class"], "xs": xs, "xq": xq}
