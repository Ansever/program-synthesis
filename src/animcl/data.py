from enum import Enum
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd


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


class AnimalsDataset(Dataset):
    def __init__(self, data_dir: str | Path, english_labels: list[EnglishLabel]):
        self.data_dir = Path(data_dir)
        italian_labels = translate_labels(english_labels)
        self.num_labels = len(italian_labels)

        data = []
        for label_idx, italian_label in enumerate(italian_labels):
            label_dir = self.data_dir / italian_label.value
            for fpath in label_dir.glob("*"):
                if fpath.is_file():
                    data.append(
                        {
                            "italian_label": italian_label,
                            "label_idx": label_idx,
                            "path": str(fpath),
                        }
                    )
        self.data_df = pd.DataFrame(data)

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.data_df.loc[idx, "path"]
        image = read_image(img_path)
        labels = np.zeros(self.num_labels, dtype=int)
        labels[self.data_df.loc[idx, "label_idx"]] = 1
        return image, torch.tensor(labels)


if __name__ == "__main__":
    dataset = AnimalsDataset(
        Path(__file__).parent.parent.parent / "data" / "raw-img",
        [EnglishLabel.DOG, EnglishLabel.CAT],
    )
    print(len(dataset))
    for i in range(5):
        image, labels = dataset[i]
        print(f"Image shape: {image.shape}, Labels: {labels}")
