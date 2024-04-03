import torch
from torch.utils.data import Dataset
from pathlib import Path
import typer
import pandas as pd
import torch.nn.functional as F

class RegimenDataset(Dataset):
    """Regimen dataset."""

    """Initializes instance of class RegimenDataset.
    Args:
    input_path (str): Path to the csv file with the regimen data.
    nrows (int): Number of rows to read per mini-batch from the csv file.
    """

    def __init__(
        self,
        input_path: Path = typer.Argument(..., dir_okay=False, help="CSV file to load"),
        nrows: int = typer.Argument(),
    ):

        df = pd.read_csv(input_path)
        # todo: create per day per person instances
        df = df[["drug_id", "route_id", "day"]]
        # todo: partition by person and nrows
        df['id'] = df.index // nrows
        subsequences = [sub_df.drop(columns='id') for _, sub_df in df.groupby('id')]
        subsequences = [torch.from_numpy(sub_df.to_numpy()) for sub_df in subsequences]

        self.data = torch.zeros((len(subsequences), 3, 224, 224))
        for i, subsequence in enumerate(subsequences):
            pad_left = (224 - subsequence.size(1)) // 2
            pad_right = 224 - pad_left - subsequence.size(1)
            pad_top = (224 - subsequence.size(0)) // 2
            pad_bottom = 224 - pad_top - subsequence.size(0)
            # pad_dims = (0, 224 - subsequence.size(1), 0, 224 - subsequence.size(0))
            padded_subsequence = F.pad(subsequence, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
            padded_subsequence = padded_subsequence.repeat(3, 1, 1)
            self.data[i] = padded_subsequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
