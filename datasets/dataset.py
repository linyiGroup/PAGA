from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TCRpHLADataset(Dataset):
    def __init__(
        self,
        csv_path,
        embedding_dict_path,
        embedding_root,
        max_length=34,
        embedding_dim=960,
    ):
        self.data = pd.read_csv(csv_path)
        self.embedding_root = Path(embedding_root)
        self.max_length = max_length
        self.label_col = "Label"
        self.embedding_dim = embedding_dim

        required_columns = {
            "CDR3",
            "MT_pep",
            "HLA_sequence",
            "Label",
        }

        missing = required_columns - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        with open(embedding_dict_path, "rb") as file:
            self.embedding_dict = pickle.load(file)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def clean_sequence(sequence):
        return str(sequence).strip().upper().replace(" ", "")

    @lru_cache(maxsize=32768)
    def load_embedding(self, sequence):
        if sequence not in self.embedding_dict:
            raise KeyError(
                f"Sequence not found in embedding dictionary: {sequence}"
            )

        emb_path = self.embedding_dict[sequence]

        path = Path(emb_path)

        if not path.is_absolute():
            path = self.embedding_root / path

        embedding = np.load(path).astype(np.float32)

        if embedding.shape != (self.max_length, self.embedding_dim):
            raise ValueError(
                f"Unexpected ESMc embedding shape for {sequence}: "
                f"{embedding.shape}. Expected: ({self.max_length}, 960)"
            )


        if embedding.shape[0] != self.max_length:
            raise ValueError(
                f"Unexpected embedding length for {sequence}: "
                f"{embedding.shape[0]}. Expected length: {self.max_length}"
            )

        return torch.from_numpy(embedding)

    def create_mask(self, sequence):
        length = min(
            len(self.clean_sequence(sequence)),
            self.max_length,
        )

        mask = torch.zeros(
            self.max_length,
            dtype=torch.bool,
        )
        mask[:length] = True

        return mask

    def __getitem__(self, index):
        row = self.data.iloc[index]

        tcr = self.clean_sequence(row["CDR3"])
        peptide = self.clean_sequence(row["MT_pep"])
        hla = self.clean_sequence(row["HLA_sequence"])

        return {
            "tcr_embedding": self.load_embedding(tcr),
            "peptide_embedding": self.load_embedding(peptide),
            "hla_embedding": self.load_embedding(hla),

            "tcr_mask": self.create_mask(tcr),
            "peptide_mask": self.create_mask(peptide),
            "hla_mask": self.create_mask(hla),

            "label": torch.tensor(
                float(row["Label"]),
                dtype=torch.float32,
            ),
        }