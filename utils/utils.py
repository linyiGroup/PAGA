from pathlib import Path
import os

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)

import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np

import yaml
from datasets.dataset import TCRpHLADataset
from torch.utils.data import DataLoader


class Logger(object):
    def __init__(self, log_path):
        self.log_path = log_path
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
    def write(self, message):
        print(message)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(message)

            
def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid config file: {config_path}"
        )

    return config


def build_loader(csv_path, config, shuffle):
    dataset = TCRpHLADataset(
        csv_path=csv_path,
        embedding_dict_path=config["data"]["embedding_dict_path"],
        embedding_root=config["data"]["embedding_root"],
        max_length=config["data"]["max_length"],
    )

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    return loader


def forward_model(model, batch, device):
    return model(
        hla_embedding=batch["hla_embedding"].to(device),
        peptide_embedding=batch["peptide_embedding"].to(device),
        tcr_embedding=batch["tcr_embedding"].to(device),
        hla_mask=batch["hla_mask"].to(device),
        peptide_mask=batch["peptide_mask"].to(device),
        tcr_mask=batch["tcr_mask"].to(device),
    )


def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device
):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Train"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["label"].long()

        optimizer.zero_grad()
        logits = forward_model(model, batch, device)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += (loss.item() * len(labels))

    return total_loss / len(loader.dataset)


def calculate_ppvn(y_true, y_scores, n):
    if n == 0:
        return 0.0
    sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    sorted_y_true = [y_true[i] for i in sorted_indices]
    ppvn = sum(sorted_y_true[:n]) / n
    return ppvn


@torch.no_grad()
def evaluate(
        model,
        loader,
        criterion,
        device
):
    model.eval()

    total_loss = 0
    labels = []
    probabilities = []

    for batch in tqdm(loader, desc="Valid"):
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch["label"].long()

        logits = forward_model(model, batch, device)
        loss = criterion(logits, y)

        total_loss += (loss.item() * len(y))

        prob = torch.softmax(logits, dim=1)[:, 1]

        probabilities.extend(prob.cpu().numpy())
        labels.extend(y.cpu().numpy())

    labels = np.array(labels)
    probabilities = np.array(probabilities)
    predictions = (probabilities >= 0.5).astype(int)

    n_positive = int(np.sum(labels == 1))
    ppvn_score = calculate_ppvn(labels, probabilities, n_positive)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
        "roc_auc": roc_auc_score(labels, probabilities),
        "aupr": average_precision_score(labels, probabilities),
        "ppvn": ppvn_score  # 新增 PPVN
    }

    metrics["labels"] = labels
    metrics["probabilities"] = probabilities

    return metrics
