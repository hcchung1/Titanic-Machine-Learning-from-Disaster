from __future__ import annotations

from typing import Iterable, List, Tuple

import torch # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch.nn as nn # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch.nn.functional as F # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from torch.amp import GradScaler, autocast # pyright: ignore[reportMissingModuleSource, reportMissingImports]


class TitanicMLP(nn.Module):
    """Simple feed-forward network for tabular Titanic features."""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (256, 128, 64), dropout: float = 0.3):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification (0 = died, 1 = survived)
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(features)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_examples += labels.size(0)
        total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * total_correct / total_examples
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device, return_predictions: bool = False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_labels: List[int] | None = [] if return_predictions else None
    all_preds: List[int] | None = [] if return_predictions else None

    use_amp = device.type == 'cuda'

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(features)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(features)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_examples += labels.size(0)
            total_correct += (preds == labels).sum().item()
            if return_predictions:
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * total_correct / total_examples
    if return_predictions:
        return avg_loss, accuracy, all_labels or [], all_preds or []
    return avg_loss, accuracy


def test(model, test_loader, device):
    model.eval()
    predictions: List[Tuple[int, int]] = []
    with torch.no_grad():
        for features, passenger_ids in test_loader:
            features = features.to(device)
            outputs = model(features)
            preds = outputs.argmax(dim=1).cpu().tolist()
            for pid, pred in zip(passenger_ids.tolist(), preds):
                predictions.append((int(pid), int(pred)))
    return predictions
