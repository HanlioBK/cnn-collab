"""小工具函式。"""
import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """計算一個 batch 的準確率。"""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()
