"""Backward-compatible facade re-exporting dataset helpers from ``srp26.data``."""

from srp26.data import MTSDataset, create_dataloader

__all__ = ["MTSDataset", "create_dataloader"]
