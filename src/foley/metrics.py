"""Compatibility module exposing evaluation metrics."""
from metrics import (accuracy_confusion, aligned_rmse, check_wav_format, classify,
                     nn_train, pairwise_rmse)

__all__ = ["accuracy_confusion", "aligned_rmse", "check_wav_format", "classify",
           "nn_train", "pairwise_rmse"]
