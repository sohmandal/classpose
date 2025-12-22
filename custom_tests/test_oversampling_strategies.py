"""
Test and compare oversampling strategies.
"""

import numpy as np
import pandas as pd
from classpose.utils import (
    oversample_classes,
    get_train_probs,
    get_instance_counts,
)

TEST_DATA_DIR = "data/conic"

labels = np.load(f"{TEST_DATA_DIR}/labels.npy")

instance_counts, class_weights, weights = get_train_probs(
    labels.transpose(0, 3, 1, 2)
)
instance_counts_p05, class_weights_p05, weights_p05 = get_train_probs(
    labels.transpose(0, 3, 1, 2), power=0.5
)
instance_counts_p2, class_weights_p2, weights_p2 = get_train_probs(
    labels.transpose(0, 3, 1, 2), power=2
)
original_train_counts = instance_counts.sum(0)
oversampled_train_counts = get_instance_counts(
    oversample_classes(
        labels, labels.transpose(0, 3, 1, 2), n_extra_classes=4, seed=42
    )[1][:, 1]
).sum(0)
resampled_train_counts = instance_counts[
    np.random.default_rng(42)
    .choice(np.arange(labels.shape[0]), p=weights, size=labels.shape[0])
    .astype(int)
].sum(0)
resampled_train_counts_p2 = instance_counts_p2[
    np.random.default_rng(42)
    .choice(np.arange(labels.shape[0]), p=weights_p2, size=labels.shape[0])
    .astype(int)
].sum(0)
resampled_train_counts_p05 = instance_counts_p05[
    np.random.default_rng(42)
    .choice(np.arange(labels.shape[0]), p=weights_p05, size=labels.shape[0])
    .astype(int)
].sum(0)


digits = 3
original_train_counts = original_train_counts / original_train_counts.sum()
oversampled_train_counts = (
    oversampled_train_counts / oversampled_train_counts.sum()
)
resampled_train_counts = resampled_train_counts / resampled_train_counts.sum()
resampled_train_counts_p2 = (
    resampled_train_counts_p2 / resampled_train_counts_p2.sum()
)
resampled_train_counts_p05 = (
    resampled_train_counts_p05 / resampled_train_counts_p05.sum()
)

df = pd.DataFrame(
    {
        "original": original_train_counts,
        "oversampled": oversampled_train_counts,
        "resampled_p05": resampled_train_counts_p05,
        "resampled": resampled_train_counts,
        "resampled_p2": resampled_train_counts_p2,
    }
)
df = df.round(digits)

print(df)
