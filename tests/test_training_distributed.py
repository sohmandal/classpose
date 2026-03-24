from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from classpose.dataset import (
    DistributedEpochSampler,
    SequentialDistributedSampler,
)
from classpose.train import train_class_seg


def _make_synthetic_sample(index: int) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((3, 32, 32), dtype=np.float32)
    image[1] = np.linspace(0.0, 1.0, 32, dtype=np.float32)[None, :]
    image[2] = np.linspace(0.0, 1.0, 32, dtype=np.float32)[:, None]

    instance = np.zeros((32, 32), dtype=np.int16)
    classes = np.zeros((32, 32), dtype=np.int16)

    center_y = 10 + (index % 3)
    center_x = 11 + (index % 4)
    rr, cc = np.ogrid[:32, :32]
    mask = (rr - center_y) ** 2 + (cc - center_x) ** 2 <= 16
    instance[mask] = 1
    classes[mask] = 1 + (index % 2)
    image[0, mask] = 1.0

    return image, np.stack([instance, classes], axis=0)


def _make_synthetic_dataset(
    train_count: int = 4, test_count: int = 2
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    samples = [_make_synthetic_sample(i) for i in range(train_count + test_count)]
    train_samples = samples[:train_count]
    test_samples = samples[train_count:]
    train_images = [image for image, _ in train_samples]
    train_labels = [label for _, label in train_samples]
    test_images = [image for image, _ in test_samples]
    test_labels = [label for _, label in test_samples]
    return train_images, train_labels, test_images, test_labels


class ToyClassposeNet(nn.Module):
    def __init__(self, n_cell_classes: int = 3, device: str = "cpu"):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.out = nn.Conv2d(8, 3, kernel_size=1)
        self.out_class = nn.Conv2d(8, n_cell_classes, kernel_size=1)
        self.W2 = nn.Parameter(torch.tensor(1.0))
        self.n_cell_classes = n_cell_classes
        self.diam_mean = torch.tensor([30.0])
        self.diam_labels = nn.Parameter(
            torch.tensor([30.0]), requires_grad=False
        )
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        features = self.encoder(x.float())
        seg_logits = self.out(features) * self.W2
        class_logits = self.out_class(features)
        combined = torch.cat((class_logits, seg_logits), dim=1)
        aux = torch.zeros((x.shape[0], 8), device=x.device)
        return combined, aux

    def save_model(self, filename: str | Path):
        torch.save(self.state_dict(), filename)
        
        
def test_distributed_epoch_sampler_is_deterministic():
    probs = np.array([0.05, 0.15, 0.30, 0.50], dtype=np.float64)
    sampler_a = DistributedEpochSampler(
        dataset_length=4,
        batch_size=1,
        train_probs=probs,
        nimg_per_epoch=8,
        rank=0,
        num_replicas=2,
        seed=7,
    )
    sampler_b = DistributedEpochSampler(
        dataset_length=4,
        batch_size=1,
        train_probs=probs,
        nimg_per_epoch=8,
        rank=0,
        num_replicas=2,
        seed=7,
    )

    sampler_a.set_epoch(3)
    sampler_b.set_epoch(3)

    assert sampler_a.local_indices().tolist() == sampler_b.local_indices().tolist()


def test_distributed_epoch_sampler_shards_without_overlap_for_uniform_sampling():
    sampler_rank0 = DistributedEpochSampler(
        dataset_length=8,
        batch_size=2,
        train_probs=None,
        nimg_per_epoch=8,
        rank=0,
        num_replicas=2,
        seed=11,
    )
    sampler_rank1 = DistributedEpochSampler(
        dataset_length=8,
        batch_size=2,
        train_probs=None,
        nimg_per_epoch=8,
        rank=1,
        num_replicas=2,
        seed=11,
    )

    sampler_rank0.set_epoch(0)
    sampler_rank1.set_epoch(0)

    rank0_indices = sampler_rank0.local_indices().tolist()
    rank1_indices = sampler_rank1.local_indices().tolist()

    assert set(rank0_indices).isdisjoint(set(rank1_indices))
    assert sorted(rank0_indices + rank1_indices) == list(range(8))


def test_distributed_epoch_sampler_truncates_to_full_distributed_batches():
    sampler = DistributedEpochSampler(
        dataset_length=10,
        batch_size=2,
        train_probs=np.full(10, 0.1, dtype=np.float64),
        nimg_per_epoch=10,
        rank=0,
        num_replicas=2,
        seed=5,
    )

    assert len(sampler) == 4
    assert len(sampler.local_indices()) == 4


def test_distributed_epoch_sampler_errors_when_no_full_global_batch_exists():
    with pytest.raises(ValueError, match="Lower batch_size"):
        DistributedEpochSampler(
            dataset_length=3,
            batch_size=2,
            train_probs=np.full(3, 1 / 3, dtype=np.float64),
            nimg_per_epoch=3,
            rank=0,
            num_replicas=2,
            seed=1,
        )


def test_sequential_distributed_sampler_covers_validation_set_once():
    sampler_rank0 = SequentialDistributedSampler(
        dataset_length=5,
        rank=0,
        num_replicas=2,
    )
    sampler_rank1 = SequentialDistributedSampler(
        dataset_length=5,
        rank=1,
        num_replicas=2,
    )

    rank0_indices = sampler_rank0.indices()
    rank1_indices = sampler_rank1.indices()

    assert set(rank0_indices).isdisjoint(set(rank1_indices))
    assert sorted(rank0_indices + rank1_indices) == list(range(5))


def test_train_class_seg_single_process_smoke(tmp_path):
    train_images, train_labels, test_images, test_labels = _make_synthetic_dataset()
    net = ToyClassposeNet(device="cpu")

    model_path, train_losses, test_losses = train_class_seg(
        net=net,
        train_data=train_images,
        train_labels=train_labels,
        test_data=test_images,
        test_labels=test_labels,
        batch_size=2,
        n_epochs=1,
        save_path=tmp_path,
        model_name="single_process_smoke",
        save_every=1,
        num_workers=0,
        min_train_masks=0,
        validate_every_epoch=True,
        use_uncertainty_weighting=True,
        device=torch.device("cpu"),
        distributed=False,
        rank=0,
        world_size=1,
        config_snapshot={"test_case": "single_process_smoke"},
        random_seed=123,
    )

    model_dir = tmp_path / "single_process_smoke"
    assert Path(model_path).exists()
    assert (model_dir / "checkpoint_last.train.pt").exists()
    assert (model_dir / "checkpoint_best.train.pt").exists()
    assert train_losses.shape == (1,)
    assert test_losses.shape == (1,)
