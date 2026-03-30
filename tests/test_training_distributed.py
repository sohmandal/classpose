import os
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from classpose.dataset import (
    DistributedEpochSampler,
    SequentialDistributedSampler,
    ClassposeTrainingDataset,
)
from classpose.distributed import cleanup_distributed, setup_distributed
from classpose.train import train_class_seg


def _make_synthetic_sample(index: int) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((3, 32, 32), dtype=np.float32)
    image[1] = np.linspace(0.0, 1.0, 32, dtype=np.float32)[None, :]
    image[2] = np.linspace(0.0, 1.0, 32, dtype=np.float32)[:, None]

    labels = np.zeros((5, 32, 32), dtype=np.float32)

    center_y = 10 + (index % 3)
    center_x = 11 + (index % 4)
    rr, cc = np.ogrid[:32, :32]
    mask = (rr - center_y) ** 2 + (cc - center_x) ** 2 <= 16

    # instance mask (channel 0)
    labels[0, mask] = 1.0

    # class (channel 1)
    labels[1, mask] = 1 + (index % 2)

    # instance (1 to n where n is the number of instances)
    labels[2, mask] = 1.0

    # flows (channels 3, 4)
    labels[3, mask] = 0.5  # dummy flow y
    labels[4, mask] = 0.5  # dummy flow x

    return image, labels


def _make_synthetic_dataset(
    train_count: int = 4, test_count: int = 2
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]
]:
    samples = [
        _make_synthetic_sample(i) for i in range(train_count + test_count)
    ]
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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_cuda_ddp_smoke(
    rank: int,
    world_size: int,
    master_port: int,
    output_dir: str,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    context = setup_distributed(
        device_arg="cuda",
        backend="nccl",
        timeout_seconds=120,
    )
    try:
        (
            train_images,
            train_labels,
            test_images,
            test_labels,
        ) = _make_synthetic_dataset()

        train_dataset = ClassposeTrainingDataset(
            data_array=np.array(train_images),
            label_array=np.array(train_labels),
            diameter_array=np.full(len(train_images), 30.0),
        )
        test_dataset = ClassposeTrainingDataset(
            data_array=np.array(test_images),
            label_array=np.array(test_labels),
            diameter_array=np.full(len(test_images), 30.0),
        )

        train_sampler = DistributedEpochSampler(
            dataset_length=len(train_images),
            batch_size=2,
            train_probs=None,
            nimg_per_epoch=len(train_images),
            rank=context.rank,
            num_replicas=context.world_size,
            seed=123,
        )
        train_sampler.set_epoch(0)
        np.save(
            Path(output_dir) / f"rank_{context.rank}_train_indices.npy",
            train_sampler.local_indices(),
        )

        val_sampler = SequentialDistributedSampler(
            dataset_length=len(test_images),
            rank=context.rank,
            num_replicas=context.world_size,
        )
        np.save(
            Path(output_dir) / f"rank_{context.rank}_val_indices.npy",
            np.asarray(val_sampler.indices(), dtype=np.int64),
        )

        net = ToyClassposeNet(device=str(context.device))
        ddp_net = DDP(
            net,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )
        train_class_seg(
            net=ddp_net,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=2,
            n_epochs=1,
            save_path=output_dir,
            model_name="cuda_ddp_smoke",
            save_every=1,
            num_workers=0,
            validate_every_epoch=True,
            use_uncertainty_weighting=True,
            device=context.device,
            distributed=context.distributed,
            rank=context.rank,
            world_size=context.world_size,
            config_snapshot={"test_case": "cuda_ddp_smoke"},
            random_seed=123,
        )
    finally:
        cleanup_distributed()


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

    assert (
        sampler_a.local_indices().tolist() == sampler_b.local_indices().tolist()
    )


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
    (
        train_images,
        train_labels,
        test_images,
        test_labels,
    ) = _make_synthetic_dataset()

    train_dataset = ClassposeTrainingDataset(
        data_array=np.array(train_images),
        label_array=np.array(train_labels),
        diameter_array=np.full(len(train_images), 30.0),
    )
    test_dataset = ClassposeTrainingDataset(
        data_array=np.array(test_images),
        label_array=np.array(test_labels),
        diameter_array=np.full(len(test_images), 30.0),
    )

    net = ToyClassposeNet(device="cpu")

    model_path, train_losses, test_losses = train_class_seg(
        net=net,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=2,
        n_epochs=1,
        save_path=tmp_path,
        model_name="single_process_smoke",
        save_every=1,
        num_workers=0,
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


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not dist.is_available()
    or not dist.is_nccl_available(),
    reason="requires more than one CUDA GPU with NCCL",
)
def test_train_class_seg_cuda_ddp_smoke(tmp_path):
    output_dir = str(tmp_path)
    world_size = 2
    master_port = _find_free_port()

    mp.spawn(
        _run_cuda_ddp_smoke,
        args=(world_size, master_port, output_dir),
        nprocs=world_size,
        join=True,
    )

    model_dir = tmp_path / "cuda_ddp_smoke"
    assert (model_dir / "cuda_ddp_smoke").exists()
    assert (model_dir / "cuda_ddp_smoke_best").exists()
    assert (model_dir / "checkpoint_last.train.pt").exists()
    assert (model_dir / "checkpoint_best.train.pt").exists()

    train_rank0 = np.load(tmp_path / "rank_0_train_indices.npy")
    train_rank1 = np.load(tmp_path / "rank_1_train_indices.npy")
    assert set(train_rank0.tolist()).isdisjoint(set(train_rank1.tolist()))
    assert sorted(train_rank0.tolist() + train_rank1.tolist()) == [0, 1, 2, 3]

    val_rank0 = np.load(tmp_path / "rank_0_val_indices.npy")
    val_rank1 = np.load(tmp_path / "rank_1_val_indices.npy")
    assert set(val_rank0.tolist()).isdisjoint(set(val_rank1.tolist()))
    assert sorted(val_rank0.tolist() + val_rank1.tolist()) == [0, 1]

    checkpoint = torch.load(
        model_dir / "checkpoint_last.train.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert len(checkpoint["rng_state_by_rank"]) == 2
    assert checkpoint["config_snapshot"]["test_case"] == "cuda_ddp_smoke"
