from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from cellpose.transforms import normalize_img, random_rotate_and_resize
from torch.utils.data import Dataset, Sampler

from classpose.transforms import create_stardist_augmentation, get_config


def _build_augment_pipeline(augment_pipeline_config: str | None):
    if augment_pipeline_config is None:
        return None
    return create_stardist_augmentation(get_config(augment_pipeline_config))


def augment_single_image(
    imgs: np.ndarray,
    lbls: np.ndarray,
    diams: float,
    diam_mean: float,
    rescale: bool,
    scale_range: float | list[float] | None,
    bsize: int,
    normalize_params: dict[str, Any],
    augment: bool,
    augment_pipeline: Any,
) -> tuple[np.ndarray, np.ndarray]:
    rsc = np.array(
        [diams / diam_mean if rescale else 1.0],
        dtype=np.float32,
    )
    if augment:
        if augment_pipeline is not None:
            imgs = augment_pipeline.transform(imgs)
        imgi, lbl = random_rotate_and_resize(
            [imgs],
            Y=[lbls],
            rescale=rsc,
            scale_range=scale_range,
            xy=(bsize, bsize),
        )[:2]
        image = imgi[0]
        label = lbl[0]
    else:
        image = imgs
        label = lbls

    image = normalize_img(image, **normalize_params)
    return np.ascontiguousarray(image), np.ascontiguousarray(label)


class ClassposeTrainingDataset(Dataset):
    def __init__(
        self,
        data_array: np.ndarray,
        label_array: np.ndarray,
        diameter_array: np.ndarray,
        augment_pipeline_config: str | None = None,
        diam_mean: float = 100,
        rescale: bool = True,
        scale_range: float | list[float] | None = None,
        bsize: int = 128,
        normalize_params: dict[str, Any] = {"normalize": True},
        augment: bool = True,
    ):
        self.data_array = data_array
        self.label_array = label_array
        self.diameter_array = diameter_array
        self.augment_pipeline_config = augment_pipeline_config
        self.diam_mean = diam_mean
        self.rescale = rescale
        self.scale_range = scale_range
        self.bsize = bsize
        self.normalize_params = normalize_params
        self.augment = augment
        self._augment_pipeline = None

    def __len__(self) -> int:
        return len(self.data_array)

    def _get_augment_pipeline(self):
        if not self.augment or self.augment_pipeline_config is None:
            return None
        if self._augment_pipeline is None:
            self._augment_pipeline = _build_augment_pipeline(
                self.augment_pipeline_config
            )
        return self._augment_pipeline

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return augment_single_image(
            self.data_array[index],
            self.label_array[index][1:],
            float(self.diameter_array[index]),
            diam_mean=self.diam_mean,
            rescale=self.rescale,
            scale_range=self.scale_range,
            bsize=self.bsize,
            normalize_params=self.normalize_params,
            augment=self.augment,
            augment_pipeline=self._get_augment_pipeline(),
        )


class DistributedEpochSampler(Sampler[int]):
    def __init__(
        self,
        dataset_length: int,
        batch_size: int,
        train_probs: np.ndarray | None = None,
        nimg_per_epoch: int | None = None,
        rank: int = 0,
        num_replicas: int = 1,
        seed: int = 0,
    ):
        if dataset_length <= 0:
            raise ValueError("dataset_length must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        self.dataset_length = dataset_length
        self.batch_size = batch_size
        self.train_probs = None
        if train_probs is not None:
            train_probs = np.asarray(train_probs, dtype=np.float64)
            if train_probs.shape[0] != dataset_length:
                raise ValueError(
                    "train_probs must have the same length as the dataset"
                )
            if np.any(train_probs < 0):
                raise ValueError("train_probs must be non-negative")
            if float(train_probs.sum()) <= 0.0:
                raise ValueError("train_probs must sum to a positive value")
            self.train_probs = train_probs / train_probs.sum()

        self.nimg_per_epoch = (
            dataset_length if nimg_per_epoch is None else int(nimg_per_epoch)
        )
        if self.nimg_per_epoch <= 0:
            raise ValueError("nimg_per_epoch must be positive")
        if self.train_probs is None and self.nimg_per_epoch > dataset_length:
            raise ValueError(
                "nimg_per_epoch cannot exceed the dataset size without oversampling"
            )

        self.rank = rank
        self.num_replicas = num_replicas
        self.seed = seed
        self.epoch = 0
        self.global_batch_size = self.num_replicas * self.batch_size
        self._local_num_samples = self._compute_local_num_samples()

    def _build_global_indices(self, epoch: int | None = None) -> np.ndarray:
        epoch = self.epoch if epoch is None else epoch
        rng = np.random.default_rng(self.seed + epoch)
        all_indices = np.arange(self.dataset_length, dtype=np.int64)

        if self.train_probs is None:
            global_indices = rng.permutation(all_indices)[: self.nimg_per_epoch]
        else:
            global_indices = rng.choice(
                all_indices,
                size=self.nimg_per_epoch,
                p=self.train_probs,
            )

        usable_size = global_indices.shape[0] - (
            global_indices.shape[0] % self.global_batch_size
        )
        if usable_size == 0:
            raise ValueError(
                "The epoch does not contain enough samples for even one full "
                f"distributed batch. Lower batch_size ({self.batch_size}), lower "
                f"world_size ({self.num_replicas}), or increase nimg_per_epoch "
                f"({self.nimg_per_epoch})."
            )

        return np.asarray(global_indices[:usable_size], dtype=np.int64)

    def _compute_local_num_samples(self) -> int:
        return self._build_local_indices(epoch=0).shape[0]

    def _build_local_indices(self, epoch: int | None = None) -> np.ndarray:
        global_indices = self._build_global_indices(epoch=epoch)
        reshaped = global_indices.reshape(-1, self.num_replicas, self.batch_size)
        return reshaped[:, self.rank, :].reshape(-1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def local_indices(self, epoch: int | None = None) -> np.ndarray:
        return self._build_local_indices(epoch=epoch)

    def __iter__(self) -> Iterator[int]:
        return iter(self._build_local_indices().tolist())

    def __len__(self) -> int:
        return self._local_num_samples


class SequentialDistributedSampler(Sampler[int]):
    def __init__(
        self,
        dataset_length: int,
        rank: int = 0,
        num_replicas: int = 1,
    ):
        if dataset_length < 0:
            raise ValueError("dataset_length must be non-negative")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        self.dataset_length = dataset_length
        self.rank = rank
        self.num_replicas = num_replicas

        base = dataset_length // num_replicas
        remainder = dataset_length % num_replicas
        self.start_index = rank * base + min(rank, remainder)
        self.end_index = self.start_index + base + (
            1 if rank < remainder else 0
        )

    def indices(self) -> list[int]:
        return list(range(self.start_index, self.end_index))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices())

    def __len__(self) -> int:
        return self.end_index - self.start_index
