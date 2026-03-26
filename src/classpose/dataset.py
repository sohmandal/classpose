from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterator

import numpy as np
import h5py
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


class ClassposeDataset(Dataset):
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def subset(self, indices):
        dataset_copy = deepcopy(self)
        dataset_copy.indices = indices
        dataset_copy.length = len(indices)
        return dataset_copy


class ClassposeTrainingDataset(ClassposeDataset):
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

        self.length = len(self.data_array)
        self.indices = np.arange(0, self.length, dtype=np.uint32)

        self.n_classes = int(
            max([np.max(lbl[1]) for lbl in self.label_array]) + 1
        )

    def __len__(self) -> int:
        return self.length

    def _get_augment_pipeline(self):
        if not self.augment or self.augment_pipeline_config is None:
            return None
        if self._augment_pipeline is None:
            self._augment_pipeline = _build_augment_pipeline(
                self.augment_pipeline_config
            )
        return self._augment_pipeline

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        index = self.indices[index]
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

    @property
    def images(self):
        return self.data_array

    @property
    def labels(self):
        return [lbl[:2].astype(np.int16) for lbl in self.label_array]


class ClassposeHDF5Dataset(ClassposeDataset):
    def __init__(
        self,
        hdf5_path: str,
        diameter_array: np.ndarray | None = None,
        augmentation_strategy: str | None = None,
        diam_mean: float = 30,
        rescale: bool = False,
        scale_range: float | list[float] | None = 0.5,
        bsize: int = 256,
        normalize_params: dict[str, Any] = {"normalize": True},
        augment: bool = True,
        keep_open: bool = False,
    ):
        self.hdf5_path = hdf5_path
        self.augmentation_strategy = augmentation_strategy
        self.diam_mean = diam_mean
        self.rescale = rescale
        self.scale_range = scale_range
        self.bsize = bsize
        self.normalize_params = normalize_params
        self.augment = augment
        self.keep_open = keep_open
        self._augment_pipeline = None
        self._hdf5_file = None

        self.length = self._get_length()
        self.indices = np.arange(self.length, dtype=np.int32)

        if diameter_array is not None:
            self.diameter_array = diameter_array
        else:
            self.diameter_array = np.ones(self.length) * diam_mean

        self._n_classes = None

    @property
    def hdf5_file(self):
        if self.keep_open:
            if self._hdf5_file is None:
                self._hdf5_file = h5py.File(self.hdf5_path, "r")
            return self._hdf5_file
        else:
            return h5py.File(self.hdf5_path, "r")

    def _get_length(self):
        if self.keep_open:
            return len(self.hdf5_file["images"])
        else:
            with h5py.File(self.hdf5_path, "r") as f:
                return len(f["images"])

    def _get_item(
        self, i: int | list[int] | slice
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(i, slice):
            i = [i for i in range(slice.start, slice.stop, slice.step)]
        if self.keep_open:
            if isinstance(i, (list, np.ndarray)):
                images = np.stack([self.hdf5_file["images"][idx] for idx in i])
                labels = np.stack([self.hdf5_file["labels"][idx] for idx in i])
            else:
                images = self.hdf5_file["images"][i]
                labels = self.hdf5_file["labels"][i]
            return images, labels
        else:
            with h5py.File(self.hdf5_path, "r") as f:
                if isinstance(i, (list, np.ndarray)):
                    images = np.stack([f["images"][idx] for idx in i])
                    labels = np.stack([f["labels"][idx] for idx in i])
                else:
                    images = f["images"][i]
                    labels = f["labels"][i]
                return images, labels

    def __len__(self) -> int:
        return self.length

    def _get_augment_pipeline(self):
        if not self.augment or self.augmentation_strategy is None:
            return None
        if self._augment_pipeline is None:
            self._augment_pipeline = _build_augment_pipeline(
                self.augmentation_strategy
            )
        return self._augment_pipeline

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        img, lbl = self._get_item(self.indices[index])

        return augment_single_image(
            img,
            lbl[1:],
            float(self.diameter_array[self.indices[index]]),
            diam_mean=self.diam_mean,
            rescale=self.rescale,
            scale_range=self.scale_range,
            bsize=self.bsize,
            normalize_params=self.normalize_params,
            augment=self.augment,
            augment_pipeline=self._get_augment_pipeline(),
        )

    @property
    def images(self):
        return [self._get_item(i)[0] for i in range(len(self))]

    @property
    def labels(self):
        out = [
            self._get_item(i)[1][:2].astype(np.int16) for i in range(len(self))
        ]
        return out

    @property
    def n_classes(self):
        if self._n_classes is None:
            self._n_classes = int(
                max([lbl[1].max() for lbl in self.labels]) + 1
            )
        return self._n_classes


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
        reshaped = global_indices.reshape(
            -1, self.num_replicas, self.batch_size
        )
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
        self.end_index = (
            self.start_index + base + (1 if rank < remainder else 0)
        )

    def indices(self) -> list[int]:
        return list(range(self.start_index, self.end_index))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices())

    def __len__(self) -> int:
        return self.end_index - self.start_index
