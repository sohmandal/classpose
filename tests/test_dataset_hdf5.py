import os
import h5py
import numpy as np
import tempfile
import pytest

from classpose.dataset import ClassposeHDF5Dataset


def test_classpose_hdf5_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = os.path.join(tmpdir, "test_dataset.h5")

        num_samples = 5
        channels, height, width = 3, 32, 32

        images = np.random.rand(num_samples, channels, height, width).astype(
            np.float32
        )
        labels = np.zeros((num_samples, 5, height, width), dtype=np.int32)

        for i in range(num_samples):
            labels[i, 0, 10:20, 10:20] = 1
            labels[i, 1, 10:20, 10:20] = 1
            labels[i, 1, 10:20, 10:20] = 1
            labels[i, 2, 10:20, 10:20] = 0.5
            labels[i, 3, 10:20, 10:20] = 0.5

        labels = labels.astype(np.float32)

        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset(
                "images", data=images, chunks=(1, channels, height, width)
            )
            f.create_dataset(
                "labels", data=labels, chunks=(1, 5, height, width)
            )

        dataset = ClassposeHDF5Dataset(
            hdf5_path=hdf5_path, diam_mean=30.0, augment=False, bsize=32
        )

        assert len(dataset) == num_samples

        img, lbl = dataset[0]

        assert img.shape == (channels, height, width)
        assert lbl.shape == (
            4,
            height,
            width,
        )  # instance, class, flow_y, flow_x
        assert img.dtype == np.float32

        dataset_aug = ClassposeHDF5Dataset(
            hdf5_path=hdf5_path,
            diam_mean=30.0,
            augment=True,
            bsize=32,
            rescale=False,
        )

        img_aug, lbl_aug = dataset_aug[1]
        assert img_aug.shape == (channels, height, width)
        assert lbl_aug.shape == (4, height, width)

        custom_diams = np.array([20.0, 25.0, 30.0, 35.0, 40.0])
        dataset_diams = ClassposeHDF5Dataset(
            hdf5_path=hdf5_path,
            diameter_array=custom_diams,
            augment=False,
            bsize=32,
        )
        assert len(dataset_diams.diameter_array) == num_samples
        assert np.allclose(dataset_diams.diameter_array, custom_diams)


def test_classpose_hdf5_lazy_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = os.path.join(tmpdir, "test_lazy.h5")

        num_samples = 3
        channels, height, width = 3, 32, 32

        images = np.random.rand(num_samples, channels, height, width).astype(
            np.float32
        )
        labels = np.zeros((num_samples, 5, height, width), dtype=np.int32)

        labels[0, 1, 5:10, 5:10] = 1
        labels[1, 1, 5:10, 5:10] = 2
        labels[2, 1, 5:10, 5:10] = 1

        labels[0, 0, 5:10, 5:10] = 1
        labels[1, 0, 5:10, 5:10] = 1
        labels[2, 0, 5:10, 5:10] = 2

        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("images", data=images)
            f.create_dataset("labels", data=labels)

        dataset_compute = ClassposeHDF5Dataset(
            hdf5_path=hdf5_path, bsize=32, augment=False
        )

        assert dataset_compute.n_classes == 3
        assert dataset_compute.class_counts is not None
        assert dataset_compute.instance_counts is not None

        mock_class_counts = np.array([10, 20, 30])
        mock_instance_counts = np.array([[1, 2], [3, 4], [5, 6]])

        with h5py.File(hdf5_path, "a") as f:
            f.create_dataset("class_counts", data=mock_class_counts)
            f.create_dataset("instance_counts", data=mock_instance_counts)

        dataset_direct = ClassposeHDF5Dataset(
            hdf5_path=hdf5_path, bsize=32, augment=False
        )

        assert dataset_direct.n_classes == 3
        assert np.array_equal(dataset_direct.class_counts, mock_class_counts)
        assert np.array_equal(
            dataset_direct.instance_counts, mock_instance_counts
        )


def test_classpose_hdf5_subset_lazy_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = os.path.join(tmpdir, "test_subset_lazy.h5")

        num_samples = 4
        channels, height, width = 3, 32, 32

        images = np.random.rand(num_samples, channels, height, width).astype(
            np.float32
        )
        labels = np.zeros((num_samples, 5, height, width), dtype=np.int32)

        labels[0, 1, 5:10, 5:10] = 1
        labels[1, 1, 5:10, 5:10] = 2
        labels[2, 1, 5:10, 5:10] = 1
        labels[3, 1, 5:10, 5:10] = 2

        labels[0, 0, 5:10, 5:10] = 1
        labels[1, 0, 5:10, 5:10] = 1
        labels[2, 0, 5:10, 5:10] = 2
        labels[3, 0, 5:10, 5:10] = 2

        mock_class_counts = np.array([10, 20, 30])
        mock_instance_counts = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("images", data=images)
            f.create_dataset("labels", data=labels)
            f.create_dataset("class_counts", data=mock_class_counts)
            f.create_dataset("instance_counts", data=mock_instance_counts)

        dataset = ClassposeHDF5Dataset(
            hdf5_path=hdf5_path, bsize=32, augment=False
        )

        _ = dataset.class_counts
        _ = dataset.instance_counts

        subset_indices = [0, 2]
        subset_ds = dataset.subset(subset_indices)

        expected_subset_instance_counts = mock_instance_counts[subset_indices]
        assert np.array_equal(
            subset_ds.instance_counts, expected_subset_instance_counts
        )

        assert not np.array_equal(subset_ds.class_counts, mock_class_counts)


def test_classpose_hdf5_subset_detailed():
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = os.path.join(tmpdir, "test_subset_detailed.h5")

        num_samples = 6
        channels, height, width = 3, 32, 32

        images = np.random.rand(num_samples, channels, height, width).astype(
            np.float32
        )
        labels = np.zeros((num_samples, 5, height, width), dtype=np.int32)

        for i in range(num_samples):
            labels[i, 1, 0:10, 0:10] = 1
            labels[i, 0, 0:10, 0:10] = 1
            labels[i, 1, 10:20, 10:20] = 2
            labels[i, 0, 10:20, 10:20] = 2

        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("images", data=images)
            f.create_dataset("labels", data=labels)

        dataset = ClassposeHDF5Dataset(
            hdf5_path=hdf5_path, bsize=32, augment=False
        )

        sub1_indices = [1, 3, 5]
        sub1 = dataset.subset(sub1_indices)

        assert len(sub1) == 3
        assert len(sub1.images) == 3
        assert len(sub1.labels) == 3

        assert sub1.instance_counts.shape == (3, 3)
        assert sub1.class_counts.shape == (3,)

        sub2_indices = [0, 2]
        sub2 = sub1.subset(sub2_indices)

        assert len(sub2) == 2
        assert len(sub2.images) == 2
        assert len(sub2.labels) == 2

        assert sub2.instance_counts.shape == (2, 3)
        assert sub2.class_counts.shape == (3,)

        expected_sub2_instance_counts = np.array([[1, 1, 1], [1, 1, 1]])
        assert np.array_equal(
            sub2.instance_counts, expected_sub2_instance_counts
        )

        expected_sub2_class_counts = np.array([2 * (32 * 32 - 200), 200, 200])
        assert np.array_equal(sub2.class_counts, expected_sub2_class_counts)


def test_classpose_hdf5_subset_edge_cases():
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = os.path.join(tmpdir, "test_edge_cases.h5")

        num_samples = 5
        channels, height, width = 3, 32, 32

        images = np.random.rand(num_samples, channels, height, width).astype(
            np.float32
        )
        labels = np.zeros((num_samples, 5, height, width), dtype=np.int32)

        labels[:, 1, 0:10, 0:10] = 1  # class 1

        mock_instance_counts = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        )

        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("images", data=images)
            f.create_dataset("labels", data=labels)
            f.create_dataset("instance_counts", data=mock_instance_counts)

        dataset = ClassposeHDF5Dataset(
            hdf5_path=hdf5_path, bsize=32, augment=False
        )

        empty_subset = dataset.subset([])
        assert len(empty_subset) == 0
        assert empty_subset.instance_counts.shape[0] == 0

        with pytest.raises(
            ValueError,
            match="max\\(\\) iterable argument is empty|need at least one array to concatenate",
        ):
            _ = empty_subset.class_counts
