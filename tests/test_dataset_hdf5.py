import os
import h5py
import numpy as np
import tempfile

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
