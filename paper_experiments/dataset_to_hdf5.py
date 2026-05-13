import os
import argparse
import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path

from classpose.train_utils import _process_train_test
from classpose.log import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ClassPose dataset to HDF5 format. "
        "This will only work if all images are identical in size."
    )
    parser.add_argument(
        "--data_dirs",
        type=str,
        required=True,
        nargs="+",
        help="Directories containing images.npy and labels.npy",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        required=True,
        help="Number of classes",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output HDF5 file",
    )
    return parser.parse_args()


def resize_and_increment(dataset, key, new_items, n: int):
    item_shape = dataset[key].shape[1:]
    logger.info("Appending %d items with shape %s to %s", n, item_shape, key)
    old_size = dataset[key].shape[0]
    dataset[key].resize((old_size + n, *item_shape))
    for i, item in enumerate(new_items):
        dataset[key][old_size + i] = item


def get_class_counts(Y: list[np.ndarray], n_classes: int) -> np.ndarray:
    """
    Get class counts directly from class array.

    Args:
        Y (np.ndarray): Array of labels (1 channel: class).
        n_classes (int): Number of classes.

    Returns:
        np.ndarray: Array of class counts.
    """
    return np.bincount(
        np.concatenate([y[y >= 0].ravel() for y in Y]), minlength=n_classes
    )


def get_instance_counts(
    processed_labels: np.ndarray | list[np.ndarray],
    classes: np.ndarray | list[np.ndarray],
    n_classes: int | None = None,
) -> np.ndarray:
    """
    Simplified version of get instance counts.

    Args:
        processed_labels (np.ndarray): instance + flows array or list of such
            arrays.
        classes (np.ndarray): class array or list of class arrays.
        n_classes (int | None, optional): Number of classes. If None, it will be
            inferred from the labels. Defaults to None.

    Returns:
        np.ndarray: Array of instance counts.
    """
    if n_classes is None:
        n_classes = np.max([im.max() for im in classes]) + 1
    counts = np.zeros((len(processed_labels), n_classes))
    for i in range(len(processed_labels)):
        ins = processed_labels[i][0]
        cla = classes[i][0]
        for j in range(n_classes):
            counts[i, j] = np.unique(ins[cla == j]).size
    return counts


def load_data(data_dir):
    images_path = os.path.join(data_dir, "images.npy")
    labels_path = os.path.join(data_dir, "labels.npy")

    images = np.load(images_path, allow_pickle=True)
    if np.issubdtype(images[0].dtype, np.object_):
        new_images = np.empty(len(images), dtype=object)
        for i, im in enumerate(images):
            new_images[i] = np.ascontiguousarray(im).astype(np.float32)
        images = new_images

    labels = np.load(labels_path, allow_pickle=True)
    if np.issubdtype(labels[0].dtype, np.object_):
        new_labels = np.empty(len(labels), dtype=object)
        for i, im in enumerate(labels):
            new_labels[i] = np.ascontiguousarray(im).astype(np.int32)
        labels = new_labels

    transp = (2, 0, 1)
    train_images = [np.transpose(im, transp) for im in images]
    train_labels = [np.transpose(im, transp) for im in labels]

    del images
    del labels

    return train_images, train_labels


def main():
    args = parse_args()

    class_count_accumulator = np.zeros(args.n_classes)
    logger.info(f"Loading data from {len(args.data_dirs)} dirs...")
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_file, "w") as f:
        # load a signal dataset
        train_images, train_labels = load_data(args.data_dirs[0])
        h, w = train_images[0].shape[1:]
        img_shape = (3, h, w)
        lbl_shape = (5, h, w)
        f.create_dataset(
            "images",
            shape=(0, *img_shape),
            maxshape=(None, *img_shape),
            chunks=(1, *img_shape),
        )
        f.create_dataset(
            "labels",
            shape=(0, *lbl_shape),
            maxshape=(None, *lbl_shape),
            chunks=(1, *lbl_shape),
        )
        f.create_dataset(
            "instance_counts",
            shape=(0, args.n_classes),
            maxshape=(None, args.n_classes),
        )
        for i, data_dir in enumerate(args.data_dirs):
            logger.info("Processing %s", data_dir)
            if i > 0:
                train_images, train_labels = load_data(data_dir)
            logger.info("Processing data (computing flows, filtering)...")
            out = _process_train_test(
                train_data=train_images,
                train_labels=train_labels,
                min_train_masks=0,
                compute_flows=True,
                normalize_params={"normalize": False},
            )

            processed_images = out[0]
            processed_labels = out[1]
            processed_classes = out[5]

            # instantiate as an iterator to avoid loading all labels in memory
            final_labels = (
                np.concatenate((y[:1], y_class, y[1:]), axis=0)
                for y, y_class in zip(processed_labels, processed_classes)
            )

            logger.info("Computing class counts")
            class_count_accumulator += get_class_counts(
                processed_classes, args.n_classes
            )

            logger.info("Getting instance counts")
            instance_counts = get_instance_counts(
                processed_labels, processed_classes, n_classes=args.n_classes
            )

            logger.info("Writing data")
            n = len(processed_images)
            resize_and_increment(f, "images", processed_images, n=n)
            resize_and_increment(f, "labels", final_labels, n=n)
            resize_and_increment(f, "instance_counts", instance_counts, n=n)

            del processed_images
            del instance_counts

        logger.info("Storing class counts")
        f["class_counts"] = class_count_accumulator

    logger.info("Done!")


if __name__ == "__main__":
    main()
