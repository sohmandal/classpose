import os
import argparse
import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path

from classpose.train import _process_train_test
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
        "--output_file",
        type=str,
        required=True,
        help="Path to output HDF5 file",
    )
    return parser.parse_args()


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

    return train_images, train_labels


def main():
    args = parse_args()

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
        for data_dir in args.data_dirs:
            logger.info("Processing %s", data_dir)
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

            final_labels = [
                np.concatenate((y[:1], y_class, y[1:]), axis=0)
                for y, y_class in tqdm(
                    zip(processed_labels, processed_classes),
                    total=len(processed_labels),
                )
            ]

            image_data = np.stack(processed_images)
            label_data = np.stack(final_labels)

            logger.info(
                "Appending input array of size %s to HDF5", image_data.shape
            )
            f["images"].resize(
                (f["images"].shape[0] + image_data.shape[0], *img_shape)
            )
            f["images"][-image_data.shape[0] :] = image_data
            logger.info(
                "Appending label array of size %s to HDF5", image_data.shape
            )
            f["labels"].resize(
                (f["labels"].shape[0] + label_data.shape[0], *lbl_shape)
            )
            f["labels"][-label_data.shape[0] :] = label_data

    logger.info("Done!")


if __name__ == "__main__":
    main()
