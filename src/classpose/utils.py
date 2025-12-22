"""
Utils for training, inference, evaluation and visualization.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from skimage.measure import label
from tqdm import tqdm
from openslide import OpenSlide

from classpose.log import get_logger

utils_logger = get_logger(__name__)


def get_default_device(
    device: str | torch.device | None = None,
) -> torch.device:
    """
    Get the default device to use for computation.

    Args:
        device (str | torch.device | None, optional): Device to use for
            computation. Defaults to None.

    Returns:
        torch.device: The default device to use for computation.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device(device: str | None = None) -> list[torch.device]:
    """
    Get device to use for inference.

    Args:
        device (str | None): Device to use for inference. If None, will use the
            faster available device.

    Returns:
        list[torch.device]: List of devices to use for inference.
    """
    devices = []
    if device is not None:
        if ":" in device:
            device, idxs = device.split(":")
            idxs = idxs.split(",")
            devices = [torch.device(device + ":" + idx) for idx in idxs]
        else:
            devices = [torch.device(device)]
        return devices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        devices = [torch.device("cuda")]
        utils_logger.info("Using CUDA device.")
    elif torch.backends.mps.is_available():
        devices = [torch.device("mps")]
        utils_logger.info("Using MPS device.")
    else:
        devices = [torch.device("cpu")]
        utils_logger.info("Using CPU device.")
    return devices


def make_sparse(
    labels: list[np.ndarray] | np.ndarray,
    fraction: float = 0.1,
    seed: int | np.random.Generator | None = None,
) -> list[np.ndarray] | np.ndarray:
    """
    Make labels sparse by setting a fraction of the semantic labels to 0.

    Args:
        labels (list[np.ndarray] | np.ndarray): List of label arrays or a
            single label array. Expected shape is (N, 2, H, W) where N is the
            number of images, 2 is the number of channels (instance and semantic
            annotations), H and W are the height and width of the images.
        fraction (float, optional): Fraction of instances whose labels are set
            to background. Defaults to 0.1.
        seed (int | np.random.Generator | None, optional): Random seed.
            Defaults to None.

    Returns:
        list[np.ndarray] | np.ndarray: List of sparse label arrays or a single
            sparse label array.
    """
    utils_logger.debug("Making labels sparse")
    if fraction > 1.0 or fraction < 0.0:
        raise ValueError("Fraction must be between 0 and 1")
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, int):
        rng = np.random.default_rng(seed)
    else:
        rng = seed
    is_list = isinstance(labels, list)
    labels = np.array(labels)
    n_prior = 0
    n_after = 0
    utils_logger.debug(f"{np.unique(labels[:, 0])}")
    utils_logger.debug(f"{np.unique(labels[:, 1])}")
    for i in range(labels.shape[0]):
        instance_maps = labels[i, 0]
        n_instances = instance_maps.max() + 1
        n_prior += n_instances
        all_instances = np.arange(1, n_instances, dtype=np.int32)
        kept_instances = rng.choice(
            all_instances, size=int(n_instances * (1 - fraction)), replace=False
        )
        labels[i, 1][np.isin(instance_maps, kept_instances)] = 0
        new_labels = np.zeros_like(instance_maps)
        for idx, new_idx in zip(
            kept_instances, range(len(kept_instances), 0, -1)
        ):
            new_labels[instance_maps == idx] = new_idx
        labels[i, 0] = new_labels
        n_after += np.unique(labels[i, 0]).max()
    utils_logger.debug(f"Number of instances before: {n_prior}")
    utils_logger.debug(f"Number of instances after: {n_after}")
    if is_list:
        return [label for label in labels]
    return labels


def oversample_classes(
    X: np.ndarray,
    Y: np.ndarray,
    n_extra_classes: int = 4,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample classes in Y.

    Based on https://github.com/stardist/stardist/blob/conic-2022/examples/conic-2022/train.ipynb

    Args:
        X (np.ndarray): Array of images.
        Y (np.ndarray): Array of labels (2 channels: instance and class).
        n_extra_classes (int, optional): Number of extra classes to oversample.
            Defaults to 4.
        seed (int | np.random.Generator | None, optional): Random seed.
            Defaults to None.

    Returns:
        tuple: A tuple containing the oversampled images and labels.
    """
    y0 = Y[:, 1]
    rng = np.random.default_rng(seed)

    # get the most infrequent classes
    class_counts = get_class_counts(y0, y0.max() + 1)
    extra_classes = np.argsort(class_counts)[:n_extra_classes]
    all(
        class_counts[c] > 0 or utils_logger.critical(f"count 0 for class {c}")
        for c in extra_classes
    )

    # how many extra samples (more for infrequent classes)
    n_extras = np.sqrt(np.sum(class_counts[1:]) / class_counts[extra_classes])
    n_extras = n_extras / np.max(n_extras)
    utils_logger.info(f"oversample classes: {extra_classes}")
    idx_take = np.arange(len(X))

    for c, n_extra in zip(extra_classes, n_extras):
        # oversample probability is ~ number of instances
        prob = np.sum(y0[:, ::2, ::2] == c, axis=(1, 2))
        prob = np.clip(prob, 0, np.percentile(prob, 99.8))
        prob = prob**2
        # prob[prob<np.percentile(prob,90)] = 0
        prob = prob / np.sum(prob)
        n_extra = int(n_extra * len(X))
        utils_logger.info(f"adding {n_extra} images of class {c}")
        idx_extra = rng.choice(np.arange(len(X)), n_extra, p=prob)
        idx_take = np.append(idx_take, idx_extra)

    X, Y = map(lambda x: x[idx_take], (X, Y))
    return X, Y


def get_class_counts(Y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Get class counts.

    Args:
        Y (np.ndarray): Array of labels (2 channels: instance and class).
        n_classes (int): Number of classes.

    Returns:
        np.ndarray: Array of class counts.
    """
    return np.bincount(Y, minlength=n_classes)


def get_class_weights(
    Y: np.ndarray | list[np.ndarray], n_classes: int
) -> np.ndarray:
    """
    Get class weights.

    Based on https://github.com/stardist/stardist/blob/conic-2022/examples/conic-2022/train.ipynb

    Args:
        Y (np.ndarray): Array of labels (2 channels: instance and class).
        n_classes (int): Number of classes.

    Returns:
        np.ndarray: Array of class weights.
    """
    class_counts = get_class_counts(
        np.concatenate([y[1].ravel() for y in Y]), n_classes
    )
    inv_freq = np.median(class_counts) / class_counts
    inv_freq = inv_freq**0.5
    class_weights = inv_freq.round(4)
    utils_logger.info(f"class weights = {class_weights.tolist()}")
    return class_weights


def plot_images(images: np.ndarray, labels: np.ndarray | None = None):
    """
    Plot images and labels.

    Args:
        images (np.ndarray): Array of images.
        labels (np.ndarray | None, optional): Array of labels. Defaults to None.
    """
    nimg = images.shape[0]
    if labels is not None:
        assert images.shape[0] == labels.shape[0]
        _, ax = plt.subplots(nimg, 2, figsize=(4, 10))
    else:
        _, ax = plt.subplots(nimg, 1, figsize=(4, 10))
    for i in range(nimg):
        if labels is not None:
            ax[i, 0].imshow(images[i])
            ax[i, 1].imshow(labels[i], alpha=0.5)
            ax[i, 0].set_title("Image")
            ax[i, 1].set_title("Label")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
        else:
            ax[i].imshow(images[i])
            ax[i].set_title("Image")
            ax[i].axis("off")
    plt.tight_layout()
    plt.show()


def get_instance_counts(
    labels: np.ndarray, label_instances: bool = False
) -> np.ndarray:
    """
    Get instance counts.

    Args:
        labels (np.ndarray): Array of labels (2 channels: instance and class).
        label_instances (bool, optional): Whether to label instances. Defaults to False.

    Returns:
        np.ndarray: Array of instance counts.
    """
    n_classes = np.max([im[1].max() for im in labels]) + 1
    counts = np.zeros((len(labels), n_classes))
    for i in range(len(labels)):
        if label_instances:
            instances = label(labels[i][0])
        else:
            instances = labels[i][0]
        classes = labels[i][1]
        for j in range(n_classes):
            counts[i, j] = np.unique(instances[classes == j]).size
    return counts


def compute_custom_oversampling_probabilities(
    labels: np.ndarray, power: float = 1
) -> np.ndarray:
    """
    Compute custom oversampling probabilities using instance-weighted class balancing.

    Args:
        labels (np.ndarray): Array of labels (2 channels: instance and class).
        power (float, optional): Power to raise the weights to. Defaults to 1.

    Returns:
        np.ndarray: Normalized probability array for training sample selection.
    """
    classes = np.concatenate([im[1].flatten() for im in labels])
    n_classes = classes.max() + 1
    class_counts = get_class_counts(classes, n_classes)
    class_weights = 1 / class_counts
    class_weights[0] = 0
    instance_counts = get_instance_counts(labels)
    weights = np.sum(instance_counts * class_weights[None], 1)
    weights = weights**power
    weights = weights / weights.sum()
    utils_logger.info(
        f"Custom oversampling - probability range: {weights.min():.6f} to {weights.max():.6f}"
    )
    return weights


def compute_stardist_oversampling_probabilities(
    train_labels: list[np.ndarray] | np.ndarray, n_rare_classes: int = 4
) -> np.ndarray:
    """
    Compute oversampling probabilities using StarDist's methodology for rare class emphasis.
    Based on https://github.com/stardist/stardist/blob/conic-2022/examples/conic-2022/train.ipynb:

    Args:
        train_labels: ClassPose labels with shape (N, C, H, W), channel 1 = class labels
        n_rare_classes: Number of rarest classes to oversample (default: 4)

    Returns:
        Probability array of shape (N,) normalized to sum to 1.0 for use as train_probs
    """
    # Handle both array and list formats
    if isinstance(train_labels, list):
        train_labels = np.array(train_labels)

    # Extract class channel - adapt from StarDist's Y0[:, :, :, 1] to ClassPose's [:, 1, :, :]
    class_channel = train_labels[:, 1, :, :]  # Shape: (N, H, W)
    n_images = class_channel.shape[0]

    # Count instances per class using 4x downsampling (matches StarDist exactly)
    # StarDist: np.bincount(Y0[:, ::4, ::4, 1].ravel(), minlength=len(CLASS_NAMES))
    class_counts = np.bincount(
        class_channel[:, ::4, ::4].ravel(),
        minlength=int(class_channel.max()) + 1,
    )

    # Identify the most infrequent classes (excluding background class 0)
    rare_classes = np.argsort(class_counts[1:])[:n_rare_classes] + 1

    # Validate that rare classes have non-zero counts
    for c in rare_classes:
        if class_counts[c] == 0:
            utils_logger.warning(f"Rare class {c} has zero instances")

    # Compute scaling factors using square root methodology (matches StarDist)
    # StarDist: n_extras = np.sqrt(np.sum(class_counts[1:]) / class_counts[extra_classes])
    total_instances = np.sum(class_counts[1:])  # Exclude background
    scaling_factors = np.sqrt(total_instances / class_counts[rare_classes])

    utils_logger.info(f"Oversampling rare classes: {rare_classes.tolist()}")
    utils_logger.info(f"Scaling factors: {scaling_factors.tolist()}")

    # Start with uniform probabilities
    train_probs = np.ones(n_images, dtype=np.float64)

    # Apply StarDist's probability computation for each rare class
    for c, scaling_factor in zip(rare_classes, scaling_factors):
        # Count instances of rare class per image using 2x downsampling (matches StarDist)
        # StarDist: prob = np.sum(Y0[:, ::2, ::2, 1] == c, axis=(1, 2))
        prob = np.sum(class_channel[:, ::2, ::2] == c, axis=(1, 2)).astype(
            np.float64
        )

        # Apply StarDist's exact methodology
        prob = np.clip(prob, 0, np.percentile(prob, 99.8))  # Clip outliers
        prob = prob**2  # Square for emphasis

        # Only update probabilities for images containing this rare class
        if np.sum(prob) > 0:
            prob = prob / np.max(prob)
            # Weight the probabilities by the scaling factor
            train_probs *= 1.0 + scaling_factor * prob

    # Final normalization to ensure probabilities sum to 1.0
    train_probs = train_probs / np.sum(train_probs)

    utils_logger.info(
        f"Probability statistics - min: {train_probs.min():.6f}, "
        f"max: {train_probs.max():.6f}, median: {np.median(train_probs):.6f}"
    )

    return train_probs


def download_if_unavailable(path: str, url: str) -> str:
    """
    Downloads a file from a URL if it is not available.

    Args:
        path (str): Path to the file.
        url (str): URL to download the file from.

    Returns:
        str: Path to the file.
    """
    if not os.path.exists(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        utils_logger.info("Downloading model %s", path)
        response = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in tqdm(
                response.iter_content(chunk_size=8192),
                unit_scale=True,
                desc="Downloading model",
            ):
                f.write(chunk)
    return path


def get_slide_resolution(openslide_slide: OpenSlide) -> tuple[float, float]:
    """
    Get the slide resolution in microns per pixel.

    Args:
        openslide_slide (OpenSlide): OpenSlide object.

    Returns:
        tuple[float, float]: Resolution in microns per pixel.
    """
    props = openslide_slide.properties
    x, y = None, None
    if "openslide.mpp-x" in props and "openslide.mpp-y" in props:
        x, y = float(props["openslide.mpp-x"]), float(props["openslide.mpp-y"])
        utils_logger.info("Slide resolution extracted from openslide.mpp")
    elif "tiff.XResolution" in props and "tiff.YResolution" in props:
        x, y = float(props["tiff.XResolution"]), float(
            props["tiff.YResolution"]
        )
        utils_logger.info(
            "Slide resolution extracted from tiff.XResolution and tiff.YResolution"
        )
        if props["tiff.ResolutionUnit"].lower() == "centimenter":
            x, y = 10000 / x, 10000 / y
            utils_logger.info(
                "Slide resolution converted from centimeters to microns"
            )
        elif props["tiff.ResolutionUnit"].lower() == "inch":
            x, y = 25400 / x, 25400 / y
            utils_logger.info(
                "Slide resolution converted from inches to microns"
            )
    if x is None or y is None:
        utils_logger.warning("Slide does not have MPP information")
        raise ValueError("Slide does not have MPP information")
    utils_logger.info("Slide resolution: %s", (x, y))
    return x, y
