"""
Utils for training, inference, evaluation and visualization.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from tqdm import tqdm
from openslide import OpenSlide

from classpose.log import get_logger

utils_logger = get_logger(__name__)

ALLOW_UNSAFE_REQUESTS = os.getenv("ALLOW_UNSAFE_REQUESTS", "false").lower() in [
    "true",
    "1",
]
if ALLOW_UNSAFE_REQUESTS:
    utils_logger.warning(
        "Unsafe requests enabled. This is not recommended for production use."
    )


GEOJSON_OUTPUT_TEMPLATES = {
    "cell_contours": os.getenv(
        "CLASSPOSE_CELL_CONTOURS_GEOJSON",
        "{base_name}_cell_contours.geojson",
    ),
    "cell_centroids": os.getenv(
        "CLASSPOSE_CELL_CENTROIDS_GEOJSON",
        "{base_name}_cell_centroids.geojson",
    ),
    "tissue_contours": os.getenv(
        "CLASSPOSE_TISSUE_CONTOURS_GEOJSON",
        "{base_name}_tissue_contours.geojson",
    ),
    "artefact_contours": os.getenv(
        "CLASSPOSE_ARTEFACT_CONTOURS_GEOJSON",
        "{base_name}_artefact_contours.geojson",
    ),
    "roi": os.getenv(
        "CLASSPOSE_ROI_GEOJSON",
        "{base_name}_roi.geojson",
    ),
}


def get_geojson_output_filename(output_kind: str, base_name: str) -> str:
    """
    Get the filename for a GeoJSON output file.

    The output_kind must be one of the keys in ``GEOJSON_OUTPUT_TEMPLATES``.

    Args:
        output_kind (str): The kind of output (e.g., "cell_contours", "cell_centroids").
        base_name (str): The base name of the file.

    Returns:
        str: The filename for the GeoJSON output file.
    """
    if output_kind not in GEOJSON_OUTPUT_TEMPLATES:
        valid_options = ", ".join(GEOJSON_OUTPUT_TEMPLATES.keys())
        err = f"Invalid output kind: {output_kind}. Valid options are: {valid_options}"
        utils_logger.error(err)
        raise ValueError(err)
    template = GEOJSON_OUTPUT_TEMPLATES[output_kind]
    return template.format(base_name=base_name)


def get_geojson_output_path_from_prefix(
    output_prefix: str | Path, output_kind: str
) -> Path:
    """
    Get the path for a GeoJSON output file from a prefix.

    The output_kind must be one of the keys in ``GEOJSON_OUTPUT_TEMPLATES``.

    Args:
        output_prefix (str | Path): The prefix of the output file.
        output_kind (str): The kind of output (e.g., "cell_contours", "cell_centroids").

    Returns:
        Path: The path for the GeoJSON output file.
    """
    output_prefix = Path(output_prefix)
    return output_prefix.with_name(
        get_geojson_output_filename(output_kind, output_prefix.name)
    )


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


def download_if_unavailable(
    path: str, url: str, description: str = "Downloading model"
) -> str:
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
        utils_logger.info("%s %s", description, path)
        try:
            response = requests.get(url, stream=True, verify=True)
        except:
            if not ALLOW_UNSAFE_REQUESTS:
                utils_logger.error(f"Cannot download slide from {url}")
                utils_logger.error(
                    "Downloading using unsafe requests requires setting ALLOW_UNSAFE_REQUESTS to True"
                )
                raise ValueError(
                    "Downloading using unsafe requests requires setting ALLOW_UNSAFE_REQUESTS to True"
                )
            utils_logger.warning(
                "Downloading using unsafe requests. This is not recommended for production use."
            )
            response = requests.get(url, stream=True, verify=False)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        with open(path, "wb") as f:
            with tqdm(
                total=total_size, unit_scale=True, unit="B", desc=description
            ) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
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
        utils_logger.debug("Slide resolution extracted from openslide.mpp")
    elif "tiff.XResolution" in props and "tiff.YResolution" in props:
        x, y = float(props["tiff.XResolution"]), float(
            props["tiff.YResolution"]
        )
        utils_logger.debug(
            "Slide resolution extracted from tiff.XResolution and tiff.YResolution"
        )
        if props["tiff.ResolutionUnit"].lower() == "centimeter":
            x, y = 10000 / x, 10000 / y
            utils_logger.debug(
                "Slide resolution converted from centimeters to microns"
            )
        elif props["tiff.ResolutionUnit"].lower() == "inch":
            x, y = 25400 / x, 25400 / y
            utils_logger.debug(
                "Slide resolution converted from inches to microns"
            )
        else:
            utils_logger.error(
                "Slide resolution unit not recognized: %s",
                props["tiff.ResolutionUnit"],
            )
            raise ValueError(
                "Slide resolution unit not recognized: {}".format(
                    props["tiff.ResolutionUnit"]
                )
            )
    if x is None or y is None:
        utils_logger.error("Slide does not have MPP information")
        raise ValueError("Slide does not have MPP information")
    utils_logger.debug("Slide resolution: %s", (x, y))
    return x, y
