"""
HE Staining augmentation adapted from StarDist's CoNIC implementation.
"""

import warnings

import cv2
import numpy as np
from sklearn.decomposition import NMF

# Suppress sklearn convergence warnings globally for this module
warnings.filterwarnings(
    "ignore", "Maximum number of iterations", module="sklearn"
)
warnings.filterwarnings("ignore", "convergence", module="sklearn")


def _assert_uint8_image(x):
    """Assert that input is a uint8 RGB image."""
    assert x.ndim == 3 and x.shape[-1] == 3 and x.dtype.type is np.uint8


def rgb_to_density(x):
    """Convert RGB uint8 image to optical density."""
    _assert_uint8_image(x)
    x = np.maximum(x, 1)
    return np.maximum(-1 * np.log(x / 255), 1e-6)


def density_to_rgb(x):
    """Convert optical density image back to RGB uint8."""
    return np.clip(255 * np.exp(-x), 0, 255).astype(np.uint8)


def rgb_to_lab(x):
    """Convert RGB uint8 image to LAB color space."""
    _assert_uint8_image(x)
    return cv2.cvtColor(x, cv2.COLOR_RGB2LAB)


def lab_to_rgb(x):
    """Convert LAB uint8 image back to RGB color space."""
    _assert_uint8_image(x)
    return cv2.cvtColor(x, cv2.COLOR_LAB2RGB)


def extract_stains(x, subsample=128, l1_reg=0.001, tissue_threshold=200):
    """Extract H&E stain matrix and stain concentrations using NMF.

    Following StarDist's exact implementation.

    Args:
        x: RGB uint8 image
        subsample: Subsampling factor for NMF fitting
        l1_reg: L1 regularization for NMF
        tissue_threshold: LAB lightness threshold to identify tissue pixels

    Returns:
        tuple: (H, stains)
            - H: stain matrix (2, 3)
            - stains: stain concentrations (H, W, 2)
    """
    _assert_uint8_image(x)

    model = NMF(
        n_components=2,
        init="random",
        random_state=0,
        alpha_W=l1_reg,
        alpha_H=0,
        l1_ratio=1,
    )

    # optical density
    density = rgb_to_density(x)

    # only select darker regions
    tissue_mask = rgb_to_lab(x)[..., 0] < tissue_threshold

    values = density[tissue_mask]

    # Handle edge case where no tissue pixels are found
    if len(values) == 0:
        # Use all pixels if no tissue detected
        values = density.reshape(-1, 3)

    # compute stain matrix on subsampled values (way faster)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(values) > subsample:
            model.fit(values[::subsample])
        else:
            model.fit(values)

    H = model.components_

    # normalize rows
    H = H / np.linalg.norm(H, axis=1, keepdims=True)
    if H[0, 0] < H[1, 0]:
        H = H[[1, 0]]

    # get stains on full image
    Hinv = np.linalg.pinv(H)
    stains = density.reshape((-1, 3)) @ Hinv
    stains = stains.reshape(x.shape[:2] + (2,))

    return H, stains


def stains_to_rgb(stains, stain_matrix):
    """Reconstruct RGB image from stain concentrations and stain matrix.

    Following StarDist's exact implementation.
    """
    assert stains.ndim == 3 and stains.shape[-1] == 2
    assert stain_matrix.shape == (2, 3)
    return density_to_rgb(stains @ stain_matrix)


def augment_stains(
    x,
    amount_matrix=0.15,
    amount_stains=0.4,
    n_samples=1,
    subsample=128,
    rng=None,
):
    """Augment HE staining of RGB image.

    Following StarDist's exact implementation.

    Args:
        x: RGB uint8 image array of shape (H, W, 3)
        amount_matrix: Amount of matrix perturbation
        amount_stains: Amount of stain perturbation
        n_samples: Number of augmented samples to generate
        subsample: Number of pixels to subsample for NMF
        rng: Random number generator

    Returns:
        Augmented RGB uint8 image(s)
    """
    _assert_uint8_image(x)
    if rng is None:
        rng = np.random

    M, stains = extract_stains(x, subsample=subsample)

    M = np.expand_dims(M, 0) + amount_matrix * rng.uniform(
        -1, 1, (n_samples, 2, 3)
    )
    M = np.maximum(M, 0)

    stains = np.expand_dims(stains, 0) * (
        1 + amount_stains * rng.uniform(-1, 1, (n_samples, 1, 1, 2))
    )
    stains = np.maximum(stains, 0)

    if n_samples == 1:
        return stains_to_rgb(stains[0], M[0])
    else:
        return np.stack(
            tuple(stains_to_rgb(s, m) for s, m in zip(stains, M)), 0
        )


class HEStainingTransform:
    """HE staining augmentation transform."""

    def __init__(
        self,
        amount_matrix: float = 0.15,
        amount_stains: float = 0.4,
        probability: float = 0.9,
        seed: int | None = None,
    ):
        """
        Initialize HE staining transform.

        Args:
            amount_matrix: Amount of matrix perturbation (0.15 from StarDist)
            amount_stains: Amount of stain perturbation (0.4 from StarDist)
            probability: Probability of applying augmentation (0.9 from StarDist)
            seed: Random seed for reproducibility
        """
        self.amount_matrix = amount_matrix
        self.amount_stains = amount_stains
        self.probability = probability

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Apply HE staining augmentation to a single image.

        Args:
            image: RGB image array of shape (H, W, 3) or (3, H, W)

        Returns:
            Augmented image with same shape as input
        """
        # Skip augmentation based on probability
        if self.rng.random() > self.probability:
            return image

        # Handle channel dimension
        if image.shape[0] == 3:  # Channel first
            image_hwc = np.transpose(image, (1, 2, 0))
            channels_first = True
        else:  # Channel last
            image_hwc = image
            channels_first = False

        # Convert to uint8 if needed
        if image_hwc.dtype != np.uint8:
            if image_hwc.max() <= 1.0:
                image_hwc = (image_hwc * 255).astype(np.uint8)
            else:
                image_hwc = image_hwc.astype(np.uint8)

        # Apply HE staining augmentation
        try:
            # Convert numpy RNG to old-style for compatibility with StarDist code
            legacy_rng = np.random.RandomState(
                self.rng.integers(0, 2**32 - 1)
            )

            augmented = augment_stains(
                image_hwc,
                amount_matrix=self.amount_matrix,
                amount_stains=self.amount_stains,
                rng=legacy_rng,
            )

            # Convert back to original format
            if image.dtype == np.float32 or image.dtype == np.float64:
                augmented = augmented.astype(image.dtype) / 255.0
            else:
                augmented = augmented.astype(image.dtype)

        except Exception as e:
            warnings.warn(
                f"HE staining augmentation failed: {e}, returning original image"
            )
            augmented = image_hwc

            # Convert back to original format for failed case
            if image.dtype == np.float32 or image.dtype == np.float64:
                augmented = augmented.astype(image.dtype) / 255.0
            else:
                augmented = augmented.astype(image.dtype)

        # Convert back to original channel order
        if channels_first:
            augmented = np.transpose(augmented, (2, 0, 1))

        return augmented

    def transform_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Apply HE staining augmentation to a batch of images.

        Args:
            images: Batch of images with shape (N, H, W, 3) or (N, 3, H, W)

        Returns:
            Batch of augmented images with same shape as input
        """
        return np.array([self.transform(img) for img in images])
