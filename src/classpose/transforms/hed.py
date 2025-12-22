"""
Implements color space conversion between HED and RGB.

Adapted from https://github.com/DIAGNijmegen/pathology-he-auto-augment
"""

import numpy as np
from scipy import linalg
from skimage.exposure import rescale_intensity

RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]
).astype(np.float32)
HED_FROM_RGB = linalg.inv(RGB_FROM_HED).astype(np.float32)


def rgb2hed(
    rgb: np.ndarray, simple_mode: bool = False
) -> tuple[np.ndarray, float]:
    """
    Convert RGB to HED.

    Args:
        rgb (numpy.ndarray): RGB image.
        simple_mode (bool): If True, assumes input is in [0,1] range and uses simplified logic.
                           If False, uses complex logic for handling potential negative values.

    Returns:
        tuple[numpy.ndarray, float]: HED image and shift.
    """
    return separate_stains(rgb, HED_FROM_RGB, simple_mode)


def hed2rgb(hed: np.ndarray, shift: float) -> np.ndarray:
    """
    Convert HED to RGB.

    Args:
        hed (numpy.ndarray): HED image.
        shift (float): Shift.

    Returns:
        numpy.ndarray: RGB image.
    """
    return combine_stains(hed, RGB_FROM_HED, shift)


def separate_stains(
    rgb: np.ndarray, conv_matrix: np.ndarray, simple_mode: bool = False
) -> np.ndarray:
    """
    Separate stains from an RGB image.

    Args:
        rgb (numpy.ndarray): RGB image.
        conv_matrix (numpy.ndarray): Conversion matrix.
        simple_mode (bool): If True, assumes input is in [0,1] range and uses simplified logic.
                           If False, uses complex logic for handling potential negative values.

    Returns:
        numpy.ndarray: Stains.
    """
    rgb = rgb.astype(np.float32)

    if simple_mode:
        # Simple logic for [0,1] min-max normalized inputs
        # Mimic original implementation: direct -log() without shift
        # Add small epsilon to avoid log(0) and ensure positive values
        rgb = np.clip(rgb, 1e-6, 1.0)
        stains = np.dot(np.reshape(-np.log(rgb), (-1, 3)), conv_matrix)
        shift = 0.0  # No shift needed for simple mode
    else:
        # Complex logic for Cellpose normalization that may contain negative values
        shift = rgb.min()
        if shift < 0:
            shift = np.abs(shift) + 1
        else:
            shift = 1
        rgb += shift
        stains = np.dot(np.reshape(-np.log(rgb), (-1, 3)), conv_matrix)

    return np.reshape(stains, rgb.shape), shift


def combine_stains(
    stains: np.ndarray, conv_matrix: np.ndarray, shift: float
) -> np.ndarray:
    """
    Combine stains into an RGB image.

    Args:
        stains (numpy.ndarray): Stains.
        conv_matrix (numpy.ndarray): Conversion matrix.
        shift (float): Shift.

    Returns:
        numpy.ndarray: RGB image.
    """
    stains = stains.astype(np.float32)
    logrgb2 = np.dot(-np.reshape(stains, (-1, 3)), conv_matrix)
    rgb2 = np.exp(logrgb2)

    if shift == 0.0:
        # For simple mode: direct conversion back without shift
        return np.clip(np.reshape(rgb2, stains.shape), 0.0, 1.0)
    else:
        # For complex mode: apply shift and rescale intensity
        return rescale_intensity(
            np.reshape(rgb2 - shift, stains.shape), in_range=(-1, 1)
        )


def check_range(
    range_tuple: tuple[float, float], expected_range: tuple[float, float]
) -> bool:
    """
    Check if the range tuple is valid.

    Args:
        range_tuple (tuple[float, float]): Range tuple.
        expected_range (tuple[float, float]): Expected range.

    Returns:
        bool: True if the range tuple is valid, False otherwise.
    """
    if len(range_tuple) != 2:
        raise ValueError(f"The range tuple {range_tuple} must be length 2.")
    if range_tuple[0] > range_tuple[1]:
        raise ValueError(f"The range tuple {range_tuple} is not valid.")
    if range_tuple[0] < expected_range[0] or range_tuple[1] > expected_range[1]:
        raise ValueError(
            f"The range tuple {range_tuple} is not within the expected range {expected_range}."
        )


class HEDTransform:
    """
    HEDTransform.
    """

    def __init__(
        self,
        sigma_ranges: list[tuple[float, float]],
        bias_ranges: list[tuple[float, float]],
        cutoff_range: tuple[float, float],
        seed: int | np.random.Generator | None = None,
        channel_dimension: int = 2,
        simple_mode: bool = False,
    ):
        """
        Initialize the HEDTransform.

        Args:
            sigma_ranges (list[tuple[float, float]]): List of ranges for sigmas.
            bias_ranges (list[tuple[float, float]]): List of ranges for biases.
            cutoff_range (tuple[float, float]): Cutoff range for the patch.
            seed (int | np.random.Generator | None): Seed for the random number generator.
            channel_dimension (int): Channel dimension.
            simple_mode (bool): If True, assumes input is in [0,1] range from min-max normalization.
                               If False, uses complex logic for Cellpose normalization with potential negatives.
        """
        self.sigma_ranges = sigma_ranges
        self.bias_ranges = bias_ranges
        self.cutoff_range = cutoff_range
        self.seed = seed
        self.channel_dimension = channel_dimension
        self.simple_mode = simple_mode

        assert self.channel_dimension in [
            0,
            2,
        ], "Channel dimension must be 0 or 2."

        if seed is None:
            self.rng = np.random.default_rng()
        elif isinstance(seed, int):
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = seed

        check_range(range_tuple=self.cutoff_range, expected_range=(0.0, 1.0))
        for range_tuple in self.sigma_ranges:
            check_range(range_tuple=range_tuple, expected_range=(-1.0, 1.0))
        for range_tuple in self.bias_ranges:
            check_range(range_tuple=range_tuple, expected_range=(-1.0, 1.0))

    def apply_sigma_and_bias(
        self, patch_hed: np.ndarray, sigm: list[float], bias: list[float]
    ) -> np.ndarray:
        """
        Apply sigma and bias to the patch.

        Args:
            patch_hed (np.ndarray): Patch in HED color space.
            sigm (list[float]): Standard deviations for each channel.
            bias (list[float]): Biases for each channel.

        Returns:
            np.ndarray: Patch with applied sigma and bias.
        """
        if sigm != 0.0:
            patch_hed *= 1.0 + sigm

        if bias != 0.0:
            patch_hed += bias

        return patch_hed

    def sample_sigma(self) -> list[float]:
        """
        Sample random sigmas for each channel.

        Returns:
            list[float]: Sigmas for each channel.
        """
        return [
            self.rng.uniform(
                low=self.sigma_ranges[i][0], high=self.sigma_ranges[i][1]
            )
            for i in range(3)
        ]

    def sample_bias(self) -> list[float]:
        """
        Sample random biases for each channel.

        Returns:
            list[float]: Biases for each channel.
        """
        return [
            self.rng.uniform(
                low=self.bias_ranges[i][0], high=self.bias_ranges[i][1]
            )
            for i in range(3)
        ]

    def transform(
        self,
        patch: np.ndarray,
    ) -> np.ndarray:
        """
        Apply color deformation on the patch.

        Args:
            patch (np.ndarray): Patch to transform.
            sigmas (list[float]): Standard deviations for each channel.
            biases (list[float]): Biases for each channel.
            cutoff_range (tuple[float, float]): Cutoff range for the patch.

        Returns:
            np.ndarray: Transformed patch.
        """
        # Check if the patch is inside the cutoff values.
        patch_mean = np.mean(patch)
        original_dtype = patch.dtype
        if original_dtype == np.uint8:
            patch = patch / 255.0
            patch_mean = patch_mean / 255.0
        elif original_dtype not in [np.float16, np.float32, np.float64]:
            raise ValueError(f"Unsupported patch dtype: {patch.dtype}")
        if self.cutoff_range[0] <= patch_mean <= self.cutoff_range[1]:
            # Reorder the patch to channel last format if necessary
            if self.channel_dimension == 2:
                patch_image = patch
                transposed = False
            else:
                patch_image = np.transpose(a=patch, axes=(1, 2, 0))
                transposed = True
            patch_hed, shift = rgb2hed(
                rgb=patch_image, simple_mode=self.simple_mode
            )

            # Generate random sigmas and biases.
            sigmas = self.sample_sigma()
            biases = self.sample_bias()

            # Augment the Haematoxylin, Eosin and DAB channels.
            for i in range(3):
                patch_hed[:, :, i] = self.apply_sigma_and_bias(
                    patch_hed[:, :, i], sigmas[i], biases[i]
                )

            # Convert back to RGB color coding and order back to channels first order.
            patch_rgb = hed2rgb(hed=patch_hed, shift=shift)
            patch_rgb = np.clip(a=patch_rgb, a_min=0.0, a_max=1.0)

            if transposed:
                patch_transformed = np.transpose(a=patch_rgb, axes=(2, 0, 1))
            else:
                patch_transformed = patch_rgb

            if original_dtype == np.uint8:
                patch_transformed = (patch_transformed * 255).astype(np.uint8)

            return patch_transformed

        else:
            return patch

    def transform_batch(
        self,
        patches: np.ndarray,
    ) -> np.ndarray:
        """
        Apply color deformation on the patches.

        Args:
            patches (np.ndarray): Patches to transform.

        Returns:
            np.ndarray: Transformed patches.
        """
        return np.array([self.transform(patch) for patch in patches])
