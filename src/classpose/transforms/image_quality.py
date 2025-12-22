"""
Image quality augmentations adapted from StarDist's CoNIC implementation.
Includes Gaussian blur, additive noise, and hue-brightness-saturation adjustments.
"""

import warnings
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.color import hsv2rgb, rgb2hsv
from torchvision.transforms.v2 import functional as tvf


class GaussianBlurTransform:
    """Gaussian blur augmentation transform."""

    def __init__(
        self,
        sigma_range: tuple[float, float] = (0, 2),
        probability: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize Gaussian blur transform.

        Args:
            sigma_range: Range of sigma values for Gaussian blur (0, 2) from StarDist
            probability: Probability of applying augmentation (0.1 from StarDist)
            seed: Random seed for reproducibility
        """
        self.sigma_range = sigma_range
        self.probability = probability

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to a single image.

        Args:
            image: Image array of shape (H, W, C) or (C, H, W)

        Returns:
            Blurred image with same shape as input
        """
        # Skip augmentation based on probability
        if self.rng.random() > self.probability:
            return image

        # Sample sigma value
        sigma = self.rng.uniform(*self.sigma_range)

        # Handle channel dimension
        if (
            image.shape[0] <= 4 and len(image.shape) == 3
        ):  # Likely channel first
            # Apply blur to spatial dimensions only
            blurred = np.zeros_like(image)
            for c in range(image.shape[0]):
                blurred[c] = gaussian_filter(image[c], sigma=sigma)
        else:  # Channel last or 2D
            if len(image.shape) == 3:  # Channel last
                blurred = np.zeros_like(image)
                for c in range(image.shape[2]):
                    blurred[:, :, c] = gaussian_filter(
                        image[:, :, c], sigma=sigma
                    )
            else:  # 2D
                blurred = gaussian_filter(image, sigma=sigma)

        return blurred

    def transform_batch(self, images: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to a batch of images."""
        return np.array([self.transform(img) for img in images])


class AdditiveNoiseTransform:
    """Additive Gaussian noise augmentation transform."""

    def __init__(
        self,
        sigma: float = 0.01,
        probability: float = 0.8,
        seed: Optional[int] = None,
    ):
        """
        Initialize additive noise transform.

        Args:
            sigma: Standard deviation of Gaussian noise (0.01 from StarDist)
            probability: Probability of applying augmentation (0.8 from StarDist)
            seed: Random seed for reproducibility
        """
        self.sigma = sigma
        self.probability = probability

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Apply additive noise to a single image.

        Args:
            image: Image array

        Returns:
            Noisy image with same shape and dtype as input
        """
        # Skip augmentation based on probability
        if self.rng.random() > self.probability:
            return image

        # Generate noise
        noise = self.rng.normal(0, self.sigma, image.shape).astype(image.dtype)

        # Add noise and clip to valid range
        noisy_image = image + noise

        # Clip based on image dtype and range
        if image.dtype == np.uint8:
            noisy_image = np.clip(noisy_image, 0, 255)
        elif image.dtype in [np.float32, np.float64]:
            if image.max() <= 1.0:  # Normalized to [0, 1]
                noisy_image = np.clip(noisy_image, 0, 1)
            # Otherwise assume unnormalized float, don't clip
        return noisy_image.astype(image.dtype)

    def transform_batch(self, images: np.ndarray) -> np.ndarray:
        """Apply additive noise to a batch of images."""
        return np.array([self.transform(img) for img in images])


class HueBrightnessSaturationTransform:
    """Hue, brightness, and saturation augmentation transform."""

    def __init__(
        self,
        hue: float = 0.0,
        brightness: float = 0.1,
        saturation: tuple[float, float] = (1.0, 1.0),
        probability: float = 0.9,
        seed: Optional[int] = None,
    ):
        """
        Initialize HBS transform.

        Args:
            hue: Hue adjustment range (0 from StarDist - no hue change)
            brightness: Brightness adjustment range (0.1 from StarDist)
            saturation: Saturation multiplier range ((1,1) from StarDist - no saturation change)
            probability: Probability of applying augmentation (0.9 from StarDist)
            seed: Random seed for reproducibility
        """
        self.hue = hue
        self.brightness = brightness
        self.saturation = saturation
        self.probability = probability

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def _hbs_adjust(self, image: np.ndarray) -> np.ndarray:
        """
        Apply HBS adjustment.
        """
        # Sample adjustment values
        h_hue = self.rng.uniform(-self.hue, self.hue) if self.hue > 0 else 0
        h_brightness = 1 + self.rng.uniform(-self.brightness, self.brightness)
        h_saturation = self.rng.uniform(*self.saturation)

        image_shape = image.shape
        if image_shape[0] != 3:
            image = np.transpose(image, (2, 0, 1))

        # Convert to tensor
        image_tensor = torch.as_tensor(image, dtype=torch.float32)

        # Ensure values are in [0, 1] range for TF operations
        if image.dtype == np.uint8:
            image_tensor = image_tensor / 255.0
        elif image.max() > 1.0:
            image_tensor = image_tensor / 255.0

        # Apply adjustments
        if h_hue != 0:
            image_tensor = tvf.adjust_hue(image_tensor, h_hue)
        if h_brightness != 0:
            image_tensor = tvf.adjust_brightness(image_tensor, h_brightness)
        if h_saturation != 1.0:
            image_tensor = tvf.adjust_saturation(image_tensor, h_saturation)

        # Convert back to numpy
        result = image_tensor.numpy()

        if image_shape[0] != 3:
            result = np.transpose(result, (1, 2, 0))

        # Convert back to original dtype and range
        if image.dtype == np.uint8:
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
        elif image.max() > 1.0:
            result = np.clip(result * 255, 0, 255).astype(image.dtype)
        else:
            result = np.clip(result, 0, 1).astype(image.dtype)

        return result

    def _hbs_adjust_numpy(self, image: np.ndarray) -> np.ndarray:
        """Apply simplified HBS adjustment using scikit-image."""
        # Sample adjustment values
        image_dtype = image.dtype
        h_hue = self.rng.uniform(-self.hue, self.hue) if self.hue > 0 else 0
        h_brightness = self.rng.uniform(-self.brightness, self.brightness)
        h_saturation = self.rng.uniform(*self.saturation)

        # Apply brightness adjustment
        hsv_image = rgb2hsv(image)
        hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0] + h_hue, 0, 1)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * h_saturation, 0, 1)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + h_brightness, 0, 1)
        adjusted = hsv2rgb(hsv_image)
        if image_dtype == np.uint8:
            adjusted = np.clip(adjusted * 255, 0, 255)
        else:
            adjusted = np.clip(adjusted, 0, 1)
        return adjusted.astype(image_dtype)

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Apply HBS adjustment to a single image.

        Args:
            image: RGB image array of shape (H, W, 3) or (3, H, W)

        Returns:
            Adjusted image with same shape as input
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

        # Ensure RGB format (3 channels)
        if image_hwc.shape[2] != 3:
            sh = image_hwc.shape[2]
            warnings.warn(
                f"HBS augmentation expects 3 channel RGB image, got {sh} channels"
            )
            return image

        # Apply HBS adjustment
        adjusted = self._hbs_adjust(image_hwc)

        # Convert back to original channel order
        if channels_first:
            adjusted = np.transpose(adjusted, (2, 0, 1))

        return adjusted

    def transform_batch(self, images: np.ndarray) -> np.ndarray:
        """Apply HBS adjustment to a batch of images."""
        return np.array([self.transform(img) for img in images])


class ImageQualityAugmentation:
    """Combined image quality augmentation pipeline."""

    def __init__(
        self,
        gaussian_blur_config: Optional[dict] = None,
        additive_noise_config: Optional[dict] = None,
        hbs_config: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize combined image quality augmentation.

        Args:
            gaussian_blur_config: Configuration for Gaussian blur
            additive_noise_config: Configuration for additive noise
            hbs_config: Configuration for HBS adjustment
            seed: Random seed for reproducibility
        """
        # Default configurations matching StarDist parameters
        gaussian_blur_config = gaussian_blur_config or {}
        additive_noise_config = additive_noise_config or {}
        hbs_config = hbs_config or {}

        # Initialize individual transforms
        self.gaussian_blur = GaussianBlurTransform(
            seed=seed, **gaussian_blur_config
        )
        self.additive_noise = AdditiveNoiseTransform(
            seed=seed, **additive_noise_config
        )
        self.hbs = HueBrightnessSaturationTransform(seed=seed, **hbs_config)

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Apply all image quality augmentations in sequence."""
        # Apply in order: blur -> noise -> HBS
        image = self.gaussian_blur.transform(image)
        image = self.additive_noise.transform(image)
        image = self.hbs.transform(image)
        return image

    def transform_batch(self, images: np.ndarray) -> np.ndarray:
        """Apply image quality augmentations to a batch of images."""
        return np.array([self.transform(img) for img in images])
