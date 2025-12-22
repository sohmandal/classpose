"""
Simplified StarDist-style augmentation manager for ClassPose.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from classpose.transforms.he_staining import HEStainingTransform
from classpose.transforms.hed import HEDTransform
from classpose.transforms.image_quality import (
    AdditiveNoiseTransform,
    GaussianBlurTransform,
    HueBrightnessSaturationTransform,
)


class StarDistAugmentation:
    """Simplified StarDist augmentation manager."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with a single configuration dictionary.

        Args:
            config: Configuration dictionary with all augmentation settings
        """
        self.config = config
        self.rng = np.random.default_rng(config.get("seed", None))

        # Build augmentation pipeline as a list of transforms
        self.transforms = self._build_pipeline()

    def _build_pipeline(self) -> List[Callable]:
        """Build the augmentation pipeline based on configuration."""
        pipeline = []

        # Color augmentation (HED or H&E staining)
        color_transform = self._build_color_transform()
        if color_transform:
            pipeline.append(color_transform)

        # Image quality augmentations
        pipeline.extend(self._build_quality_transforms())

        return pipeline

    def _build_color_transform(self) -> Optional[Callable]:
        """Build color space augmentation transform."""
        use_hed = self.config.get(
            "use_hed", True
        )  # Default to True for backward compatibility
        use_he_staining = self.config.get("use_he_staining", False)

        if not use_hed and not use_he_staining:
            return None

        # Initialize transforms
        hed_transform = None
        he_staining_transform = None

        if use_hed:
            hed_config = self.config.get("hed_config", {})
            hed_transform = HEDTransform(**hed_config)

        if use_he_staining:
            he_config = self.config.get("he_staining_config", {})
            he_staining_transform = HEStainingTransform(**he_config)

        # Return appropriate color transform function
        if use_hed and use_he_staining:
            hed_prob = self.config.get("hed_probability", 0.5)
            return lambda images: (
                hed_transform.transform_batch(images)
                if self.rng.random() < hed_prob
                else he_staining_transform.transform_batch(images)
            )
        elif use_hed:
            return hed_transform.transform_batch
        else:
            return he_staining_transform.transform_batch

    def _build_quality_transforms(self) -> List[Callable]:
        """Build image quality augmentation transforms."""
        transforms = []

        # Gaussian blur
        if self.config.get("use_gaussian_blur", False):
            blur_config = self.config.get("gaussian_blur_config", {})
            blur_transform = GaussianBlurTransform(**blur_config)
            transforms.append(blur_transform.transform_batch)

        # Additive noise
        if self.config.get("use_additive_noise", False):
            noise_config = self.config.get("additive_noise_config", {})
            noise_transform = AdditiveNoiseTransform(**noise_config)
            transforms.append(noise_transform.transform_batch)

        # HBS adjustment
        if self.config.get("use_hbs", False):
            hbs_config = self.config.get("hbs_config", {})
            hbs_transform = HueBrightnessSaturationTransform(**hbs_config)
            transforms.append(hbs_transform.transform_batch)

        return transforms

    def transform_batch(self, images: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline to batch of images."""
        for transform in self.transforms:
            images = transform(images)
        return images

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline to single image."""
        images_batch = np.expand_dims(image, axis=0)
        augmented_batch = self.transform_batch(images_batch)
        return augmented_batch[0]


# Simplified factory function
def create_stardist_augmentation(
    config: Optional[Dict[str, Any]] = None
) -> StarDistAugmentation:
    """Create StarDist augmentation with simplified configuration.

    Args:
        config: Augmentation configuration dictionary
    """
    if config is None:
        config = {
            "use_hed": True,
            "use_he_staining": False,
            "use_gaussian_blur": False,
            "use_additive_noise": False,
            "use_hbs": False,
            "seed": 42,
            "hed_config": {
                "sigma_ranges": [(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)],
                "bias_ranges": [(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)],
                "cutoff_range": (0.15, 0.85),
            },
        }

    return StarDistAugmentation(config)
