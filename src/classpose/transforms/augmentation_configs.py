"""
Configuration templates for ClassPose augmentation strategies.

This module provides pre-defined configuration dictionaries for different
augmentation strategies that can be used with the StarDist augmentation
integration in ClassPose training.
"""

# HED-only augmentation (original ClassPose behavior)
HED_VALUE = 0.25
HED_VALUE_R = (-HED_VALUE, HED_VALUE)
HED_ONLY_CONFIG = {
    "use_hed": True,
    "use_he_staining": False,
    "use_gaussian_blur": False,
    "use_additive_noise": False,
    "use_hbs": False,
    "seed": 42,
    "channel_dimension": 0,
    "hed_config": {
        "sigma_ranges": [HED_VALUE_R, HED_VALUE_R, HED_VALUE_R],
        "bias_ranges": [HED_VALUE_R, HED_VALUE_R, HED_VALUE_R],
        "cutoff_range": (0.15, 0.85),
    },
}

# Enhanced augmentation with HED, HE staining (50/50 probability), and image quality augmentations
ENHANCED_CONFIG = {
    "use_hed": True,
    "use_he_staining": True,
    "use_gaussian_blur": True,
    "use_additive_noise": True,
    "use_hbs": True,
    "seed": 42,
    "channel_dimension": 0,
    "hed_probability": 0.5,
    "hed_config": {
        "sigma_ranges": [HED_VALUE_R, HED_VALUE_R, HED_VALUE_R],
        "bias_ranges": [HED_VALUE_R, HED_VALUE_R, HED_VALUE_R],
        "cutoff_range": (0.15, 0.85),
    },
    "he_staining_config": {
        "amount_matrix": 0.15,
        "amount_stains": 0.4,
        "probability": 0.9,
    },
    "gaussian_blur_config": {
        "sigma_range": (0, 2),
        "probability": 0.1,
    },
    "additive_noise_config": {
        "sigma": 0.01,
        "probability": 0.8,
    },
    "hbs_config": {
        "hue": 0.1,
        "brightness": 0.1,
        "saturation": (0.9, 1.1),
        "probability": 0.9,
    },
}

# Configuration registry for easy access
AUGMENTATION_CONFIGS = {
    "hed_only": HED_ONLY_CONFIG,
    "enhanced": ENHANCED_CONFIG,
}


def get_config(config_name: str) -> dict:
    """
    Get a pre-defined augmentation configuration.

    Args:
        config_name (str): Name of the configuration. Available options:
            - 'hed_only': Original ClassPose HED-only augmentation
            - 'enhanced': Enhanced augmentation with HED, HE staining (50/50 probability), and image quality augmentations

    Returns:
        dict: Configuration dictionary for the specified augmentation strategy.

    Raises:
        ValueError: If config_name is not recognized.
    """
    if config_name not in AUGMENTATION_CONFIGS:
        available = list(AUGMENTATION_CONFIGS.keys())
        raise ValueError(
            f"Unknown config '{config_name}'. Available: {available}"
        )

    return AUGMENTATION_CONFIGS[config_name].copy()


def create_custom_config(**kwargs) -> dict:
    """
    Create a custom augmentation configuration by modifying the enhanced config.

    Args:
        **kwargs: Configuration parameters to override. Available parameters:
            - use_hed (bool): Whether to use HED color space augmentation
            - use_he_staining (bool): Whether to use H&E staining augmentation
            - use_gaussian_blur (bool): Whether to use Gaussian blur
            - use_additive_noise (bool): Whether to use additive noise
            - use_hbs (bool): Whether to use hue-brightness-saturation augmentation
            - hed_probability (float): Probability of using HED vs HE staining when both enabled
            - seed (int): Random seed for reproducibility
            - channel_dimension (int): Channel dimension (0 for channel-first, -1 for channel-last)
            - *_config (dict): Specific configuration for each augmentation type

    Returns:
        dict: Custom configuration dictionary.

    Example:
        >>> config = create_custom_config(
        ...     use_gaussian_blur=False,
        ...     additive_noise_config={'sigma': 0.005, 'probability': 0.5}
        ... )
    """
    base_config = ENHANCED_CONFIG.copy()

    # Update top-level parameters
    for key, value in kwargs.items():
        if not key.endswith("_config"):
            base_config[key] = value

    # Update nested config dictionaries
    for key, value in kwargs.items():
        if key.endswith("_config") and key in base_config:
            base_config[key].update(value)
        elif key.endswith("_config"):
            base_config[key] = value

    return base_config
