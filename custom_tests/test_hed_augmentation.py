"""
Test HED augmentation.
"""

import matplotlib.pyplot as plt
import numpy as np
from cellpose.transforms import normalize_img
from classpose.transforms import HEDTransform

TEST_DATA_DIR = "data/conic/train"
MODEL_PATH = "models/cellpose_1748444549.3961618"
SHOW_IMAGES = True

images = np.load(f"{TEST_DATA_DIR}/images.npy")

VALUE = 0.1
VALUE_R = (-VALUE, VALUE)
transform = HEDTransform(
    sigma_ranges=[VALUE_R, VALUE_R, VALUE_R],
    bias_ranges=[VALUE_R, VALUE_R, VALUE_R],
    cutoff_range=(0.15, 0.85),
    seed=42,
    channel_dimension=2,
)

for i in range(images.shape[0]):
    transformed_image = transform.transform(images[i])
    transformed_normalised_image = transform.transform(images[i] / 255.0)
    transformed_standardised_image = transform.transform(
        normalize_img(images[i])
    )
    difference = images[i] / 255 - transformed_image
    difference = np.abs(difference)
    if SHOW_IMAGES:
        fig, ax = plt.subplots(1, 5, figsize=(12, 6))
        ax[0].imshow(images[i])
        ax[0].set_title("Original")
        ax[1].imshow(transformed_image)
        ax[1].set_title("Transformed")
        ax[2].imshow(transformed_normalised_image)
        ax[2].set_title("Transformed Normalised")
        ax[3].imshow(transformed_standardised_image)
        ax[3].set_title("Transformed Standardised")
        ax[4].imshow(difference)
        ax[4].set_title("Difference")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.show()
