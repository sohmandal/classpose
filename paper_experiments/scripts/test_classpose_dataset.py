import numpy as np
from classpose.dataset import ClassposeDataset

# Create mock data
# Let's create 5 sample images and labels
n_samples = 5
image_size = (256, 256)

import os

data_dir = "../data/conic/train"  # or wherever your data is
images = np.load(os.path.join(data_dir, "images.npy"))
labels = np.load(os.path.join(data_dir, "labels.npy"))
diameters = np.load(os.path.join(data_dir, "diameters.npy"))


# Dataset parameters
batch_size = 4
n_workers = 4

if __name__ == "__main__":
    # Get the configuration for the augmentation strategy

    dataset = ClassposeDataset(
        data_array=images,
        label_array=labels,
        diameter_array=diameters,
        diam_mean=float(diameters.mean()),
        rescale=True,
        scale_range=None,
        bsize=128,
        normalize_params={"normalize": True},
        augment=True,
        augment_pipeline_config="enhanced",
        n_proc=n_workers,
        batch_size=batch_size,
    )

    # Test the dataset
    print("Testing ClassposeDataset...")

    # Put indices into the queue
    dataset.put(list(range(n_samples)))

    # Get processed batches
    for _ in range(n_samples // batch_size):
        batch = dataset.q_out.get()
        print(f"\nGot batch of shape: {batch.shape}")
        print(f"Batch contents: {batch.shape}")
        print(f"Data range: min={batch[0][0].min()}, max={batch[0][0].max()}")
        print(f"Label range: min={batch[0][1].min()}, max={batch[0][1].max()}")

    print("\nTest complete!")
