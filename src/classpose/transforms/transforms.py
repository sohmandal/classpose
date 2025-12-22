import torch


def unaugment_class_tiles(y: torch.Tensor) -> torch.Tensor:
    """Reverse test-time augmentations for averaging. Skips flow-flipping.

    Args:
        y (float32): Array of shape (ntiles_y, ntiles_x, nclasses, Ly, Lx).

    Returns:
        float32: Array of shape (ntiles_y, ntiles_x, nclasses, Ly, Lx).
    """
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if j % 2 == 0 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, :]
            elif j % 2 == 1 and i % 2 == 0:
                y[j, i] = y[j, i, :, :, ::-1]
            elif j % 2 == 1 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, ::-1]
    return y
