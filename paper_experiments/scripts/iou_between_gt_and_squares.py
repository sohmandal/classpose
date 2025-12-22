import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


def get_ious(instances: np.ndarray) -> list[float]:
    dist = []
    u = np.unique(instances)
    u = u[u > 0]
    for i in u:
        instance = instances == i
        intersection_area = np.sum(instance)
        x, y = np.where(instance)
        start = (x.min(), y.min())
        end = (x.max(), y.max())
        union_area = (end[0] - start[0] + 1) * (end[1] - start[1] + 1)
        iou = intersection_area / union_area
        dist.append(iou)

    return dist


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute IoU between ground truth and squares."
    )
    parser.add_argument(
        "--gt_path", type=str, required=True, help="Path to ground truth masks."
    )
    args = parser.parse_args()

    gt_masks = np.load(args.gt_path)[..., 0]

    with Pool(8) as pool:
        dist = pool.map(get_ious, tqdm(gt_masks))
    dist = np.concatenate(dist)

    for q in [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]:
        print(f"Quantile ({q}): {np.percentile(dist, q * 100)}")
    print(f"Mean: {np.mean(dist)}")
