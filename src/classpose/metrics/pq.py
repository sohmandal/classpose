from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from classpose.metrics.stats_utils import get_multi_pq_info, get_pq, remap_label
from classpose.metrics.utils import (
    check_and_coherce_if_necessary,
    filter_out_unlabelled_cells,
)


class MulticlassPQCalculator:
    """
    Callable that computes multi-class PQ info for a single image pair.

    Designed to be used with :func:`multiprocessing.Pool.imap_unordered` so
    that per-image PQ computation can be parallelised.
    """

    def __init__(
        self,
        nr_classes: int,
        match_iou: float,
        no_border_instances: bool = False,
    ):
        """
        Initialise the calculator.

        Args:
            nr_classes (int): Number of foreground classes.
            match_iou (float): IoU threshold for matching instances.
            no_border_instances (bool, optional): If ``True``, instances
                touching the image border are removed before metric
                computation. Defaults to False.
        """
        self.nr_classes = nr_classes
        self.match_iou = match_iou
        self.no_border_instances = no_border_instances

    def __call__(
        self, gt_pred_idx: tuple[np.ndarray, np.ndarray, int]
    ) -> tuple[list, int]:
        """
        Compute PQ info for one ground-truth / prediction pair.

        Args:
            gt_pred_idx (tuple[np.ndarray, np.ndarray, int]): A tuple of
                ``(gt_mask, pred_mask, image_index)``.

        Returns:
            tuple[list, int]: Per-class PQ statistics and the image index.
        """
        gt, pred, idx = gt_pred_idx
        if self.no_border_instances:
            gt = remove_border_instances(gt)
            pred = remove_border_instances(pred)
        pq_info = get_multi_pq_info(
            gt, pred, nr_classes=self.nr_classes, match_iou=self.match_iou
        )
        return pq_info, idx


def remove_border_instances(mask: np.ndarray) -> np.ndarray:
    """
    Zero-out instances that touch the image border.

    An instance is considered a border instance if any of its pixels lie on
    the first or last row/column of the mask.

    Args:
        mask (np.ndarray): Instance mask of shape ``(H, W)`` or
            ``(H, W, C)``.  When 3-dimensional, the first channel is
            treated as the instance channel and all channels are zeroed
            for border instances.

    Returns:
        np.ndarray: Mask with border instances set to 0.
    """
    if len(mask.shape) == 3:
        instances = mask[..., 0]
    else:
        instances = mask
    border_instances = np.unique(
        np.concatenate(
            [instances[0], instances[:, 0], instances[-1], instances[:, -1]]
        )
    )
    border_instances = border_instances[border_instances != 0]
    mask[np.isin(instances, border_instances)] = 0
    return mask


def compute_binary_pq_metrics(
    gt_masks: np.ndarray | list[np.ndarray],
    pred_masks: np.ndarray | list[np.ndarray],
    match_iou: float = 0.5,
    no_border_instances: bool = False,
) -> pd.DataFrame:
    """
    Compute binary PQ metrics for a batch of masks. Expects both input
    masks to have shapes HxW.

    Args:
        gt_masks (np.ndarray | list[np.ndarray]): Ground truth masks.
        pred_masks (np.ndarray | list[np.ndarray]): Predicted masks.
        match_iou (float, optional): IoU threshold for matching. Defaults to 0.5.
        no_border_instances (bool, optional): Whether to remove border instances.
            These are defined as instances which have any pixel on the border.
            Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with PQ metrics.
    """
    results = []

    # The expected shape has to be 3 if we want to use
    gt_masks = check_and_coherce_if_necessary(gt_masks, 2)
    pred_masks = check_and_coherce_if_necessary(pred_masks, 2)

    for i in tqdm(range(len(gt_masks)), desc="Computing metrics"):
        gt = gt_masks[i]
        pred = pred_masks[i]

        if no_border_instances:
            gt = remove_border_instances(gt)
            pred = remove_border_instances(pred)

        # Ensure masks have proper instance IDs (contiguous)
        gt = remap_label(gt)
        pred = remap_label(pred)

        # Get PQ metrics
        pq_stats, counts, iou_sum = get_pq(gt, pred, match_iou=match_iou)
        dq, sq, pq = pq_stats
        tp, fp, fn = counts

        results.append(
            {
                "image_id": i,
                "pq": pq,
                "dq": dq,
                "sq": sq,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp),
                "recall": tp / (tp + fn),
                "f1": (2 * tp) / (2 * tp + fp + fn),
                "iou_sum": iou_sum,
                "avg_iou": iou_sum / tp if tp > 0 else 0.0,
            }
        )

    return pd.DataFrame(results)


def compute_multiclass_pq_metrics(
    gt_masks: np.ndarray | list[np.ndarray],
    pred_masks: np.ndarray | list[np.ndarray],
    match_iou: float = 0.5,
    nr_classes: int = 6,
    n_workers: int = 0,
    no_border_instances: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute multi-class PQ metrics for a batch of masks. Expects both input
    masks to have shapes HxWx2, where the first channel is the instance
    segmentation mask and the second channel is the class mask.

    Args:
        gt_masks (np.ndarray | list[np.ndarray]): Ground truth masks.
        pred_masks (np.ndarray | list[np.ndarray]): Predicted masks.
        match_iou (float, optional): IoU threshold for matching. Defaults to 0.5.
        nr_classes (int, optional): Number of classes. Defaults to 6.
        n_workers (int, optional): Number of workers for parallel processing. Defaults to 0.
        no_border_instances (bool, optional): Whether to remove border instances.
            These are defined as instances which have any pixel on the border.
            Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with global PQ metrics per class.
    """
    # Initialize arrays to store aggregated stats
    tp_per_class = np.zeros(nr_classes)
    fp_per_class = np.zeros(nr_classes)
    fn_per_class = np.zeros(nr_classes)
    iou_sum_per_class = np.zeros(nr_classes)

    gt_masks = check_and_coherce_if_necessary(gt_masks, 3)
    pred_masks = check_and_coherce_if_necessary(pred_masks, 3)

    gt_masks, pred_masks = filter_out_unlabelled_cells(gt_masks, pred_masks)

    per_image_results = []

    pq_calculator = MulticlassPQCalculator(
        nr_classes, match_iou, no_border_instances
    )

    inputs_for_pq_calculator = zip(gt_masks, pred_masks, range(len(gt_masks)))

    if n_workers < 2:
        map_fn = map(pq_calculator, inputs_for_pq_calculator)
    else:
        pool = Pool(n_workers)
        map_fn = pool.imap_unordered(pq_calculator, inputs_for_pq_calculator)

    n = len(gt_masks)
    for pq_info, i in tqdm(map_fn, desc="Computing metrics", total=n):
        image_metrics = {"image_id": i}

        # Aggregate stats for each class
        for class_idx in range(nr_classes):
            tp_per_class[class_idx] += pq_info[class_idx][0]
            fp_per_class[class_idx] += pq_info[class_idx][1]
            fn_per_class[class_idx] += pq_info[class_idx][2]
            iou_sum_per_class[class_idx] += pq_info[class_idx][3]

            tp = pq_info[class_idx][0]
            fp = pq_info[class_idx][1]
            fn = pq_info[class_idx][2]
            iou_sum = pq_info[class_idx][3]
            avg_iou = iou_sum / tp if tp > 0 else 0.0

            class_num = class_idx + 1
            image_metrics[f"class_{class_num}_tp"] = tp
            image_metrics[f"class_{class_num}_fp"] = fp
            image_metrics[f"class_{class_num}_fn"] = fn
            image_metrics[f"class_{class_num}_avg_iou"] = avg_iou

        per_image_results.append(image_metrics)

    per_image_results = sorted(per_image_results, key=lambda x: x["image_id"])

    # Calculate PQ metrics for each class
    results = []
    for class_idx in range(nr_classes):
        tp = tp_per_class[class_idx]
        fp = fp_per_class[class_idx]
        fn = fn_per_class[class_idx]
        iou_sum = iou_sum_per_class[class_idx]

        # Calculate DQ (Detection Quality)
        dq = tp / ((tp + 0.5 * fp + 0.5 * fn) + 1.0e-6)

        # Calculate SQ (Segmentation Quality)
        sq = iou_sum / (tp + 1.0e-6)

        # Calculate PQ (Panoptic Quality)
        pq = dq * sq

        results.append(
            {
                "class_id": class_idx + 1,
                "pq": pq,
                "dq": dq,
                "sq": sq,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp),
                "recall": tp / (tp + fn),
                "f1": (2 * tp) / (2 * tp + fp + fn),
                "iou_sum": iou_sum,
            }
        )

    # Calculate average metrics across all classes
    avg_results = {
        "class_id": "avg",
        "pq": np.mean([r["pq"] for r in results]),
        "dq": np.mean([r["dq"] for r in results]),
        "sq": np.mean([r["sq"] for r in results]),
        "tp": np.sum([r["tp"] for r in results]),
        "fp": np.sum([r["fp"] for r in results]),
        "fn": np.sum([r["fn"] for r in results]),
        "precision": np.mean([r["precision"] for r in results]),
        "recall": np.mean([r["recall"] for r in results]),
        "f1": np.mean([r["f1"] for r in results]),
        "iou_sum": np.sum([r["iou_sum"] for r in results]),
    }

    results.append(avg_results)

    global_df = pd.DataFrame(results)
    per_image_df = pd.DataFrame(per_image_results)

    return global_df, per_image_df
