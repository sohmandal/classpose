"""
Compute Panoptic Quality (PQ) metrics between ground truth and predicted masks.

This script calculates:
- PQ (Panoptic Quality): Combination of DQ and SQ (PQ = DQ * SQ)
- DQ (Detection/Recognition Quality): F1-score measuring how well instances are detected
- SQ (Segmentation Quality): IoU measuring the quality of segmentation for matched instances
- TP (True Positives): Number of correctly detected instances
- FP (False Positives): Number of predicted instances without ground truth matches
- FN (False Negatives): Number of ground truth instances without predicted matches
- Precision (Precision): Precision for the model (TP / (TP + FP))
- Recall (Recall): Recall for the model (TP / (TP + FN))
- F1 (F1-score): F1-score for the model (2 * (Precision * Recall) / (Precision + Recall))

Usage:
    python compute_pq_metrics.py --gt_path /path/to/ground_truth_masks --pred_path /path/to/predicted_masks

    Optional arguments:
    --match_iou MATCH_IOU   IoU threshold for matching instances (default: 0.5)
    --output OUTPUT         Path to save results as CSV (default: None)
    --binary                If set, treat masks as binary instance segmentation without classes
"""

import argparse
import numpy as np
from classpose.log import get_logger
from pathlib import Path
from tqdm import trange

logger = get_logger(__name__)

from classpose.metrics.pq import (
    compute_binary_pq_metrics,
    compute_multiclass_pq_metrics,
)
from classpose.metrics.utils import load_masks


def main(args):
    logger.info(f"Loading ground truth masks from {args.gt_path}")
    gt_masks = load_masks(args.gt_path)

    logger.info(f"Loading predicted masks from {args.pred_path}")
    pred_masks = load_masks(args.pred_path)

    nr_classes = int(np.max([m[..., 1].max() for m in gt_masks]))

    if args.label_map:
        logger.info(f"Applying label map: {args.label_map}")
        label_map = {0: 0}
        unique_values = [0]
        for i in args.label_map:
            k, v = i.split("=")
            label_map[int(k)] = int(v)
            if int(v) not in unique_values:
                unique_values.append(int(v))
        unique_values = np.array(unique_values)
        lm_vec_pred = np.vectorize(label_map.get)
        logger.info(f"Label map: {label_map}")
        for i in trange(
            len(pred_masks), desc="Applying label map to predicted masks"
        ):
            pred_masks[i] = pred_masks[i].astype(int)
            pred_masks[i][..., 1] = lm_vec_pred(pred_masks[i][..., 1])
        for i in trange(
            len(gt_masks), desc="Applying label map to ground truth masks"
        ):
            gt_masks[i][..., 1] = np.where(
                np.isin(gt_masks[i][..., 1], unique_values),
                gt_masks[i][..., 1],
                0,
            )

    if args.ignore_classes:
        for i in args.ignore_classes:
            gt_masks[..., 1][gt_masks[..., 1] == i] = 0
            pred_masks[..., 1][pred_masks[..., 1] == i] = 0
    # Check that masks have the same shape
    if isinstance(gt_masks, list) and isinstance(pred_masks, list):
        if len(gt_masks) != len(pred_masks):
            raise ValueError(
                f"Number of ground truth masks ({len(gt_masks)}) doesn't match predicted masks ({len(pred_masks)})"
            )
    elif gt_masks.shape != pred_masks.shape:
        raise ValueError(
            f"Ground truth mask shape {gt_masks.shape} doesn't match predicted mask shape {pred_masks.shape}"
        )

    # Compute metrics
    if args.binary:
        logger.info(
            f"Computing binary PQ metrics with IoU threshold {args.match_iou}"
        )
        results = compute_binary_pq_metrics(
            gt_masks,
            pred_masks,
            match_iou=args.match_iou,
        )

        # Print results
        print("\nResults:")
        print(results.to_string(index=False))

        # Save results if output path is provided
        if args.output:
            results.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")

    else:
        logger.info(
            f"Computing multi-class PQ metrics with IoU threshold {args.match_iou} for {nr_classes} classes"
        )

        global_results, per_image_results = compute_multiclass_pq_metrics(
            gt_masks,
            pred_masks,
            match_iou=args.match_iou,
            nr_classes=nr_classes,
            n_workers=args.n_workers,
        )

        # Print results
        print("\nGlobal Results:")
        print(global_results.to_string(index=False))

        print("\nPer-Image Results:")
        print(per_image_results.head().to_string(index=False))

        # Save results if output path is provided
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            global_results.to_csv(args.output, index=False)
            logger.info(f"Global results saved to {args.output}")

            # Generate per-image output filename
            base_name = args.output.rsplit(".", 1)[0]
            ext = args.output.rsplit(".", 1)[1] if "." in args.output else "csv"
            per_image_output = f"{base_name}_per_image.{ext}"

            per_image_results.to_csv(per_image_output, index=False)
            logger.info(f"Per-image results saved to {per_image_output}")


def main_with_args():
    parser = argparse.ArgumentParser(
        description="Compute PQ (Panoptic Quality) metrics between ground truth and predicted masks."
    )
    parser.add_argument(
        "--gt_path",
        required=True,
        help="Path to ground truth masks (directory or file)",
    )
    parser.add_argument(
        "--pred_path",
        required=True,
        help="Path to predicted masks (directory or file)",
    )
    parser.add_argument(
        "--match_iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching instances",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as CSV",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Treat masks as binary instance segmentation without classes",
    )
    parser.add_argument(
        "--ignore_classes",
        type=int,
        default=None,
        nargs="+",
        help="Classes to ignore.",
    )
    parser.add_argument(
        "--label_map",
        type=str,
        nargs="+",
        default=None,
        help="Label map for mutli-class conversion. Should be a list of k=v index pairs."
        "Example: --label_map 0=0 1=1 2=2.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of workers for parallel processing",
    )

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    main_with_args()
