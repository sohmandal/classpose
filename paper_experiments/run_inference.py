import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from skimage.transform import resize

from classpose.log import add_file_handler, get_logger
from classpose.utils import get_device

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = get_logger("classpose")

for name in logging.root.manager.loggerDict:
    if "cellpose" in name or "classpose" in name:
        logging.getLogger(name).setLevel(logging.INFO)

SHOW_IMAGES_DEFAULT = False

from classpose.metrics.pq import compute_multiclass_pq_metrics
from classpose.models import ClassposeModel
from classpose.entrypoints.predict_wsi import infer_structure

CLASS_COLORS = {
    0: [0, 0, 0],  # background - black
    1: [1, 1, 0],  # Neutrophil - yellow
    2: [1, 0, 0],  # Epithelial - red
    3: [0, 0, 0.5],  # Lymphocyte - navy
    4: [1, 0, 1],  # Plasma - magenta
    5: [1, 0.5, 0],  # Eosinophil - orange
    6: [0, 1, 0],  # Connective - green
}


def apply_class_colormap(class_mask, colors_dict):
    h, w = class_mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.float32)
    for class_id, color_val in colors_dict.items():
        mask = class_mask == class_id
        colored_mask[mask] = np.array(color_val, dtype=np.float32)
    return colored_mask


def load_labels(labels_path: str) -> np.ndarray:
    """
    Load labels from a numpy file.

    Args:
        labels_path (str): Path to the labels file.

    Returns:
        np.ndarray: Labels loaded from the file.

    Raises:
        FileNotFoundError: If the labels file is not found.
    """
    if os.path.exists(labels_path):
        labels = np.load(labels_path, allow_pickle=True)
        logger.info(f"Loaded {len(labels)} labels from {labels_path}")
    else:
        logger.error(f"Labels file not found: {labels_path}")
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    return labels


def get_rescale_ratio(training_to_inference_mpp: str):
    ratio = 1.0
    if ":" in training_to_inference_mpp:
        training_mpp, inference_mpp = training_to_inference_mpp.split(":")
        training_mpp = float(training_mpp)
        inference_mpp = float(inference_mpp)
        if training_mpp != inference_mpp:
            ratio = inference_mpp / training_mpp
    else:
        ratio = float(training_to_inference_mpp)
    return ratio


def rescale_if_necessary(image: np.ndarray, training_to_inference_mpp: str):
    """
    Rescale images if necessary.

    Args:
        image (np.ndarray): image to be rescaled.
        training_to_inference_mpp (str): multiplier to rescale images from
            training resolution to inference resolution. For example, if the
            training images are 0.5 microns per pixel and the inference images
            are 1 micron per pixel, then the value should be "0.5". If the
            training images are 0.5 microns per pixel and the inference images
            are 2 microns per pixel, then the value should be "0.5:2".

    Returns:
        np.ndarray: Rescaled images.
    """
    if training_to_inference_mpp is not None:
        ratio = get_rescale_ratio(training_to_inference_mpp)
        if ratio != 1.0:
            sh = image.shape
            new_sh = (int(sh[0] * ratio), int(sh[1] * ratio))
            image = resize(image.astype(np.float64), new_sh)
    return image


def rescale_label_if_necessary(label: np.ndarray, new_sh: tuple[int, int]):
    """
    Rescale labels if necessary.

    Args:
        label (np.ndarray): label to be rescaled.
        new_sh (tuple[int, int]): new shape of the label.

    Returns:
        np.ndarray: Rescaled images.
    """
    sh = label.shape
    if sh != new_sh:
        label = resize(label, new_sh, order=0)
    return label


def main(args):
    if args.log_path:
        add_file_handler(logger, args.log_path)
    logger.info(f"Starting inference with model: {args.model_path}")
    logger.info(f"Loading data from: {args.test_data_dir}")
    if args.metrics_output_dir:
        logger.info(f"Metrics will be saved in: {args.metrics_output_dir}")
        os.makedirs(args.metrics_output_dir, exist_ok=True)

    images_path = os.path.join(args.test_data_dir, "images.npy")

    if os.path.exists(images_path):
        images = np.load(images_path, allow_pickle=True)
        logger.info(f"Loaded {len(images)} images from {images_path}")
    else:
        logger.error(f"Images file not found: {images_path}")
        raise FileNotFoundError(f"Images file not found: {images_path}")

    feature_transformation_structure, n_classes = infer_structure(
        args.model_path
    )
    device = get_device(args.device)[0]
    model = ClassposeModel(
        gpu=device.type == "cuda",
        pretrained_model=args.model_path,
        device=device,
        nclasses=n_classes,
        feature_transformation_structure=feature_transformation_structure,
        bf16=args.bf16,
    )

    if os.environ.get("COMPILE_MODEL", "") == "1":
        logger.info("Compiling model...")
        model.net = torch.compile(model.net)

    all_gt_for_pq = []
    all_pred_for_pq = []

    start_time = time.time()
    logger.info("Starting evaluation loop...")
    for i, image in enumerate(images):
        logger.info(f"Processing image {i+1}/{len(images)}")

        image_sh = image.shape[:2]
        masks, _, class_masks, _ = model.eval(
            rescale_if_necessary(image, args.training_to_inference_mpp),
            batch_size=16,
            resample=True,
            normalize=True,
            invert=False,
            diameter=None,
            augment=args.tta,
            min_size=15,
            tile_overlap=0.1,
            bsize=256,
            compute_masks=True,
        )
        # rescale to original to prevent approximation errors
        masks = rescale_label_if_necessary(masks, image_sh)
        class_masks = rescale_label_if_necessary(class_masks, image_sh)

        pred_combined_hw2 = np.stack((masks, class_masks), axis=-1)
        all_pred_for_pq.append(pred_combined_hw2)

    end_time = time.time()

    if args.predictions_output_dir:
        logger.info("Saving predictions.")
        Path(args.predictions_output_dir).mkdir(parents=True, exist_ok=True)
        np.save(
            os.path.join(args.predictions_output_dir, "predictions.npy"),
            np.array(all_pred_for_pq, dtype=object),
        )

    if args.metrics_output_dir:
        labels_path = os.path.join(args.test_data_dir, "labels.npy")
        labels = load_labels(labels_path)
        if len(images) != len(labels):
            msg = f"Mismatch in number of images ({len(images)}) and labels ({len(labels)})."
            logger.error(msg)
            raise ValueError(msg)
        if labels.ndim == 4 and labels.shape[-1] >= 2:
            n_classes = int(labels[..., 1].max()) + 1
            logger.info(f"Determined number of classes: {n_classes}")
        elif labels.ndim != 1:
            msg = f"Labels array has unexpected shape {labels.shape}. Expected (N, H, W, 2)."
            logger.error(msg)
            raise ValueError(msg)

        for i in range(len(labels)):
            current_gt_for_pq = labels[i]
            all_gt_for_pq.append(current_gt_for_pq)

        logger.info("Preparing data for metrics computation.")
        gt_for_pq = np.array(
            [x.astype(int) for x in all_gt_for_pq], dtype=object
        )
        pred_for_pq = np.array(
            [x.astype(int) for x in all_pred_for_pq], dtype=object
        )

        if args.ignore_classes:
            for i in args.ignore_classes:
                for i in range(len(pred_for_pq)):
                    gt_for_pq[i][..., 1][gt_for_pq[i][..., 1] == i] = 0
                    pred_for_pq[i][..., 1][pred_for_pq[i][..., 1] == i] = 0

        logger.info("Running metrics computation.")
        global_metrics, per_image_metrics = compute_multiclass_pq_metrics(
            gt_for_pq,
            pred_for_pq,
            match_iou=args.match_iou,
            nr_classes=n_classes - 1,
            n_workers=args.n_workers_metrics,
        )

        if args.model_name is None:
            model_basename = os.path.basename(args.model_path)
            model_name_without_ext = ".".join(
                os.path.splitext(model_basename)[:-1]
            )
        else:
            model_name_without_ext = args.model_name

        print(global_metrics.to_string(index=False))

        metrics_output = os.path.join(
            args.metrics_output_dir,
            model_name_without_ext + "_metrics.csv",
        )
        per_image_metrics_output = os.path.join(
            args.metrics_output_dir,
            model_name_without_ext + "_per_image_metrics.csv",
        )

        global_metrics.to_csv(metrics_output, index=False)
        logger.info(f"Global results saved to {metrics_output}")

        per_image_metrics.to_csv(per_image_metrics_output, index=False)
        logger.info(f"Per-image results saved to {per_image_metrics_output}")

    logger.info(
        "Evaluation loop completed in %.2f seconds", end_time - start_time
    )
    logger.info(
        "Average time per image: %.2f seconds",
        (end_time - start_time) / len(images),
    )
    logger.info("Inference and visualization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Classpose inference and visualization."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained Classpose model weights.",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
        help="Path to the test dataset directory (must contain images.npy and optionally labels.npy).",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Triggers half precision inference.",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Triggers test time augmentation.",
    )
    parser.add_argument(
        "--predictions_output_dir",
        type=str,
        default=None,
        help="Path to save the predictions. If not provided, predictions will not be saved.",
    )
    parser.add_argument(
        "--metrics_output_dir",
        type=str,
        default=None,
        help="Directory to save the metrics CSV file. If not provided, metrics will not be computed.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model. If not provided, the model name will be extracted from the model path.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Path to the log file. If not provided, logs will not be saved.",
    )
    parser.add_argument(
        "--n_workers_metrics",
        type=int,
        default=0,
        help="Number of workers for parallel metrics computation. If 0, uses a single CPU core.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., cuda:0, mps, cpu). If not provided, uses"
        " the best available device.",
    )
    parser.add_argument(
        "--match_iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching instances during metrics computation.",
    )
    parser.add_argument(
        "--ignore_classes",
        type=int,
        default=None,
        nargs="+",
        help="Classes to ignore. These are set to 0 during metrics calculation.",
    )
    parser.add_argument(
        "--training_to_inference_mpp",
        type=str,
        default=None,
        help="Training to inference MPP ratio."
        "If a single number, it will be used to rescale inference images before prediction."
        "If two colon-separated numbers are provided (i.e. 0.25:0.5), their ratio will be used.",
    )

    args = parser.parse_args()

    main(args)
