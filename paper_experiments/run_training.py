import os
import numpy as np
import torch
import logging
import argparse
import time
import sys
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP

# Default values, can be overridden by command-line arguments
DEFAULT_DATA_DIR = "../data/conic_miccai/train"
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 4
DEFAULT_BASE_LEARNING_RATE = 5e-5
DEFAULT_LR_SCALING = "none"
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "models/classpose_miccai"
DEFAULT_MAKE_SPARSE = False
DEFAULT_MODEL_NAME = None
DEFAULT_USE_CLASS_WEIGHTS = True

if os.getenv("TF32", "0") == "1":
    torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train Classpose model.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing images.npy and labels.npy",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="Fraction of data to use for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr_scaling",
        type=str,
        choices=["none", "sqrt"],
        default=DEFAULT_LR_SCALING,
        help="Distributed-only learning-rate scaling mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--make_sparse",
        action="store_true",
        default=DEFAULT_MAKE_SPARSE,
        help="Whether to make the training labels sparse",
    )
    parser.add_argument(
        "--subsample_fraction",
        type=float,
        default=None,
        help="Fraction of data to subsample",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Name for the saved model (will use timestamp-based name if not provided)",
    )
    parser.add_argument(
        "--freeze",
        type=str,
        choices=["none", "backbone", "segmentation_head", "neck"],
        default=["none"],
        nargs="+",
        help="Select which parts to freeze. Backbone does not include neck",
    )
    parser.add_argument(
        "--oversampling_method",
        type=str,
        choices=["none", "stardist", "custom"],
        default="custom",
        help="Oversampling method: 'none' (uniform), 'stardist' (rare class focus), 'custom' (instance-weighted)",
    )
    parser.add_argument(
        "--n_rare_classes",
        type=int,
        default=4,
        help="Number of rarest classes to oversample (used with 'stardist' method)",
    )
    parser.add_argument(
        "--oversampling_power",
        type=float,
        default=1.0,
        help="Power to raise weights (used with 'custom' method)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Save model checkpoint every N epochs (default: 100)",
    )
    parser.add_argument(
        "--save_each",
        action="store_true",
        default=False,
        help="Save model to separate file at each checkpoint (creates multiple files)",
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        default=False,
        help="Disable class weighting for classification loss calculation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for training. Accepted values are 'auto', 'cpu', 'mps', or 'cuda'. To choose CUDA GPUs, set CUDA_VISIBLE_DEVICES and pass --device cuda.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a training-state checkpoint ending in '.train.pt' for resuming training. Inference weight files remain separate.",
    )
    parser.add_argument(
        "--augmentation_strategy",
        type=str,
        choices=["hed_only", "enhanced"],
        default="enhanced",
        help="Augmentation strategy to use: 'hed_only' for original ClassPose behavior, 'enhanced' for HED+HE staining+image quality augmentations",
    )
    parser.add_argument(
        "--min_train_masks",
        type=int,
        default=5,
        help="Minimum number of masks per instance in training data",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data augmentation."
        "If > 0, uses multiprocessing to parallelize data augmentation."
        "If 0, uses single process.",
    )
    parser.add_argument(
        "--feature_transformation_structure",
        type=int,
        nargs="+",
        default=None,
        help="Feature transformation structure to use.",
    )
    parser.add_argument(
        "--use_uncertainty_weighting",
        action="store_true",
        default=False,
        help="Use task uncertainty weighting for automatic loss balancing (default: False)",
    )
    parser.add_argument(
        "--validate_every_epoch",
        action="store_true",
        default=False,
        help="If set, run validation every epoch. Otherwise, use default validation schedule (epoch 5 and every 10 epochs)",
    )
    return parser.parse_args()


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


from classpose import train
from classpose.distributed import (
    broadcast_object,
    cleanup_distributed,
    is_main_process,
    setup_distributed,
)
from classpose.models import ClassposeModel
from classpose.utils import make_sparse
from classpose.log import get_logger, add_file_handler


def main(args):
    torchrun_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed_requested = torchrun_world_size > 1
    if args.device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(
            "Training only accepts --device=auto, --device=cpu, --device=mps, "
            "or --device=cuda. To choose CUDA GPUs, set CUDA_VISIBLE_DEVICES "
            "and use --device cuda."
        )
    if distributed_requested and args.device in {"cpu", "mps"}:
        raise ValueError(
            "Distributed training currently supports CUDA only. "
            "Keep CPU/MPS on the single-process path."
        )

    context = None
    try:
        context = setup_distributed(
            device_arg="cuda" if distributed_requested else args.device,
        )
        logger = get_logger("classpose")

        effective_model_name = args.model_name
        if effective_model_name is None and is_main_process():
            effective_model_name = f"classpose_{time.time()}"
        effective_model_name = broadcast_object(effective_model_name)
        args.model_name = effective_model_name

        model_specific_dir = os.path.join(args.output_dir, effective_model_name)
        os.makedirs(model_specific_dir, exist_ok=True)
        log_file_path = os.path.join(model_specific_dir, "training_run.log")
        if is_main_process():
            add_file_handler(logger, log_file_path)

        raw_command = (
            " ".join([sys.executable] + sys.argv)
            if sys.executable
            else " ".join(sys.argv)
        )

        logger.info("--- ClassPose Training Run Initializing ---")
        logger.info(f"Script: {os.path.abspath(__file__)}")
        logger.info(f"Raw Command: {raw_command}")
        logger.info(f"Parsed Arguments: {vars(args)}")
        logger.info(f"Base Output Directory: {os.path.abspath(args.output_dir)}")
        logger.info(
            f"Model Specific Directory: {os.path.abspath(model_specific_dir)}"
        )
        logger.info(f"Logging to file: {os.path.abspath(log_file_path)}")
        logger.info(
            "Distributed context: distributed=%s. Backend=%s. Rank=%s. Local_rank=%s. World_size=%s. Device=%s.",
            context.distributed,
            context.backend,
            context.rank,
            context.local_rank,
            context.world_size,
            context.device,
        )
        resolved_learning_rate = DEFAULT_BASE_LEARNING_RATE
        if context.distributed:
            effective_global_batch = args.batch_size * context.world_size
            logger.info(
                "In distributed mode, --batch_size is per-rank. Effective global batch = %s.",
                effective_global_batch,
            )
            if args.lr_scaling == "sqrt":
                resolved_learning_rate = DEFAULT_BASE_LEARNING_RATE * (
                    np.sqrt(effective_global_batch / DEFAULT_BATCH_SIZE)
                )
                if effective_global_batch != DEFAULT_BATCH_SIZE:
                    logger.info(
                        "Resolved distributed learning rate %.6g from base %.6g. ",
                        resolved_learning_rate,
                        DEFAULT_BASE_LEARNING_RATE,
                    )
            else:
                logger.info(
                    "Distributed learning rate scaling disabled. Using learning rate %.6g.",
                    resolved_learning_rate,
                )
        logger.info("-------------------------------------------")

        if args.oversampling_method == "stardist" and args.n_rare_classes <= 0:
            raise ValueError(
                "n_rare_classes must be positive when using 'stardist' oversampling"
            )
        if args.oversampling_method == "custom" and args.oversampling_power < 0:
            raise ValueError(
                "oversampling_power must be non-negative when using 'custom' oversampling"
            )

        logger.info("Loading data")

        images_path = os.path.join(args.data_dir, "images.npy")
        labels_path = os.path.join(args.data_dir, "labels.npy")

        if os.path.exists(images_path):
            images = np.load(images_path, allow_pickle=True)
            if np.issubdtype(images[0].dtype, np.object_):
                new_images = np.empty(len(images), dtype=object)
                for i, im in enumerate(images):
                    im = np.ascontiguousarray(im).astype(np.float32)
                    new_images[i] = im
                images = new_images
        else:
            raise FileNotFoundError(f"Images not found: should be {images_path}")

        if os.path.exists(labels_path):
            labels = np.load(labels_path, allow_pickle=True)
            if np.issubdtype(labels[0].dtype, np.object_):
                new_labels = np.empty(len(labels), dtype=object)
                for i, im in enumerate(labels):
                    im = np.ascontiguousarray(im).astype(np.int64)
                    new_labels[i] = im
                labels = new_labels
            if np.issubdtype(labels[0].dtype, np.floating):
                logger.info("Labels are floating, converting to int64")
                n_unique_float_labels = len(np.unique(labels))
                labels = [im.astype(np.int64) for im in labels]
                if len(np.unique(labels)) != n_unique_float_labels:
                    error = (
                        "Different number of unique labels after conversion to int64 - please check labels!"
                    )
                    logger.critical(error)
                    raise ValueError(error)
        else:
            raise FileNotFoundError(f"Labels not found: should be {labels_path}")

        if args.subsample_fraction is not None:
            logger.info(f"Subsampling data to {args.subsample_fraction} fraction")
            n = images.shape[0]
            all_indices = np.arange(n, dtype=np.int32)
            rng = np.random.default_rng(args.seed)
            rng.shuffle(all_indices)
            idxs = all_indices[: int(args.subsample_fraction * n)]
            images = images[idxs]
            labels = labels[idxs]

        n = images.shape[0]
        all_indices = np.arange(n, dtype=np.int32)
        transp = (2, 0, 1)
        if args.train_fraction < 1.0:
            train_idx, test_idx = train_test_split(
                all_indices,
                train_size=args.train_fraction,
                random_state=args.seed,
            )
            train_images = [np.transpose(im, transp) for im in images[train_idx]]
            train_labels = [np.transpose(im, transp) for im in labels[train_idx]]
            test_images = [np.transpose(im, transp) for im in images[test_idx]]
            test_labels = [np.transpose(im, transp) for im in labels[test_idx]]
        else:
            train_images = [np.transpose(im, transp) for im in images]
            train_labels = [np.transpose(im, transp) for im in labels]
            test_images = None
            test_labels = None

        if args.make_sparse:
            logger.info("Making training labels sparse.")
            train_labels = make_sparse(
                train_labels, fraction=0.1, seed=args.seed
            )
        else:
            logger.info("Skipping making training labels sparse.")

        train_probs = None
        if args.oversampling_method == "stardist":
            from classpose.utils import compute_stardist_oversampling_probabilities

            logger.info(
                "Computing oversampling probabilities using StarDist methodology"
            )
            train_probs = compute_stardist_oversampling_probabilities(
                train_labels, n_rare_classes=args.n_rare_classes
            )
            logger.info(
                f"StarDist oversampling enabled - probability range: {train_probs.min():.6f} to {train_probs.max():.6f}"
            )
        elif args.oversampling_method == "custom":
            from classpose.utils import compute_custom_oversampling_probabilities

            logger.info(
                "Computing oversampling probabilities using custom instance-weighted method"
            )
            train_probs = compute_custom_oversampling_probabilities(
                train_labels, power=args.oversampling_power
            )
            logger.info(
                f"Custom oversampling enabled - probability range: {train_probs.min():.6f} to {train_probs.max():.6f}"
            )
        else:
            logger.info("Using uniform sampling (no oversampling)")

        n_classes = int(max([im[1].max() for im in train_labels]) + 1)

        class_weights = None
        if not args.no_class_weights:
            from classpose.utils import get_class_weights

            logger.info(
                "Computing class weights using inverse frequency with square root scaling"
            )
            class_weights = get_class_weights(train_labels, n_classes)
            logger.info(f"Class weights computed: {class_weights.tolist()}")
        else:
            logger.info("Class weighting disabled - using uniform weights")

        model = ClassposeModel(
            gpu=False,
            pretrained_model="cpsam",
            device=context.device,
            nclasses=n_classes,
            feature_transformation_structure=args.feature_transformation_structure,
        )
        model.net.freeze(
            "backbone" in args.freeze,
            "segmentation_head" in args.freeze,
            "neck" in args.freeze,
        )

        if context.distributed:
            # Current freeze modes disable parameters before wrapping, and the
            # active training graph uses all remaining trainable parameters.
            model.net = DDP(
                model.net,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                find_unused_parameters=False,
            )
            if os.environ.get("COMPILE_MODEL", "") == "1":
                logger.warning(
                    "COMPILE_MODEL=1 is disabled in distributed mode."
                )
        elif os.environ.get("COMPILE_MODEL", "") == "1":
            logger.info("Compiling model")
            model.net = torch.compile(model.net)

        config_snapshot = deepcopy(vars(args))
        config_snapshot.update(
            {
                "resolved_device": str(context.device),
                "distributed_active": context.distributed,
                "rank": context.rank,
                "world_size": context.world_size,
                "backend": context.backend,
            }
        )

        train.train_class_seg(
            net=model.net,
            train_data=train_images,
            train_labels=train_labels,
            test_data=test_images,
            test_labels=test_labels,
            train_probs=train_probs,
            batch_size=args.batch_size,
            learning_rate=resolved_learning_rate,
            n_epochs=args.epochs,
            save_path=args.output_dir,
            model_name=args.model_name,
            save_every=args.save_every,
            save_each=args.save_each,
            class_weights=class_weights,
            augmentation_strategy=args.augmentation_strategy,
            num_workers=args.num_workers,
            use_uncertainty_weighting=args.use_uncertainty_weighting,
            validate_every_epoch=args.validate_every_epoch,
            min_train_masks=args.min_train_masks,
            log_file_path=log_file_path if is_main_process() else None,
            random_seed=args.seed,
            device=context.device,
            distributed=context.distributed,
            rank=context.rank,
            world_size=context.world_size,
            resume_checkpoint=args.resume_checkpoint,
            config_snapshot=config_snapshot,
        )
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
