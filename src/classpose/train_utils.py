import os
import numpy as np
import torch
from skimage import io
from skimage.measure import label
from tqdm import trange
from sklearn.model_selection import train_test_split
from cellpose import dynamics, utils
from cellpose.train import _reshape_norm
from classpose import models
from classpose.log import get_logger
from classpose.dataset import ClassposeTrainingDataset, ClassposeDataset
from classpose.utils import get_default_device

logger = get_logger(__name__)


def _split_labels(
    labels: list[np.ndarray], mask_classes: bool = True
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Split labels into class and mask arrays.

    Args:
        labels (list[np.ndarray]): List of label arrays.
        mask_classes (bool, optional): masks class labels. Defaults to True.
            Class labels are masked out if their corresponding foreground pixel
            is not foreground.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: Tuple of class and mask arrays.
    """
    classes = [labels[i][-1:] for i in range(len(labels))]
    labels = [labels[i][:-1] for i in range(len(labels))]
    if mask_classes:
        for i in range(len(classes)):
            classes[i] = np.int16(classes[i])
            # mask class labels where instance foreground is 0
            classes[i][np.logical_and(labels[i][0] == 0, classes[i] > 0)] = -100
            # mask class labels where class foreground is 0 but instance foreground is > 0
            classes[i][np.logical_and(labels[i][0] > 0, classes[i] == 0)] = -100
    return labels, classes


def _process_train_test(
    train_data: list[np.ndarray] | None = None,
    train_labels: list[np.ndarray] | None = None,
    train_files: list[str] | None = None,
    train_labels_files: list[str] | None = None,
    train_probs: np.ndarray | None = None,
    train_classes: np.ndarray | None = None,
    test_data: list[np.ndarray] | None = None,
    test_labels: list[np.ndarray] | None = None,
    test_files: list[str] | None = None,
    test_labels_files: list[str] | None = None,
    test_probs: np.ndarray | None = None,
    test_classes: np.ndarray | None = None,
    load_files: bool = True,
    min_train_masks: int = 5,
    compute_flows: bool = False,
    normalize_params: dict = {"normalize": False},
    channel_axis: int | None = None,
    device: torch.device | None = None,
):
    """
    Process train and test data.

    The main change when compared with Cellpose is that, if not provided,
    class labels are expected to be the last channel in the label array. As
    such, the label array should have 2 (instance + class) or 4
    (instance + flows (2) + class) channels.

    Args:
        train_data (list[np.ndarray] or None): List of training data arrays.
        train_labels (list[np.ndarray] or None): List of training label arrays.
        train_files (list[str] or None): List of training file paths.
        train_labels_files (list[str] or None): List of training label file paths.
        train_probs (np.ndarray or None): Array of training probabilities.
        train_classes (np.ndarray or None): Array of training classes.
        test_data (list[np.ndarray] or None): List of test data arrays.
        test_labels (list[np.ndarray] or None): List of test label arrays.
        test_files (list[str] or None): List of test file paths.
        test_labels_files (list[str] or None): List of test label file paths.
        test_probs (np.ndarray or None): Array of test probabilities.
        test_classes (np.ndarray or None): Array of test classes.
        load_files (bool): Whether to load data from files.
        min_train_masks (int): Minimum number of masks required for training images.
        compute_flows (bool): Whether to compute flows.
        channels (list[int] or None): List of channel indices to use.
        channel_axis (int or None): Axis of channel dimension.
        rgb (bool): Convert training/testing images to RGB.
        normalize_params (dict): Dictionary of normalization parameters.
        device (torch.device): Device to use for computation.

    Returns:
        tuple: A tuple containing the processed train and test data and sampling probabilities and diameters.
    """
    device = get_default_device(device)

    if train_data is not None and train_labels is not None:
        # if data is loaded
        nimg = len(train_data)
        nimg_test = len(test_data) if test_data is not None else None
    else:
        # otherwise use files
        nimg = len(train_files)
        if train_labels_files is None:
            train_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif"
                for tf in train_files
            ]
            train_labels_files = [
                tf for tf in train_labels_files if os.path.exists(tf)
            ]
        if (
            test_data is not None or test_files is not None
        ) and test_labels_files is None:
            test_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in test_files
            ]
            test_labels_files = [
                tf for tf in test_labels_files if os.path.exists(tf)
            ]
        if not load_files:
            logger.info("Using files instead of loading dataset")
        else:
            # load all images
            logger.info("Loading images and labels")
            train_data = [io.imread(train_files[i]) for i in trange(nimg)]
            train_labels = [
                io.imread(train_labels_files[i]) for i in trange(nimg)
            ]
        nimg_test = len(test_files) if test_files is not None else None
        if load_files and nimg_test:
            test_data = [io.imread(test_files[i]) for i in trange(nimg_test)]
            test_labels = [
                io.imread(test_labels_files[i]) for i in trange(nimg_test)
            ]

    ### check that arrays are correct size
    if (train_labels is not None and nimg != len(train_labels)) or (
        train_labels_files is not None and nimg != len(train_labels_files)
    ):
        error_message = "train data and labels not same length"
        logger.critical(error_message)
        raise ValueError(error_message)
    if (test_labels is not None and nimg_test != len(test_labels)) or (
        test_labels_files is not None and nimg_test != len(test_labels_files)
    ):
        logger.warning("test data and labels not same length, not using")
        test_data, test_files = None, None
    if train_labels is not None:
        if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
            error_message = (
                "training data or labels are not at least two-dimensional"
            )
            logger.critical(error_message)
            raise ValueError(error_message)
        if train_data[0].ndim > 3:
            error_message = "training data is more than three-dimensional (should be 2D or 3D array)"
            logger.critical(error_message)
            raise ValueError(error_message)

        # perform here classpose-specific checks
        # we first check that labels have three dimensions
        if train_labels[0].ndim == 3:
            # check that the first channel is the instance label
            if train_labels[0].shape[0] == 2 or train_labels[0].shape[0] == 4:
                if train_classes is None:
                    train_labels, train_classes = _split_labels(train_labels)
                else:
                    train_labels, _ = _split_labels(train_labels)
                if test_labels is not None:
                    if test_classes is None:
                        test_labels, test_classes = _split_labels(test_labels)
                    else:
                        test_labels, _ = _split_labels(test_labels)
            else:
                error_message = "training labels should have two or four channels for classpose if classes are not provided"
                logger.critical(error_message)
                raise ValueError(error_message)
        else:
            error_message = "training labels should have three dimensions"
            logger.critical(error_message)
            raise ValueError(error_message)

    ### check that flows are computed
    if train_labels is not None:
        train_labels = dynamics.labels_to_flows(
            train_labels, files=train_files, device=device
        )
        if test_labels is not None:
            test_labels = dynamics.labels_to_flows(
                test_labels, files=test_files, device=device
            )
    elif compute_flows:
        for k in trange(nimg):
            tl = dynamics.labels_to_flows(
                io.imread(train_labels_files[k]),
                files=train_files,
                device=device,
            )
        if test_files is not None:
            for k in trange(nimg_test):
                tl = dynamics.labels_to_flows(
                    io.imread(test_labels_files[k]),
                    files=test_files,
                    device=device,
                )

    ### compute diameters
    nmasks = np.zeros(nimg)
    diam_train = np.zeros(nimg)
    logger.info("Computing diameters")
    for k in trange(nimg):
        tl = (
            train_labels[k][0]
            if train_labels is not None
            else io.imread(train_labels_files[k])[0]
        )
        diam_train[k], dall = utils.diameters(tl)
        nmasks[k] = len(dall)
    diam_train[diam_train < 5] = 5.0
    if test_data is not None:
        diam_test = np.array(
            [
                utils.diameters(test_labels[k][0])[0]
                for k in trange(len(test_labels))
            ]
        )
        diam_test[diam_test < 5] = 5.0
    elif test_labels_files is not None:
        diam_test = np.array(
            [
                utils.diameters(io.imread(test_labels_files[k])[0])[0]
                for k in trange(len(test_labels_files))
            ]
        )
        diam_test[diam_test < 5] = 5.0
    else:
        diam_test = None

    ### check to remove training images with too few masks
    if min_train_masks > 0:
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            logger.warning(
                f"{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set"
            )
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            if train_data is not None:
                train_data = [train_data[i] for i in ikeep]
                train_labels = [train_labels[i] for i in ikeep]
            if train_files is not None:
                train_files = [train_files[i] for i in ikeep]
            if train_labels_files is not None:
                train_labels_files = [train_labels_files[i] for i in ikeep]
            if train_probs is not None:
                train_probs = train_probs[ikeep]
            if train_classes is not None:
                train_classes = [train_classes[i] for i in ikeep]
            diam_train = diam_train[ikeep]
            nimg = len(train_data)

    ### normalize probabilities
    train_probs = (
        1.0 / nimg * np.ones(nimg, "float64")
        if train_probs is None
        else train_probs
    )
    train_probs /= train_probs.sum()
    if test_files is not None or test_data is not None:
        test_probs = (
            1.0 / nimg_test * np.ones(nimg_test, "float64")
            if test_probs is None
            else test_probs
        )
        test_probs /= test_probs.sum()

    ### reshape and normalize train / test data
    normed = False
    if normalize_params["normalize"]:
        logger.info(f"Normalizing {normalize_params}")
    if train_data is not None:
        train_data = _reshape_norm(
            train_data,
            channel_axis=channel_axis,
            normalize_params=normalize_params,
        )
        normed = True
    if test_data is not None:
        test_data = _reshape_norm(
            test_data,
            channel_axis=channel_axis,
            normalize_params=normalize_params,
        )

    return (
        train_data,
        train_labels,
        train_files,
        train_labels_files,
        train_probs,
        train_classes,
        diam_train,
        test_data,
        test_labels,
        test_files,
        test_labels_files,
        test_probs,
        test_classes,
        diam_test,
        normed,
    )


def _build_dataset(
    data: list[np.ndarray],
    labels: list[np.ndarray],
    diameters: np.ndarray,
    diam_mean: float,
    rescale: bool,
    scale_range: float | list[float] | None,
    bsize: int,
    normalize_params: dict,
    augment_pipeline_config: str | None,
) -> ClassposeTrainingDataset:
    return ClassposeTrainingDataset(
        data_array=data,
        label_array=labels,
        diameter_array=diameters,
        diam_mean=diam_mean,
        rescale=rescale,
        scale_range=scale_range,
        bsize=bsize,
        normalize_params=normalize_params,
        augment=True,
        augment_pipeline_config=augment_pipeline_config,
    )


def get_class_counts(Y: list[np.ndarray], n_classes: int) -> np.ndarray:
    """
    Get class counts.

    Args:
        Y (np.ndarray): Array of labels (2 channels: instance and class).
        n_classes (int): Number of classes.

    Returns:
        np.ndarray: Array of class counts.
    """
    return np.bincount(
        np.concatenate([y[1].ravel() for y in Y]), minlength=n_classes
    )


def get_instance_counts(
    labels: np.ndarray | list[np.ndarray],
    label_instances: bool = False,
    n_classes: int | None = None,
) -> np.ndarray:
    """
    Get instance counts.

    Args:
        labels (np.ndarray): Array of labels (2 channels: instance and class) or
            list of such arrays.
        label_instances (bool, optional): Whether to label instances. Defaults to False.
        n_classes (int | None, optional): Number of classes. If None, it will be
            inferred from the labels. Defaults to None.

    Returns:
        np.ndarray: Array of instance counts.
    """
    if n_classes is None:
        n_classes = np.max([im[1].max() for im in labels]) + 1
    counts = np.zeros((len(labels), n_classes))
    for i in range(len(labels)):
        if label_instances:
            instances = label(labels[i][0])
        else:
            instances = labels[i][0]
        classes = labels[i][1]
        for j in range(n_classes):
            counts[i, j] = np.unique(instances[classes == j]).size
    return counts


def get_class_weights(class_counts: np.ndarray) -> np.ndarray:
    """
    Get class weights.

    Based on https://github.com/stardist/stardist/blob/conic-2022/examples/conic-2022/train.ipynb

    Args:
        class_counts (np.ndarray): Array of class counts.
        n_classes (int): Number of classes.

    Returns:
        np.ndarray: Array of class weights.
    """
    logger.info(
        "Computing class weights using inverse frequency with square root scaling"
    )
    inv_freq = np.median(class_counts) / class_counts
    inv_freq = inv_freq**0.5
    class_weights = inv_freq.round(4)
    logger.info(f"class weights = {class_weights.tolist()}")
    logger.info(f"Class weights computed: {class_weights.tolist()}")
    return class_weights


def compute_oversampling_probabilities(
    class_counts: np.ndarray, instance_counts: np.ndarray, power: float = 1
) -> np.ndarray:
    """
    Compute custom oversampling probabilities using instance-weighted class balancing.

    Args:
        class_counts (np.ndarray): class count array (1 dimensional, one entry
            for each class).
        instance_counts (np.ndarray): instance count array (2 dimensional, one
            entry for each sample and class).
        power (float, optional): Power to raise the weights to. Defaults to 1.

    Returns:
        np.ndarray: Normalized probability array for training sample selection.
    """
    logger.info(f"Computing oversampling probabilities with power {power}")
    class_weights = 1 / class_counts
    class_weights[0] = 0
    weights = np.sum(instance_counts * class_weights[None], 1)
    weights = weights**power
    weights = weights / weights.sum()
    logger.info(
        f"Custom oversampling - probability range: {weights.min():.6f} to {weights.max():.6f}"
    )
    return weights


def process_and_build_dataset(
    images: list[np.ndarray],
    labels: list[np.ndarray],
    diam_mean: float = 30,
    device: torch.device | None = None,
    normalize: bool | dict = True,
    compute_flows: bool = True,
    channel_axis: int | None = None,
    rescale: bool = False,
    scale_range: float | list[float] | None = 0.5,
    bsize: int = 256,
    augmentation_strategy: str | None = None,
) -> ClassposeTrainingDataset:
    """
    Process raw image and label arrays into a ClassposeTrainingDataset.

    """
    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default.copy()
        normalize_params["normalize"] = normalize

    transp = (2, 0, 1)
    images = [np.transpose(im, transp) for im in images]
    labels = [np.transpose(im, transp) for im in labels]

    out = _process_train_test(
        train_data=images,
        train_labels=labels,
        train_files=None,
        train_labels_files=None,
        train_probs=None,
        train_classes=None,
        test_data=None,
        test_labels=None,
        test_files=None,
        test_labels_files=None,
        test_probs=None,
        test_classes=None,
        load_files=False,
        min_train_masks=0,
        compute_flows=compute_flows,
        channel_axis=channel_axis,
        normalize_params={"normalize": False},
        device=device,
    )

    (
        proc_images,
        proc_labels,
        _,
        _,
        _,
        proc_classes,
        diameters,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = out

    proc_labels_concat = [
        np.concatenate((y[:1], y_class, y[1:]), axis=0)
        for y, y_class in zip(proc_labels, proc_classes)
    ]

    dataset = _build_dataset(
        data=proc_images,
        labels=proc_labels_concat,
        diameters=diameters,
        diam_mean=diam_mean,
        rescale=rescale,
        scale_range=scale_range,
        bsize=bsize,
        normalize_params=normalize_params,
        augment_pipeline_config=augmentation_strategy,
    )

    return dataset


def load_data_arrays(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    images_path = os.path.join(data_dir, "images.npy")
    labels_path = os.path.join(data_dir, "labels.npy")

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Images or labels not found in {data_dir}")

    images = np.load(images_path, allow_pickle=True)
    if np.issubdtype(images[0].dtype, np.object_):
        new_images = np.empty(len(images), dtype=object)
        for i, im in enumerate(images):
            new_images[i] = np.ascontiguousarray(im).astype(np.float32)
        images = new_images

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
            error = "Different number of unique labels after conversion to int64 - please check labels!"
            logger.critical(error)
            raise ValueError(error)

    return images, labels


def subsample_dataset(
    dataset: ClassposeDataset, subsample_fraction: float | None, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if subsample_fraction is None:
        return dataset
    logger.info(f"Subsampling dataset to {subsample_fraction} fraction")
    n = len(dataset)
    all_indices = np.arange(n, dtype=np.int32)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_indices)
    idxs = all_indices[: int(subsample_fraction * n)]
    dataset = dataset.subset(idxs)
    return dataset


def split_dataset(
    dataset: ClassposeDataset, train_fraction: float, seed: int
) -> tuple[ClassposeDataset, ClassposeDataset | None,]:
    if train_fraction < 1.0:
        n = len(dataset)
        all_indices = np.arange(n, dtype=np.int32)
        train_idx, test_idx = train_test_split(
            all_indices,
            train_size=train_fraction,
            random_state=seed,
        )
        train_dataset = dataset.subset(train_idx)
        test_dataset = dataset.subset(test_idx)
    else:
        train_dataset = dataset
        test_dataset = None

    return train_dataset, test_dataset


def oversample_classes(
    X: np.ndarray,
    Y: np.ndarray,
    n_extra_classes: int = 4,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample classes in Y.

    Based on https://github.com/stardist/stardist/blob/conic-2022/examples/conic-2022/train.ipynb

    Args:
        X (np.ndarray): Array of images.
        Y (np.ndarray): Array of labels (2 channels: instance and class).
        n_extra_classes (int, optional): Number of extra classes to oversample.
            Defaults to 4.
        seed (int | np.random.Generator | None, optional): Random seed.
            Defaults to None.

    Returns:
        tuple: A tuple containing the oversampled images and labels.
    """
    y0 = Y[:, 1]
    rng = np.random.default_rng(seed)

    # get the most infrequent classes
    class_counts = get_class_counts(y0, y0.max() + 1)
    extra_classes = np.argsort(class_counts)[:n_extra_classes]
    all(
        class_counts[c] > 0 or logger.critical(f"count 0 for class {c}")
        for c in extra_classes
    )

    # how many extra samples (more for infrequent classes)
    n_extras = np.sqrt(np.sum(class_counts[1:]) / class_counts[extra_classes])
    n_extras = n_extras / np.max(n_extras)
    logger.info(f"oversample classes: {extra_classes}")
    idx_take = np.arange(len(X))

    for c, n_extra in zip(extra_classes, n_extras):
        # oversample probability is ~ number of instances
        prob = np.sum(y0[:, ::2, ::2] == c, axis=(1, 2))
        prob = np.clip(prob, 0, np.percentile(prob, 99.8))
        prob = prob**2
        # prob[prob<np.percentile(prob,90)] = 0
        prob = prob / np.sum(prob)
        n_extra = int(n_extra * len(X))
        logger.info(f"adding {n_extra} images of class {c}")
        idx_extra = rng.choice(np.arange(len(X)), n_extra, p=prob)
        idx_take = np.append(idx_take, idx_extra)

    X, Y = map(lambda x: x[idx_take], (X, Y))
    return X, Y
