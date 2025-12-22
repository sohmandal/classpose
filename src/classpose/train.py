import os
import time
import random
from pathlib import Path


import numpy as np
import torch
from cellpose import dynamics, models, utils
from cellpose.train import _get_batch, _loss_fn_seg, _reshape_norm
from cellpose.transforms import normalize_img, random_rotate_and_resize
from skimage import io
from torch import nn
from tqdm import trange

from classpose.dataset import ClassposeDataset
from classpose.log import get_logger, add_file_handler
from classpose.utils import get_default_device

train_logger = get_logger(__name__)


class LossAggregator(nn.Module):
    """
    Multi-loss aggregator for automatic multi-task loss weighting.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    by Kendall et al. (2018).

    """

    def __init__(self, n_losses: int, optimise: bool = True):
        """
        Args:
            n_losses (int): Number of losses to aggregate
            optimise (bool): Whether to learn uncertainty weights (True) or use equal weighting (False)
        """
        super().__init__()
        self.n_losses = n_losses
        self.optimise = optimise

        self.log_var = nn.Parameter(
            torch.zeros(n_losses), requires_grad=optimise
        )

    def forward(self, *losses):
        """
        Forward pass applying uncertainty weighting.

        Args:
            *losses: Variable number of loss tensors to aggregate

        Returns:
            Combined weighted loss tensor
        """
        assert (
            len(losses) == self.n_losses
        ), f"Expected {self.n_losses} losses, got {len(losses)}"

        losses = torch.stack(list(losses))

        precision = torch.exp(-self.log_var)
        weighted_losses = precision * losses

        if self.optimise:
            # Add log variance terms for learnable case
            weighted_losses = weighted_losses + self.log_var

        return weighted_losses.sum()

    def get_uncertainty_factors(self, seg_trainable=True):
        """
        Get learned uncertainty weights for logging purposes.

        """
        with torch.no_grad():
            weights = torch.exp(-self.log_var)

            result = {}
            weight_idx = 0

            if seg_trainable:
                result["seg_weight"] = weights[weight_idx].item()
                weight_idx += 1

            result["ce_weight"] = weights[weight_idx].item()
            weight_idx += 1
            result["tversky_weight"] = weights[weight_idx].item()

            return result


def _loss_fn_tversky(
    lbl: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    class_weights: torch.Tensor | None = None,
    alpha: float = 0.3,
    gamma: float = 1.33,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Calculates the Tversky loss function between true labels lbl and prediction y.

    Adapted from [1].

    [1] https://arxiv.org/pdf/1810.07842

    Args:
        lbl (torch.Tensor): True labels.
        y (torch.Tensor): Predicted values.
        n_classes (int): Number of classes.
        alpha (float, optional): Alpha parameter for Tversky loss. Defaults to 0.3.
        gamma (float, optional): Gamma parameter for Tversky loss. Defaults to 1.33.
        eps (float, optional): Epsilon parameter for Tversky loss. Defaults to 1e-6.

    Returns:
        torch.Tensor: Loss value.
    """
    beta = 1 - alpha
    valid_mask = (lbl[:, 0] != -100).float()[:, None]
    lbl[:, 0][lbl[:, 0] == -100] = 0.0
    lbl_one_hot = nn.functional.one_hot(
        lbl[:, 0].long(), num_classes=n_classes
    ).permute(0, 3, 1, 2)
    y_cl = y[:, :-3]
    # Convert logits to probabilities using softmax
    y_probs = torch.softmax(y_cl, dim=1)
    tp = torch.sum(y_probs * lbl_one_hot * valid_mask, dim=(2, 3))
    fp = torch.sum(y_probs * (1 - lbl_one_hot) * valid_mask, dim=(2, 3))
    fn = torch.sum((1 - y_probs) * lbl_one_hot * valid_mask, dim=(2, 3))
    loss = 1.0 - torch.true_divide(tp, tp + alpha * fp + beta * fn)
    loss = torch.clip(loss, eps, 1 - eps)
    loss = loss.pow(1 / gamma)
    if class_weights is not None:
        loss = loss * class_weights

    return loss.mean()


def _loss_fn_class(
    lbl: torch.Tensor,
    y: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Calculates the loss function between true labels lbl and prediction y.

    The mask is used to ignore CE values whenever there are no annotations in lbl.
    In other words, when a mask is provided the only pixels whose CE is not
    considered are those where mask == 1 and lbl[:, 0] == 0.

    Args:
        lbl (torch.Tensor): True labels.
        y (torch.Tensor): Predicted values.
        class_weights (torch.Tensor, optional): Class weights tensor. Defaults to None.

    Returns:
        torch.Tensor: Loss value.
    """
    criterion3 = nn.CrossEntropyLoss(
        reduction="mean", weight=class_weights, ignore_index=-100
    )
    loss3 = criterion3(y[:, :-3], lbl[:, 0].long())

    return loss3


def split_labels(
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
            train_logger.info(">>> using files instead of loading dataset")
        else:
            # load all images
            train_logger.info(">>> loading images and labels")
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
        train_logger.critical(error_message)
        raise ValueError(error_message)
    if (test_labels is not None and nimg_test != len(test_labels)) or (
        test_labels_files is not None and nimg_test != len(test_labels_files)
    ):
        train_logger.warning("test data and labels not same length, not using")
        test_data, test_files = None, None
    if train_labels is not None:
        if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
            error_message = (
                "training data or labels are not at least two-dimensional"
            )
            train_logger.critical(error_message)
            raise ValueError(error_message)
        if train_data[0].ndim > 3:
            error_message = "training data is more than three-dimensional (should be 2D or 3D array)"
            train_logger.critical(error_message)
            raise ValueError(error_message)

        # perform here classpose-specific checks
        # we first check that labels have three dimensions
        if train_labels[0].ndim == 3:
            # check that the first channel is the instance label
            if train_labels[0].shape[0] == 2 or train_labels[0].shape[0] == 4:
                if train_classes is None:
                    train_labels, train_classes = split_labels(train_labels)
                else:
                    train_labels, _ = split_labels(train_labels)
                if test_labels is not None:
                    if test_classes is None:
                        test_labels, test_classes = split_labels(test_labels)
                    else:
                        test_labels, _ = split_labels(test_labels)
            else:
                error_message = "training labels should have two or four channels for classpose if classes are not provided"
                train_logger.critical(error_message)
                raise ValueError(error_message)
        else:
            error_message = "training labels should have three dimensions"
            train_logger.critical(error_message)
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
    train_logger.info(">>> computing diameters")
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
            train_logger.warning(
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
        train_logger.info(f">>> normalizing {normalize_params}")
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


def _process_single_image(args):
    """
    Helper function for multiprocessing - processes a single image with geometric transforms and normalization.

    Args:
        args: Tuple of (img, lbl, rescale_factor, scale_range, bsize, normalize_params)

    Returns:
        Tuple of (processed_image, processed_label)
    """
    img, lbl, r, scale_range, bsize, normalize_params = args
    img_, lbl_ = random_rotate_and_resize(
        [img], Y=[lbl], rescale=[r], scale_range=scale_range, xy=(bsize, bsize)
    )[:2]
    img_ = normalize_img(img_[0], **normalize_params)
    return img_, lbl_[0]


def _get_batch_and_augment(
    data: list[np.ndarray] | None,
    labels: list[np.ndarray] | None,
    files: list[str] | None,
    labels_files: list[str] | None,
    kwargs: dict,
    inds: list[int],
    diams: np.ndarray,
    diam_mean: float,
    rescale: bool,
    scale_range: list[float] | None,
    bsize: int,
    normalize_params: dict,
    augment: bool,
    augment_pipeline,
):
    imgs, lbls = _get_batch(
        inds,
        data=data,
        labels=labels,
        files=files,
        labels_files=labels_files,
        **kwargs,
    )
    diams = np.array([diams[i] for i in inds])
    rsc = diams / diam_mean if rescale else np.ones(len(diams), "float32")
    # augmentations
    if augment:
        if augment_pipeline is not None:
            imgs = augment_pipeline.transform_batch(imgs)
        # Parallelize random_rotate_and_resize and normalization
        args_list = [
            (imgs[i], lbls[i], rsc[i], scale_range, bsize, normalize_params)
            for i in range(len(imgs))
        ]
        imgi, lbl = zip(*(_process_single_image(args) for args in args_list))
        imgi, lbl = list(imgi), list(lbl)
    else:
        imgi = imgs
        lbl = lbls
        imgi = [normalize_img(img, **normalize_params) for img in imgi]
    imgi = np.stack(imgi)
    lbl = np.stack(lbl)
    return imgi, lbl


def seed_everything(seed: int):
    """
    Seed all random number generators for reproducibility.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.mps.manual_seed(seed)


def train_class_seg(
    net: torch.nn.Module,
    train_data: list[np.ndarray] | None = None,
    train_labels: list[np.ndarray] | None = None,
    train_files: list[str] | None = None,
    train_labels_files: list[str] | None = None,
    train_probs: list[float] | None = None,
    train_classes: list[int] | None = None,
    test_data: list[np.ndarray] | None = None,
    test_labels: list[np.ndarray] | None = None,
    test_files: list[str] | None = None,
    test_labels_files: list[str] | None = None,
    test_probs: list[float] | None = None,
    test_classes: list[int] | None = None,
    channel_axis: int | None = None,
    load_files: bool = True,
    batch_size: int = 1,
    learning_rate: float | list[float] = 5e-5,
    SGD: bool = False,
    n_epochs: int = 100,
    weight_decay: float = 0.1,
    normalize: bool | dict = True,
    compute_flows: bool = False,
    save_path: str | None = None,
    save_every: int = 100,
    save_each: bool = False,
    nimg_per_epoch: int | None = None,
    nimg_test_per_epoch: int | None = None,
    rescale: bool = False,
    scale_range: list[float] | None = None,
    bsize: int = 256,
    min_train_masks: int = 5,
    model_name: str | None = None,
    class_weights: list[float] | None = None,
    augmentation_strategy: str = "hed_only",
    num_workers: int = 4,
    use_uncertainty_weighting: bool = False,
    validate_every_epoch: bool = False,
    log_file_path: str | None = None,
    random_seed: int = 42,
):
    """
    Train the network with images for segmentation.

    Args:
        net (torch.nn.Module): The network model to train.
        train_data (list[np.ndarray], optional): List of arrays (2D or 3D) - images for training. Defaults to None.
        train_labels (list[np.ndarray], optional): List of arrays (2D or 3D) - labels for train_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        train_files (list[str], optional): List of strings - file names for images in train_data (to save flows for future runs). Defaults to None.
        train_labels_files (list, optional): List of training label file paths. Defaults to None.
        train_probs (list[float], optional): List of floats - probabilities for each image to be selected during training. Defaults to None.
        test_data (list[np.ndarray], optional): List of arrays (2D or 3D) - images for testing. Defaults to None.
        test_labels (list[np.ndarray], optional): List of arrays (2D or 3D) - labels for test_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        test_files (list[str], optional): List of strings - file names for images in test_data (to save flows for future runs). Defaults to None.
        test_labels_files (list, optional): List of test label file paths. Defaults to None.
        test_probs (list[float], optional): List of floats - probabilities for each image to be selected during testing. Defaults to None.
        load_files (bool, optional): Boolean - whether to load images and labels from files. Defaults to True.
        batch_size (int, optional): Integer - number of patches to run simultaneously on the GPU. Defaults to 8.
        learning_rate (float | list[float], optional): Float or list/np.ndarray - learning rate for training. Defaults to 0.005.
        n_epochs (int, optional): Integer - number of times to go through the whole training set during training. Defaults to 2000.
        weight_decay (float, optional): Float - weight decay for the optimizer. Defaults to 1e-5.
        momentum (float, optional): Float - momentum for the optimizer. Defaults to 0.9.
        SGD (bool, optional): Deprecated in v4.0.1+ - AdamW always used.
        normalize (bool | dict, optional): Boolean or dictionary - whether to normalize the data. Defaults to True.
        compute_flows (bool, optional): Boolean - whether to compute flows during training. Defaults to False.
        save_path (str, optional): String - where to save the trained model. Defaults to None.
        save_every (int, optional): Integer - save the network every [save_every] epochs. Defaults to 100.
        save_each (bool, optional): Boolean - save the network to a new filename at every [save_each] epoch. Defaults to False.
        nimg_per_epoch (int, optional): Integer - minimum number of images to train on per epoch. Defaults to None.
        nimg_test_per_epoch (int, optional): Integer - minimum number of images to test on per epoch. Defaults to None.
        rescale (bool, optional): Boolean - whether or not to rescale images during training. Defaults to True.
        min_train_masks (int, optional): Integer - minimum number of masks an image must have to use in the training set. Defaults to 5.
        model_name (str, optional): String - name of the network. Defaults to None.
        class_weights (list[float], optional): List of class weights for weighted loss computation. Defaults to None.
        augmentation_strategy (str, optional): Pre-defined augmentation strategy name. Options: 'hed_only', 'enhanced'. Defaults to 'hed_only'.
        num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        use_uncertainty_weighting (bool, optional): Whether to use task uncertainty weighting for automatic loss balancing. If True, the model learns optimal weights for segmentation, classification CE, and Tversky losses automatically. If False, uses equal weighting (1.0 each). Defaults to False.
        log_file_path (str, optional): Path to the log file. Defaults to None.
    Returns:
        tuple: A tuple containing the path to the saved model weights, training losses, and test losses.

    """
    if log_file_path is not None:
        add_file_handler(train_logger, log_file_path)

    if SGD:
        train_logger.warning("SGD is deprecated, using AdamW instead")

    device = net.device

    scale_range = 0.5 if scale_range is None else scale_range

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default.copy()
        normalize_params["normalize"] = normalize

    out = _process_train_test(
        train_data=train_data,
        train_labels=train_labels,
        train_files=train_files,
        train_labels_files=train_labels_files,
        train_probs=train_probs,
        train_classes=train_classes,
        test_data=test_data,
        test_labels=test_labels,
        test_files=test_files,
        test_labels_files=test_labels_files,
        test_probs=test_probs,
        test_classes=test_classes,
        load_files=load_files,
        min_train_masks=min_train_masks,
        compute_flows=compute_flows,
        channel_axis=channel_axis,
        normalize_params={"normalize": False},
        device=net.device,
    )
    (
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
    ) = out
    train_labels = [
        np.concatenate((y[:1], y_class, y[1:]), axis=0)
        for y, y_class in zip(train_labels, train_classes)
    ]
    if test_labels is not None:
        test_labels = [
            np.concatenate((y[:1], y_class, y[1:]), axis=0)
            for y, y_class in zip(test_labels, test_classes)
        ]
    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {
            "normalize_params": normalize_params,
            "channel_axis": channel_axis,
        }

    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    if class_weights is not None and isinstance(
        class_weights, (list, np.ndarray, tuple)
    ):
        class_weights = np.float32(class_weights)
        class_weights = torch.from_numpy(class_weights).to(device).float()

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = (
        nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch
    )

    # learning rate schedule
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 99:
        LR = LR[:-50]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))

    train_logger.info(
        f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}"
    )
    train_logger.info(
        f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}"
    )
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Check if segmentation head is trainable
    seg_trainable = (
        any(p.requires_grad for p in net.out.parameters())
        or net.W2.requires_grad
    )

    # Determine number of active losses
    n_active_losses = 2
    if seg_trainable:
        n_active_losses += 1

    # Initialize loss aggregator
    loss_aggregator = LossAggregator(
        n_losses=n_active_losses, optimise=use_uncertainty_weighting
    ).to(device)

    if use_uncertainty_weighting:
        # Add uncertainty parameters to optimizer with lower learning rate
        optimizer.add_param_group(
            {
                "params": loss_aggregator.parameters(),
                "lr": learning_rate
                * 0.1,  # Use lower LR for uncertainty params
            }
        )
        train_logger.info(
            f">>> Using task uncertainty weighting for {n_active_losses} active losses"
        )
    else:
        train_logger.info(
            f">>> Using equal weighting for {n_active_losses} active losses"
        )

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)

    # Create a directory named after the model and save weights inside it
    model_dir = save_path / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = model_dir / model_name

    train_logger.info(f">>> saving model to {filename}")

    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    best_val_loss = np.inf

    # Initialize augmentation pipeline based on strategy
    train_logger.info(
        f">>> Using augmentation strategy: '{augmentation_strategy}'"
    )

    # Log multiprocessing configuration
    if num_workers > 0:
        train_logger.info(
            f">>> Using multiprocessing for training with {num_workers} workers"
        )
    else:
        train_logger.info(">>> Using single-threaded processing for training")

    train_dataset = ClassposeDataset(
        data_array=train_data,
        label_array=train_labels,
        diameter_array=diam_train,
        diam_mean=net.diam_mean.item(),
        rescale=rescale,
        scale_range=scale_range,
        bsize=bsize,
        normalize_params=normalize_params,
        augment=True,
        augment_pipeline_config=augmentation_strategy,
        n_proc=num_workers,
        batch_size=batch_size,
    )

    seed_everything(random_seed)

    for iepoch in range(n_epochs):
        rng = np.random.default_rng(iepoch)
        if nimg != nimg_per_epoch:
            # choose random images for epoch with probability train_probs
            rperm = rng.choice(
                np.arange(0, nimg), size=(nimg_per_epoch,), p=train_probs
            )
        else:
            if train_probs is None:
                rperm = rng.permutation(np.arange(0, nimg))
            else:
                # otherwise oversample
                rperm = rng.choice(
                    np.arange(0, nimg), size=(nimg,), p=train_probs
                )
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch]  # set learning rate
        net.train()

        # Initialize epoch loss accumulators
        epoch_train_seg_loss = 0.0
        epoch_train_ce_loss = 0.0
        epoch_train_tversky_loss = 0.0
        epoch_train_total_loss = 0.0
        num_batches = 0

        # make rperm divisible by batch_size so there are no hanging images
        rperm = rperm[: nimg_per_epoch - nimg_per_epoch % batch_size]
        train_dataset.put(rperm.tolist(), reset_queues=True)

        for _ in trange(0, rperm.shape[0], batch_size):
            imgi, lbl = train_dataset.get()
            # network and loss optimization
            X = torch.from_numpy(imgi).to(device)
            lbl = torch.from_numpy(lbl).to(device)
            y = net(X)[0]

            batch_seg_loss = 0.0
            batch_ce_loss = 0.0
            batch_tversky_loss = 0.0

            if seg_trainable:
                loss_seg = _loss_fn_seg(lbl, y, device)
                batch_seg_loss = loss_seg.item()
            else:
                loss_seg = torch.tensor(0.0, device=device, requires_grad=True)

            loss_ce = _loss_fn_class(lbl, y, class_weights=class_weights)
            loss_tversky = _loss_fn_tversky(
                lbl,
                y,
                class_weights=class_weights,
                n_classes=net.n_cell_classes,
            )
            batch_ce_loss = loss_ce.item()
            batch_tversky_loss = loss_tversky.item()

            # Choose weighting strategy
            active_losses = []
            if seg_trainable:
                active_losses.append(loss_seg)
            active_losses.append(loss_ce)
            active_losses.append(loss_tversky)

            loss = loss_aggregator(*active_losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_total_loss = loss.item()

            # Accumulate losses for epoch averaging
            epoch_train_seg_loss += batch_seg_loss
            epoch_train_ce_loss += batch_ce_loss
            epoch_train_tversky_loss += batch_tversky_loss
            epoch_train_total_loss += batch_total_loss
            num_batches += 1

            train_loss = batch_total_loss
            train_loss *= len(imgi)

            # keep track of average training loss across epochs
            lavg += train_loss
            nsum += len(imgi)
            # per epoch training loss
            train_losses[iepoch] += train_loss

        # Average epoch losses
        avg_train_seg_loss = (
            epoch_train_seg_loss / num_batches if num_batches > 0 else 0.0
        )
        avg_train_ce_loss = (
            epoch_train_ce_loss / num_batches if num_batches > 0 else 0.0
        )
        avg_train_tversky_loss = (
            epoch_train_tversky_loss / num_batches if num_batches > 0 else 0.0
        )
        avg_train_total_loss = (
            epoch_train_total_loss / num_batches if num_batches > 0 else 0.0
        )

        train_losses[iepoch] /= nimg_per_epoch

        # Log training losses
        if use_uncertainty_weighting:
            current_weights = loss_aggregator.get_uncertainty_factors(
                seg_trainable=seg_trainable
            )
            train_logger.info(
                f"Epoch {iepoch}, Segmentation Loss: {avg_train_seg_loss:.4f}, Classification CE Loss: {avg_train_ce_loss:.4f}, Tversky Loss: {avg_train_tversky_loss:.4f}, Total Loss: {avg_train_total_loss:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s"
            )

            weight_parts = []
            if "seg_weight" in current_weights:
                weight_parts.append(f"Seg: {current_weights['seg_weight']:.3f}")
            if "ce_weight" in current_weights:
                weight_parts.append(f"CE: {current_weights['ce_weight']:.3f}")
            if "tversky_weight" in current_weights:
                weight_parts.append(
                    f"Tversky: {current_weights['tversky_weight']:.3f}"
                )

            if weight_parts:
                train_logger.info(
                    f"Epoch {iepoch} Uncertainty Weights - {', '.join(weight_parts)}"
                )
        else:
            train_logger.info(
                f"Epoch {iepoch}, Segmentation Loss: {avg_train_seg_loss:.4f}, Classification CE Loss: {avg_train_ce_loss:.4f}, Tversky Loss: {avg_train_tversky_loss:.4f}, Total Loss: {avg_train_total_loss:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s"
            )

        if validate_every_epoch or iepoch == 5 or iepoch % 10 == 0:
            lavgt = 0.0
            val_seg_loss = 0.0
            val_ce_loss = 0.0
            val_tversky_loss = 0.0
            val_total_loss = 0.0
            val_batches = 0

            if test_data is not None or test_files is not None:
                rng = np.random.default_rng(42)
                if nimg_test != nimg_test_per_epoch:
                    rperm = rng.choice(
                        np.arange(0, nimg_test),
                        size=(nimg_test_per_epoch,),
                        p=test_probs,
                    )
                else:
                    rperm = rng.permutation(np.arange(0, nimg_test))
                for ibatch in range(0, len(rperm), batch_size):
                    with torch.no_grad():
                        net.eval()
                        inds = rperm[ibatch : ibatch + batch_size]
                        imgi, lbl = _get_batch_and_augment(
                            data=test_data,
                            labels=test_labels,
                            files=test_files,
                            labels_files=test_labels_files,
                            kwargs=kwargs,
                            inds=inds,
                            diams=diam_test,
                            diam_mean=net.diam_mean.item(),
                            rescale=rescale,
                            scale_range=scale_range,
                            bsize=bsize,
                            normalize_params=normalize_params,
                            augment=True,
                            augment_pipeline=None,
                        )
                        X = torch.from_numpy(imgi).to(device)
                        lbl = torch.from_numpy(lbl).to(device)
                        y = net(X)[0]

                        batch_val_seg_loss = 0.0
                        batch_val_ce_loss = 0.0
                        batch_val_tversky_loss = 0.0

                        if seg_trainable:
                            loss_seg_val = _loss_fn_seg(lbl, y, device)
                            batch_val_seg_loss = loss_seg_val.item()
                        else:
                            loss_seg_val = torch.tensor(
                                0.0, device=device, requires_grad=True
                            )

                        loss_ce_val = _loss_fn_class(
                            lbl, y, class_weights=class_weights
                        )
                        loss_tversky_val = _loss_fn_tversky(
                            lbl,
                            y,
                            class_weights=class_weights,
                            n_classes=net.n_cell_classes,
                        )
                        batch_val_ce_loss = loss_ce_val.item()
                        batch_val_tversky_loss = loss_tversky_val.item()

                        # Choose weighting strategy for validation
                        active_losses = []
                        if seg_trainable:
                            active_losses.append(loss_seg_val)
                        active_losses.append(loss_ce_val)
                        active_losses.append(loss_tversky_val)

                        loss = loss_aggregator(*active_losses)

                        test_loss = loss.item()
                        test_loss *= len(imgi)
                        lavgt += test_loss

                        # Accumulate validation losses
                        val_seg_loss += batch_val_seg_loss
                        val_ce_loss += batch_val_ce_loss
                        val_tversky_loss += batch_val_tversky_loss
                        val_total_loss += loss.item()
                        val_batches += 1

                lavgt /= len(rperm)
                test_losses[iepoch] = lavgt

                # Average validation losses
                avg_val_seg_loss = (
                    val_seg_loss / val_batches if val_batches > 0 else 0.0
                )
                avg_val_ce_loss = (
                    val_ce_loss / val_batches if val_batches > 0 else 0.0
                )
                avg_val_tversky_loss = (
                    val_tversky_loss / val_batches if val_batches > 0 else 0.0
                )
                avg_val_total_loss = (
                    val_total_loss / val_batches if val_batches > 0 else 0.0
                )

                # Log validation losses
                if use_uncertainty_weighting:
                    current_weights = loss_aggregator.get_uncertainty_factors(
                        seg_trainable=seg_trainable
                    )
                    train_logger.info(
                        f"Epoch {iepoch} Validation, Segmentation Loss: {avg_val_seg_loss:.4f}, Classification CE Loss: {avg_val_ce_loss:.4f}, Tversky Loss: {avg_val_tversky_loss:.4f}, Total Loss: {avg_val_total_loss:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s"
                    )

                    weight_parts = []
                    if "seg_weight" in current_weights:
                        weight_parts.append(
                            f"Seg: {current_weights['seg_weight']:.3f}"
                        )
                    if "ce_weight" in current_weights:
                        weight_parts.append(
                            f"CE: {current_weights['ce_weight']:.3f}"
                        )
                    if "tversky_weight" in current_weights:
                        weight_parts.append(
                            f"Tversky: {current_weights['tversky_weight']:.3f}"
                        )

                    if weight_parts:
                        train_logger.info(
                            f"Epoch {iepoch} Validation Uncertainty Weights - {', '.join(weight_parts)}"
                        )
                else:
                    train_logger.info(
                        f"Epoch {iepoch} Validation, Segmentation Loss: {avg_val_seg_loss:.4f}, Classification CE Loss: {avg_val_ce_loss:.4f}, Tversky Loss: {avg_val_tversky_loss:.4f}, Total Loss: {avg_val_total_loss:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s"
                    )

                # Save best model based on validation loss
                if avg_val_total_loss < best_val_loss:
                    best_val_loss = avg_val_total_loss
                    filename_best = model_dir / f"{model_name}_best"
                    train_logger.info(
                        f"New best validation loss: {best_val_loss:.4f}. Saving model to {filename_best}"
                    )
                    net.save_model(filename_best)

            lavg /= nsum
            lavg, nsum = 0, 0

        if iepoch == n_epochs - 1 or (iepoch % save_every == 0 and iepoch != 0):
            if (
                save_each and iepoch != n_epochs - 1
            ):  # separate files as model progresses
                filename0 = str(filename) + f"_epoch_{iepoch:04d}"
            else:
                filename0 = filename
            train_logger.info(f"saving network parameters to {filename0}")
            net.save_model(filename0)

    train_logger.info(f"Saving the final model to {filename}")
    net.save_model(filename)

    return filename, train_losses, test_losses
