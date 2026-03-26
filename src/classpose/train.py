import os
import time
import random
from pathlib import Path
from typing import Any


import numpy as np
import torch
import torch.distributed as dist
from cellpose.train import _loss_fn_seg
from torch import nn
from torch.utils.data import DataLoader

from classpose.dataset import (
    ClassposeTrainingDataset,
    ClassposeHDF5Dataset,
    DistributedEpochSampler,
    SequentialDistributedSampler,
)
from classpose.distributed import (
    all_reduce_sum,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    seed_worker,
    sync_module_grads,
    unwrap_model,
)
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
            optimise (bool): Whether to learn uncertainty weights (True) or use
                equal weighting (False)
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
        and torch.backends.mps.is_available()
    ):
        torch.mps.manual_seed(seed)


def _set_optimizer_lrs(
    optimizer: torch.optim.Optimizer, learning_rate: float
) -> None:
    for param_group in optimizer.param_groups:
        lr_scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = learning_rate * lr_scale


def _build_dataloader(
    dataset,
    batch_size: int,
    sampler,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def _capture_rng_state() -> dict[str, Any]:
    rng_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda_all"] = torch.cuda.get_rng_state_all()
    return rng_state


def _restore_rng_state(rng_state: dict[str, Any] | None) -> None:
    if rng_state is None:
        return
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch_state = rng_state["torch"]
    if isinstance(torch_state, torch.Tensor):
        torch_state = torch_state.detach().cpu().to(dtype=torch.uint8)
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda_all" in rng_state:
        cuda_states = []
        for state in rng_state["cuda_all"]:
            if isinstance(state, torch.Tensor):
                cuda_states.append(state.detach().cpu().to(dtype=torch.uint8))
            else:
                cuda_states.append(state)
        torch.cuda.set_rng_state_all(cuda_states)


def _gather_rng_states(distributed: bool) -> list[dict[str, Any]]:
    local_state = _capture_rng_state()
    if not distributed:
        return [local_state]

    gathered_states: list[dict[str, Any] | None] = [None] * get_world_size()
    dist.all_gather_object(gathered_states, local_state)
    return [state for state in gathered_states if state is not None]


def _save_training_checkpoint(
    checkpoint_path: Path,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_aggregator: nn.Module,
    epoch: int,
    best_val_loss: float,
    train_losses: np.ndarray,
    test_losses: np.ndarray,
    config_snapshot: dict[str, Any] | None,
    distributed: bool,
) -> None:
    rng_state_by_rank = _gather_rng_states(distributed)
    if not is_main_process():
        return

    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": unwrap_model(net).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_aggregator_state_dict": loss_aggregator.state_dict(),
        "best_val_loss": float(best_val_loss),
        "train_losses": train_losses,
        "test_losses": test_losses,
        "config_snapshot": config_snapshot,
        "rng_state_by_rank": rng_state_by_rank,
    }
    torch.save(checkpoint, checkpoint_path)


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _load_training_checkpoint(
    checkpoint_path: str,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_aggregator: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    unwrap_model(net).load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)

    loss_aggregator_state = checkpoint.get("loss_aggregator_state_dict")
    if loss_aggregator_state is not None:
        loss_aggregator.load_state_dict(loss_aggregator_state)

    rng_state_by_rank = checkpoint.get("rng_state_by_rank")
    if rng_state_by_rank:
        rank = get_rank()
        if rank < len(rng_state_by_rank):
            _restore_rng_state(rng_state_by_rank[rank])

    return checkpoint


def _should_validate(iepoch: int, validate_every_epoch: bool) -> bool:
    return validate_every_epoch or iepoch == 5 or iepoch % 10 == 0


def train_class_seg(
    net: torch.nn.Module,
    train_dataset: ClassposeHDF5Dataset | ClassposeTrainingDataset,
    train_probs: np.ndarray | None = None,
    test_dataset: ClassposeHDF5Dataset | ClassposeTrainingDataset | None = None,
    batch_size: int = 1,
    learning_rate: float | list[float] = 5e-5,
    n_epochs: int = 100,
    weight_decay: float = 0.1,
    save_path: str | None = None,
    save_every: int = 100,
    save_each: bool = False,
    nimg_per_epoch: int | None = None,
    nimg_test_per_epoch: int | None = None,
    scale_range: list[float] | None = None,
    model_name: str | None = None,
    class_weights: list[float] | None = None,
    num_workers: int = 4,
    use_uncertainty_weighting: bool = False,
    validate_every_epoch: bool = False,
    log_file_path: str | None = None,
    random_seed: int = 42,
    device: torch.device | None = None,
    distributed: bool | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    resume_checkpoint: str | None = None,
    config_snapshot: dict[str, Any] | None = None,
):
    """
    Train the network with images for segmentation.

    Args:
        net (torch.nn.Module): The network model to train.
        train_dataset (ClassposeHDF5Dataset | ClassposeTrainingDataset): Dataset containing training images and labels.
        train_probs (np.ndarray | None, optional): Probabilities for each image to be selected during training. Defaults to None.
        test_dataset (ClassposeHDF5Dataset | ClassposeTrainingDataset | None, optional): Dataset containing testing images and labels. Defaults to None.
        batch_size (int, optional): Number of training samples processed per optimizer step on the current process. Defaults to 1.
        learning_rate (float | list[float], optional): Base learning rate used to build the warmup/decay schedule. Defaults to 5e-5.
        n_epochs (int, optional): Number of training epochs to run. Defaults to 100.
        weight_decay (float, optional): Weight decay for AdamW. Defaults to 0.1.
        save_path (str | None, optional): String - where to save the trained model. Defaults to None.
        save_every (int, optional): Integer - save the network every [save_every] epochs. Defaults to 100.
        save_each (bool, optional): Boolean - save the network to a new filename at every [save_each] epoch. Defaults to False.
        nimg_per_epoch (int | None, optional): Integer - minimum number of images to train on per epoch. Defaults to None.
        nimg_test_per_epoch (int | None, optional): Integer - minimum number of images to test on per epoch. Defaults to None.
        scale_range (list[float] | None, optional): Range of scales for data augmentation. Defaults to None.
        model_name (str | None, optional): String - name of the network. Defaults to None.
        class_weights (list[float] | None, optional): List of class weights for weighted loss computation. Defaults to None.
        num_workers (int, optional): Number of DataLoader workers used for data loading. Defaults to 4.
        use_uncertainty_weighting (bool, optional): Whether to use task uncertainty weighting for automatic loss balancing. If True, the model learns optimal weights for segmentation, classification CE, and Tversky losses automatically. If False, uses equal weighting (1.0 each). Defaults to False.
        validate_every_epoch (bool, optional): If True, run validation at every epoch instead of the legacy sparse schedule. Defaults to False.
        log_file_path (str | None, optional): Path to the log file. Defaults to None.
        random_seed (int, optional): Base random seed. The effective seed is offset by rank in distributed training. Defaults to 42.
        device (torch.device | None, optional): Explicit device for training and checkpoint restore. If None, it is derived from the model/device helpers. Defaults to None.
        distributed (bool | None, optional): Whether distributed training is active. If None, the function infers it from torch.distributed state. Defaults to None.
        rank (int | None, optional): Rank of the current process in distributed training. If None, it is inferred from torch.distributed state. Defaults to None.
        world_size (int | None, optional): Total number of distributed processes. If None, it is inferred from torch.distributed state. Defaults to None.
        resume_checkpoint (str | None, optional): Path to a `.train.pt` training-state checkpoint to resume from. Defaults to None.
        config_snapshot (dict[str, Any] | None, optional): Run configuration persisted into resume checkpoints for traceability. Defaults to None.
    Returns:
        tuple: A tuple containing the path to the saved model weights, training losses, and test losses.

    """
    distributed = is_distributed() if distributed is None else distributed
    rank = get_rank() if rank is None else rank
    world_size = get_world_size() if world_size is None else world_size

    if log_file_path is not None and is_main_process():
        add_file_handler(train_logger, log_file_path)

    raw_net = unwrap_model(net)
    if device is None:
        try:
            device = next(raw_net.parameters()).device
        except StopIteration:
            device = get_default_device(None)
    device = get_default_device(device)
    seed_everything(random_seed + rank)

    scale_range = 0.5 if scale_range is None else scale_range

    oversampling_active = train_probs is not None

    raw_net.diam_labels.data = torch.tensor(
        [train_dataset.diameter_array.mean()], device=device
    )

    if class_weights is not None and isinstance(
        class_weights, (list, np.ndarray, tuple)
    ):
        class_weights = np.float32(class_weights)
        class_weights = torch.from_numpy(class_weights).to(device).float()

    nimg = len(train_dataset)
    nimg_test = len(train_dataset)
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    if nimg_test is not None and nimg_test_per_epoch not in (None, nimg_test):
        train_logger.warning(
            "nimg_test_per_epoch is ignored in the DataLoader validation path; "
            "the full validation set is evaluated once per validation epoch."
        )
    nimg_test_per_epoch = nimg_test

    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for _ in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 99:
        LR = LR[:-50]
        for _ in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))

    train_logger.info(
        f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}, "
        f"distributed={distributed}, rank={rank}, world_size={world_size}"
    )
    train_logger.info(
        f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}"
    )
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    seg_trainable = (
        any(p.requires_grad for p in raw_net.out.parameters())
        or raw_net.W2.requires_grad
    )

    n_active_losses = 2
    if seg_trainable:
        n_active_losses += 1

    loss_aggregator = LossAggregator(
        n_losses=n_active_losses, optimise=use_uncertainty_weighting
    ).to(device)

    if use_uncertainty_weighting:
        optimizer.add_param_group(
            {
                "params": loss_aggregator.parameters(),
                "lr_scale": 0.1,
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
    model_dir = save_path / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = model_dir / model_name
    checkpoint_last = model_dir / "checkpoint_last.train.pt"
    checkpoint_best = model_dir / "checkpoint_best.train.pt"

    train_logger.info(f">>> saving model to {filename}")

    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    best_val_loss = np.inf

    if num_workers > 0:
        train_logger.info(
            f"Using multiprocessing for training with {num_workers} workers"
        )
    else:
        train_logger.info("Using single-threaded processing for training")

    train_sampler = DistributedEpochSampler(
        dataset_length=nimg,
        train_probs=train_probs if oversampling_active else None,
        nimg_per_epoch=nimg_per_epoch,
        batch_size=batch_size,
        rank=rank,
        num_replicas=world_size,
        seed=random_seed,
    )

    pin_memory = device.type == "cuda"
    train_loader = _build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=random_seed + rank,
    )

    val_loader = None
    if test_dataset is not None:
        val_sampler = (
            SequentialDistributedSampler(
                dataset_length=nimg_test,
                rank=rank,
                num_replicas=world_size,
            )
            if distributed
            else None
        )
        val_loader = _build_dataloader(
            dataset=test_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=random_seed + rank + 10_000,
        )

    start_epoch = 0
    if resume_checkpoint is not None:
        if not resume_checkpoint.endswith(".train.pt"):
            raise ValueError(
                "Resume checkpoints must be training-state artifacts ending in '.train.pt'. "
            )
        checkpoint = _load_training_checkpoint(
            checkpoint_path=resume_checkpoint,
            net=net,
            optimizer=optimizer,
            loss_aggregator=loss_aggregator,
            device=device,
        )
        best_val_loss = float(checkpoint.get("best_val_loss", np.inf))
        saved_train_losses = checkpoint.get("train_losses")
        if saved_train_losses is not None:
            n_saved = min(len(train_losses), len(saved_train_losses))
            train_losses[:n_saved] = saved_train_losses[:n_saved]
        saved_test_losses = checkpoint.get("test_losses")
        if saved_test_losses is not None:
            n_saved = min(len(test_losses), len(saved_test_losses))
            test_losses[:n_saved] = saved_test_losses[:n_saved]
        start_epoch = int(checkpoint["epoch"]) + 1
        train_logger.info(
            f">>> Resuming training from {resume_checkpoint} at epoch {start_epoch}"
        )

    if start_epoch >= n_epochs:
        raise ValueError(
            f"Resume checkpoint already completed epoch {start_epoch - 1}; "
            f"requested n_epochs={n_epochs} leaves no training steps to run."
        )

    last_completed_epoch = start_epoch - 1

    for iepoch in range(start_epoch, n_epochs):
        train_sampler.set_epoch(iepoch)
        _set_optimizer_lrs(optimizer, LR[iepoch])
        net.train()

        train_seg_sum = 0.0
        train_ce_sum = 0.0
        train_tversky_sum = 0.0
        train_total_sum = 0.0
        train_sample_count = 0

        for X, lbl in train_loader:
            X = X.to(device, non_blocking=pin_memory).float()
            lbl = lbl.to(device, non_blocking=pin_memory)
            y = net(X)[0]

            if seg_trainable:
                loss_seg = _loss_fn_seg(lbl, y, device)
                batch_seg_loss = loss_seg.item()
            else:
                loss_seg = None
                batch_seg_loss = 0.0

            loss_ce = _loss_fn_class(lbl, y, class_weights=class_weights)
            loss_tversky = _loss_fn_tversky(
                lbl,
                y,
                class_weights=class_weights,
                n_classes=raw_net.n_cell_classes,
            )

            active_losses = []
            if loss_seg is not None:
                active_losses.append(loss_seg)
            active_losses.append(loss_ce)
            active_losses.append(loss_tversky)
            loss = loss_aggregator(*active_losses)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if distributed and use_uncertainty_weighting:
                sync_module_grads(loss_aggregator)
            optimizer.step()

            batch_size_local = int(X.shape[0])
            train_sample_count += batch_size_local
            train_seg_sum += batch_seg_loss * batch_size_local
            train_ce_sum += loss_ce.item() * batch_size_local
            train_tversky_sum += loss_tversky.item() * batch_size_local
            train_total_sum += loss.item() * batch_size_local

        train_totals = torch.tensor(
            [
                train_seg_sum,
                train_ce_sum,
                train_tversky_sum,
                train_total_sum,
                train_sample_count,
            ],
            device=device,
            dtype=torch.float64,
        )
        train_totals = all_reduce_sum(train_totals)
        total_train_samples = int(train_totals[4].item())
        avg_train_seg_loss = (
            train_totals[0].item() / total_train_samples
            if total_train_samples > 0
            else 0.0
        )
        avg_train_ce_loss = (
            train_totals[1].item() / total_train_samples
            if total_train_samples > 0
            else 0.0
        )
        avg_train_tversky_loss = (
            train_totals[2].item() / total_train_samples
            if total_train_samples > 0
            else 0.0
        )
        avg_train_total_loss = (
            train_totals[3].item() / total_train_samples
            if total_train_samples > 0
            else 0.0
        )
        train_losses[iepoch] = avg_train_total_loss

        if is_main_process():
            if use_uncertainty_weighting:
                current_weights = loss_aggregator.get_uncertainty_factors(
                    seg_trainable=seg_trainable
                )
                train_logger.info(
                    f"Epoch {iepoch}, Segmentation Loss: {avg_train_seg_loss:.4f}, "
                    f"Classification CE Loss: {avg_train_ce_loss:.4f}, "
                    f"Tversky Loss: {avg_train_tversky_loss:.4f}, "
                    f"Total Loss: {avg_train_total_loss:.4f}, "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}, time {time.time()-t0:.2f}s"
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
                        f"Epoch {iepoch} Uncertainty Weights - {', '.join(weight_parts)}"
                    )
            else:
                train_logger.info(
                    f"Epoch {iepoch}, Segmentation Loss: {avg_train_seg_loss:.4f}, "
                    f"Classification CE Loss: {avg_train_ce_loss:.4f}, "
                    f"Tversky Loss: {avg_train_tversky_loss:.4f}, "
                    f"Total Loss: {avg_train_total_loss:.4f}, "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}, time {time.time()-t0:.2f}s"
                )

        if (
            _should_validate(iepoch, validate_every_epoch)
            and val_loader is not None
        ):
            net.eval()
            val_seg_sum = 0.0
            val_ce_sum = 0.0
            val_tversky_sum = 0.0
            val_total_sum = 0.0
            val_sample_count = 0

            with torch.no_grad():
                for X, lbl in val_loader:
                    X = X.to(device, non_blocking=pin_memory).float()
                    lbl = lbl.to(device, non_blocking=pin_memory)
                    y = net(X)[0]

                    if seg_trainable:
                        loss_seg_val = _loss_fn_seg(lbl, y, device)
                        batch_val_seg_loss = loss_seg_val.item()
                    else:
                        loss_seg_val = None
                        batch_val_seg_loss = 0.0

                    loss_ce_val = _loss_fn_class(
                        lbl, y, class_weights=class_weights
                    )
                    loss_tversky_val = _loss_fn_tversky(
                        lbl,
                        y,
                        class_weights=class_weights,
                        n_classes=raw_net.n_cell_classes,
                    )

                    active_losses = []
                    if loss_seg_val is not None:
                        active_losses.append(loss_seg_val)
                    active_losses.append(loss_ce_val)
                    active_losses.append(loss_tversky_val)
                    loss = loss_aggregator(*active_losses)

                    batch_size_local = int(X.shape[0])
                    val_sample_count += batch_size_local
                    val_seg_sum += batch_val_seg_loss * batch_size_local
                    val_ce_sum += loss_ce_val.item() * batch_size_local
                    val_tversky_sum += (
                        loss_tversky_val.item() * batch_size_local
                    )
                    val_total_sum += loss.item() * batch_size_local

            val_totals = torch.tensor(
                [
                    val_seg_sum,
                    val_ce_sum,
                    val_tversky_sum,
                    val_total_sum,
                    val_sample_count,
                ],
                device=device,
                dtype=torch.float64,
            )
            val_totals = all_reduce_sum(val_totals)
            total_val_samples = int(val_totals[4].item())
            avg_val_seg_loss = (
                val_totals[0].item() / total_val_samples
                if total_val_samples > 0
                else 0.0
            )
            avg_val_ce_loss = (
                val_totals[1].item() / total_val_samples
                if total_val_samples > 0
                else 0.0
            )
            avg_val_tversky_loss = (
                val_totals[2].item() / total_val_samples
                if total_val_samples > 0
                else 0.0
            )
            avg_val_total_loss = (
                val_totals[3].item() / total_val_samples
                if total_val_samples > 0
                else 0.0
            )
            test_losses[iepoch] = avg_val_total_loss

            if is_main_process():
                if use_uncertainty_weighting:
                    current_weights = loss_aggregator.get_uncertainty_factors(
                        seg_trainable=seg_trainable
                    )
                    train_logger.info(
                        f"Epoch {iepoch} Validation, Segmentation Loss: {avg_val_seg_loss:.4f}, "
                        f"Classification CE Loss: {avg_val_ce_loss:.4f}, "
                        f"Tversky Loss: {avg_val_tversky_loss:.4f}, "
                        f"Total Loss: {avg_val_total_loss:.4f}, "
                        f"LR={optimizer.param_groups[0]['lr']:.6f}, time {time.time()-t0:.2f}s"
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
                        f"Epoch {iepoch} Validation, Segmentation Loss: {avg_val_seg_loss:.4f}, "
                        f"Classification CE Loss: {avg_val_ce_loss:.4f}, "
                        f"Tversky Loss: {avg_val_tversky_loss:.4f}, "
                        f"Total Loss: {avg_val_total_loss:.4f}, "
                        f"LR={optimizer.param_groups[0]['lr']:.6f}, time {time.time()-t0:.2f}s"
                    )

            if avg_val_total_loss < best_val_loss:
                best_val_loss = avg_val_total_loss
                filename_best = model_dir / f"{model_name}_best"
                if is_main_process():
                    train_logger.info(
                        f"New best validation loss: {best_val_loss:.4f}. Saving model to {filename_best}"
                    )
                    raw_net.save_model(filename_best)
                _save_training_checkpoint(
                    checkpoint_path=checkpoint_best,
                    net=net,
                    optimizer=optimizer,
                    loss_aggregator=loss_aggregator,
                    epoch=iepoch,
                    best_val_loss=best_val_loss,
                    train_losses=train_losses,
                    test_losses=test_losses,
                    config_snapshot=config_snapshot,
                    distributed=distributed,
                )

        if iepoch != n_epochs - 1 and iepoch % save_every == 0 and iepoch != 0:
            if save_each:
                filename0 = Path(str(filename) + f"_epoch_{iepoch:04d}")
                checkpoint_epoch = (
                    model_dir / f"checkpoint_epoch_{iepoch:04d}.train.pt"
                )
            else:
                filename0 = filename
                checkpoint_epoch = checkpoint_last

            if is_main_process():
                train_logger.info(f"saving network parameters to {filename0}")
                raw_net.save_model(filename0)

            _save_training_checkpoint(
                checkpoint_path=checkpoint_last,
                net=net,
                optimizer=optimizer,
                loss_aggregator=loss_aggregator,
                epoch=iepoch,
                best_val_loss=best_val_loss,
                train_losses=train_losses,
                test_losses=test_losses,
                config_snapshot=config_snapshot,
                distributed=distributed,
            )
            if save_each:
                _save_training_checkpoint(
                    checkpoint_path=checkpoint_epoch,
                    net=net,
                    optimizer=optimizer,
                    loss_aggregator=loss_aggregator,
                    epoch=iepoch,
                    best_val_loss=best_val_loss,
                    train_losses=train_losses,
                    test_losses=test_losses,
                    config_snapshot=config_snapshot,
                    distributed=distributed,
                )

        last_completed_epoch = iepoch

    if is_main_process():
        train_logger.info(f"Saving the final model to {filename}")
        raw_net.save_model(filename)
    _save_training_checkpoint(
        checkpoint_path=checkpoint_last,
        net=net,
        optimizer=optimizer,
        loss_aggregator=loss_aggregator,
        epoch=last_completed_epoch,
        best_val_loss=best_val_loss,
        train_losses=train_losses,
        test_losses=test_losses,
        config_snapshot=config_snapshot,
        distributed=distributed,
    )

    return filename, train_losses, test_losses
