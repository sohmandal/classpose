from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

DEFAULT_DDP_TIMEOUT_SECONDS = 1800


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def _resolve_single_process_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this system")
        return torch.device("mps")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")
        torch.cuda.empty_cache()
        return torch.device("cuda")

    if device_arg.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")
        gpu_id = int(device_arg.split(":", maxsplit=1)[1])
        if gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU {gpu_id} is not available. Available GPUs: 0-{torch.cuda.device_count()-1}"
            )
        torch.cuda.empty_cache()
        return torch.device("cuda", gpu_id)

    if device_arg.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")
        gpu_id = int(device_arg)
        if gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU {gpu_id} is not available. Available GPUs: 0-{torch.cuda.device_count()-1}"
            )
        torch.cuda.empty_cache()
        return torch.device("cuda", gpu_id)

    raise ValueError(
        f"Invalid device '{device_arg}'. Use 'auto', 'cpu', 'mps', 'cuda', 'cuda:N', or a GPU ID like '0'."
    )


def setup_distributed(
    device_arg: str,
    backend: str | None = None,
    timeout_seconds: int | None = None,
) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size <= 1:
        return DistributedContext(
            distributed=False,
            device=_resolve_single_process_device(device_arg),
        )

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend_name = backend or ("nccl" if torch.cuda.is_available() else "gloo")

    if backend_name == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for NCCL distributed training")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif device_arg == "cpu":
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Distributed training requires CUDA in the runtime training path"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    if not is_distributed():
        kwargs: dict[str, Any] = {"backend": backend_name}
        effective_timeout_seconds = (
            DEFAULT_DDP_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        kwargs["timeout"] = timedelta(seconds=effective_timeout_seconds)
        dist.init_process_group(**kwargs)

    return DistributedContext(
        distributed=True,
        device=device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend_name,
    )


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    raw_model = model
    while True:
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module
            continue
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
            continue
        return raw_model


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    reduced = tensor.clone()
    if is_distributed():
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    reduced = all_reduce_sum(tensor)
    if is_distributed():
        reduced /= get_world_size()
    return reduced


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not is_distributed():
        return obj
    obj_list = [obj if get_rank() == src else None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def sync_module_grads(module: nn.Module) -> None:
    if not is_distributed():
        return

    world_size = get_world_size()
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad /= world_size


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
