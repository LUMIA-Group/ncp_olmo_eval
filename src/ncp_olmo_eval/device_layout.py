"""CUDA device mapping for independent evaluation workers."""

from __future__ import annotations

import os


def _visible_cuda_device_count() -> int:
    import torch

    return int(torch.cuda.device_count())


def local_cuda_device_index() -> int:
    """Map a local worker rank onto one of the visible CUDA devices."""

    device_count = _visible_cuda_device_count()
    if device_count <= 0:
        raise RuntimeError("Core evaluation requires at least one visible CUDA device")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    processes_per_gpu = int(os.environ.get("CONCEPTLM_PROCESSES_PER_GPU", "1"))
    if processes_per_gpu <= 0:
        raise ValueError(
            "CONCEPTLM_PROCESSES_PER_GPU must be positive, got "
            f"{processes_per_gpu}"
        )
    local_world_size = int(
        os.environ.get(
            "LOCAL_WORLD_SIZE",
            str(device_count * processes_per_gpu),
        )
    )
    expected_local_world_size = device_count * processes_per_gpu
    if local_world_size != expected_local_world_size:
        raise RuntimeError(
            "local process layout does not match visible GPUs: "
            f"LOCAL_WORLD_SIZE={local_world_size}, visible_gpus={device_count}, "
            f"processes_per_gpu={processes_per_gpu}, "
            f"expected={expected_local_world_size}"
        )
    if not 0 <= local_rank < local_world_size:
        raise ValueError(
            f"LOCAL_RANK={local_rank} is outside LOCAL_WORLD_SIZE={local_world_size}"
        )
    return local_rank % device_count
