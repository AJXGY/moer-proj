from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

_TORCH = None
_DIST = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch as torch_module

        _TORCH = torch_module
    return _TORCH


def _dist():
    global _DIST
    if _DIST is None:
        import torch.distributed as dist_module

        _DIST = dist_module
    return _DIST


def _has_musa_runtime() -> bool:
    return hasattr(_torch(), "musa")


def _cuda_available() -> bool:
    try:
        return bool(_torch().cuda.is_available())
    except Exception:
        return False


def _musa_available() -> bool:
    if not _has_musa_runtime():
        return False
    try:
        return bool(_torch().musa.is_available())
    except Exception:
        return False


def preferred_device_backend() -> str:
    override = str(os.environ.get("MVP_DEVICE_BACKEND", "")).strip().lower()
    if override in {"cuda", "musa"}:
        return override
    if shutil.which("mthreads-gmi"):
        return "musa"
    if shutil.which("nvidia-smi"):
        return "cuda"
    if _musa_available():
        return "musa"
    if _cuda_available():
        return "cuda"
    return "cuda"


def is_backend_available(backend: str | None = None) -> bool:
    resolved = str(backend or preferred_device_backend()).lower()
    if resolved == "musa":
        return _musa_available()
    return _cuda_available()


def default_device_string() -> str:
    return f"{preferred_device_backend()}:0"


def visible_devices_env_var(backend: str | None = None) -> str:
    resolved = (backend or preferred_device_backend()).lower()
    return "MUSA_VISIBLE_DEVICES" if resolved == "musa" else "CUDA_VISIBLE_DEVICES"


def device_module(backend: str | Any | None = None):
    resolved = backend
    if hasattr(backend, "type"):
        resolved = backend.type
    torch = _torch()
    resolved = str(resolved or preferred_device_backend()).lower()
    if resolved == "musa" and _has_musa_runtime():
        return torch.musa
    return torch.cuda


def distributed_backend_for_device(backend: str | Any | None = None) -> str:
    resolved = backend
    if hasattr(backend, "type"):
        resolved = backend.type
    resolved = str(resolved or preferred_device_backend()).lower()
    if resolved == "musa":
        # On this S3000 environment, torch.distributed may report MCCL capability
        # but c10d still fails with "Unknown c10d backend type MCCL". Default to
        # gloo unless the user explicitly opts into MCCL for a known-good runtime.
        forced = str(os.environ.get("MVP_MUSA_DIST_BACKEND", "")).strip().lower()
        if forced in {"gloo", "mccl"}:
            return forced
        return "gloo"
    return "nccl"


def set_device(index: int, backend: str | Any | None = None) -> None:
    module = device_module(backend)
    if hasattr(module, "set_device"):
        module.set_device(index)


def synchronize(device: str | Any | None = None) -> None:
    module = device_module(device)
    if hasattr(module, "synchronize"):
        module.synchronize()


def empty_cache(device: str | Any | None = None) -> None:
    module = device_module(device)
    if hasattr(module, "empty_cache"):
        module.empty_cache()


def make_timing_event(device: str | Any | None = None):
    module = device_module(device)
    event_ctor = getattr(module, "Event", None)
    if event_ctor is None:
        return None
    return event_ctor(enable_timing=True)


def get_device_properties(device: str | Any) -> Any:
    module = device_module(device)
    return module.get_device_properties(device)


def profiler_activities_for_device(device: str | Any | None = None) -> list[Any]:
    from torch.profiler import ProfilerActivity

    resolved = device
    if hasattr(device, "type"):
        resolved = device.type
    resolved = str(resolved or preferred_device_backend()).lower()
    activities = [ProfilerActivity.CPU]
    if resolved == "cuda" and hasattr(ProfilerActivity, "CUDA"):
        activities.append(ProfilerActivity.CUDA)
    return activities


def system_gpu_inventory() -> list[dict[str, Any]]:
    probe_env = os.environ.copy()
    probe_env.pop("LD_LIBRARY_PATH", None)
    if shutil.which("mthreads-gmi"):
        try:
            completed = subprocess.run(
                ["mthreads-gmi", "-q"],
                capture_output=True,
                text=True,
                check=False,
                env=probe_env,
            )
        except OSError:
            completed = None
        if completed and completed.returncode == 0:
            rows: list[dict[str, Any]] = []
            current: dict[str, Any] | None = None
            for line in completed.stdout.splitlines():
                gpu_match = re.match(r"GPU(\d+)\s", line.strip())
                if gpu_match:
                    if current is not None:
                        rows.append(current)
                    current = {
                        "index": int(gpu_match.group(1)),
                        "name": f"MTT GPU {gpu_match.group(1)}",
                        "backend": "musa",
                    }
                    continue
                if current is None:
                    continue
                if "Product Name" in line:
                    _, _, value = line.partition(":")
                    name = value.strip()
                    if name:
                        current["name"] = name
            if current is not None:
                rows.append(current)
            if rows:
                return rows
    if shutil.which("nvidia-smi"):
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
                env=probe_env,
            )
        except OSError:
            completed = None
        if completed and completed.returncode == 0:
            rows = []
            for line in completed.stdout.splitlines():
                text = line.strip()
                if not text:
                    continue
                index_text, _, name = text.partition(",")
                try:
                    index = int(index_text.strip())
                except ValueError:
                    continue
                rows.append(
                    {
                        "index": index,
                        "name": name.strip() or f"GPU {index}",
                        "backend": "cuda",
                    }
                )
            if rows:
                return rows
    if shutil.which("mthreads-gmi"):
        return [
            {"index": 0, "name": "MTT S3000", "backend": "musa"},
            {"index": 1, "name": "MTT S3000", "backend": "musa"},
        ]
    return []


def project_python_candidates() -> list[str]:
    candidates = [
        Path(__file__).resolve().parent / "tools" / "python_with_env.sh",
        Path.home() / "miniconda3" / "envs" / "llama_4gpu" / "bin" / "python",
        Path.home() / "miniconda3" / "envs" / "llama_4gpu" / "bin" / "python3",
        Path.home() / "anaconda3" / "envs" / "llama_4gpu" / "bin" / "python",
        Path.home() / "anaconda3" / "envs" / "llama_4gpu" / "bin" / "python3",
    ]
    return [str(path) for path in candidates if path.exists()]
