#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def run_command(command):
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }


def package_available(name):
    return importlib.util.find_spec(name) is not None


def detect_accelerator():
    result = {
        "backend": "cpu",
        "device_count": 0,
        "devices": [],
        "torch_available": package_available("torch"),
        "torch_musa_available": package_available("torch_musa"),
    }
    if not result["torch_available"]:
        return result

    # Attempt to import torch but handle import errors gracefully (e.g., missing shared libs)
    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive
        result["torch_import_error"] = str(exc)
        # leave backend as cpu and return available info
        return result

    if result["torch_musa_available"]:
        try:
            import torch_musa  # noqa: F401

            count = torch.musa.device_count()
            result["backend"] = "musa"
            result["device_count"] = count
            result["devices"] = [f"musa:{idx}" for idx in range(count)]
            return result
        except Exception as exc:
            result["musa_error"] = str(exc)

    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            result["backend"] = "cuda"
            result["device_count"] = count
            result["devices"] = [f"cuda:{idx}" for idx in range(count)]
            return result
    except Exception as exc:  # pragma: no cover - defensive
        result["cuda_error"] = str(exc)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path to preflight.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    checks = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "packages": {
            "torch": package_available("torch"),
            "torch_musa": package_available("torch_musa"),
            "transformers": package_available("transformers"),
            "accelerate": package_available("accelerate"),
        },
        "tools": {
            "mthreads-gmi": shutil.which("mthreads-gmi"),
            "nvidia-smi": shutil.which("nvidia-smi"),
            "docker": shutil.which("docker"),
        },
        "commands": {
            "uname": run_command(["uname", "-a"]),
            "lscpu": run_command(["lscpu"]),
            "ip": run_command(["ip", "-brief", "addr"]) if shutil.which("ip") else None,
            "mthreads-gmi": run_command(["mthreads-gmi"]) if shutil.which("mthreads-gmi") else None,
            "nvidia-smi": run_command(["nvidia-smi"]) if shutil.which("nvidia-smi") else None,
        },
        "accelerator": detect_accelerator(),
    }

    checks["criteria"] = {
        "single_card_visible": checks["accelerator"]["device_count"] >= 1,
        "dual_card_visible": checks["accelerator"]["device_count"] >= 2,
        "python_dependencies_ready": all(
            checks["packages"][name] for name in ("torch", "transformers")
        ),
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(checks, handle, ensure_ascii=False, indent=2)

    print(args.output)


if __name__ == "__main__":
    main()
