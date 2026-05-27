#!/usr/bin/env python3
"""
5.1.6 前置检查模块 - 训练环境预检查
MTT-TRAIN-RUN-TEST Preflight Check

检查项：
- CPU识别
- 驱动程序
- Python依赖
- GPU/NPU可见性（单卡/双卡）
- 模型文件完整性
"""

import argparse
import importlib.util
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone


def run_command(command):
    """执行shell命令并返回结果"""
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            shell=True
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }


def package_available(name):
    """检查Python包是否可用"""
    return importlib.util.find_spec(name) is not None


def detect_accelerator():
    """检测加速器（GPU/NPU）"""
    result = {
        "backend": "cpu",
        "device_count": 0,
        "devices": [],
        "torch_available": package_available("torch"),
        "torch_musa_available": package_available("torch_musa"),
    }
    
    if not result["torch_available"]:
        return result
    
    try:
        import torch
        
        # 检查CUDA
        if torch.cuda.is_available():
            result["backend"] = "cuda"
            result["device_count"] = torch.cuda.device_count()
            for i in range(result["device_count"]):
                try:
                    device_name = torch.cuda.get_device_name(i)
                    device_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    result["devices"].append({
                        "id": i,
                        "name": device_name,
                        "memory_gb": round(device_mem, 2)
                    })
                except Exception:
                    result["devices"].append({"id": i, "name": "Unknown", "memory_gb": 0})
        
        # 检查MUSA
        if result["torch_musa_available"]:
            import torch_musa
            try:
                musa_count = 0
                musa_devices = []

                if hasattr(torch, "musa") and torch.musa.is_available():
                    musa_count = int(torch.musa.device_count())
                    for i in range(musa_count):
                        name = "MUSA Device"
                        try:
                            name = torch.musa.get_device_name(i)
                        except Exception:
                            pass
                        musa_devices.append({"id": i, "name": name, "memory_gb": 0})

                # 某些环境下 torch.musa 查询可能失败，回退到 mthreads-gmi 解析。
                if musa_count == 0:
                    probe = run_command("mthreads-gmi")
                    if probe.get("returncode") == 0 and probe.get("stdout"):
                        ids = []
                        for line in probe["stdout"].splitlines():
                            m = re.match(r"^\s*(\d+)\s+MTT", line)
                            if m:
                                ids.append(int(m.group(1)))
                        if ids:
                            musa_count = len(ids)
                            musa_devices = [
                                {"id": i, "name": f"MTT Device {i}", "memory_gb": 0}
                                for i in ids
                            ]

                if musa_count > 0:
                    result["backend"] = "musa"
                    result["device_count"] = musa_count
                    result["devices"] = musa_devices
            except Exception:
                pass
    except Exception as e:
        pass
    
    return result


def get_system_info():
    """获取系统信息"""
    return {
        "os": platform.system(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def check_dependencies():
    """检查Python依赖"""
    required_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
    ]
    
    available = []
    missing = []
    
    for module_name, display_name in required_packages:
        if package_available(module_name):
            available.append(display_name)
        else:
            missing.append(display_name)
    
    return {
        "available": available,
        "missing": missing,
        "all_ready": len(missing) == 0,
    }


def check_model(model_path):
    """检查模型文件完整性"""
    result = {
        "exists": False,
        "path": model_path,
        "files": [],
        "size_mb": 0,
        "complete": False,
    }
    
    if not os.path.exists(model_path):
        return result
    
    result["exists"] = True
    
    # 检查关键文件
    required_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
    ]
    
    optional_files = [
        "model.safetensors",
        "model.pt",
        "pytorch_model.bin",
        "model-*.safetensors",
        "consolidated.00.pth",
    ]
    
    found_model_files = False
    
    for file_pattern in optional_files:
        if "*" in file_pattern:
            import glob
            if glob.glob(os.path.join(model_path, file_pattern)):
                found_model_files = True
                break
        elif os.path.exists(os.path.join(model_path, file_pattern)):
            found_model_files = True
            result["files"].append(file_pattern)
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            result["files"].append(file)
    
    # 计算模型大小
    total_size = 0
    for root, dirs, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
            except Exception:
                pass
    
    result["size_mb"] = round(total_size / (1024 * 1024), 2)
    result["complete"] = found_model_files and len(result["files"]) > 0
    
    return result


def check_single_card_visible(device_count):
    """检查单卡GPU是否可见"""
    return {
        "visible": device_count >= 1,
        "required_gpus": 1,
        "available_gpus": device_count,
        "status": "ready" if device_count >= 1 else "not_ready",
    }


def check_dual_card_visible(device_count):
    """检查双卡GPU是否可见"""
    return {
        "visible": device_count >= 2,
        "required_gpus": 2,
        "available_gpus": device_count,
        "status": "ready" if device_count >= 2 else "not_ready",
    }


def main():
    parser = argparse.ArgumentParser(description='5.1.6 Preflight Check')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    default_model_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../..",
            "clj-proj",
            "model",
            "Meta-Llama-3.1-8B",
        )
    )
    parser.add_argument('--model_path', type=str, default=default_model_path,
                       help='Model path')
    args = parser.parse_args()
    
    # 收集所有检查结果
    preflight_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_id": "MTT-TRAIN-RUN-TEST",
        "criteria": {},
        "details": {},
    }
    
    # 系统信息
    preflight_report["details"]["system"] = get_system_info()
    
    # 依赖检查
    deps = check_dependencies()
    preflight_report["details"]["dependencies"] = deps
    preflight_report["criteria"]["python_dependencies_ready"] = deps["all_ready"]
    
    # 加速器检查
    accelerator = detect_accelerator()
    preflight_report["details"]["accelerator"] = accelerator
    
    # 模型检查
    model_info = check_model(args.model_path)
    preflight_report["details"]["model"] = model_info
    preflight_report["criteria"]["model_ready"] = model_info["complete"]
    
    # 单卡可见性
    single_check = check_single_card_visible(accelerator["device_count"])
    preflight_report["details"]["single_card"] = single_check
    preflight_report["criteria"]["single_card_visible"] = single_check["visible"]
    
    # 双卡可见性
    dual_check = check_dual_card_visible(accelerator["device_count"])
    preflight_report["details"]["dual_card"] = dual_check
    preflight_report["criteria"]["dual_card_visible"] = dual_check["visible"]
    
    # 整体准备状态
    all_ready = (
        deps["all_ready"] and
        model_info["complete"] and
        single_check["visible"]
    )
    preflight_report["criteria"]["all_ready"] = all_ready
    
    # 输出结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(preflight_report, f, indent=2, ensure_ascii=False)
    
    print(json.dumps(preflight_report, indent=2, ensure_ascii=False))
    return 0 if all_ready else 1


if __name__ == '__main__':
    exit(main())
