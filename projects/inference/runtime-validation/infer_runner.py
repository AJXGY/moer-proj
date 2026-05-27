#!/usr/bin/env python3
import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import platform
import re
import traceback
from datetime import datetime, timezone


def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    return prompts


def detect_backend(requested):
    if requested != "auto":
        return requested
    if importlib.util.find_spec("torch_musa") is not None:
        return "musa"
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def load_model_bundle(model_path, backend, device_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if backend == "musa":
        import torch_musa  # noqa: F401

    device = f"{backend}:{device_id}" if backend != "cpu" else "cpu"
    model_dtype = torch.float16 if backend == "musa" else "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return tokenizer, model, device


def build_inputs(tokenizer, prompt, device):
    messages = [
        {"role": "system", "content": "你是严格答题助手。只输出最终答案本身，不要重复题目，不要解释，不要添加标点或代码。"},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        attention_mask = (
            (encoded != tokenizer.pad_token_id).long()
            if tokenizer.pad_token_id is not None else encoded.new_ones(encoded.shape)
        )
        return {"input_ids": encoded.to(device), "attention_mask": attention_mask.to(device)}
    encoded = tokenizer(f"{prompt}\n答案：", return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def real_infer(tokenizer, model, device, prompt, max_new_tokens):
    import torch
    import time

    inputs = build_inputs(tokenizer, prompt, device)
    input_len = inputs["input_ids"].shape[-1]
    eos_token_ids = []
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    except Exception:
        eot_id = None
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in eos_token_ids:
        eos_token_ids.append(eot_id)

    # Device sync before timing to flush prior async ops
    try:
        if hasattr(torch, "musa") and getattr(torch, "musa") is not None:
            try:
                torch.musa.synchronize()
            except Exception:
                pass
        elif torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    except Exception:
        pass

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_ids or tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Device sync after generation to ensure accurate timing
    try:
        if hasattr(torch, "musa") and getattr(torch, "musa") is not None:
            try:
                torch.musa.synchronize()
            except Exception:
                pass
        elif torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    except Exception:
        pass

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    generated_ids = output_ids[0][input_len:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return decoded, elapsed_ms


def dry_infer(item, mode_name, device_label):
    expected = item.get("expected_contains", [])
    if expected:
        return expected[0]
    return f"[dry-run][{mode_name}][{device_label}]"


def normalize_response(item, text):
    stripped = text.strip()
    if not stripped:
        return ""
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    candidates = []
    if lines:
        candidates.append(lines[0])
    candidates.append(stripped)

    expected = item.get("expected_contains", [])
    if not expected:
        return candidates[0]

    for token in expected:
        normalized_token = token.strip()
        for candidate in candidates:
            if candidate == normalized_token:
                return normalized_token
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(normalized_token)}(?![A-Za-z0-9_])"
        match = re.search(pattern, stripped)
        if match:
            return match.group(0)
    return candidates[0]


def validate_response(item, normalized_text):
    expected = item.get("expected_contains", [])
    if not expected:
        return True
    return normalized_text in expected


def worker_main(worker_id, prompts, args, queue):
    backend = detect_backend(args.device_type)
    device_ids = args.device_ids or list(range(args.num_devices))
    device_id = device_ids[worker_id]
    device_label = f"{backend}:{device_id}" if backend != "cpu" else "cpu"

    results = []
    errors = []
    started_at = datetime.now(timezone.utc).isoformat()
    model_load_ms = None
    try:
        use_real = (
            not args.dry_run
            and importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None
        )
        tokenizer = model = device = None
        if use_real:
            import time as _time
            load_start = _time.perf_counter()
            tokenizer, model, device = load_model_bundle(args.model_path, backend, device_id)
            load_end = _time.perf_counter()
            model_load_ms = (load_end - load_start) * 1000.0
        for item in prompts:
            if use_real:
                raw_text, gen_ms = real_infer(
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    prompt=item["prompt"],
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                raw_text = dry_infer(item, args.mode_name, device_label)
            normalized_text = normalize_response(item, raw_text)
            validated = validate_response(item, normalized_text)
            results.append(
                {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "response": normalized_text,
                    "raw_response": raw_text,
                    "expected_contains": item.get("expected_contains", []),
                    "validation_passed": validated,
                    "worker_id": worker_id,
                    "device": device_label,
                    "success": True,
                    "gen_ms": gen_ms if use_real else None,
                }
            )
    except Exception as exc:
        errors.append(
            {
                "worker_id": worker_id,
                "device": device_label,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    payload = {
        "worker_id": worker_id,
        "device": device_label,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "errors": errors,
    }
    if use_real:
        payload["model_load_ms"] = model_load_ms
    queue.put(payload)


def chunk_prompts(prompts, parts):
    chunks = [[] for _ in range(parts)]
    for idx, item in enumerate(prompts):
        chunks[idx % parts].append(item)
    return chunks


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode-name", required=True, choices=["single", "dual"])
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--device-type", default="auto", choices=["auto", "musa", "cuda", "cpu"])
    parser.add_argument("--device-ids", default="")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = load_prompts(args.prompts_file)
    parsed_device_ids = [int(item) for item in args.device_ids.split(",") if item.strip()]
    args.device_ids = parsed_device_ids
    if not args.device_ids:
        args.device_ids = list(range(args.num_devices))

    prompt_chunks = chunk_prompts(prompts, args.num_devices)
    queue = mp.get_context("spawn").Queue()
    procs = []
    for worker_id in range(args.num_devices):
        proc = mp.get_context("spawn").Process(
            target=worker_main,
            args=(worker_id, prompt_chunks[worker_id], args, queue),
        )
        proc.start()
        procs.append(proc)

    worker_payloads = []
    for _ in procs:
        worker_payloads.append(queue.get())

    for proc in procs:
        proc.join()

    outputs = []
    errors = []
    for payload in sorted(worker_payloads, key=lambda item: item["worker_id"]):
        outputs.extend(payload["results"])
        errors.extend(payload["errors"])

    validation_passed = all(item.get("validation_passed", False) for item in outputs) if outputs else False
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode_name": args.mode_name,
        "hostname": platform.node(),
        "model_path": os.path.abspath(args.model_path),
        "prompts_file": os.path.abspath(args.prompts_file),
        "num_prompts": len(prompts),
        "num_devices": args.num_devices,
        "device_ids": args.device_ids,
        "dry_run": args.dry_run,
        "success": not errors and len(outputs) == len(prompts),
        "validation_passed": validation_passed,
        "validated_outputs_count": sum(1 for item in outputs if item.get("validation_passed")),
        "outputs_count": len(outputs),
        "errors": errors,
        "worker_payloads": worker_payloads,
    }

    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    with open(os.path.join(args.output_dir, "outputs.jsonl"), "w", encoding="utf-8") as handle:
        for item in outputs:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output_dir, "run.log"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2))
        handle.write("\n")

    print(os.path.join(args.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
