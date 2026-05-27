#!/usr/bin/env python3
import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone


def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    return prompts


def detect_backend(requested):
    if requested != "auto":
        return requested
    try:
        import torch_musa  # noqa: F401
        import torch

        if hasattr(torch, "musa") and torch.musa.is_available():
            return "musa"
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


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

    # sync before
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

    # sync after
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


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--device-type", default="auto")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--output-dir", default="artifacts/hotbench_run")
    parser.add_argument("--dry-run", action="store_true", help="Skip importing torch and return dry responses.")
    parser.add_argument("--simulate", action="store_true", help="Simulate generation latency using mb/op/predictor.py and model config (no torch needed).")
    parser.add_argument("--simulate-peak-tflops", type=float, default=40.0, help="Simulated device peak TFLOPS for predictor (default 40.0)")
    parser.add_argument("--simulate-peak-bw-gbps", type=float, default=800.0, help="Simulated device peak bandwidth in GB/s (default 800)")
    parser.add_argument("--simulate-kernel-overhead-us", type=float, default=20.0, help="Simulated kernel launch overhead in microseconds")
    parser.add_argument("--simulate-num-sms", type=int, default=108, help="Simulated SM count")
    parser.add_argument("--simulate-intra-bw-gbps", type=float, default=250.0, help="Simulated intra-node bandwidth GB/s")
    parser.add_argument("--simulate-seq-len", type=int, default=64, help="Sequence length to use for simulation when tokenizer is unavailable")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = load_prompts(args.prompts_file)

    backend = detect_backend(args.device_type)
    device = f"{backend}:{args.device_id}" if backend != "cpu" else "cpu"

    # Dry-run mode: skip torch entirely and return placeholder responses
    if args.dry_run:
        model_load_ms = 0.0
        gen_times = []
        outputs = []
        for idx in range(max(1, args.num_requests)):
            item = prompts[idx % len(prompts)] if prompts else {"id": f"hot-{idx}", "prompt": "测试"}
            prompt_text = item.get("prompt", "")
            resp = item.get("expected_contains", [None])[0] or f"[dry-run][{device}]"
            gen_ms = 0.0
            gen_times.append(gen_ms)
            outputs.append({"id": item.get("id", f"hot-{idx}"), "prompt": prompt_text, "response": resp, "gen_ms": gen_ms})

        stats = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_path": os.path.abspath(args.model_path),
            "device": device,
            "model_load_ms": model_load_ms,
            "num_requests": args.num_requests,
            "warmup": args.warmup,
            "gen_times_ms": gen_times,
            "avg_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "median_ms": 0.0,
        }

        write_json(os.path.join(args.output_dir, "summary.json"), stats)
        with open(os.path.join(args.output_dir, "outputs.jsonl"), "w", encoding="utf-8") as f:
            for item in outputs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(os.path.join(args.output_dir, "summary.json"))
        return

    # Simulation mode: do not import torch; estimate generation latency using mb/op/predictor.py and model config
    if args.simulate:
        try:
            # load model config.json from model_path without importing torch
            cfg_path = os.path.join(args.model_path, "config.json")
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"config.json not found in model path: {args.model_path}")
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)

            hidden_size = int(cfg.get("hidden_size", 4096))
            num_heads = int(cfg.get("num_attention_heads", 32))
            num_layers = int(cfg.get("num_hidden_layers", 32))
            intermediate_size = int(cfg.get("intermediate_size", hidden_size * 4))
            max_pos = int(cfg.get("max_position_embeddings", 2048))

            # dynamic import of predictor module from repository
            import importlib.util
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            predictor_path = os.path.join(repo_root, "mb", "op", "predictor.py")
            if not os.path.exists(predictor_path):
                raise FileNotFoundError("predictor.py not found for simulation: " + predictor_path)
            spec = importlib.util.spec_from_file_location("mb_op_predictor", predictor_path)
            predictor_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(predictor_mod)

            # build a minimal hardware profile object expected by PredictorEngine
            class SimpleHW:
                pass

            hw = SimpleHW()
            hw.peak_tflops = float(args.simulate_peak_tflops)
            hw.peak_bw_gbps = float(args.simulate_peak_bw_gbps)
            hw.kernel_overhead_us = float(args.simulate_kernel_overhead_us)
            hw.num_sms = int(args.simulate_num_sms)
            hw.device_name = "simulated-musa"
            hw.device_capability = (2, 1)
            hw.supports_flash_attention = False if hw.device_capability < (2, 2) else True
            hw.intra_node_bw_gbps = float(args.simulate_intra_bw_gbps)
            hw.nccl_latency_us = 10.0

            engine = predictor_mod.PredictorEngine(hw)
            # register kernels used in prediction (explicit mapping to actual class names)
            kernel_map = {
                "mm": predictor_mod.GEMMKernel,
                "sdpa": predictor_mod.SDPAKernel,
                "rmsnorm": predictor_mod.RMSNormKernel,
                "softmax": predictor_mod.SoftmaxKernel,
                "ffn": predictor_mod.FFNKernel,
                "add": predictor_mod.ADDKernel,
            }
            for k, cls in kernel_map.items():
                engine.register_kernel(k, cls)

            head_dim = max(1, hidden_size // max(1, num_heads))

            gen_times = []
            outputs = []
            for idx in range(max(1, args.num_requests)):
                item = prompts[idx % len(prompts)] if prompts else {"id": f"hot-{idx}", "prompt": "测试"}
                prompt_text = item.get("prompt", "")
                # crude token estimate: characters/4
                approx_tokens = max(1, int(len(prompt_text) / 4))
                seq_len = min(max(1, approx_tokens + args.max_new_tokens), max_pos)

                # predict layer-wise costs
                sdpa_us = engine.predict_us("sdpa", 1, num_heads, seq_len, head_dim)
                ffn_us = engine.predict_us("ffn", 1, seq_len, intermediate_size)
                rms_us = engine.predict_us("rmsnorm", 1, seq_len, hidden_size)
                add_us = engine.predict_us("add", 1, seq_len, hidden_size)

                per_layer_us = sdpa_us + ffn_us + rms_us + add_us
                total_us = per_layer_us * num_layers + getattr(engine, "gemm_overhead_us", 0.0)
                gen_ms = total_us / 1000.0

                gen_times.append(gen_ms)
                outputs.append({"id": item.get("id", f"hot-{idx}"), "prompt": prompt_text, "response": item.get("expected_contains", [""])[0] or "", "gen_ms": gen_ms})

            stats = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model_path": os.path.abspath(args.model_path),
                "device": "simulated",
                "model_load_ms": 0.0,
                "num_requests": args.num_requests,
                "warmup": args.warmup,
                "gen_times_ms": gen_times,
                "avg_ms": sum(gen_times) / len(gen_times) if gen_times else 0.0,
                "min_ms": min(gen_times) if gen_times else 0.0,
                "max_ms": max(gen_times) if gen_times else 0.0,
            }
            sorted_vals = sorted(gen_times)
            if sorted_vals:
                n = len(sorted_vals)
                if n % 2 == 1:
                    median = sorted_vals[n // 2]
                else:
                    median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
            else:
                median = 0.0
            stats["median_ms"] = median

            write_json(os.path.join(args.output_dir, "summary.json"), stats)
            with open(os.path.join(args.output_dir, "outputs.jsonl"), "w", encoding="utf-8") as f:
                for item in outputs:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(os.path.join(args.output_dir, "summary.json"))
            return
        except Exception:
            traceback.print_exc()
            raise

    # Real mode: import torch and run real inference
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if backend == "musa":
            import torch_musa  # noqa: F401

        model_dtype = torch.float16 if backend == "musa" else None

        load_t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.to(device)
        load_t1 = time.perf_counter()
        model_load_ms = (load_t1 - load_t0) * 1000.0

        # warmup
        for _ in range(max(0, args.warmup)):
            _prompt = prompts[_ % len(prompts)]["prompt"] if prompts else "测试"
            try:
                _ = real_infer(tokenizer, model, device, _prompt, args.max_new_tokens)
            except Exception:
                pass

        gen_times = []
        outputs = []
        for idx in range(max(1, args.num_requests)):
            item = prompts[idx % len(prompts)] if prompts else {"id": f"hot-{idx}", "prompt": "测试"}
            prompt_text = item.get("prompt", "")
            try:
                decoded, gen_ms = real_infer(tokenizer, model, device, prompt_text, args.max_new_tokens)
            except Exception as exc:
                decoded = ""
                gen_ms = None
                print(f"Error during generation idx={idx}: {exc}")
            gen_times.append(gen_ms if gen_ms is not None else 0.0)
            outputs.append({
                "id": item.get("id", f"hot-{idx}"),
                "prompt": prompt_text,
                "response": decoded,
                "gen_ms": gen_ms,
            })

        stats = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_path": os.path.abspath(args.model_path),
            "device": device,
            "model_load_ms": model_load_ms,
            "num_requests": args.num_requests,
            "warmup": args.warmup,
            "gen_times_ms": gen_times,
            "avg_ms": sum(gen_times) / len(gen_times) if gen_times else 0.0,
            "min_ms": min(gen_times) if gen_times else 0.0,
            "max_ms": max(gen_times) if gen_times else 0.0,
        }

        # median
        sorted_vals = sorted(gen_times)
        if sorted_vals:
            n = len(sorted_vals)
            if n % 2 == 1:
                median = sorted_vals[n // 2]
            else:
                median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
        else:
            median = 0.0
        stats["median_ms"] = median

        write_json(os.path.join(args.output_dir, "summary.json"), stats)
        with open(os.path.join(args.output_dir, "outputs.jsonl"), "w", encoding="utf-8") as f:
            for item in outputs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(os.path.join(args.output_dir, "summary.json"))
    except Exception as exc:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
