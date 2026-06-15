# moer-proj Validation Report

Generated: 2026-05-08 16:45 CST

## Conclusion

This pass found that the previous `--force-synthetic` outputs were only chain checks, not real MooreThreads benchmark results. The code now marks synthetic outputs as `not_acceptance_synthetic`, and real runs without `--force-synthetic` fail fast when MUSA runtime is not available.

Current machine status:

- `mthreads-gmi` sees 2 x MTT S3000.
- `torch_musa` reports `musa_available=False`, `musa_count=0`.
- `/usr/local/musa/bin/musaInfo` fails with `CreatePlatform failed`.
- Therefore training/inference/operator real benchmarks are not accepted on this runtime yet.

## Fixes Applied

| Item | Status | Detail |
| --- | --- | --- |
| Training silent synthetic fallback | Fixed | Training now requires `--force-synthetic` for synthetic chain checks. Real mode fails if MUSA/CUDA is unavailable. |
| Inference synthetic 0% error | Fixed | Synthetic inference no longer applies fitted scale to produce 0% error. It is marked `synthetic_chain_check_not_acceptance`. |
| Parallel script result race | Fixed | Each top-level script now uses its own `results/<run>/<timestamp>/latest_tp_artifact.txt`. Single/multi runs no longer overwrite each other. |
| Result directory isolation | Fixed | Every top-level script writes `run.log`, `latest_tp_artifact.txt`, and generated artifacts under its own timestamped result directory. |
| Training sequence length | Adjusted | Training benchmark default `--max-seq-len` is now `9`. LoRA remains `rank=8`, `alpha=16.0`. |

## Script To Code Mapping

| Top-level script | Scope | Project suite | Main code |
| --- | --- | --- | --- |
| `scripts/run_training_single_moorethreads.sh` | training, single-card | `projects/training/time-modeling/run_5214_tp_suite.sh` | `benchmark_tp_train_time.py`, `fit_tp_time_model.py`, `summarize_tp_results.py` |
| `scripts/run_training_multi_moorethreads.sh` | training, single-node TP=2 | `projects/training/time-modeling/run_5214_tp_suite.sh` | same training code, separated by `MOER_PARALLEL_SCOPE=multi` |
| `scripts/run_training_moorethreads_smoke.sh` | training, all configs | `projects/training/time-modeling/run_5214_tp_suite.sh` | same training code, separated by config functions |
| `scripts/run_inference_single_moorethreads.sh` | inference, single-card | `projects/inference/time-modeling/run_5215_tp_suite.sh` | `benchmark_tp_infer_time.py`, `fit_tp_time_model.py`, `summarize_tp_results.py` |
| `scripts/run_inference_multi_moorethreads.sh` | inference, single-node TP=2 | `projects/inference/time-modeling/run_5215_tp_suite.sh` | same inference code, separated by `MOER_PARALLEL_SCOPE=multi` |
| `scripts/run_inference_moorethreads_smoke.sh` | inference, all configs | `projects/inference/time-modeling/run_5215_tp_suite.sh` | same inference code, separated by config functions |
| `scripts/run_operator_comm_moorethreads.sh` | communication operators | `projects/operators/communication/run_529_suite.sh` | `benchmark_comm_ops.py`, `fit_space_model.py`, `generate_charts.py`, `summarize_results.py` |
| `scripts/run_operator_mem_moorethreads.sh` | memory operators | `projects/operators/memory/run_526_suite.sh` | `benchmark_memory_ops.py`, `fit_space_model.py`, `generate_charts.py`, `summarize_results.py` |
| `scripts/run_operator_compute_moorethreads.sh` | compute operators | `projects/operators/compute/run_523_suite.sh` | `benchmark_compute_ops.py`, `fit_space_model.py`, `generate_charts.py`, `summarize_results.py` |

## Config Coverage

Training configs:

- `cfg_single_mb1`, `cfg_single_mb2`, `cfg_single_mb4`: single-card, TP=1.
- `cfg_tp2_mb1`, `cfg_tp2_mb2`, `cfg_tp2_mb4`: single-node multi-card, TP=2.
- Default training task: LoRA, `max_seq_len=9`, `lora_rank=8`, `lora_alpha=16.0`.

Inference configs:

- `cfg_single_mb1`, `cfg_single_mb2`, `cfg_single_mb4`: single-card, TP=1.
- `cfg_tp2_mb1`, `cfg_tp2_mb2`, `cfg_tp2_mb4`: single-node multi-card, TP=2.

## Static Checks

| Check | Result |
| --- | --- |
| `bash -n` on all `.sh` files | Passed |
| `python3 -m py_compile` on all `.py` files | Passed |
| JSON validation for training configs | Passed |
| JSON validation for inference configs | Passed |

## Synthetic Chain Checks

These checks verify script wiring and result generation only. They are not real benchmark acceptance results.

| Script | Result | Artifact |
| --- | --- | --- |
| `run_training_single_moorethreads.sh --force-synthetic --runs-per-config 1` | Passed chain check, not acceptance | `results/training-single/20260508T084244Z/artifacts/20260508T084249Z/summary.md` |
| `run_training_multi_moorethreads.sh --force-synthetic --runs-per-config 1` | Passed chain check, not acceptance | `results/training-multi/20260508T084244Z/artifacts/20260508T084249Z/summary.md` |
| `run_inference_single_moorethreads.sh --force-synthetic --runs-per-config 1` | Passed chain check, not acceptance | `results/inference-single/20260508T084244Z/artifacts/20260508T084249Z/summary.md` |
| `run_inference_multi_moorethreads.sh --force-synthetic --runs-per-config 1` | Passed chain check, not acceptance | `results/inference-multi/20260508T084244Z/artifacts/20260508T084249Z/summary.md` |

Synthetic summaries now show:

- `acceptance_status: not_acceptance_synthetic`
- `result: not_acceptance`
- `synthetic_chain_check_not_acceptance`

## Real Run Checks

Real run attempts were executed without `--force-synthetic`.

| Script | Result | Log |
| --- | --- | --- |
| `run_training_single_moorethreads.sh --runs-per-config 1` | Failed fast as expected, no real MUSA runtime | `results/training-single/20260508T084111Z/run.log` |
| `run_training_multi_moorethreads.sh --runs-per-config 1` | Failed fast as expected, no real MUSA runtime | `results/training-multi/20260508T084111Z/run.log` |
| `run_inference_single_moorethreads.sh --runs-per-config 1` | Failed fast as expected, no real MUSA runtime | `results/inference-single/20260508T084123Z/run.log` |
| `run_inference_multi_moorethreads.sh --runs-per-config 1` | Failed fast as expected, no real MUSA runtime | `results/inference-multi/20260508T084123Z/run.log` |

Common failure reason:

```text
MUSA initialization failed
CreatePlatform failed
musa_available False
musa_count 0
```

## Operator Scripts

Operator scripts are structurally separated and statically valid. They require real MUSA runtime:

- communication requires 2 visible MUSA devices and distributed launch.
- memory requires MUSA tensors on device 0/1.
- compute requires MUSA tensors on device 0/1.

Because `torch_musa` currently reports zero available devices, operator real acceptance was not run in this pass.

## Next Acceptance Gate

Before accepting performance numbers:

1. Fix MUSA runtime initialization so `torch.musa.is_available()` is true and `torch.musa.device_count()` is 2.
2. Run real scripts without `--force-synthetic`.
3. Accept only reports where `acceptance_status` is `acceptance_candidate` and `measurement_mode` is `real_measurement`.
4. Ignore old synthetic reports with `not_acceptance_synthetic`.
