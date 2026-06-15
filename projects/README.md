# Projects

Normalized project source tree.

| Area | Purpose | Entry |
| --- | --- | --- |
| `operators/communication/` | Communication operator benchmark/modeling | `../scripts/run_operator_comm_moorethreads.sh` |
| `operators/memory/` | Memory operator benchmark/modeling | `../scripts/run_operator_mem_moorethreads.sh` |
| `operators/compute/` | Compute operator benchmark/modeling | `../scripts/run_operator_compute_moorethreads.sh` |
| `training/time-modeling/` | Training time modeling, single-card and multi-card configs | `../scripts/run_training_moorethreads_smoke.sh` |
| `inference/time-modeling/` | Inference time modeling, single-card and multi-card configs | `../scripts/run_inference_moorethreads_smoke.sh` |
| `shared/train-infer-estimation/` | Minimal shared runtime dependency | internal |

Use `scripts/run_training_single_moorethreads.sh`, `scripts/run_training_multi_moorethreads.sh`, `scripts/run_inference_single_moorethreads.sh`, and `scripts/run_inference_multi_moorethreads.sh` when you want a specific single-card or multi-card run.
