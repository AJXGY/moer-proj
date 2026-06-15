# moer-proj

Clean MooreThreads benchmark project layout.

## Layout

```text
projects/
  operators/
    communication/
    memory/
    compute/
  training/
    time-modeling/
  inference/
    time-modeling/

scripts/
  run_training_single_moorethreads.sh
  run_training_multi_moorethreads.sh
  run_operator_comm_moorethreads.sh
  run_operator_mem_moorethreads.sh
  run_operator_compute_moorethreads.sh
  run_training_moorethreads_smoke.sh
  run_inference_single_moorethreads.sh
  run_inference_multi_moorethreads.sh
  run_inference_moorethreads_smoke.sh
results/
```

## Scope

- `operators/`: three operator classes, communication, memory, and compute.
- `training/time-modeling/`: training single-card and multi-card timing/modeling configs.
- `inference/time-modeling/`: inference single-card and multi-card timing/modeling configs.
- `projects/shared/train-infer-estimation/`: minimal shared runtime used by operator, training, and inference scripts.

Every top-level script creates a timestamped result folder under `results/` and writes a `run.log` plus generated artifacts.

Model weights are not included. Put local weights at:

```text
clj-proj/model/Meta-Llama-3.1-8B
```
