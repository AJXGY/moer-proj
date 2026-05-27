# moerxiancheng-clj-xyj-proj

Research and task workspace for `clj-proj`, `xyj`, and related benchmarking or modeling scripts.

## Included

- Source code and task documents
- Config files and runnable scripts
- Charts and summaries that are suitable for version control
- LoRA-style training/inference validation summaries for MUSA runs

## Normalized Layout

Use these paths as the stable entry points for new work:

```text
projects/
  operators/
    communication/
    memory/
    compute/
  training/
    runtime-validation/
    lora-runtime-validation/
    model-structure/
    time-modeling/
  inference/
    runtime-validation/
    model-structure/
    time-modeling/
  shared/
    train-infer-estimation/
  reports/

scripts/
  check_project_layout.sh
  run_inference_runtime_moorethreads.sh
  run_inference_model_structure_moorethreads.sh
  run_operator_comm_moorethreads.sh
  run_operator_mem_moorethreads.sh
  run_operator_compute_moorethreads.sh
  run_training_runtime_moorethreads.sh
  run_training_lora_moorethreads.sh
  run_training_model_structure_moorethreads.sh
  run_training_moorethreads_smoke.sh
  run_inference_moorethreads_smoke.sh
```

The dated `clj-proj/5.x` and `xyj/5.x` directories now only keep compatibility wrappers, ignored caches, or local model files. The source projects live under `projects/`, and `scripts/` provides repository-level runnable entry points.

## Current Training Scope

- Training runtime tests use LoRA-style lightweight fine-tuning rather than full 8B parameter updates.
- The Llama3.1-8B backbone is loaded for real forward computation, while only low-rank adapter/probe parameters are updated.
- This keeps the tests aligned with the LoRA training requirement and avoids excessive full fine-tuning runtime.

## Excluded

- Local model weights
- Generated runtime artifacts and logs
- Python cache files
- Crash dump files
