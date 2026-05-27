# Projects

This directory is the normalized project index for the benchmark and modeling work.

The historical task directories are kept in place for traceability. New entry points should be added here first, and runnable commands should go through `../scripts/`.

## Layout

- `operators/communication/`: communication operator benchmarks and modeling.
- `operators/memory/`: memory operator benchmarks and modeling.
- `operators/compute/`: compute operator benchmarks and modeling.
- `training/runtime-validation/`: training single-card and single-node multi-card runtime validation.
- `training/lora-runtime-validation/`: LoRA training runtime validation from the `xyj` task line.
- `training/model-structure/`: training task processing model output validation.
- `training/time-modeling/`: training runtime measurement and time modeling.
- `inference/runtime-validation/`: inference single-card and single-node multi-card runtime validation.
- `inference/model-structure/`: inference task processing model output validation.
- `inference/time-modeling/`: inference runtime measurement and time modeling.
- `shared/train-infer-estimation/`: shared train/infer/operator MVP platform.
- `reports/`: cross-project reports formerly stored at the legacy root.

## Project Mapping

| Normalized area | Historical source | Main script |
| --- | --- | --- |
| `operators/communication/` | `../clj-proj/5.2.9/` | `../scripts/run_operator_comm_moorethreads.sh` |
| `operators/memory/` | `../clj-proj/5.2.6/` | `../scripts/run_operator_mem_moorethreads.sh` |
| `operators/compute/` | `../clj-proj/5.2.3/` | `../scripts/run_operator_compute_moorethreads.sh` |
| `training/runtime-validation/` | `../clj-proj/5.1.6/` | `../scripts/run_training_runtime_moorethreads.sh` |
| `training/lora-runtime-validation/` | `../xyj/5.1.6/` | `../scripts/run_training_lora_moorethreads.sh` |
| `training/model-structure/` | `../clj-proj/5.1.11/` | `../scripts/run_training_model_structure_moorethreads.sh` |
| `training/time-modeling/` | `../clj-proj/5.2.14/` | `../scripts/run_training_moorethreads_smoke.sh` |
| `inference/runtime-validation/` | `../clj-proj/5.1.5/` | `../scripts/run_inference_runtime_moorethreads.sh` |
| `inference/model-structure/` | `../xyj/5.1.12/` | `../scripts/run_inference_model_structure_moorethreads.sh` |
| `inference/time-modeling/` | `../xyj/5.2.15/` | `../scripts/run_inference_moorethreads_smoke.sh` |
| `shared/train-infer-estimation/` | `../clj-proj/train-infer-estimation-release-2026-04-11/` | `shared/train-infer-estimation/tools/python_with_env.sh` |
