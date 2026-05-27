# Scripts

Repository-level runnable entry points.

Each script resolves the repository root and delegates to the current implementation directory. Keep new automation here so callers do not need to know the historical task folder names.

Run `check_project_layout.sh` after moving files or changing entry points.

## Main Entries

- `run_inference_runtime_moorethreads.sh`
- `run_inference_model_structure_moorethreads.sh`
- `run_inference_moorethreads_smoke.sh`
- `run_training_runtime_moorethreads.sh`
- `run_training_lora_moorethreads.sh`
- `run_training_model_structure_moorethreads.sh`
- `run_training_moorethreads_smoke.sh`
- `run_operator_compute_moorethreads.sh`
- `run_operator_mem_moorethreads.sh`
- `run_operator_comm_moorethreads.sh`
