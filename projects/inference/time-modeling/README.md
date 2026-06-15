# Inference Time Modeling

Inference timing/modeling project for MooreThreads.

## Card Modes

- Single-card: use configs with `world_size=1`, `tp_size=1`, or `nproc_per_node=1`.
- Multi-card: use TP/multi-process configs with `world_size=2`, `tp_size=2`, or `nproc_per_node=2`.

## Run

```bash
../../../scripts/run_inference_moorethreads_smoke.sh
```

