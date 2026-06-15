# Inference Time Modeling Summary

- generated_at: 2026-06-10T05:09:03.042131+00:00
- artifact: /home/o_mabin/moer-proj/results/inference-multi/20260610T050725Z/artifacts/20260610T050729Z
- backend: musa
- device_count: 2
- scopes: multi
- tensor_parallel_sizes: 2
- acceptance_status: acceptance_candidate
- result: pass

## Config Results

| config | scope | TP | MB | T_real(ms) | T_sim(ms) | error | mode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cfg_tp2_mb1 | multi | 2 | 1 | 97.833 | 97.460 | 0.38% | torch_infer_mvp_estimate_only_single_raw_scaled_by_mb + one_parameter_tp_scale |
| cfg_tp2_mb2 | multi | 2 | 2 | 195.783 | 194.919 | 0.44% | torch_infer_mvp_estimate_only_single_raw_scaled_by_mb + one_parameter_tp_scale |
| cfg_tp2_mb4 | multi | 2 | 4 | 389.313 | 389.838 | 0.13% | torch_infer_mvp_estimate_only_single_raw_scaled_by_mb + one_parameter_tp_scale |

## Files

- benchmark: /home/o_mabin/moer-proj/results/inference-multi/20260610T050725Z/artifacts/20260610T050729Z/tp_benchmark_results.json
- model: /home/o_mabin/moer-proj/results/inference-multi/20260610T050725Z/artifacts/20260610T050729Z/tp_time_model_results.json
