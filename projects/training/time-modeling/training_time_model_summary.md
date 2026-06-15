# Training Time Modeling Summary

- generated_at: 2026-06-10T02:19:35.210624+00:00
- artifact: /home/o_mabin/moer-proj/results/training-multi/20260610T021734Z/artifacts/20260610T021738Z
- backend: musa
- device_count: 2
- scopes: multi
- tensor_parallel_sizes: 2
- acceptance_status: acceptance_candidate
- result: pass

## Config Results

| config | scope | TP | MB | T_real(ms) | T_sim(ms) | error | mode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cfg_tp2_mb1 | multi | 2 | 1 | 460.055 | 505.434 | 9.86% | analytical_only_llama_layer_expanded + one_parameter_train_single_node_tp_scale |
| cfg_tp2_mb2 | multi | 2 | 2 | 1055.836 | 1011.447 | 4.20% | analytical_only_llama_layer_expanded + one_parameter_train_single_node_tp_scale |
| cfg_tp2_mb4 | multi | 2 | 4 | 2249.487 | 2027.050 | 9.89% | analytical_only_llama_layer_expanded + one_parameter_train_single_node_tp_scale |

## Files

- benchmark: /home/o_mabin/moer-proj/results/training-multi/20260610T021734Z/artifacts/20260610T021738Z/tp_benchmark_results.json
- model: /home/o_mabin/moer-proj/results/training-multi/20260610T021734Z/artifacts/20260610T021738Z/tp_time_model_results.json
