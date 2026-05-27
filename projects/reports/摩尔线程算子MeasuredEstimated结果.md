# 摩尔线程（run id=5.2.3-20260506T230225Z）

## 算子

| 算子 | Measured | Estimated | Error |
| --- | ---: | ---: | ---: |
| attention output proj gemm | 47.1311 ms | 44.6171 ms | 5.33% |
| flash attention | 3.3394 ms | 3.3559 ms | 0.50% |
| mlp down gemm | 162.3980 ms | 156.9406 ms | 3.36% |
| mlp gate gemm | 153.5905 ms | 159.8942 ms | 4.10% |
| mlp up gemm | 152.8451 ms | 160.1653 ms | 4.79% |

说明：本表取自 `clj-proj/5.2.3/artifacts/20260415T100500Z/space_model_results.json` 的双卡结果，口径为 `seq_len=1024`、`float16`。
