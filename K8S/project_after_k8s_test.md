# K8S 更新后项目脚本运行测试报告

## 测试结论

2026-04-30 在 `ICT109` 上对 `moerxiancheng-clj-xyj-proj` 中 `clj-proj`、`xyj` 的代表性测试脚本进行了运行验证。本轮共执行 9 组脚本，退出码均为 `0`。

结论：K8S 更新后，当前工作区内已验证的 `clj-proj`、`xyj` 脚本链路可以正常运行；MUSA 后端可见 2 张 `MTT S3000`，真实训练/TP 推理小样本可以跑通。

## 测试环境

| 项目 | 内容 |
| --- | --- |
| 测试时间 | 2026-04-30 01:00-01:29 UTC |
| 机器 | ICT109 |
| 用户 | o_mabin |
| 项目目录 | `/home/o_mabin/moerxiancheng-clj-xyj-proj` |
| 日志目录 | `moerxiancheng-clj-xyj-proj/K8S/project_test_logs/20260430T_project_after_k8s` |
| 加速后端 | MUSA |
| 设备 | 2 x MTT S3000 |

## 测试结果汇总

| 序号 | 项目 | 脚本/命令 | 类型 | 结果 | 产物/日志 |
| --- | --- | --- | --- | --- | --- |
| 1 | `clj-proj/5.1.5` | `bash run_515_suite.sh --dry-run --max-new-tokens 1` | 推理 dry-run，single/dual | 通过 | `artifacts/20260430T010002Z`，`clj_515_dryrun.log` |
| 2 | `clj-proj/5.1.6` | `bash run_516_suite.sh` | 真实训练小样本，single/dual/TP | 通过 | `artifacts/20260430T012756Z`，`clj_516_suite.log` |
| 3 | `clj-proj/5.2.3` | `bash run_523_suite.sh` | 计算算子 benchmark | 通过 | `clj_523_suite.log` |
| 4 | `clj-proj/5.2.6` | `bash run_526_suite.sh` | 内存算子 benchmark | 通过 | `clj_526_suite.log` |
| 5 | `clj-proj/5.2.9` | `bash run_529_suite.sh` | 分布式通信 benchmark | 通过 | `clj_529_suite.log` |
| 6 | `clj-proj/5.2.14` | `benchmark_tp_train_time.py --runs-per-config 1` + fit/summarize | TP 训练最小样本 | 通过 | `artifacts/20260430T011008Z`，`clj_5214_tp_min.log` |
| 7 | `xyj/5.1.6` | `bash run_516_suite.sh --dry-run` | 训练 dry-run，single/dual/TP | 通过 | `artifacts/20260430T012019Z`，`xyj_516_dryrun.log` |
| 8 | `xyj/5.1.12` | `bash run_512_suite.sh` | 推理建模/图表/汇总 | 通过 | `5.1.12任务进展.md`，`xyj_512_suite.log` |
| 9 | `xyj/5.2.15` | `bash run_5215_tp_suite.sh --runs-per-config 1 --warmups 0 --max-seq-len 8` | TP 推理最小样本 | 通过 | `artifacts/20260430T012142Z`，`xyj_5215_tp_min.log` |

## 关键验证点

### `clj-proj/5.1.5` 推理 dry-run

- single 模式：`success=true`，`validation_passed=true`，输出 3 条。
- dual 模式：`success=true`，`validation_passed=true`，输出 3 条。
- 产物目录：`/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.5/artifacts/20260430T010002Z`

### `clj-proj/5.1.6` 真实训练

真实加载 `Meta-Llama-3.1-8B`，在 MUSA 上完成 single、dual、TP 三种训练小样本。

| 模式 | 后端 | 设备数 | 步数 | 结果 | checkpoint |
| --- | --- | --- | --- | --- | --- |
| single | musa | 2 | 2 | `success=true` | `single_adapter_checkpoint.pt` |
| dual/PP | musa | 2 | 2 | `success=true` | `dual_adapter_checkpoint.pt` |
| TP | musa | 2 | 2 | `success=true` | `tp_adapter_checkpoint.pt` |

产物目录：`/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.6/artifacts/20260430T012756Z`

### `clj-proj/5.2.9` 分布式通信

`torch.distributed.run --nproc_per_node=2 benchmark_comm_ops.py` 执行完成，退出码为 `0`。

日志中出现主机名解析提示：

```text
Unable to resolve hostname to a (local) address. Using the loopback address as fallback.
```

该提示未导致脚本失败，但建议后续将 `ICT109` 写入 `/etc/hosts` 或配置 `GLOO_SOCKET_IFNAME`，避免分布式通信在多机/多网卡场景下绑定到非预期接口。

### `clj-proj/5.2.14` TP 训练最小样本

使用 `--runs-per-config 1` 做最小样本验证，3 个 TP 配置均生成结果：

| 配置 | 结果 | 单次耗时 |
| --- | --- | --- |
| `cfg_tp2_mb1` | 通过 | 约 98.50 ms |
| `cfg_tp2_mb2` | 通过 | 约 196.85 ms |
| `cfg_tp2_mb4` | 通过 | 约 393.76 ms |

产物目录：`/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.14/artifacts/20260430T011008Z`

### `xyj/5.1.6` 训练 dry-run

dry-run 模式下完成 single、dual、TP 三条链路，均检测到 MUSA 后端和 2 张 `MTT S3000`。

| 模式 | distributed | parallel mode | 结果 |
| --- | --- | --- | --- |
| single | false | single | `success=true` |
| dual | true | pp | `success=true` |
| TP | true | tp | `success=true` |

产物目录：`/home/o_mabin/moerxiancheng-clj-xyj-proj/xyj/5.1.6/artifacts/20260430T012019Z`

### `xyj/5.2.15` TP 推理最小样本

真实加载模型，完成 3 个 TP 推理配置：

| 配置 | 结果 | 单次耗时 |
| --- | --- | --- |
| `cfg_tp2_mb1` | 通过 | 约 15088.70 ms |
| `cfg_tp2_mb2` | 通过 | 约 195.57 ms |
| `cfg_tp2_mb4` | 通过 | 约 388.05 ms |

产物目录：`/home/o_mabin/moerxiancheng-clj-xyj-proj/xyj/5.2.15/artifacts/20260430T012142Z`

## 全部退出码

```text
clj_515_dryrun.rc=0
clj_516_suite.rc=0
clj_5214_tp_min.rc=0
clj_523_suite.rc=0
clj_526_suite.rc=0
clj_529_suite.rc=0
xyj_512_suite.rc=0
xyj_516_dryrun.rc=0
xyj_5215_tp_min.rc=0
```

## 注意事项

1. 本轮为了验证 K8S 更新后项目是否还能正常跑，优先选择了代表性脚本、dry-run 和最小样本；没有把所有重型 real suite 按默认参数完整跑满。
2. `clj-proj/5.2.14` 和 `xyj/5.2.15` 使用了 reduced runs：`runs-per-config=1`，用于功能验证，不作为正式性能基准。
3. 日志中多次出现 `No ROCm runtime is found`，但当前后端为 MUSA，该提示未影响测试结果。
4. 日志中出现 `Flash attention only supports architecture with mp version >= 2.2`，当前 MTT S3000 为 mp 2.1，脚本仍能完成。
5. `clj-proj/5.2.9` 出现主机名解析到 loopback 的提示，建议修复 `ICT109` 主机名解析。
6. 如果要做完整回归，还可以继续跑 `xyj/5.1.12/run_512_real_suite.sh`、`xyj/5.2.15/run_5215_real_suite.sh`、`clj-proj/5.2.14/run_5214_tp_suite.sh` 默认完整参数版本。

## 可发群简短结论

```text
已在 109/ICT109 上对 moerxiancheng-clj-xyj-proj 里的 clj-proj、xyj 做了一轮 K8S 更新后的项目脚本验证。共跑了 9 组代表性脚本，包括 clj 的推理 dry-run、真实训练 single/dual/TP、计算/内存/通信 benchmark、TP 训练小样本，以及 xyj 的训练 dry-run、5.1.12 建模脚本、5.2.15 TP 推理小样本。所有脚本退出码均为 0，MUSA 后端可识别 2 张 MTT S3000，真实训练和 TP 推理最小样本均能跑通。

注意：分布式通信日志里有 ICT109 主机名解析到 loopback 的提示，建议后续修复 /etc/hosts 或配置 GLOO_SOCKET_IFNAME；本轮 TP 性能类任务采用最小 runs 做功能验证，不作为正式性能基准。
```
