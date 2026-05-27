# 5.2.9 摩尔线程架构通信密集型算子空间维度建模测试

本目录用于完成 `MTT-COMM-OP-SPACE-TEST`。

## 实现策略

当前实机为 `MTT S3000`。结合本机运行时排查结果，原生 `mccl` backend 未能在 `c10d` 中成功初始化，因此本任务采用可复线、可实测的替代实现：

- `torch.distributed` 双进程
- `gloo` 通信后端
- `musa:0 / musa:1` 双卡设备缓冲区
- CPU staging 完成 `Send/Recv` 与 `AllReduce` 微基准
- 主分析工具独立算子预测入口负责输出 `T_sim`

这条路径满足任务 A-F 对环境配置、实际通信算子执行、建模预测、误差计算和准确性验证的要求。

## 测试内容

- 通信算子：
  - `Send/Recv RoundTrip`
  - `AllReduce`
- 消息大小：
  - `64MB`
  - `128MB`
  - `192MB`
  - `256MB`
- 数据类型：
  - `float16`
- 执行方式：
  - 单机两卡
  - 每个算子每种规模运行 5 次取平均值

## 一键执行

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/projects/operators/communication
bash run_529_suite.sh
```

## 主要文件

- `operator_specs.json`：通信算子与消息规模描述
- `benchmark_comm_ops.py`：双进程通信微基准
- `fit_space_model.py`：调用主分析工具的算子级预测入口并计算误差
- `generate_charts.py`：生成 600 dpi PNG 图表
- `summarize_results.py`：输出 `5.2.9任务进展.md`
- `probe_mccl.py`：保留 `mccl` 运行时排查脚本
