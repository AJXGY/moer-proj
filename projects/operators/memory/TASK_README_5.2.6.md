# 5.2.6 摩尔线程架构访存密集型算子空间维度建模测试

本目录用于完成 `MTT-MEM-OP-SPACE-TEST`。

交付内容包括：

- 访存密集型算子微基准脚本
- 单卡 / 单机双卡实测数据
- 基于主分析工具独立算子预测入口的空间维度预测结果与误差统计
- 600 dpi PNG 图表
- `5.2.6任务进展.md`

## 一键执行

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/projects/operators/memory
bash run_526_suite.sh
```

## 主要文件

- `operator_specs.json`：代表性访存密集型算子描述
- `benchmark_memory_ops.py`：实测单卡 / 双卡算子耗时
- `fit_space_model.py`：调用主分析工具的算子级预测入口并计算误差
- `generate_charts.py`：生成 600 dpi PNG 图表
- `summarize_results.py`：输出 `5.2.6任务进展.md`
