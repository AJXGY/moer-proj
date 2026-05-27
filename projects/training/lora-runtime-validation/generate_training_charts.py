#!/usr/bin/env python3
"""
5.1.6 图表生成模块 - 训练性能可视化
MTT-TRAIN-RUN-TEST Charts Generator

生成性能对比图表
"""

import argparse
import json
import os
from datetime import datetime, timezone


def load_json(path):
    """加载JSON文件"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


def generate_comparison_report(artifact_dir, output_path):
    """生成对比报告"""
    
    report = []
    report.append("# 5.1.6 训练性能对比总结\n\n")
    report.append(f"**生成时间**：{datetime.now(timezone.utc).isoformat()}\n\n")
    
    # 搜索所有artifact目录
    artifact_list = []
    if os.path.isdir(artifact_dir):
        for item in sorted(os.listdir(artifact_dir)):
            item_path = os.path.join(artifact_dir, item)
            if os.path.isdir(item_path) and item not in ['manual-dual-real', 'manual-single-real']:
                # 尝试读取汇总信息
                single_summary = load_json(os.path.join(item_path, 'single', 'summary.json'))
                dual_summary = load_json(os.path.join(item_path, 'dual', 'summary.json'))
                
                if single_summary or dual_summary:
                    artifact_list.append({
                        'timestamp': item,
                        'single': single_summary,
                        'dual': dual_summary,
                    })
    
    if not artifact_list:
        report.append("**警告**：未找到任何训练结果\n")
        return ''.join(report)
    
    # 生成表格
    report.append("## 历次训练记录\n\n")
    report.append("| 时间戳 | 单卡状态 | 双卡状态 | 单卡耗时(s) | 双卡耗时(s) |\n")
    report.append("|-------|--------|--------|-----------|----------|\n")
    
    for artifact in artifact_list:
        timestamp = artifact['timestamp']
        single_status = "成功" if (artifact['single'] and artifact['single'].get('success')) else "待测"
        dual_status = "成功" if (artifact['dual'] and artifact['dual'].get('success')) else "待测"
        single_time = artifact['single'].get('execution_time_seconds', 0) if artifact['single'] else "-"
        dual_time = artifact['dual'].get('execution_time_seconds', 0) if artifact['dual'] else "-"
        
        report.append(f"| {timestamp} | {single_status} | {dual_status} | {single_time} | {dual_time} |\n")
    
    report.append("\n")
    
    # 性能分析
    if len(artifact_list) >= 2:
        report.append("## 性能趋势分析\n\n")
        report.append("### 最新执行结果\n\n")
        
        latest = artifact_list[-1]
        if latest['single']:
            report.append(f"**单卡训练**：\n")
            report.append(f"- 执行时间：{latest['single'].get('execution_time_seconds', '-')} 秒\n")
            report.append(f"- Dry-run模式：{latest['single'].get('dry_run', False)}\n")
            report.append(f"- 输出文件数：{len(latest['single'].get('outputs', []))}\n\n")
        
        if latest['dual']:
            report.append(f"**双卡训练**：\n")
            report.append(f"- 执行时间：{latest['dual'].get('execution_time_seconds', '-')} 秒\n")
            report.append(f"- Dry-run模式：{latest['dual'].get('dry_run', False)}\n")
            report.append(f"- 输出文件数：{len(latest['dual'].get('outputs', []))}\n\n")
    
    # 建议
    report.append("## 改进建议\n\n")
    report.append("1. **性能优化**\n")
    report.append("   - 监控显存占用率，调整batch_size以提高利用率\n")
    report.append("   - 启用gradient checkpointing以减少显存需求\n")
    report.append("   - 考虑混合精度训练(Mixed Precision)提高吞吐量\n\n")
    
    report.append("2. **通信优化**（双卡）\n")
    report.append("   - 检查MCCL/NCCL版本是否最新\n")
    report.append("   - 调整AllReduce算法以优化通信效率\n")
    report.append("   - 使用GPU直接通信(GPU Direct)如果硬件支持\n\n")
    
    report.append("3. **模型优化**\n")
    report.append("   - 尝试LoRA微调替代全量训练以减少计算量\n")
    report.append("   - 考虑使用更高效的优化器(如AdamW)\n")
    report.append("   - 实验不同的学习率调度策略\n\n")
    
    report.append("4. **环境优化**\n")
    report.append("   - 确保使用最新的PyTorch和MUSA驱动版本\n")
    report.append("   - 验证CPU/GPU亲和性设置\n")
    report.append("   - 检查系统的NUMA配置(如适用)\n")
    
    return ''.join(report)


def main():
    parser = argparse.ArgumentParser(description='5.1.6 Charts Generator')
    parser.add_argument('--artifact_dir', type=str, default='./artifacts',
                       help='Artifact directory')
    parser.add_argument('--output', type=str, required=True, help='Output HTML/Markdown file')
    args = parser.parse_args()
    
    # 生成对比报告
    report = generate_comparison_report(args.artifact_dir, args.output)
    
    # 写入文件
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n✓ 报告已生成：{args.output}")
    return 0


if __name__ == '__main__':
    exit(main())
