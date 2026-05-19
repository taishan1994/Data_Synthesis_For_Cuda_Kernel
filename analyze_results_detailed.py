import os
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def analyze_results_detailed(output_dir):
    """
    详细分析结果文件，按照level分层统计
    
    Args:
        output_dir: 输出目录路径
    """
    
    if not os.path.exists(output_dir):
        print(f"输出目录不存在: {output_dir}")
        return
    
    # 获取所有 JSON 文件
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"在目录 {output_dir} 中没有找到 JSON 文件")
        return
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    print("=" * 80)
    
    # 按照 level 分组
    level_groups = defaultdict(list)
    
    for json_file in json_files:
        file_path = os.path.join(output_dir, json_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取 level
            level = data.get('level', 'unknown')
            level_groups[level].append(data)
        
        except Exception as e:
            print(f"❌ 读取文件 {json_file} 失败: {e}")
            continue
    
    # 按照排序的level顺序输出
    for level in sorted(level_groups.keys()):
        print(f"\n{'=' * 80}")
        print(f"Level: {level}")
        print(f"样本数量: {len(level_groups[level])}")
        print('=' * 80)
        
        analyze_level_detailed(level_groups[level])
    
    # 总体统计
    print(f"\n{'=' * 80}")
    print("总体统计")
    print('=' * 80)
    analyze_all_detailed(level_groups)
    
    # 整体最佳性能分析
    analyze_overall_best_performance(level_groups)
    
    # 生成图表
    generate_charts(level_groups, output_dir)

def analyze_level_detailed(samples):
    """
    详细分析单个level的样本
    
    Args:
        samples: 该level的所有样本列表
    """
    
    # 统计每一轮的数据
    max_rounds = 5
    round_stats = {
        i: {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'speedups': [],
            'fast1': 0,
            'fast1.2': 0,
            'fast1.5': 0,
            'fast2': 0,
            'error_categories': defaultdict(int)
        }
        for i in range(1, max_rounds + 1)
    }
    
    for sample in samples:
        messages = sample.get('messages', [])
        num_rounds = sample.get('num_rounds', 0)
        
        # 分析每一轮（除了最后一轮）
        for round_idx in range(1, min(num_rounds, max_rounds)):
            round_stats[round_idx]['total'] += 1
            
            # 获取该轮的反馈
            feedback = extract_round_feedback(messages, round_idx)
            
            if feedback is None:
                continue
            
            # 统计通过率
            compiled = feedback.get('compiled', False)
            correctness = feedback.get('correctness', False)
            speedup = feedback.get('speedup', 0)
            
            if compiled and correctness:
                round_stats[round_idx]['passed'] += 1
                round_stats[round_idx]['speedups'].append(speedup)
                
                # 统计加速比分布
                if speedup >= 2.0:
                    round_stats[round_idx]['fast2'] += 1
                    round_stats[round_idx]['fast1.5'] += 1
                    round_stats[round_idx]['fast1.2'] += 1
                    round_stats[round_idx]['fast1'] += 1
                elif speedup >= 1.5:
                    round_stats[round_idx]['fast1.5'] += 1
                    round_stats[round_idx]['fast1.2'] += 1
                    round_stats[round_idx]['fast1'] += 1
                elif speedup >= 1.2:
                    round_stats[round_idx]['fast1.2'] += 1
                    round_stats[round_idx]['fast1'] += 1
                elif speedup >= 1.0:
                    round_stats[round_idx]['fast1'] += 1
            else:
                round_stats[round_idx]['failed'] += 1
                
                # 统计失败原因
                error_category = categorize_error(feedback)
                round_stats[round_idx]['error_categories'][error_category] += 1
    
    # 打印每一轮的统计
    for round_idx in range(1, max_rounds + 1):
        stats = round_stats[round_idx]
        if stats['total'] == 0:
            continue
        
        print(f"\n第 {round_idx} 轮:")
        print(f"  总样本数: {stats['total']}")
        print(f"  通过数: {stats['passed']}")
        print(f"  失败数: {stats['failed']}")
        print(f"  通过率: {stats['passed'] / stats['total'] * 100:.1f}%")
        print(f"  加速比分布:")
        print(f"    fast1 (>=1.0x): {stats['fast1']} ({stats['fast1'] / stats['total'] * 100:.1f}%)")
        print(f"    fast1.2 (>=1.2x): {stats['fast1.2']} ({stats['fast1.2'] / stats['total'] * 100:.1f}%)")
        print(f"    fast1.5 (>=1.5x): {stats['fast1.5']} ({stats['fast1.5'] / stats['total'] * 100:.1f}%)")
        print(f"    fast2 (>=2.0x): {stats['fast2']} ({stats['fast2'] / stats['total'] * 100:.1f}%)")
        
        if stats['speedups']:
            print(f"  平均加速比: {np.mean(stats['speedups']):.2f}x")
            print(f"  中位数加速比: {np.median(stats['speedups']):.2f}x")
            print(f"  最大加速比: {np.max(stats['speedups']):.2f}x")
            print(f"  最小加速比: {np.min(stats['speedups']):.2f}x")
        
        if stats['error_categories']:
            print(f"  失败原因分类:")
            for error_type, count in sorted(stats['error_categories'].items(), key=lambda x: -x[1]):
                print(f"    {error_type}: {count} ({count / stats['failed'] * 100:.1f}%)")

def analyze_all_detailed(level_groups):
    """
    分析所有level的总体统计
    
    Args:
        level_groups: 按照level分组的样本字典
    """
    
    all_samples = []
    for level, samples in level_groups.items():
        all_samples.extend(samples)
    
    print(f"总样本数: {len(all_samples)}")
    
    # 统计最终加速比分布
    final_speedups = []
    for sample in all_samples:
        speedup = sample.get('final_speedup', 0)
        final_speedups.append(speedup)
    
    if final_speedups:
        print(f"\n最终加速比分布:")
        fast1_count = sum(1 for s in final_speedups if s >= 1.0)
        fast1_2_count = sum(1 for s in final_speedups if s >= 1.2)
        fast1_5_count = sum(1 for s in final_speedups if s >= 1.5)
        fast2_count = sum(1 for s in final_speedups if s >= 2.0)
        
        print(f"  fast1 (>=1.0x): {fast1_count} ({fast1_count / len(final_speedups) * 100:.1f}%)")
        print(f"  fast1.2 (>=1.2x): {fast1_2_count} ({fast1_2_count / len(final_speedups) * 100:.1f}%)")
        print(f"  fast1.5 (>=1.5x): {fast1_5_count} ({fast1_5_count / len(final_speedups) * 100:.1f}%)")
        print(f"  fast2 (>=2.0x): {fast2_count} ({fast2_count / len(final_speedups) * 100:.1f}%)")
        print(f"  平均加速比: {np.mean(final_speedups):.2f}x")
        print(f"  中位数加速比: {np.median(final_speedups):.2f}x")
        print(f"  最大加速比: {np.max(final_speedups):.2f}x")
        print(f"  最小加速比: {np.min(final_speedups):.2f}x")
    
    # 统计轮数分布
    num_rounds_dist = defaultdict(int)
    for sample in all_samples:
        num_rounds = sample.get('num_rounds', 0)
        num_rounds_dist[num_rounds] += 1
    
    print(f"\n轮数分布:")
    for rounds in sorted(num_rounds_dist.keys()):
        count = num_rounds_dist[rounds]
        print(f"  {rounds} 轮: {count} ({count / len(all_samples) * 100:.1f}%)")

def analyze_overall_best_performance(level_groups):
    """
    分析整体最佳性能：5轮中只要1轮通过就视为通过，取5轮中最好的speedup作为最终加速比
    
    Args:
        level_groups: 按照level分组的样本字典
    """
    
    print(f"\n{'=' * 80}")
    print("整体最佳性能分析（5轮中只要1轮通过就视为通过）")
    print('=' * 80)
    
    # 按照level分组分析
    for level in sorted(level_groups.keys()):
        samples = level_groups[level]
        
        print(f"\n{level.upper()}:")
        print(f"  样本数量: {len(samples)}")
        
        # 统计每个样本的最佳性能
        best_speedups = []
        passed_samples = 0
        
        for sample in samples:
            messages = sample.get('messages', [])
            num_rounds = sample.get('num_rounds', 0)
            
            # 检查前5轮中是否有任何一轮通过
            has_passed = False
            best_speedup = 0
            
            for round_idx in range(1, min(num_rounds, 5)):
                feedback = extract_round_feedback(messages, round_idx)
                
                if feedback:
                    compiled = feedback.get('compiled', False)
                    correctness = feedback.get('correctness', False)
                    speedup = feedback.get('speedup', 0)
                    
                    if compiled and correctness:
                        has_passed = True
                        if speedup > best_speedup:
                            best_speedup = speedup
            
            if has_passed:
                passed_samples += 1
                best_speedups.append(best_speedup)
        
        # 统计结果
        total_samples = len(samples)
        pass_rate = passed_samples / total_samples * 100 if total_samples > 0 else 0
        
        print(f"  通过样本数: {passed_samples}")
        print(f"  通过率: {pass_rate:.1f}%")
        
        if best_speedups:
            fast1_count = sum(1 for s in best_speedups if s >= 1.0)
            fast1_2_count = sum(1 for s in best_speedups if s >= 1.2)
            fast1_5_count = sum(1 for s in best_speedups if s >= 1.5)
            fast2_count = sum(1 for s in best_speedups if s >= 2.0)
            
            print(f"  最佳加速比分布（基于总样本数）:")
            print(f"    fast1 (>=1.0x): {fast1_count} ({fast1_count / total_samples * 100:.1f}%)")
            print(f"    fast1.2 (>=1.2x): {fast1_2_count} ({fast1_2_count / total_samples * 100:.1f}%)")
            print(f"    fast1.5 (>=1.5x): {fast1_5_count} ({fast1_5_count / total_samples * 100:.1f}%)")
            print(f"    fast2 (>=2.0x): {fast2_count} ({fast2_count / total_samples * 100:.1f}%)")
            print(f"    平均加速比: {np.mean(best_speedups):.2f}x")
            print(f"    中位数加速比: {np.median(best_speedups):.2f}x")
            print(f"    最大加速比: {np.max(best_speedups):.2f}x")
            print(f"    最小加速比: {np.min(best_speedups):.2f}x")
    
    # 总体统计
    print(f"\n{'=' * 80}")
    print("总体最佳性能统计")
    print('=' * 80)
    
    all_best_speedups = []
    all_passed_samples = 0
    all_total_samples = 0
    
    for level, samples in level_groups.items():
        all_total_samples += len(samples)
        
        for sample in samples:
            messages = sample.get('messages', [])
            num_rounds = sample.get('num_rounds', 0)
            
            has_passed = False
            best_speedup = 0
            
            for round_idx in range(1, min(num_rounds, 5)):
                feedback = extract_round_feedback(messages, round_idx)
                
                if feedback:
                    compiled = feedback.get('compiled', False)
                    correctness = feedback.get('correctness', False)
                    speedup = feedback.get('speedup', 0)
                    
                    if compiled and correctness:
                        has_passed = True
                        if speedup > best_speedup:
                            best_speedup = speedup
            
            if has_passed:
                all_passed_samples += 1
                all_best_speedups.append(best_speedup)
    
    overall_pass_rate = all_passed_samples / all_total_samples * 100 if all_total_samples > 0 else 0
    
    print(f"总样本数: {all_total_samples}")
    print(f"通过样本数: {all_passed_samples}")
    print(f"总体通过率: {overall_pass_rate:.1f}%")
    
    if all_best_speedups:
        fast1_count = sum(1 for s in all_best_speedups if s >= 1.0)
        fast1_2_count = sum(1 for s in all_best_speedups if s >= 1.2)
        fast1_5_count = sum(1 for s in all_best_speedups if s >= 1.5)
        fast2_count = sum(1 for s in all_best_speedups if s >= 2.0)
        
        print(f"\n总体最佳加速比分布（基于总样本数）:")
        print(f"  fast1 (>=1.0x): {fast1_count} ({fast1_count / all_total_samples * 100:.1f}%)")
        print(f"  fast1.2 (>=1.2x): {fast1_2_count} ({fast1_2_count / all_total_samples * 100:.1f}%)")
        print(f"  fast1.5 (>=1.5x): {fast1_5_count} ({fast1_5_count / all_total_samples * 100:.1f}%)")
        print(f"  fast2 (>=2.0x): {fast2_count} ({fast2_count / all_total_samples * 100:.1f}%)")
        print(f"  平均加速比: {np.mean(all_best_speedups):.2f}x")
        print(f"  中位数加速比: {np.median(all_best_speedups):.2f}x")
        print(f"  最大加速比: {np.max(all_best_speedups):.2f}x")
        print(f"  最小加速比: {np.min(all_best_speedups):.2f}x")

def extract_round_feedback(messages, round_idx):
    """
    从messages中提取指定轮次的反馈
    
    Args:
        messages: 消息列表
        round_idx: 轮次索引（从1开始）
    
    Returns:
        反馈字典，如果找不到则返回None
    """
    # 第round_idx轮的反馈在messages中的位置
    # messages结构: user(第1轮) -> assistant(第1轮) -> user(第2轮反馈) -> assistant(第2轮) -> ...
    # 第round_idx轮的反馈在messages中的索引是：round_idx * 2
    
    feedback_idx = round_idx * 2
    
    if feedback_idx >= len(messages):
        return None
    
    feedback_msg = messages[feedback_idx]
    
    if feedback_msg.get('role') != 'user':
        return None
    
    content = feedback_msg.get('content', '')
    
    # 从content中提取JSON反馈
    # 反馈格式: "Server feedback (status/metrics/errors):\n\n{...}"
    match = re.search(r'Server feedback.*?\n\n(\{.*\})', content, re.DOTALL)
    
    if not match:
        return None
    
    try:
        feedback = json.loads(match.group(1))
        return feedback
    except:
        return None

def categorize_error(feedback):
    """
    对错误进行分类
    
    Args:
        feedback: 反馈字典
    
    Returns:
        错误类别字符串
    """
    compiled = feedback.get('compiled', False)
    correctness = feedback.get('correctness', False)
    decoy_kernel = feedback.get('decoy_kernel', False)
    error_message = feedback.get('error_message', '')
    
    if not compiled:
        # 编译失败
        if 'compilation_error' in error_message.lower():
            return 'Compilation Error'
        else:
            return 'Compilation Failed'
    elif not correctness:
        # 正确性失败
        if 'output mismatch' in error_message.lower():
            return 'Output Mismatch'
        elif 'runtime error' in error_message.lower():
            return 'Runtime Error'
        elif 'correctness_issue' in error_message.lower():
            return 'Correctness Issue'
        else:
            return 'Correctness Failed'
    elif decoy_kernel:
        return 'Decoy Kernel'
    else:
        # 其他错误
        if error_message:
            return f'Other Error: {error_message[:50]}'
        else:
            return 'Unknown Error'

def generate_charts(level_groups, output_dir):
    """
    生成可视化图表
    
    Args:
        level_groups: 按照level分组的样本字典
        output_dir: 输出目录
    """
    
    # 设置字体（使用英文标签避免中文字体问题）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表目录
    charts_dir = os.path.join(output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # 1. 每轮通过率对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    levels = sorted(level_groups.keys())
    rounds = range(1, 5)
    
    for idx, level in enumerate(levels):
        ax = axes[idx]
        samples = level_groups[level]
        
        pass_rates = []
        for round_idx in rounds:
            messages = samples[0].get('messages', []) if samples else []
            # 计算该轮的通过率
            passed = 0
            total = 0
            for sample in samples:
                num_rounds = sample.get('num_rounds', 0)
                if round_idx < num_rounds:
                    total += 1
                    feedback = extract_round_feedback(sample.get('messages', []), round_idx)
                    if feedback and feedback.get('compiled', False) and feedback.get('correctness', False):
                        passed += 1
            
            if total > 0:
                pass_rates.append(passed / total * 100)
            else:
                pass_rates.append(0)
        
        ax.bar(rounds, pass_rates, color='steelblue', alpha=0.7)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Pass Rate (%)', fontsize=12)
        ax.set_title(f'{level.upper()} - Pass Rate by Round', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # 在柱子上显示数值
        for i, rate in enumerate(pass_rates):
            ax.text(rounds[i], rate + 2, f'{rate:.1f}%', 
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'pass_rate_by_round.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 加速比分布图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, level in enumerate(levels):
        ax = axes[idx]
        samples = level_groups[level]
        
        # 收集所有轮次的加速比
        all_speedups = []
        for sample in samples:
            messages = sample.get('messages', [])
            num_rounds = sample.get('num_rounds', 0)
            for round_idx in range(1, min(num_rounds, 5)):
                feedback = extract_round_feedback(messages, round_idx)
                if feedback and feedback.get('compiled', False) and feedback.get('correctness', False):
                    speedup = feedback.get('speedup', 0)
                    if speedup > 0:
                        all_speedups.append(speedup)
        
        if all_speedups:
            ax.hist(all_speedups, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Speedup', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'{level.upper()} - Speedup Distribution', fontsize=14, fontweight='bold')
            ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Speedup=1.0')
            ax.axvline(x=2.0, color='green', linestyle='--', linewidth=2, label='Speedup=2.0')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'speedup_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 失败原因分类图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, level in enumerate(levels):
        ax = axes[idx]
        samples = level_groups[level]
        
        # 统计所有轮次的失败原因
        error_counts = defaultdict(int)
        for sample in samples:
            messages = sample.get('messages', [])
            num_rounds = sample.get('num_rounds', 0)
            for round_idx in range(1, min(num_rounds, 5)):
                feedback = extract_round_feedback(messages, round_idx)
                if feedback:
                    compiled = feedback.get('compiled', False)
                    correctness = feedback.get('correctness', False)
                    if not compiled or not correctness:
                        error_category = categorize_error(feedback)
                        error_counts[error_category] += 1
        
        if error_counts:
            # 只显示前5个错误类别
            sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])[:5]
            categories = [e[0] for e in sorted_errors]
            counts = [e[1] for e in sorted_errors]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            ax.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title(f'{level.upper()} - Error Categories', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'error_categories.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n图表已保存到: {charts_dir}")
    print(f"  - pass_rate_by_round.png")
    print(f"  - speedup_distribution.png")
    print(f"  - error_categories.png")

if __name__ == "__main__":
    # 输出目录
    # output_dir = "/nfs/FM/gongoubo/cuda_kernel/parallel_drkernel_gpt5-4_kernelbench"
    output_dir = "/nfs/FM/gongoubo/cuda_kernel/parallel_drkernel_qwen3-5-397B_kernelbench"
    
    print("开始详细分析结果文件...")
    print("=" * 80)
    
    analyze_results_detailed(output_dir)
    
    print("=" * 80)
    print("分析完成！")
