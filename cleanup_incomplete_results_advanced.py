import os
import json
import argparse

def cleanup_incomplete_results(output_dir, dry_run=False, min_rounds=5, min_speedup=0.0):
    """
    删除没有完全跑完指定轮数的文件，以及最终加速比小于指定值的文件
    
    Args:
        output_dir: 输出目录路径
        dry_run: 是否只显示将要删除的文件而不实际删除
        min_rounds: 最少需要的轮数
        min_speedup: 最小需要的加速比
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
    print(f"清理条件:")
    print(f"  - 最少轮数: {min_rounds}")
    print(f"  - 最小加速比: {min_speedup}")
    print(f"  - 模式: {'预览模式 (不实际删除)' if dry_run else '实际删除模式'}")
    print("=" * 80)
    
    deleted_count = 0
    incomplete_count = 0
    low_speedup_count = 0
    
    # 统计信息
    speedup_distribution = {
        "0": 0,
        "0 < x <= 0.5": 0,
        "0.5 < x <= 1.0": 0,
        "1.0 < x <= 2.0": 0,
        "2.0 < x <= 5.0": 0,
        "x > 5.0": 0
    }
    
    rounds_distribution = {}
    
    for json_file in json_files:
        file_path = os.path.join(output_dir, json_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            should_delete = False
            reason = ""
            
            # 检查是否跑完指定轮数
            num_rounds = data.get('num_rounds', 0)
            if num_rounds < min_rounds:
                should_delete = True
                reason = f"未跑完{min_rounds}轮 (实际: {num_rounds} 轮)"
                incomplete_count += 1
            
            # 检查最终加速比
            final_speedup = data.get('final_speedup', 0)
            if final_speedup < min_speedup:
                should_delete = True
                if reason:
                    reason += f", 加速比 < {min_speedup}"
                else:
                    reason = f"加速比 < {min_speedup}"
                low_speedup_count += 1
            
            # 统计加速比分布
            if final_speedup == 0:
                speedup_distribution["0"] += 1
            elif final_speedup <= 0.5:
                speedup_distribution["0 < x <= 0.5"] += 1
            elif final_speedup <= 1.0:
                speedup_distribution["0.5 < x <= 1.0"] += 1
            elif final_speedup <= 2.0:
                speedup_distribution["1.0 < x <= 2.0"] += 1
            elif final_speedup <= 5.0:
                speedup_distribution["2.0 < x <= 5.0"] += 1
            else:
                speedup_distribution["x > 5.0"] += 1
            
            # 统计轮数分布
            rounds_distribution[num_rounds] = rounds_distribution.get(num_rounds, 0) + 1
            
            # 删除或预览文件
            if should_delete:
                deleted_count += 1
                if dry_run:
                    print(f"🔍 将删除: {json_file}")
                    print(f"   原因: {reason}")
                else:
                    os.remove(file_path)
                    print(f"✅ 已删除: {json_file}")
                    print(f"   原因: {reason}")
            else:
                print(f"✓ 保留: {json_file} (轮数: {num_rounds}, 加速比: {final_speedup:.2f}x)")
        
        except Exception as e:
            print(f"❌ 读取文件 {json_file} 失败: {e}")
            continue
    
    print("=" * 80)
    print(f"清理完成！")
    print(f"  - 总文件数: {len(json_files)}")
    print(f"  - {'将删除' if dry_run else '已删除'}: {deleted_count}")
    print(f"    - 未跑完{min_rounds}轮: {incomplete_count}")
    print(f"    - 加速比 < {min_speedup}: {low_speedup_count}")
    print(f"  - 保留: {len(json_files) - deleted_count}")
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("统计信息:")
    print("=" * 80)
    
    print("\n加速比分布:")
    for range_name, count in speedup_distribution.items():
        percentage = (count / len(json_files)) * 100 if json_files else 0
        print(f"  {range_name}: {count} ({percentage:.1f}%)")
    
    print("\n轮数分布:")
    for rounds in sorted(rounds_distribution.keys()):
        count = rounds_distribution[rounds]
        percentage = (count / len(json_files)) * 100 if json_files else 0
        print(f"  {rounds} 轮: {count} ({percentage:.1f}%)")
    
    # 计算保留文件的平均加速比
    if not dry_run:
        retained_files = [f for f in json_files if os.path.exists(os.path.join(output_dir, f))]
        if retained_files:
            total_speedup = 0
            valid_count = 0
            for json_file in retained_files:
                try:
                    file_path = os.path.join(output_dir, json_file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    speedup = data.get('final_speedup', 0)
                    if speedup > 0:
                        total_speedup += speedup
                        valid_count += 1
                except:
                    continue
            
            if valid_count > 0:
                avg_speedup = total_speedup / valid_count
                print(f"\n保留文件的平均加速比: {avg_speedup:.2f}x (基于 {valid_count} 个有效文件)")

def main():
    parser = argparse.ArgumentParser(description='清理不完整的结果文件')
    parser.add_argument('--output-dir', type=str, 
                       default='/nfs/FM/gongoubo/cuda_kernel/parallel_drkernel_minimax_results',
                       help='输出目录路径')
    parser.add_argument('--dry-run', action='store_true',
                       help='预览模式，只显示将要删除的文件而不实际删除')
    parser.add_argument('--min-rounds', type=int, default=5,
                       help='最少需要的轮数 (默认: 5)')
    parser.add_argument('--min-speedup', type=float, default=0.0,
                       help='最小需要的加速比 (默认: 0.0)')
    
    args = parser.parse_args()
    
    print("开始清理不完整的结果文件...")
    
    cleanup_incomplete_results(
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        min_rounds=args.min_rounds,
        min_speedup=args.min_speedup
    )
    
    print("=" * 80)
    print("清理完成！")

if __name__ == "__main__":
    main()




