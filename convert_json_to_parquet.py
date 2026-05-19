import os
import json
import pandas as pd
from pathlib import Path


def keep_best_round_only(data):
    """只保留最终加速比最大的那轮及之前的对话数据。"""
    best_round = data.get("best_round")
    messages = data.get("messages")

    if not isinstance(best_round, int) or best_round <= 0:
        return data
    if not isinstance(messages, list):
        return data

    keep_message_count = best_round * 2
    data["messages"] = messages[:keep_message_count]
    data["num_rounds"] = min(best_round, len(data["messages"]) // 2)
    return data


def convert_json_to_parquet(source_dir, output_file):
    """
    将JSON文件转换为parquet格式
    
    Args:
        source_dir: 源JSON文件目录
        output_file: 输出parquet文件路径
    """
    
    source_path = Path(source_dir)
    
    # 获取所有JSON文件
    json_files = sorted(source_path.glob("*.json"))
    
    if not json_files:
        print(f"在目录 {source_dir} 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 读取所有JSON文件
    all_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data = keep_best_round_only(data)
            data["enable_thinking"] = True
            all_data.append(data)
        except Exception as e:
            print(f"❌ 读取文件 {json_file.name} 失败: {e}")
            continue
    
    print(f"成功读取 {len(all_data)} 个文件")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"DataFrame形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为parquet格式
    df.to_parquet(output_file, index=False)
    
    print(f"✅ 成功保存到 {output_file}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    source_dir = "data/parallel_drkernel_minimax_results"
    output_file = "data/parallel_drkernel_minimax_results/drkernel.parquet"
    
    print("开始转换JSON文件到Parquet格式...")
    print("=" * 80)
    
    convert_json_to_parquet(source_dir, output_file)
    
    print("=" * 80)
    print("转换完成！")