import os
import re
from pathlib import Path

def extract_name_from_filename(filename):
    """
    从文件名中提取名称
    例如: "100_HingeLoss.py" -> "100_HingeLoss"
    """
    # 移除 .py 后缀，保留数字前缀
    name = filename.replace('.py', '')
    return name

def read_file_content(filepath):
    """
    读取文件内容
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def build_kernelbench_dict():
    """
    构建 KernelBench 字典
    """
    kernelbench_dir = Path("/nfs/FM/gongoubo/cuda_kernel/KernelBench/KernelBench")
    
    result = {}
    
    # 遍历 level1, level2, level3
    for level in ["level1", "level2", "level3"]:
        level_dir = kernelbench_dir / level
        
        if not level_dir.exists():
            print(f"Warning: {level_dir} does not exist")
            continue
        
        # 获取所有 .py 文件
        py_files = sorted(level_dir.glob("*.py"))
        
        level_dict = {}
        
        for py_file in py_files:
            # 提取名称
            name = extract_name_from_filename(py_file.name)
            
            # 读取文件内容
            content = read_file_content(py_file)
            
            if content is not None:
                level_dict[name] = content
        
        result[level] = level_dict
        
        print(f"Processed {level}: {len(level_dict)} files")
    
    return result

def main():
    print("Building KernelBench dictionary...")
    print("=" * 80)
    
    kernelbench_dict = build_kernelbench_dict()
    
    print("=" * 80)
    print(f"Total levels: {len(kernelbench_dict)}")
    
    for level, kernels in kernelbench_dict.items():
        print(f"  {level}: {len(kernels)} kernels")
    
    print("=" * 80)
    print("\nSample entries:")
    print("-" * 80)
    
    # 显示一些示例
    for level in ["level1", "level2", "level3"]:
        if level in kernelbench_dict and kernelbench_dict[level]:
            first_key = list(kernelbench_dict[level].keys())[0]
            print(f"\n{level}['{first_key}']:")
            print(f"  Content length: {len(kernelbench_dict[level][first_key])} characters")
            print(f"  First 200 chars: {kernelbench_dict[level][first_key][:200]}...")
    
    return kernelbench_dict

if __name__ == "__main__":
    kernelbench_dict = main()
    
    # 保存到文件
    import json
    
    # 由于内容可能很大，我们保存为 JSON 格式
    output_file = "/nfs/FM/gongoubo/cuda_kernel/kernelbench_dict.json"
    
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(kernelbench_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Saved successfully!")
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("Statistics:")
    print("=" * 80)
    total_files = 0
    total_chars = 0
    
    for level, kernels in kernelbench_dict.items():
        level_files = len(kernels)
        level_chars = sum(len(content) for content in kernels.values())
        total_files += level_files
        total_chars += level_chars
        
        print(f"{level}:")
        print(f"  Files: {level_files}")
        print(f"  Total characters: {level_chars:,}")
        print(f"  Average characters per file: {level_chars // level_files if level_files > 0 else 0:,}")
    
    print(f"\nTotal:")
    print(f"  Files: {total_files}")
    print(f"  Total characters: {total_chars:,}")
