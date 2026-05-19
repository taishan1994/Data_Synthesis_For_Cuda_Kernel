# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Kernel 奖励函数实现
与 KernelServer 集成，评估内核代码的质量和性能
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any
from kernel.rewards.reward_client import KernelRewardClient


# 全局客户端实例与其配置，复用连接且在配置变更时重建
_global_client = None
_global_client_cfg = {}


def extract_reference_code(solution_str: str) -> str:
    """
    从解决方案字符串中提取参考代码
    
    Args:
        solution_str: 包含提示和响应的完整字符串
        
    Returns:
        提取的参考代码
    """
    # 查找参考实现标记
    patterns = [
        r"# Reference Implementation\s*\n(.*?)(?=# Your Task|# Generate|$)",
        r"```python\s*# Reference\s*\n(.*?)```",
        r"# PyTorch Reference:\s*\n(.*?)(?=# Task|# Generate|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # 如果没有找到特定标记，尝试提取第一个 Python 代码块
    code_block_match = re.search(r"```python\s*\n(.*?)```", solution_str, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # 回退到整个字符串
    return solution_str


def extract_kernel_code(solution_str: str) -> str:
    """
    从解决方案字符串中提取内核代码
    
    Args:
        solution_str: 包含提示和响应的完整字符串
        
    Returns:
        提取的内核代码
    """
    # 查找内核实现标记
    patterns = [
        r"# Kernel Implementation\s*\n(.*?)(?=# End|$)",
        r"```python\s*# Kernel\s*\n(.*?)```",
        r"# Your implementation:\s*\n(.*?)(?=# End|$)",
        r"# Generated kernel:\s*\n(.*?)(?=# End|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # 如果没有找到特定标记，尝试提取最后一个代码块
    code_blocks = re.findall(r"```(?:\w+)?\s*\n?(.*?)```", solution_str, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    # 回退：假设整个响应就是内核代码
    return solution_str

def _ensure_include(source: str, include_line: str) -> str:
    """Add an include to the top of a generated C++/CUDA source if missing."""
    if include_line in source:
        return source
    lines = source.splitlines()
    insert_at = 0
    while insert_at < len(lines) and (
        lines[insert_at].startswith("//")
        or lines[insert_at].startswith("/*")
        or not lines[insert_at].strip()
    ):
        insert_at += 1
    lines.insert(insert_at, include_line)
    return "\n".join(lines)


def _ensure_after_includes(source: str, line_to_add: str) -> str:
    """Insert a helper line after the include block when it is not present."""
    if line_to_add in source:
        return source
    lines = source.splitlines()
    insert_at = 0
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#include"):
            insert_at = idx + 1
    lines.insert(insert_at, line_to_add)
    return "\n".join(lines)


def _repair_launcher_kernel_mismatch(cuda_code: str, binding_code: str) -> tuple[str, str]:
    """
    Match the simple post-processing used by the SFT data generator: if the
    first declared __global__ kernel is not launched anywhere, replace the first
    launch target with that kernel name.
    """
    kernel_names = re.findall(r"__global__\s+void\s+(\w+)\s*\(", cuda_code)
    if not kernel_names:
        return cuda_code, binding_code

    kernel_name = kernel_names[0]
    if re.search(rf"\b{re.escape(kernel_name)}\s*<<<", cuda_code) or re.search(
        rf"\b{re.escape(kernel_name)}\s*<<<", binding_code
    ):
        defined = set(kernel_names)
        launched = set(re.findall(r"\b(\w+)\s*<<<", cuda_code + "\n" + binding_code))
        missing = [name for name in launched if name not in defined]
        if len(defined) == 1 and missing:
            replacement = next(iter(defined))
            for missing_name in missing:
                cuda_code = re.sub(rf"\b{re.escape(missing_name)}(?=\s*<<<)", replacement, cuda_code)
                binding_code = re.sub(rf"\b{re.escape(missing_name)}(?=\s*<<<)", replacement, binding_code)
        return cuda_code, binding_code

    launch_pattern = re.compile(r"\b(\w+)\s*<<<")
    cuda_match = launch_pattern.search(cuda_code)
    if cuda_match:
        cuda_code = (
            cuda_code[: cuda_match.start()]
            + f"{kernel_name}<<<"
            + cuda_code[cuda_match.end() :]
        )
        return cuda_code, binding_code

    binding_match = launch_pattern.search(binding_code)
    if binding_match:
        binding_code = (
            binding_code[: binding_match.start()]
            + f"{kernel_name}<<<"
            + binding_code[binding_match.end() :]
        )
    return cuda_code, binding_code


def _canonicalize_cuda_agent_sections(sections: dict[str, str]) -> dict[str, str]:
    """Apply deterministic repairs that were present in offline SFT generation."""
    cuda_code = sections.get("CUDA_KERNELS", "")
    binding_code = sections.get("APPLY_BINDINGS", "")

    cuda_code, binding_code = _repair_launcher_kernel_mismatch(cuda_code, binding_code)

    if re.search(r"\b(?:u?intptr_t|u?int(?:32|64)_t)\b", cuda_code):
        cuda_code = _ensure_include(cuda_code, "#include <cstdint>")
    if re.search(r"\b(printf|fprintf|stderr)\b", cuda_code):
        cuda_code = _ensure_include(cuda_code, "#include <cstdio>")

    if "py::arg" in binding_code and "#include <torch/csrc/utils/pybind.h>" not in binding_code:
        binding_code = _ensure_include(binding_code, "#include <torch/csrc/utils/pybind.h>")
    if re.search(r"\b(printf|fprintf|stderr)\b", binding_code):
        binding_code = _ensure_include(binding_code, "#include <cstdio>")
    if re.search(r"\b(?:u?intptr_t|u?int(?:32|64)_t)\b", binding_code):
        binding_code = _ensure_include(binding_code, "#include <cstdint>")
    if re.search(r"(?<![\w:])(?:min|max)\s*\(", binding_code):
        binding_code = _ensure_include(binding_code, "#include <algorithm>")
        binding_code = _ensure_after_includes(binding_code, "using std::max; using std::min;")

    sections["CUDA_KERNELS"] = cuda_code
    sections["APPLY_BINDINGS"] = binding_code
    return sections


def extract_cuda_agent_code(solution_str: str) -> str:
    """
    专门针对 cuda_agent 提取 CUDA_KERNELS, APPLY_BINDINGS, MODEL_NEW 并组合成 backend 所需的格式
    """
    sections = {
        "CUDA_KERNELS": "",
        "APPLY_BINDINGS": "",
        "MODEL_NEW": ""
    }
    
    current_section = None
    code_lines = []
    in_code_block = False
    
    lines = solution_str.split('\n')
    
    for line in lines:
        if line.startswith("### CUDA_KERNELS"):
            if current_section and code_lines:
                sections[current_section] = '\n'.join(code_lines)
            current_section = "CUDA_KERNELS"
            code_lines = []
            in_code_block = False
        elif line.startswith("### APPLY_BINDINGS"):
            if current_section and code_lines:
                sections[current_section] = '\n'.join(code_lines)
            current_section = "APPLY_BINDINGS"
            code_lines = []
            in_code_block = False
        elif line.startswith("### MODEL_NEW"):
            if current_section and code_lines:
                sections[current_section] = '\n'.join(code_lines)
            current_section = "MODEL_NEW"
            code_lines = []
            in_code_block = False
        elif line.startswith("```"):
            in_code_block = not in_code_block
        elif in_code_block and current_section:
            code_lines.append(line)
            
    if current_section and code_lines:
        sections[current_section] = '\n'.join(code_lines)
        
    sections = _canonicalize_cuda_agent_sections(sections)

    cuda_sources = {
        "kernel.cu": sections["CUDA_KERNELS"],
        "kernel_binding.cpp": sections["APPLY_BINDINGS"]
    }
    
    return f"""### CUDA_SOURCES ###
{json.dumps(cuda_sources)}
### END_CUDA_SOURCES ###

{sections["MODEL_NEW"]}"""

def compute_kernel_reward_batch(solution_strs: list, ground_truths: list, entry_points: str, **kwargs) -> list:
    """
    批量计算内核代码奖励值
    
    Args:
        solution_strs: 解决方案字符串列表
        ground_truths: 参考实现列表
        **kwargs: 其他参数
        
    Returns:
        奖励结果列表
    """
    try:
        # 准备任务数据
        tasks = []

        # 统一从 reward_config 读取客户端配置
        reward_config = kwargs.get("reward_config", None)
        if hasattr(reward_config, "reward_model"):
            reward_config = reward_config.reward_model
        uuids = kwargs.get("uuids", None)
        is_valid = kwargs.get("is_valid", False)

        try:
            task_timeout = getattr(reward_config, "task_timeout", None)
            task_timeout_in_client = getattr(reward_config, "task_timeout_in_client", None)
        except Exception:
            task_timeout = None
            task_timeout_in_client = None

        num_perf_trials = getattr(reward_config, "num_perf_trials")
        num_correct_trials = getattr(reward_config, "num_correct_trials")
        enable_profiling = getattr(reward_config, "enable_profiling")
        verbose_errors = getattr(reward_config, "verbose_errors")
        detect_decoy_kernel = getattr(reward_config, "detect_decoy_kernel")
        reference_backend = getattr(reward_config, "reference_backend")
        backend = getattr(reward_config, "impl_backend", getattr(reward_config, "backend", "triton"))

        for i, solution_str in enumerate(solution_strs):
            # reference_code = extract_reference_code(solution_str)
            reference_code = ground_truths[i]
            if backend == "cuda_agent":
                kernel_code = extract_cuda_agent_code(solution_str)
            else:
                kernel_code = extract_kernel_code(solution_str)
            entry_point = entry_points[i]

            if uuids is not None:
                uuid = uuids[i]



            tasks.append({
                "reference_code": reference_code,
                "kernel_code": kernel_code,
                "entry_point": entry_point,
                "use_reference_cache": False,
                "uuid": uuid if uuids is not None else "",
                "is_valid": is_valid,
                "task_timeout": task_timeout,
                "task_timeout_in_client": task_timeout_in_client,
                "num_correct_trials": num_correct_trials,
                "num_perf_trials": num_perf_trials,
                "enable_profiling": enable_profiling,
                "verbose_errors": verbose_errors,
                "detect_decoy_kernel": detect_decoy_kernel,
                "reference_backend": reference_backend,
                "backend": backend,
            })
        
        # 同步调用异步函数
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 获取客户端并批量计算奖励（仅从 reward_config 取值）
        if reward_config is None:
            raise ValueError("reward_config is required")

        server_url = getattr(reward_config, "server_url", None)
        if not server_url:
            raise ValueError("server_url is required and cannot be None or empty")

        global _global_client, _global_client_cfg
        if _global_client is None or _global_client_cfg is not reward_config:
            _global_client = KernelRewardClient(reward_config=reward_config)
            _global_client_cfg = reward_config
            
        client = _global_client
        
        # 调用时传递 task_timeout
        results = loop.run_until_complete(
            client.compute_batch_rewards(tasks, use_reference_cache=False, 
                                       is_valid=is_valid, task_timeout=task_timeout, 
                                       task_timeout_in_client=task_timeout_in_client)
        )
        
        return results
        
    except Exception as e:
        logging.error(f"Error in compute_kernel_reward_batch: {e}")
        # 返回错误结果列表
        return [
            {
                "score": reward_config.reward_policy.penalties.penalty_score,
                "reward": reward_config.reward_policy.penalties.penalty_score,
                "correctness": False,
                "success": False,
                "error": str(e)
            }
            for _ in solution_strs
        ]
