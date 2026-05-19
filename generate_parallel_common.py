import contextlib
import io
import multiprocessing
import multiprocessing.pool
import sys
import traceback

import torch

# Ensure KernelGYM is in path
sys.path.insert(0, "/nfs/FM/gongoubo/cuda_kernel/KernelGYM")
from kernelgym.backend.kernelbench.cuda_agent_backend import CudaAgentBackend
from kernelgym.toolkit.kernelbench.pipeline import eval_kernel_against_ref, eval_reference_only


class NonDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(multiprocessing.pool.Pool):
    """自定义进程池，允许 worker 进程创建子进程"""

    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NonDaemonProcess
        return proc


SYSTEM_PROMPT = '''You are a PyTorch and CUDA expert. Accelerate the given PyTorch Model by creating a high-performance CUDA C++ extension, targeting the best possible performance faster than baseline.

========================
⚠️ STRICTLY FORBIDDEN
========================
NO torch operators in C++: NEVER use torch::* or torch::nn::functional::* in binding.cpp or .cu files
NO torch operations in model_new.py: Only tensor creation and your custom ops allowed
NO third-party libraries: Except cuBLAS (GEMM only) and cuDNN (Conv only)

========================
✅ ALLOWED ONLY
========================
C++: Raw CUDA kernels (for custom ops), cuBLAS (for GEMM), cuDNN (MANDATORY for Conv/ConvTranspose)
Python: torch.tensor creation, custom extension ops, tensor properties (.shape, .device)
Memory: torch::empty_like for allocation only
Focus: Implement kernels in kernels/ directory only

========================
WORKSPACE STRUCTURE
========================
.
├── binding_registry.h    # Do NOT modify - registration system
├── binding.cpp           # Do NOT modify - main module binding
├── kernels/              # YOUR WORK: Implement all kernels here
├── model.py              # DO NOT modify - Original PyTorch model
└── model_new.py          # YOUR WORK: Your optimized model using custom ops.

File Types and Usage
.cu files: CUDA kernels with __global__ functions (custom implementations)
.cpp files: cuDNN/cuBLAS API calls (NO custom kernels)
_binding.cpp files: PyTorch tensor handling and Python bindings

========================
UNIFIED WORKFLOW
========================

write CUDA kernels with __global__ functions (custom implementations)
```c++
#include <cuda_runtime.h>

// Template kernel for performance tuning
template<int BLOCK_SIZE, int TILE_SIZE>
__global__ void my_kernel_impl(float* output, const float* input, int size) {
    // Shared memory for tiling
    extern __shared__ float smem[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for large data
    for (int i = tid; i < size; i += stride) {
        // Kernel logic with optimizations
        output[i] = /* computation */;
    }
}

// C-interface launcher (no PyTorch dependencies)
extern "C" void my_kernel_launcher(
    float* output,
    const float* input,
    int size,
    int config,
    cudaStream_t stream
) {
    // Dynamic configuration selection
    int blocks = (size + 255) / 256;
    int shared_mem_size = 0;
    
    switch(config) {
        case 0: 
            shared_mem_size = 256 * sizeof(float);
            my_kernel_impl<256, 16><<<blocks, 256, shared_mem_size, stream>>>(
                output, input, size);
            break;
        case 1: 
            shared_mem_size = 128 * sizeof(float);
            my_kernel_impl<128, 32><<<blocks, 128, shared_mem_size, stream>>>(
                output, input, size);
            break;
        default:
            my_kernel_impl<256, 16><<<blocks, 256, 0, stream>>>(
                output, input, size);
    }
}
```

write apply_bings.cpp (PyTorch tensor handling and Python bindings)
```c++
// Use this two headers to replace torch/extension.h for faster compilation
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from .cu file
extern "C" void my_kernel_launcher(
    float* output,
    const float* input,
    int size,
    int config,
    cudaStream_t stream
);

// PyTorch wrapper with config parameter
torch::Tensor my_kernel_forward(torch::Tensor input, int config = 0) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    
    // Get current CUDA stream (correct way)
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call CUDA launcher with config
    my_kernel_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel(),
        config,
        stream
    );
    
    return output;
}

// Registration function
void register_my_kernel(pybind11::module& m) {
    m.def("my_kernel_forward", &my_kernel_forward, 
          "My kernel forward",
          py::arg("input"),
          py::arg("config") = 0);
}

`#include "../binding_registry.h"` is needed! 

// Auto-register
REGISTER_BINDING(my_kernel, register_my_kernel);
```

create model_new.py
```
import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, ...):  # MUST match Model signature exactly
        super().__init__()
        # Initialize parameters - preserve original structure for state_dict compatibility
        self.weight = nn.Parameter(torch.randn(...))
        self.bias = nn.Parameter(torch.zeros(...))
        
    def forward(self, x):
        # Use custom ops only - NO torch operations
        x = cuda_extension.my_kernel_forward(x, config=0)
        x = cuda_extension.gemm_forward(x, self.weight, self.bias)
        return x
```
'''

OUTPUT_PROMPT = """
========================
📦 OUTPUT FORMAT (STRICT MARKDOWN)
========================

You MUST follow this EXACT format:

### CUDA_KERNELS
```cpp
<CUDA .cu code here>

### APPLY_BINDINGS
```cpp
<apply_bindings.cpp code here>
```

### MODEL_NEW
```python
<model_new.py code here>
```

Before output:
- Ensure all 3 sections exist
- Ensure each section has valid code
- Ensure no extra text outside sections

"""

REFERENCE_PROMPT = "\nreference pytorch code:\n```python\n{}\n```"

BACK_PROMPT = '''Now you have received the server feedback for your last implementation.

CRITICAL INSTRUCTION:
Even if your previous implementation was correct and achieved a good speedup, you MUST TRY A DIFFERENT OPTIMIZATION STRATEGY in this iteration to achieve an even better performance. 
DO NOT output the exact same code as before. You should explore advanced CUDA optimization techniques such as:
- Vectorized memory accesses (e.g., float4)
- Shared memory caching and tiling
- Loop unrolling
- Thread coarsening
- Warp-level primitives
- Better grid/block configurations
and so on.

ABOUT "decoy_kernel":
In the server feedback, you might see `"decoy_kernel": true`. A "decoy kernel" means the evaluation system detected that you bypassed writing actual custom CUDA kernels and instead used PyTorch's native operators (e.g., `torch.matmul`, `F.softmax`, `torch.exp`) or their C++ equivalents (`aten::*`) as a shortcut. 
To avoid generating decoy kernels:
1. You MUST write genuine custom CUDA C++ kernels (`__global__` functions) for the core computations.
2. DO NOT use native PyTorch operators in `model_new.py` or C++ bindings to cheat the performance test. 
3. Only `cuBLAS` (for GEMM) and `cuDNN` (for Conv) are allowed as third-party calls; everything else must be implemented by you.

Here is the server feedback. Please refer to this feedback to improve the implementation:
Server feedback (status/metrics/errors):

Feel free to modify the APPLY_BINDINGS and MODEL_NEW sections if needed.

'''


def parse_response_to_json(response_content):
    sections = {
        "CUDA_KERNELS": "",
        "APPLY_BINDINGS": "",
        "MODEL_NEW": "",
    }

    current_section = None
    code_lines = []
    in_code_block = False

    lines = response_content.split("\n")

    for line in lines:
        if line.startswith("### CUDA_KERNELS"):
            if current_section and code_lines:
                sections[current_section] = "\n".join(code_lines)
            current_section = "CUDA_KERNELS"
            code_lines = []
            in_code_block = False
        elif line.startswith("### APPLY_BINDINGS"):
            if current_section and code_lines:
                sections[current_section] = "\n".join(code_lines)
            current_section = "APPLY_BINDINGS"
            code_lines = []
            in_code_block = False
        elif line.startswith("### MODEL_NEW"):
            if current_section and code_lines:
                sections[current_section] = "\n".join(code_lines)
            current_section = "MODEL_NEW"
            code_lines = []
            in_code_block = False
        elif line.startswith("```"):
            in_code_block = not in_code_block
        elif in_code_block and current_section:
            code_lines.append(line)

    if current_section and code_lines:
        sections[current_section] = "\n".join(code_lines)

    return sections


def _truncate_text(text, limit=4000):
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _serialize_metadata(metadata):
    if not isinstance(metadata, dict):
        return {}
    serialized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
        elif isinstance(value, dict):
            serialized[key] = _serialize_metadata(value)
        elif isinstance(value, list):
            serialized[key] = [
                _serialize_metadata(v) if isinstance(v, dict) else str(v)
                for v in value[:20]
            ]
        else:
            serialized[key] = str(value)
    return serialized


def _build_failed_result(error_message, *, metadata=None, error_code=None):
    return {
        "task_id": "",
        "status": "failed",
        "compiled": False,
        "correctness": False,
        "decoy_kernel": False,
        "reference_runtime": 0,
        "kernel_runtime": 0,
        "speedup": 0,
        "metadata": _serialize_metadata(metadata or {}),
        "error_message": _truncate_text(error_message, limit=8000),
        "error_code": error_code,
        "submitted_at": None,
        "completed_at": None,
        "processing_time": None,
        "status_snapshot": {},
    }


def _run_with_captured_logs(fn, *args, **kwargs):
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        value = fn(*args, **kwargs)
    logs = stdout_buffer.getvalue()
    stderr_logs = stderr_buffer.getvalue()
    if stderr_logs:
        logs = f"{logs}\n[stderr]\n{stderr_logs}" if logs else stderr_logs
    return value, logs


def _first_nonempty(*values):
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _build_detailed_error_message(metadata=None, logs=""):
    metadata = metadata or {}
    compilation_error = _first_nonempty(
        metadata.get("compilation_error"),
        metadata.get("model_load_error"),
    )
    runtime_error = _first_nonempty(
        metadata.get("runtime_error"),
        metadata.get("error_during_performance"),
    )
    correctness_issue = _first_nonempty(metadata.get("correctness_issue"))

    highlighted = []
    if compilation_error:
        highlighted.append(f"Compilation error:\n{_truncate_text(compilation_error, 5000)}")
    if runtime_error:
        highlighted.append(f"Runtime error:\n{_truncate_text(runtime_error, 5000)}")
    if correctness_issue:
        highlighted.append(f"Correctness issue:\n{_truncate_text(correctness_issue, 5000)}")
    if not highlighted and logs:
        highlighted.append(f"Evaluation logs:\n{_truncate_text(logs, 8000)}")
    if not highlighted:
        return "Unknown evaluation failure"
    return "\n\n".join(highlighted)


def _format_profiling_summary(metadata):
    metadata = metadata or {}
    profiling = metadata.get("profiling") or {}
    if not isinstance(profiling, dict) or not profiling:
        return ""

    total_cuda = profiling.get("total_cuda_time_us", 0)
    total_cpu = profiling.get("total_cpu_time_us", 0)
    kernel_count = profiling.get("kernel_count", 0)
    kernels = profiling.get("kernels") or []

    lines = [
        "Torch profile summary:",
        f"- total_cuda_time_us: {total_cuda}",
        f"- total_cpu_time_us: {total_cpu}",
        f"- kernel_count: {kernel_count}",
    ]

    top_kernels = []
    for kernel in kernels[:10]:
        if not isinstance(kernel, dict):
            continue
        name = kernel.get("name", "unknown")
        cuda_time = kernel.get("cuda_time_us", 0)
        cpu_time = kernel.get("cpu_time_us", 0)
        count = kernel.get("count", 1)
        top_kernels.append(
            f"  - {name}: cuda_time_us={cuda_time}, cpu_time_us={cpu_time}, count={count}"
        )
    if top_kernels:
        lines.append("- top_kernels:")
        lines.extend(top_kernels)
    return "\n".join(lines)


def _extract_cuda_names(cuda_kernels_code, apply_bindings_code):
    import re

    function_name_match = re.search(r'm\.def\("(\w+)"', apply_bindings_code)
    function_name = function_name_match.group(1) if function_name_match else "kernel"

    kernel_name_match = re.search(r"__global__\s+void\s+(\w+)\s*\(", cuda_kernels_code)
    kernel_name = kernel_name_match.group(1) if kernel_name_match else None

    launcher_match = re.search(r'extern\s+"C"\s+void\s+(\w+)\s*\(', apply_bindings_code)
    launcher_name = launcher_match.group(1) if launcher_match else None

    return function_name, kernel_name, launcher_name


def _fix_kernel_launch_name(cuda_kernels_code, apply_bindings_code, kernel_name, launcher_name, device_id):
    import re

    if not kernel_name or not launcher_name:
        return cuda_kernels_code, apply_bindings_code

    kernel_call_in_cuda = re.search(rf"{kernel_name}\s*<<<", cuda_kernels_code)
    kernel_call_in_binding = re.search(rf"{kernel_name}\s*<<<", apply_bindings_code)
    if kernel_call_in_cuda or kernel_call_in_binding:
        return cuda_kernels_code, apply_bindings_code

    print(f"[GPU {device_id}] ⚠️  警告: Launcher '{launcher_name}' 可能没有调用内核 '{kernel_name}'")
    print(f"[GPU {device_id}] 尝试自动修复...")

    kernel_launch_pattern = r"(\w+)\s*<<<"
    matches_cuda = list(re.finditer(kernel_launch_pattern, cuda_kernels_code))
    if matches_cuda:
        first_match = matches_cuda[0]
        old_launch = first_match.group(0)
        new_launch = f"{kernel_name}<<<"
        cuda_kernels_code = (
            cuda_kernels_code[: first_match.start()]
            + new_launch
            + cuda_kernels_code[first_match.end() :]
        )
        print(f"[GPU {device_id}] ✅ 已修复 CUDA_KERNELS 中的内核调用: {old_launch} -> {new_launch}")

    matches_binding = list(re.finditer(kernel_launch_pattern, apply_bindings_code))
    if matches_binding:
        first_match = matches_binding[0]
        old_launch = first_match.group(0)
        new_launch = f"{kernel_name}<<<"
        apply_bindings_code = (
            apply_bindings_code[: first_match.start()]
            + new_launch
            + apply_bindings_code[first_match.end() :]
        )
        print(f"[GPU {device_id}] ✅ 已修复 APPLY_BINDINGS 中的内核调用: {old_launch} -> {new_launch}")

    return cuda_kernels_code, apply_bindings_code


def evaluate_cuda_agent_example_with_device_internal(
    cuda_kernels_code,
    apply_bindings_code,
    model_new_code,
    model_ori_code,
    device_id,
    reference_result=None,
):
    """评估 CUDA-Agent 格式的 CUDA kernel 示例 (支持指定 device)"""

    function_name, kernel_name, launcher_name = _extract_cuda_names(
        cuda_kernels_code, apply_bindings_code
    )
    cuda_kernels_code, apply_bindings_code = _fix_kernel_launch_name(
        cuda_kernels_code, apply_bindings_code, kernel_name, launcher_name, device_id
    )

    cuda_sources = {
        f"{function_name}.cu": cuda_kernels_code,
        f"{function_name}_binding.cpp": apply_bindings_code,
    }

    print(f"[GPU {device_id}] 检测到函数名: {function_name}")
    if kernel_name:
        print(f"[GPU {device_id}] 检测到内核名: {kernel_name}")
    if launcher_name:
        print(f"[GPU {device_id}] 检测到 launcher: {launcher_name}")

    if reference_result is None:
        print(f"[GPU {device_id}] 评估参考实现...")
        try:
            reference_result, reference_logs = _run_with_captured_logs(
                eval_reference_only,
                original_model_src=model_ori_code,
                seed_num=42,
                num_perf_trials=100,
                verbose=False,
                device=device_id,
                entry_point="Model",
                reference_backend=None,
            )
        except Exception as e:
            print(f"[GPU {device_id}] 参考实现评估崩溃: {e}")
            return _build_failed_result(
                f"参考实现评估崩溃: {e}\n{traceback.format_exc()}",
                error_code="reference_eval_exception",
            )

        if not reference_result or not reference_result.compiled:
            print(f"[GPU {device_id}] ❌ 参考实现评估失败")
            metadata = getattr(reference_result, "metadata", {}) if reference_result else {}
            detail = (
                metadata.get("model_load_error")
                or metadata.get("runtime_error")
                or metadata.get("compilation_error")
                or "Reference evaluation returned an uncompiled result"
            )
            result = _build_failed_result(
                f"参考实现评估失败: {detail}",
                metadata=metadata,
                error_code="reference_eval_failed",
            )
            result["error_message"] = _build_detailed_error_message(metadata, reference_logs)
            return result

        print(f"[GPU {device_id}] ✅ 参考实现: {reference_result.runtime:.4f} ms")
    else:
        print(f"[GPU {device_id}] 使用预计算的参考实现结果: {reference_result.runtime:.4f} ms")

    print(f"[GPU {device_id}] 创建 CUDA-Agent Backend Adapter...")
    cuda_agent_adapter = CudaAgentBackend()

    custom_model_with_sources = f"""
### CUDA_SOURCES ###
{cuda_sources}
### END_CUDA_SOURCES ###

{model_new_code}
"""

    print(f"[GPU {device_id}] 评估生成的 CUDA Kernel...")
    try:
        kernel_result, kernel_logs = _run_with_captured_logs(
            eval_kernel_against_ref,
            original_model_src=model_ori_code,
            custom_model_src=custom_model_with_sources,
            seed_num=42,
            num_correct_trials=5,
            num_perf_trials=100,
            verbose=False,
            measure_performance=True,
            build_dir=None,
            device=device_id,
            backend="cuda_agent",
            entry_point="Model",
            enable_profiling=True,
            enable_triton_detection=False,
            backend_adapter=cuda_agent_adapter,
        )
    except Exception as e:
        print(f"[GPU {device_id}] Kernel评估崩溃: {e}")
        return _build_failed_result(
            f"Kernel评估崩溃: {e}\n{traceback.format_exc()}",
            error_code="kernel_eval_exception",
        )

    result = {
        "task_id": "",
        "status": "completed" if kernel_result is not None else "failed",
        "compiled": False,
        "correctness": False,
        "decoy_kernel": False,
        "reference_runtime": reference_result.runtime if reference_result else 0,
        "kernel_runtime": 0,
        "speedup": 0,
        "metadata": {},
        "error_message": None,
        "error_code": None,
        "submitted_at": None,
        "completed_at": None,
        "processing_time": None,
        "status_snapshot": {},
    }

    if kernel_result is None:
        result["status"] = "failed"
        result["error_code"] = "kernel_result_none"
        prefix = (
            "评估失败: eval_kernel_against_ref 返回 None。"
            "这通常意味着底层编译/加载路径提前返回，例如 lock 文件竞争、临时构建目录异常或编译加载阶段中断。"
        )
        detailed = _build_detailed_error_message({}, kernel_logs)
        result["error_message"] = _truncate_text(f"{prefix}\n\n{detailed}", limit=8000)
        return result

    result["compiled"] = kernel_result.compiled
    profiling_summary = _format_profiling_summary(getattr(kernel_result, "metadata", {}))
    if not kernel_result.compiled:
        result["error_code"] = "compilation_failed"
        detailed = _build_detailed_error_message(kernel_result.metadata, kernel_logs)
        result["error_message"] = f"{detailed}\n\n{profiling_summary}" if profiling_summary else detailed
        return result

    result["correctness"] = kernel_result.correctness
    if not kernel_result.correctness:
        result["error_code"] = "correctness_failed"
        detailed = _build_detailed_error_message(kernel_result.metadata, kernel_logs)
        result["error_message"] = f"{detailed}\n\n{profiling_summary}" if profiling_summary else detailed
        return result

    result["decoy_kernel"] = kernel_result.decoy_kernel
    result["kernel_runtime"] = kernel_result.runtime

    if kernel_result.runtime > 0:
        result["speedup"] = reference_result.runtime / kernel_result.runtime

    profiling = kernel_result.metadata.get("profiling", {})
    result["metadata"].update(
        {
            "device": kernel_result.metadata.get("device", str(device_id)),
            "gpu_name": kernel_result.metadata.get("hardware", ""),
            "backend": "cuda_agent",
            "num_perf_trials": kernel_result.metadata.get("num_perf_trials", 50),
            "hardware": kernel_result.metadata.get("hardware", ""),
            "correctness_trials": kernel_result.metadata.get("correctness_trials", ""),
            "profiling": {
                "top_10_kernels": [],
                "total_cuda_time_us": profiling.get("total_cuda_time_us", 0),
                "total_cpu_time_us": profiling.get("total_cpu_time_us", 0),
                "kernel_count": profiling.get("kernel_count", 0),
                "memory_stats": {},
            },
            "num_correct_trials": kernel_result.metadata.get("num_correct_trials", 5),
        }
    )

    for kernel in profiling.get("kernels", [])[:10]:
        result["metadata"]["profiling"]["top_10_kernels"].append(
            {
                "name": kernel.get("name", "unknown"),
                "cuda_time_us": kernel.get("cuda_time_us", 0),
                "cpu_time_us": kernel.get("cpu_time_us", 0),
                "count": kernel.get("count", 1),
            }
        )

    memory_stats = profiling.get("memory_stats", {})
    result["metadata"]["profiling"]["memory_stats"] = {
        "allocated_bytes": int(memory_stats.get("allocated_mb", 0) * 1024 * 1024),
        "reserved_bytes": int(memory_stats.get("reserved_mb", 0) * 1024 * 1024),
        "max_allocated_bytes": int(memory_stats.get("max_allocated_mb", 0) * 1024 * 1024),
    }
    if profiling_summary:
        print(f"[GPU {device_id}] {profiling_summary}")

    return result


def _sub_process_worker_wrapper(
    q,
    device_id,
    cuda_kernels_code,
    apply_bindings_code,
    model_new_code,
    model_ori_code,
    reference_result,
):
    """全局作用域的 worker 函数，供 spawn 模式下的子进程调用"""
    try:
        torch.cuda.set_device(device_id)
        res = evaluate_cuda_agent_example_with_device_internal(
            cuda_kernels_code,
            apply_bindings_code,
            model_new_code,
            model_ori_code,
            device_id,
            reference_result=reference_result,
        )
        q.put(res)
    except Exception as e:
        error_info = f"{str(e)}\n{traceback.format_exc()}"
        q.put({"status": "failed", "error_message": error_info})


def evaluate_cuda_agent_example_with_device(
    cuda_kernels_code,
    apply_bindings_code,
    model_new_code,
    model_ori_code,
    device_id,
    reference_result=None,
    timeout=600,
):
    """通过子进程评估 CUDA Kernel，彻底解决 .so 加载缓存和 CUDA 上下文问题"""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()

    p = ctx.Process(
        target=_sub_process_worker_wrapper,
        args=(
            queue,
            device_id,
            cuda_kernels_code,
            apply_bindings_code,
            model_new_code,
            model_ori_code,
            reference_result,
        ),
    )
    p.start()

    try:
        result = queue.get(timeout=timeout)
    except Exception as e:
        print(f"[GPU {device_id}] 子进程评估超时或异常: {e}")
        result = {
            "status": "failed",
            "error_message": f"Subprocess timeout or exception: {str(e)}",
        }

    p.join(timeout=5)
    if p.is_alive():
        p.terminate()

    return result


def inference(prompt, openai_client, model_id, api_params):
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]

    try:
        request_kwargs = {
            "model": model_id,
            "messages": messages,
            "max_tokens": api_params["max_tokens"],
            "temperature": api_params["temperature"],
        }

        if "reasoning_effort" in api_params:
            request_kwargs["reasoning_effort"] = api_params["reasoning_effort"]

        if api_params.get("use_top_p", True) and "top_p" in api_params:
            request_kwargs["top_p"] = api_params["top_p"]

        extra_body = {}
        if api_params.get("use_top_k", True) and "top_k" in api_params:
            extra_body["top_k"] = api_params["top_k"]
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        response = openai_client.chat.completions.create(**request_kwargs)

        message = response.choices[0].message
        try:
            think_content = response.choices[0].message.reasoning_content
        except AttributeError:
            think_content = None

        if think_content:
            main_content = message.content if message.content else ""
            return f"<think>\n{think_content}\n</think>\n\n{main_content}"
        return message.content if message.content else ""

    except Exception as e:
        print(f"Inference error: {e}")
        return ""
