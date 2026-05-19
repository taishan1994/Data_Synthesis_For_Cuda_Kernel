import requests
import json
import time
import sys

# Test CUDA code - axpby kernel
cuda_sources = {
    "axpby.cu": '''
#include <cuda_runtime.h>

template<int THREADS>
__global__ void axpby_kernel(float* out, const float* a, const float* b, float alpha, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < size; i += stride) {
        out[i] = alpha * a[i] + b[i];
    }
}

extern "C" void axpby_launcher(
    float* out,
    const float* a,
    const float* b,
    float alpha,
    int size,
    int config,
    cudaStream_t stream
) {
    if (size <= 0) return;
    switch (config) {
        case 1: {
            int threads = 128;
            int blocks = (size + threads - 1) / threads;
            axpby_kernel<128><<<blocks, threads, 0, stream>>>(out, a, b, alpha, size);
            break;
        }
        case 2: {
            int threads = 512;
            int blocks = (size + threads - 1) / threads;
            axpby_kernel<512><<<blocks, threads, 0, stream>>>(out, a, b, alpha, size);
            break;
        }
        default: {
            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            axpby_kernel<256><<<blocks, threads, 0, stream>>>(out, a, b, alpha, size);
            break;
        }
    }
}
''',
    "axpby_binding.cpp": '''
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void axpby_launcher(
    float* out,
    const float* a,
    const float* b,
    float alpha,
    int size,
    int config,
    cudaStream_t stream
);

static torch::Tensor axpby_forward(torch::Tensor a, torch::Tensor b, double alpha, int config = 0) {
    TORCH_CHECK(a.is_cuda(), "a must be CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be CUDA tensor");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

    auto out = torch::empty_like(a);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    axpby_launcher(
        out.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        static_cast<float>(alpha),
        static_cast<int>(a.numel()),
        config,
        stream
    );
    return out;
}

static void register_axpby(pybind11::module& m) {
    m.def("axpby_forward", &axpby_forward, py::arg("a"), py::arg("b"), py::arg("alpha"), py::arg("config") = 0);
}

REGISTER_BINDING(axpby, register_axpby);
'''
}

model_code = '''
import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return cuda_extension.axpby_forward(a, b, self.alpha, 0)
'''

original_model_code = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return self.alpha * a + b

def get_inputs():
    a = torch.randn(1, 128, device='cuda')
    b = torch.randn(1, 128, device='cuda')
    return [a, b]

def get_init_inputs():
    return [2.0]
'''

custom_model_with_sources = f"""
### CUDA_SOURCES ###
{json.dumps(cuda_sources)}
### END_CUDA_SOURCES ###

{model_code}
"""

def test_api():
    print("Sending evaluation request to KernelGYM API server...")
    url = "http://192.168.16.4:8001/evaluate"
    
    payload = {
        "task_id": f"test_api_{int(time.time())}",
        "reference_code": original_model_code,
        "kernel_code": custom_model_with_sources,
        "backend": "cuda_agent",  # We patched Backend enum to accept cuda_agent
        "toolkit": "kernelbench",
        "backend_adapter": "kernelbench",
        "entry_point": "Model",
        "num_correct_trials": 2,
        "num_perf_trials": 5,
        "num_warmup": 2,
        "timeout": 300,
        "enable_profiling": True,
        "enable_triton_detection": False
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        print("✅ Task submitted successfully!")
        task_id = data.get("task_id")
        print(f"Task ID: {task_id}")
        print("Waiting for result...")
        
        # Poll for status
        status_url = f"http://192.168.16.4:8001/status/{task_id}"
        result_url = f"http://192.168.16.4:8001/results/{task_id}"
        
        while True:
            time.sleep(2)
            status_resp = requests.get(status_url)
            status_data = status_resp.json()
            status = status_data.get("status")
            print(f"Status: {status}")
            
            if status in ["completed", "failed", "timeout"]:
                print("Task finished. Fetching result...")
                result_resp = requests.get(result_url)
                result_data = result_resp.json()
                
                print("\n" + "="*50)
                print("EVALUATION RESULT")
                print("="*50)
                if status == "completed":
                    print("✅ Correctness check:", result_data.get("correctness"))
                    print(f"✅ Runtime: {result_data.get('runtime', 0):.4f} ms")
                    
                    profiling = result_data.get("metadata", {}).get("profiling", {})
                    metadata = result_data.get("metadata", {})
                    if profiling:
                        print(f"✅ Profiling GPU Time: {metadata.get('custom_kernel_cuda_time_in_profiling_us', 0):.2f} us")
                        print(f"✅ Coverage: {metadata.get('custom_kernel_cuda_time_coverage', 'N/A')}")
                else:
                    print("❌ Task failed!")
                    print("Error metadata:", json.dumps(result_data.get("metadata", {}), indent=2))
                break
                
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")

if __name__ == "__main__":
    test_api()
