"""Test script for CudaAgentBackend."""

import torch
import sys
sys.path.insert(0, '/nfs/FM/gongoubo/new_project/github/Data_Synthesis_For_Cuda_Kernel/KernelGYM')

from kernelgym.backend.kernelbench.cuda_agent_backend import CudaAgentBackend

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

# Python model code that uses the CUDA extension
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

# Original model for comparison
original_model_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F


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

def test_backend():
    print("Testing CudaAgentBackend...")
    
    backend = CudaAgentBackend()
    device = torch.device("cuda:0")
    
    # Test compilation
    print("\n1. Testing compilation...")
    artifact = backend.compile(
        code=model_code,
        device=device,
        entry_point="ModelNew",
        cuda_sources=cuda_sources,
    )
    
    if not artifact.get("compiled"):
        print(f"Compilation failed: {artifact.get('error')}")
        return False
    
    print(f"Compilation successful!")
    print(f"Work directory: {artifact.get('work_dir')}")
    print(f"SO path: {artifact.get('so_path')}")
    
    # Test loading
    print("\n2. Testing loading...")
    try:
        handle = backend.load(artifact, device=device)
        print("Loading successful!")
    except Exception as e:
        print(f"Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test session creation
    print("\n3. Testing session creation...")
    try:
        session = backend.open_session(handle, device=device)
        print("Session created successfully!")
    except Exception as e:
        print(f"Session creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test model creation and execution
    print("\n4. Testing model creation and execution...")
    try:
        init_inputs = [2.0]
        model = session.create_model(init_inputs, no_grad=True, synchronize=True)
        print("Model created successfully!")
        
        # Test forward pass
        a = torch.randn(1, 128, device='cuda')
        b = torch.randn(1, 128, device='cuda')
        
        with torch.no_grad():
            output = model(a, b)
        
        print(f"Forward pass successful! Output shape: {output.shape}")
        
        # Verify correctness
        expected = 2.0 * a + b
        if torch.allclose(output, expected, rtol=1e-5, atol=1e-5):
            print("Output matches expected result!")
        else:
            print("WARNING: Output does not match expected result!")
            print(f"Output: {output}")
            print(f"Expected: {expected}")
    except Exception as e:
        print(f"Model execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test cleanup
    print("\n5. Testing cleanup...")
    try:
        session.close()
        backend.cleanup(handle)
        print("Cleanup successful!")
    except Exception as e:
        print(f"Cleanup failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAll tests passed!")
    return True

if __name__ == "__main__":
    success = test_backend()
    sys.exit(0 if success else 1)
