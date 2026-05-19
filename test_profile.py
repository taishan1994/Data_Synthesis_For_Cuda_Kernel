import sys
import json
sys.path.insert(0, '/nfs/FM/gongoubo/new_project/github/Data_Synthesis_For_Cuda_Kernel/KernelGYM')
from kernelgym.backend.kernelbench.cuda_agent_backend import CudaAgentBackend
from kernelgym.toolkit.kernelbench.pipeline import eval_kernel_against_ref

import pandas as pd
df = pd.read_parquet('/nfs/FM/gongoubo/new_project/github/Data_Synthesis_For_Cuda_Kernel/data/drkernel-coldstart-8k/drkernel-coldstart-8k.parquet')
sample = df.iloc[0]
model_ori_code = sample['original_python_code']

# Find generated code for this from drkernel_formatted_output.json
data = json.load(open('/nfs/FM/gongoubo/new_project/github/Data_Synthesis_For_Cuda_Kernel/data/drkernel_formatted_output.json'))
generated_resp = None
for msg in data['messages']:
    if msg['role'] == 'assistant':
        generated_resp = msg['content']

def parse_response_to_json(response_content):
    sections = {
        "CUDA_KERNELS": "",
        "APPLY_BINDINGS": "",
        "MODEL_NEW": ""
    }
    
    current_section = None
    code_lines = []
    in_code_block = False
    
    lines = response_content.split('\n')
    
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
    
    return sections

parsed_json = parse_response_to_json(generated_resp)
axpby_cu_code = parsed_json["CUDA_KERNELS"]
axpby_binding_code = parsed_json["APPLY_BINDINGS"]
model_new_code = parsed_json["MODEL_NEW"]

cuda_sources = {
    "axpby.cu": axpby_cu_code,
    "axpby_binding.cpp": axpby_binding_code
}

custom_model_with_sources = f"""
### CUDA_SOURCES ###
{cuda_sources}
### END_CUDA_SOURCES ###

{model_new_code}
"""

adapter = CudaAgentBackend()
result = eval_kernel_against_ref(
    original_model_src=model_ori_code,
    custom_model_src=custom_model_with_sources,
    seed_num=42,
    num_correct_trials=1,
    num_perf_trials=10,
    verbose=True,
    measure_performance=True,
    build_dir=None,
    device=0,
    backend="cuda_agent",
    entry_point="Model",
    enable_profiling=True,
    enable_triton_detection=False,
    backend_adapter=adapter,
)

print(json.dumps(result.metadata.get('profiling', {}), indent=2))
print("custom_kernel_cuda_time_in_profiling_us:", result.metadata.get('custom_kernel_cuda_time_in_profiling_us'))
