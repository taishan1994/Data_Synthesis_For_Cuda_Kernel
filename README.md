# Data_Synthesis_For_Cuda_Kernel
基于torch model生成多轮cuda kernel数据，用于SFT/RL。

天下苦kernel数据久已，本项目实现了基于torch model生成cuda kernel的数据流水线。主要是设计了cuda_agent的backend，并集成到dr kernel的KernelGYM的环境中。另外支持dr kernel的SFT以及RL。

安装环境：
```shell
cd kernelGYM
pip install -r requirements.txt
```

## CudaAgentBackend
测试cuda_agent backend：`python test_cuda_agent_backend.py`
```shell
Testing CudaAgentBackend...

1. Testing compilation...
Compiling CUDA sources in /tmp/cuda_agent_ljugoq1l/build/forced_compile with name cuda_agent_ljugoq1l
W0519 06:18:01.130000 274 torch/utils/cpp_extension.py:2425] TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
W0519 06:18:01.130000 274 torch/utils/cpp_extension.py:2425] If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'] to specific architectures.
Built so path: /tmp/cuda_agent_ljugoq1l/build/forced_compile/cuda_agent_ljugoq1l.so
Output so path: /tmp/cuda_agent_ljugoq1l/cuda_extension.so
Compilation successful!
Work directory: /tmp/cuda_agent_ljugoq1l
SO path: /tmp/cuda_agent_ljugoq1l/cuda_extension.so

2. Testing loading...
Loading successful!

3. Testing session creation...
Session created successfully!

4. Testing model creation and execution...
Model created successfully!
Forward pass successful! Output shape: torch.Size([1, 128])
Output matches expected result!

5. Testing cleanup...
Cleanup successful!

All tests passed!

```

## Profiler
测试基于cuda_agent的profile：`python test_profile.py`
```shell
[Eval] Start Evalulation! on device: 0
[Eval] Loading Original Model
[DEBUG] init inputs: []
[Eval] Original Model Loaded
[Eval] Loading and Compiling New Model with Custom CUDA Kernel
Compiling CUDA sources in /tmp/cuda_agent_oazc5emj/build/forced_compile with name cuda_agent_oazc5emj
W0519 06:29:39.897000 3836 torch/utils/cpp_extension.py:2425] TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
W0519 06:29:39.897000 3836 torch/utils/cpp_extension.py:2425] If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'] to specific architectures.
Built so path: /tmp/cuda_agent_oazc5emj/build/forced_compile/cuda_agent_oazc5emj.so
Output so path: /tmp/cuda_agent_oazc5emj/cuda_extension.so
[Eval] New Model with Custom CUDA Kernel Loaded
[Eval] Checking Correctness
[Eval] Generating Random Input with seed 734796314
device: 0
inputs: cuda:0
[PASS] trial 0: New Model matches Model
[Eval] Pass count: 1, num_correct_trials: 1
[Eval] Measuring Performance as Sample is Correct
[Profiling] Using device: 0 NVIDIA GeForce RTX 4090, warm up 3, trials 10
Trial 1: 0.0369 ms
Trial 2: 0.0285 ms
Trial 3: 0.0256 ms
Trial 4: 0.0236 ms
Trial 5: 0.0236 ms
Trial 6: 0.0236 ms
Trial 7: 0.042 ms
Trial 8: 0.0246 ms
Trial 9: 0.0235 ms
Trial 10: 0.0225 ms
[Profiling] Running 10 additional iterations for profiling...
[Profiler] Preflight CUDA op executed
[Profiler] Initializing with activities: ['ProfilerActivity.CPU', 'ProfilerActivity.CUDA']
[Profiler] Profiler started successfully
[Profiler] Context pid=%s cuda_available=%s device=%s CUDA_VISIBLE_DEVICES=%s 3836 True cuda:0 (NVIDIA GeForce RTX 4090) 
[Profiler] Profiler stopped successfully
[Profiler] key_averages: -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       aten::empty_like        17.08%     509.100us        21.49%     640.652us      64.065us       0.000us         0.00%       0.000us       0.000us           0 B           0 B     320.00 KB           0 B            10  
                                    aten::empty_strided         4.41%     131.552us         4.41%     131.552us      13.155us       0.000us         0.00%       0.000us       0.000us           0 B           0 B     320.00 KB     320.00 KB            10  
                                       cudaLaunchKernel         3.74%     111.546us        66.64%       1.987ms     198.677us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            10  
void abs_sub_kernel_coarsened<256, 4>(float*, float ...         0.00%       0.000us         0.00%       0.000us       0.000us      10.400us       100.00%      10.400us       1.040us           0 B           0 B           0 B           0 B            10  
                                Activity Buffer Request        62.90%       1.875ms        62.90%       1.875ms       1.875ms       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B             1  
                                               [memory]         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us           0 B           0 B    -320.00 KB    -320.00 KB            10  
                                  cudaDeviceSynchronize        11.87%     353.787us        11.87%     353.787us     176.893us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B             2  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.981ms
Self CUDA time total: 10.400us

[Profiling] Captured 1 CUDA kernels
[Profiling] Total CUDA time: 10.40 us
[DEBUG] profiling_metrics type: <class 'dict'>, empty: False
[DEBUG Profiling] profiling_metrics keys: dict_keys(['kernels', 'kernel_count', 'total_cpu_time_us', 'total_cuda_time_us', 'total_self_cuda_time_us', 'cuda_device_event_count', 'cuda_time_event_count', 'self_cuda_time_event_count', 'cuda_launch_api_calls', 'has_cuda_launch_api', 'event_name_sample', 'memory_stats'])
[DEBUG Profiling] kernel_count: 1
[DEBUG Profiling] triton_profiler_matches: []
[DEBUG Coverage] num_custom_kernels: 1
[DEBUG Coverage] num_total_kernels: 1
[INFO] CUDA backend: Profiler captured 1 custom kernels - accepting as valid
[Eval] Performance Stats: {'mean': 0.0274, 'std': 0.00632, 'min': 0.0225, 'max': 0.042, 'num_trials': 10}
{
  "kernels": [
    {
      "name": "void abs_sub_kernel_coarsened<256, 4>(float*, float const*, int, float)",
      "cuda_time_us": 10.399999999999864,
      "cpu_time_us": 0.0,
      "count": 10
    }
  ],
  "kernel_count": 1,
  "total_cpu_time_us": 4987.991000000002,
  "total_cuda_time_us": 10.399999999999864,
  "total_self_cuda_time_us": 0.0,
  "cuda_device_event_count": 1,
  "cuda_time_event_count": 1,
  "self_cuda_time_event_count": 0,
  "cuda_launch_api_calls": 10,
  "has_cuda_launch_api": true,
  "event_name_sample": [
    "Activity Buffer Request",
    "[memory]",
    "aten::empty_like",
    "aten::empty_strided",
    "cudaDeviceSynchronize",
    "cudaLaunchKernel",
    "void abs_sub_kernel_coarsened<256, 4>(float*, float const*, int, float)"
  ],
  "memory_stats": {
    "allocated_mb": 0.03125,
    "reserved_mb": 2.0,
    "max_allocated_mb": 0.21875,
    "max_reserved_mb": 2.0
  }
}
custom_kernel_cuda_time_in_profiling_us: 10.399999999999864

```

## 生成一条数据样例
生成一条数据用于测试流程是否有问题：`python generate_one.py`

## 批量生成数据
批量生成cuda kernel数据，支持断点续生成：`python generate_parallel_minimax.py`

如果碰到了某些意外情况导致生成的数据有问题，比如api断了，可以使用以下脚本清理错误的样本，然后重新跑失败的即可：

可以先不做删除，看看分布情况：`python cleanup_incomplete_results_advanced.py --ouput-dir xxx --dry-run`

删除掉没有跑到指定轮数的数据：`python cleanup_incomplete_results_advanced.py --ouput-dir xxx --min-rounds 5`

删除掉最终极速比小于等于某个值的数据：`python cleanup_incomplete_results_advanced.py --ouput-dir xxx --min-speed 0.0`

将数据转换为SFT需要的格式的数据：`python convert_json_to_parquet.py`

## 测试KernelBench
使用相同的方式测试不同的模型在kernnelbench上的效果：`python generate_parallel_kernelbench.py`

推理完成后可使用脚本分析结果：`python analyze_kernelbench.py`

## CudaAgent KernelGYM
启动KernelGYM沙盒环境部署cuda kernel的profile环境：

需要
- 修改：FEEDBACK_GPU_DEVICES 使用的显卡
- 修改：REDIS_HOST=部署的机器的ip地址
```shell
cd KernelGYM
bash setup.sh

bash start_all_with_monitor.sh
```

测试环境是否可用：`python3 test_api_server.py`
```shell
Sending evaluation request to KernelGYM API server...
✅ Task submitted successfully!
Task ID: test_api_1779179222
Waiting for result...
Status: completed
Task finished. Fetching result...

==================================================
EVALUATION RESULT
==================================================
✅ Correctness check: True
✅ Runtime: 0.0000 ms
✅ Profiling GPU Time: 5.76 us
✅ Coverage: Custom kernel CUDA time: 5.76us / Total time: 5.76us, Coverage: 100.00%

```

## SFT

`bash KernelGYM/drkernel/kernel/scripts/sft/8b-coldstart.sh`

主要是修改相关数据的路径以及模型路径即可。

如果不使用dr kernel自带的：即使用verl。也可以使用ms_swift下面的train.sh

## RL

`bash KernelGYM/drkernel/kernel/scripts/rl/8b_trloo_mrs_pr_prs.sh`

主要是修改相关数据的路径以及模型路径即可。

## 其它

### 生成数据时应该使用什么模型？

- 使用过GPT5.5、Claude Ops 4.7，但是这些API都不会返回思考的过程了，只有一个最终的结果，不可用。GPT5.5使用的是cliproxyapi将codex转换为api使用，可参考：https://github.com/taishan1994/python_common_code_collection/blob/main/src/codex%E4%BD%BF%E7%94%A8.md。
- 使用Qwen3-32B，模型能力太差，很难生成正确的cuda kernel代码。
- 使用GLM5，自行部署的模型速度太慢了，而且思考过程太长，导致5轮情况下，最终70%的SFT数据都超过了80k的长度，基本不可用。
- 使用minimax-2.5，思考过程比较短，且具备一定的能力，推理速度快，最终5轮数据基本上都在32k以内。

### 如何构造SFT的数据？

取5轮中加速比最高的那一轮作为最终轮，后面的轮数数据都被丢弃。

### SFT后的模型不输出think标签以及终止标签？

由于使用的是预训练的模型训练带思考标签的数据，在数据量比较少的时候，如果训练的步数过少，会导致内容虽然学习到了，但是`<think></think>`这些标签还未学习到，这是需要注意的地方。

### 多轮情况下的RL是怎么做的？
主要是有两个prompt，第一轮的prompt主要是用于引导模型生成cuda kernel，这里参考了cuda_agent的skill，将其进行了改写。后面的多轮的prompt都是一样的了，每次请求的时候会将上一轮模型的输出进行profile之后的结果拼接，再指导模型根据反馈生成对应的cuda kernel。另外，多轮的时候，think是否要拼接也是一个问题，调研了一些信息：
- deepseek-v4 pro，如果包含工具调用，才会拼接回思维链，否则只会拼接回答。
- Qwen3.5：No Thinking Content in History: In multi-turn conversations, the historical model output should only include the final output part and does not need to include the thinking content. It is implemented in the provided chat template in Jinja2. However, for frameworks that do not directly use the Jinja2 chat template, it is up to the developers to ensure that the best practice is followed. 在历史中没有思考内容：在多轮对话中，历史模型的输出应该只包括最终输出部分，不需要包括思考内容。它是在提供的 Jinja2 聊天模板中实现的。然而，对于不直接使用 Jinja2 聊天模板的框架，则由开发人员确保遵循最佳实践。
- Qwen3.6：By default, only the thinking blocks generated in handling the latest user message is retained, resulting in a pattern commonly as interleaved thinking. Qwen3.6 has been additionally trained to preserve and leverage thinking traces from historical messages. You can enable this behavior by setting the preserve_thinking option: 默认情况下，仅保留处理最新用户消息时生成的思维块，导致模式通常为交错思维。Qwen3.6 已额外训练以保留和利用历史消息中的思维轨迹。你可以通过设置 preserve_thinking 选项来启用此行为： This capability is particularly beneficial for agent scenarios, where maintaining full reasoning context can enhance decision consistency and, in many cases, reduce overall token consumption by minimizing redundant reasoning. Additionally, it can improve KV cache utilization, optimizing inference efficiency in both thinking and non-thinking modes. 这种能力对于代理场景特别有用，因为保持完整的推理上下文可以增强决策的一致性，并且在许多情况下，通过减少冗余推理来减少整体的令牌消耗。此外，它还可以提高 KV 缓存的利用率，优化推理过程中思考和非思考模式的效率。
- Kevin：使用Qwen-QWQ模型：No Thinking Content in History: In multi-turn conversations, the historical model output should only include the final output part and does not need to include the thinking content. This feature is already implemented in apply_chat_template.历史记录中没有思考内容：在多轮对话中，历史模型输出应该只包括最终输出部分，不需要包括思考内容。这个功能已经在 apply_chat_template 中实现。但是kevin在多轮RL的时候是会拼接历史的think的，如果过长的话会进行摘要。

### 数据不够怎么办？
可以合成torch model从而进一步合成cuda kernel数据，可参考：https://github.com/taishan1994/Torch_Operator_Synthesis.git

```
@misc{DSC,
  author = {Oubo Gong},
  title = {DSC: Data Synthesis For Cuda Kernel},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  url="https://github.com/taishan1994/Data_Synthesis_For_Cuda_Kernel.git",
}
```
