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
Kernel 奖励管理器，专门用于 kernel code RL 训练
复用 laser 的架构，集成 KernelServer 进行性能评估
"""

from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import re
import socket
from datetime import datetime, timezone

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


# @register("kernel")
class AsyncKernelRewardManager:
    """Kernel 奖励管理器，集成 KernelServer 进行内核性能评估"""

    def __init__(
        self,
        tokenizer,
        num_examine=5,
        compute_score=None,
        reward_fn_key="data_source",
        reward_config=None,
        **kwargs
    ) -> None:
        """
        初始化 KernelRewardManager
        
        Args:
            tokenizer: 分词器
            num_examine: 打印到控制台的样本数量
            compute_score: 自定义评分函数
            reward_fn_key: 用于识别数据源的键
            reward_config: Hydra/OmegaConf 下的 reward_model 配置（唯一客户端配置载体）
            **kwargs: 其他参数
        """

        if hasattr(reward_config, "reward_model"):
            reward_config = reward_config.reward_model

        self.reward_config = reward_config
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.is_valid = kwargs.get("is_valid", False)
        self.server_url = self.reward_config.server_url
        self.reward_policy = self.reward_config.reward_policy
        self.task_timeout = self.reward_config.task_timeout
        self.print_status = getattr(self.reward_config, "print_status", False)
        
        # 验证 server_url 不为空
        if not self.server_url:
            raise ValueError("server_url is required for KernelRewardManager")

        self.reward_weights = self.reward_config.reward_weights
        
        self.logger = logging.getLogger(__name__)
        self.sample_log_dir = self._resolve_sample_log_dir()
        self.sample_log_file = self._resolve_sample_log_file(self.sample_log_dir)

        # 打印配置信息（全部来源于 reward_config）
        self.logger.info(f"KernelRewardManager initialized with server: {self.server_url}")
        self.logger.info(f"Reward weights: {self.reward_weights}")
        if self.sample_log_file is not None:
            self.logger.info(f"Kernel sample logs will be saved to: {self.sample_log_file}")
        try:
            enhanced = self.reward_config.enhanced
            use_sandbox_rate_limit = self.reward_config.use_sandbox_rate_limit
            rate_limit = self.reward_config.rate_limit
            timeout = self.reward_config.timeout
            max_concurrent = self.reward_config.max_concurrent
            print(f"[RewardManager] cfg enhanced={enhanced} use_sandbox_rate_limit={use_sandbox_rate_limit} rate_limit={rate_limit} timeout={timeout} max_concurrent={max_concurrent}")
        except Exception:
            pass

    def _resolve_sample_log_dir(self) -> Path | None:
        sample_log_dir = os.getenv("RL_SAMPLE_LOG_DIR", "").strip()
        if not sample_log_dir:
            checkpoint_dir = os.getenv("CHECKPOINT_DIR", "").strip()
            if checkpoint_dir:
                sample_log_dir = os.path.join(checkpoint_dir, "logs", "samples")
        if not sample_log_dir:
            return None
        path = Path(sample_log_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_sample_log_file(self, sample_log_dir: Path | None) -> Path | None:
        if sample_log_dir is None:
            return None
        hostname = socket.gethostname()
        return sample_log_dir / f"samples_{hostname}_{os.getpid()}.jsonl"

    def _extract_response_sections(self, response_str: str) -> dict[str, str]:
        sections = {
            "CUDA_KERNELS": "",
            "APPLY_BINDINGS": "",
            "MODEL_NEW": "",
        }
        current_section = None
        code_lines = []
        in_code_block = False

        for line in (response_str or "").splitlines():
            if line.startswith("### CUDA_KERNELS"):
                if current_section is not None:
                    sections[current_section] = "\n".join(code_lines).strip()
                current_section = "CUDA_KERNELS"
                code_lines = []
                in_code_block = False
            elif line.startswith("### APPLY_BINDINGS"):
                if current_section is not None:
                    sections[current_section] = "\n".join(code_lines).strip()
                current_section = "APPLY_BINDINGS"
                code_lines = []
                in_code_block = False
            elif line.startswith("### MODEL_NEW"):
                if current_section is not None:
                    sections[current_section] = "\n".join(code_lines).strip()
                current_section = "MODEL_NEW"
                code_lines = []
                in_code_block = False
            elif line.startswith("```"):
                in_code_block = not in_code_block
            elif in_code_block and current_section is not None:
                code_lines.append(line)

        if current_section is not None:
            sections[current_section] = "\n".join(code_lines).strip()
        return sections

    def _classify_failure(self, response_str: str, result: dict, sections: dict[str, str]) -> tuple[str, str]:
        error = str(result.get("error") or "").strip()
        status = str(result.get("status") or "unknown")
        compiled = bool(result.get("compiled", False))
        correctness = bool(result.get("correctness", False))
        success = bool(result.get("success", False))

        if success:
            return "success", "Evaluation completed successfully"

        missing_sections = [name for name, content in sections.items() if not content.strip()]
        if "missing class ModelNew" in error:
            if "MODEL_NEW" in missing_sections:
                return "missing_model_new_section", "Missing MODEL_NEW section in raw model output"
            return "missing_model_new_class", "MODEL_NEW section exists but class ModelNew was not found"
        if "missing class Model" in error:
            return "missing_model_class", "Reference code is missing class Model"
        if missing_sections:
            return "format_missing_sections", f"Missing required sections: {', '.join(missing_sections)}"
        if "Kernel compilation failed" in error or "error:" in error or "undefined" in error:
            if "kernel_binding.cpp" in error or "binding" in error.lower():
                return "binding_compile_error", "Binding C++ compilation failed"
            if ".cu" in error or "cuda" in error.lower():
                return "cuda_compile_error", "CUDA kernel compilation failed"
            return "compile_error", "Compilation failed during environment execution"
        if "syntaxerror" in error.lower() or "invalid syntax" in error.lower():
            return "model_new_syntax_error", "Generated Python code in ModelNew has a syntax error"
        if compiled and not correctness:
            return "correctness_failure", "Code compiled but failed correctness checks"
        if status == "timeout" or "timeout" in error.lower():
            return "timeout", "Environment evaluation timed out"
        if status == "failed":
            return "task_failed", "KernelGYM backend reported task failure"
        if not response_str.strip():
            return "empty_response", "Model returned an empty response"
        return "unknown_failure", "Unclassified failure"

    def _write_sample_log(
        self,
        *,
        response_ids: list[int],
        response_str: str,
        ground_truth: str,
        entry_point: str,
        uuid: str,
        result: dict,
        reward_extra_info: dict,
        prompt_messages=None,
        generation_finish_reason=None,
        rollout_finish_reason=None,
    ) -> None:
        if self.sample_log_file is None:
            return

        try:
            conversation_messages = None
            if isinstance(prompt_messages, list):
                conversation_messages = list(prompt_messages)
                if response_str is not None:
                    conversation_messages.append({"role": "assistant", "content": response_str})
            sections = self._extract_response_sections(response_str)
            missing_sections = [name for name, content in sections.items() if not content.strip()]
            failure_category, issue_summary = self._classify_failure(response_str, result, sections)
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "uuid": uuid,
                "entry_point": entry_point,
                "response_token_count": len(response_ids),
                "response_first_token_id": response_ids[0] if response_ids else None,
                "response_last_token_id": response_ids[-1] if response_ids else None,
                "prompt_messages": prompt_messages,
                "conversation_messages": conversation_messages,
                "num_prompt_messages": len(prompt_messages) if isinstance(prompt_messages, list) else None,
                "num_conversation_messages": len(conversation_messages) if isinstance(conversation_messages, list) else None,
                "reference_code": ground_truth,
                "raw_model_output": response_str,
                "parsed_sections": sections,
                "missing_sections": missing_sections,
                "contains_cuda_kernels_section": bool(sections["CUDA_KERNELS"].strip()),
                "contains_apply_bindings_section": bool(sections["APPLY_BINDINGS"].strip()),
                "contains_model_new_section": bool(sections["MODEL_NEW"].strip()),
                "status": result.get("status"),
                "reward": result.get("score", result.get("reward")),
                "speedup": result.get("speedup"),
                "success": result.get("success"),
                "compiled": result.get("compiled"),
                "correctness": result.get("correctness"),
                "error": result.get("error"),
                "generation_finish_reason": generation_finish_reason,
                "rollout_finish_reason": rollout_finish_reason,
                "failure_category": failure_category,
                "issue_summary": issue_summary,
                "reward_extra_info": {
                    "num_custom_kernel": reward_extra_info.get("num_custom_kernel"),
                    "num_total_kernels": reward_extra_info.get("num_total_kernels"),
                    "num_coverage": reward_extra_info.get("num_coverage"),
                    "time_coverage": reward_extra_info.get("time_coverage"),
                    "is_decoy_kernel": reward_extra_info.get("is_decoy_kernel"),
                    "is_speedup_positive": reward_extra_info.get("is_speedup_positive"),
                },
                "env_result": result,
            }
            with self.sample_log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            self.logger.warning("Failed to write RL sample log: %s", exc)

    def execute_env(self, response_str: str, ground_truth: str, entry_point: str, uuid: str, response_ids: list[int]):
        """
        Execute the environment and return the result
        We split it since we hope to re-evaluate when the speedup value is anomaly large.
        """
        
        try:
            # 准备批量计算的参数
            solution_strs = [response_str]
            ground_truths = [ground_truth]
            entry_points = [entry_point]
            uuids = [uuid]
            
            # 调用评分函数
            if hasattr(self.compute_score, '__call__'):
                # 检查是否支持批量处理（更稳健地识别 partial 包裹的真实函数）
                is_batch = False
                func_name = ''
                # 直接标记优先
                if getattr(self.compute_score, "_is_batch", False):
                    is_batch = True
                # 尝试从 partial 的 raw_fn 中获取标记或名称
                underlying_func = None
                if hasattr(self.compute_score, 'func'):
                    # functools.partial(func, *args, **kwargs) 中的 func
                    underlying_func = self.compute_score.func
                    if getattr(underlying_func, "_is_batch", False):
                        is_batch = True
                # 对于 _call_with_kwargs 这类包装，raw_fn 通常在 partial.args[0]
                if hasattr(self.compute_score, 'args') and self.compute_score.args:
                    possible_raw_fn = self.compute_score.args[0]
                    if callable(possible_raw_fn):
                        underlying_func = possible_raw_fn
                        if getattr(underlying_func, "_is_batch", False):
                            is_batch = True
                # 名称兜底判断
                if hasattr(self.compute_score, '__name__'):
                    func_name = self.compute_score.__name__
                elif underlying_func is not None and hasattr(underlying_func, '__name__'):
                    func_name = underlying_func.__name__
                if 'batch' in func_name.lower():
                    is_batch = True

                # 仅传递必要控制参数：reward_config 与 is_valid
                safe_kwargs = {"reward_config": self.reward_config, "is_valid": self.is_valid}

                if is_batch:
                    results = self.compute_score(
                        solution_strs, ground_truths, entry_points,
                        uuids=uuids,
                        **safe_kwargs
                    )
                else:
                    # 单个处理
                    results = []
                    for i, (solution_str, ground_truth, entry_point) in enumerate(zip(solution_strs, ground_truths, entry_points)):
                        uuid_val = uuids[i] if i < len(uuids) else None
                        single_kwargs = {**safe_kwargs, "entry_point": entry_point, "uuid": uuid_val}
                        result = self.compute_score(
                            solution_str=solution_str,
                            ground_truth=ground_truth,
                            **single_kwargs
                        )
                        results.append(result)
            else:
                # 使用默认评分函数
                results = []
                for i, (solution_str, ground_truth, entry_point) in enumerate(zip(solution_strs, ground_truths, entry_points)):
                    uuid_val = uuids[i] if i < len(uuids) else None
                    result = default_compute_score(
                        solution_str=solution_str,
                        ground_truth=ground_truth,
                        entry_point=entry_point,
                        uuid=uuid_val,
                        is_valid=self.is_valid,
                    )
                    results.append(result)
            
        except Exception as e:
            self.logger.error(f"Error in reward computation: {e}")
            results = [
                {
                    "score": self.reward_config.reward_policy.penalties.penalty_score,
                    "reward": self.reward_config.reward_policy.penalties.penalty_score,
                    "correctness": False,
                    "success": False,
                    "compiled": False,
                    "error": str(e),
                    "num_custom_kernel": 0,
                    "num_total_kernels": 0,
                    "custom_kernel_cuda_time_in_profiling_us": 0,
                    "total_kernel_run_time_in_profiling_us": 0,
                }
                for _ in range(len(response_ids))
            ]
        
        if len(results) != 1:
            raise ValueError(f"The length of results should be 1, but got {len(results)}")
        
        return results

    def _extract_batch_field(self, data: DataProto, index: int, field: str, default=None):
        if field in data.non_tensor_batch:
            value = data.non_tensor_batch[field][index]
            return value.item() if hasattr(value, "item") else value

        if "reward_model" in data.non_tensor_batch:
            reward_model = data.non_tensor_batch["reward_model"][index]
            if isinstance(reward_model, str):
                try:
                    reward_model = json.loads(reward_model)
                except Exception:
                    reward_model = {}
            if isinstance(reward_model, dict):
                value = reward_model.get(field, default)
                return value.item() if hasattr(value, "item") else value

        return default

    def _build_batch_rewards(self, data: DataProto):
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info_list = [{} for _ in range(len(data))]

        response_ids_batch = data.batch["responses"]
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        response_strs = self.tokenizer.batch_decode(response_ids_batch, skip_special_tokens=True)

        for i in range(len(data)):
            response_ids = response_ids_batch[i][: valid_response_length[i].item()].tolist()
            response_str = response_strs[i]
            ground_truth = self._extract_batch_field(data, i, "ground_truth", "")
            entry_point = self._extract_batch_field(data, i, "entry_point", "Model")
            uuid = self._extract_batch_field(data, i, "uuid", None)
            if uuid is None:
                uuid = self._extract_batch_field(data, i, "uid", f"batch_{i}")

            result = self.__call__(
                response_ids=response_ids,
                response_str=response_str,
                ground_truth=ground_truth,
                entry_point=entry_point,
                uuid=str(uuid),
                return_dict=True,
                return_full_state=False,
            )

            sample_reward_tensor = result["reward_tensor"]
            copy_len = min(sample_reward_tensor.shape[0], reward_tensor.shape[1])
            reward_tensor[i, :copy_len] = sample_reward_tensor[:copy_len]
            reward_extra_info = {
                k: v
                for k, v in result["reward_extra_info"].items()
                if not k.endswith("_tensor")
            }
            reward_extra_info_list[i] = reward_extra_info

        return {
            "reward_tensor": reward_tensor,
            "extra_info": {"reward_extra_info": reward_extra_info_list},
        }

    # def __call__(self, data: DataProto, return_dict: bool = False, **kwargs):
    def __call__(self, 
                response_ids: list[int] | DataProto, 
                response_str: str | None = None, 
                ground_truth: str | None = None, 
                entry_point: str | None = None, 
                uuid: str | None = None, 
                return_dict: bool = True,
                return_full_state: bool = False,
                **kwargs):
        """
        Async reward manager for kernel code RL training

        Only pass necessary data to the reward manager to keep efficient in async mode.
        
        Args:
            response_ids: Response token ids
            response_str: Response string
            ground_truth: Ground truth
            entry_point: Entry point
            uuid: UUID
            return_dict: Whether to return a dictionary
            **kwargs: Additional keyword arguments for the reward function
        Returns:
            Reward tensor or a dictionary containing reward information
        """
        if isinstance(response_ids, DataProto):
            return self._build_batch_rewards(response_ids)

        assert response_str is not None
        assert ground_truth is not None
        assert entry_point is not None
        assert uuid is not None

        # 如果已经有 rm_scores，直接返回
        # if "rm_scores" in data.batch.keys():
        #     if return_dict:
        #         return {"reward_tensor": data.batch["rm_scores"]}
        #     else:
        #         return data.batch["rm_scores"]

        # 初始化返回张量，长度与响应截断长度一致，避免在后续裁剪时丢失分数
        max_response_length = kwargs.get("response_length")
        valid_response_length = len(response_ids)
        if max_response_length is not None:
            valid_response_length = min(valid_response_length, int(max_response_length))
        valid_response_length = max(valid_response_length, 1)

        reward_tensor = torch.zeros(valid_response_length, dtype=torch.float32)
        # reward_extra_info = defaultdict(list)
        reward_extra_info = {}

        # 性能指标张量
        correctness_tensor = torch.zeros(1, dtype=torch.float32)
        performance_tensor = torch.zeros(1, dtype=torch.float32)
        compilation_tensor = torch.zeros(1, dtype=torch.float32)
        
        already_print_data_sources = {}
        
        print(f"[DEBUG] entry point in reward manager: {entry_point}")
        
        # 使用计算函数进行评估

        results = self.execute_env(response_str, ground_truth, entry_point, uuid, response_ids)

        speedup = results[0].get("speedup", 0.0)

        if speedup is None:
            speedup = 0.0

        if speedup > self.reward_config.speedup_reward_upper_bound:
            print(f"[DEBUG] speedup is anomaly large, re-execute the environment")
            results = self.execute_env(response_str, ground_truth, entry_point, uuid, response_ids)
            speedup = results[0].get("speedup", 0.0)

        results = results[0]

        score = results.get("score", results.get("reward", 0.0))
        num_custom_kernel = results.get("num_custom_kernel", 0)
        num_total_kernels = results.get("num_total_kernels", 0)
        custom_kernel_cuda_time_in_profiling_us = results.get("custom_kernel_cuda_time_in_profiling_us", 0)
        total_kernel_run_time_in_profiling_us = results.get("total_kernel_run_time_in_profiling_us", 0)
        correctness = results.get("correctness", False)
        success = results.get("success", False)
        compiled = results.get("compiled", False)
        speedup = results.get("speedup", 0.0)
        if speedup is None:
            speedup = 0.0
        status = results.get("status", "unknown")
        err_msg = results.get("error")
        is_speedup_positive = (speedup >= 1.0 + self.reward_config.speedup_eps)
        is_decoy_kernel = results.get("decoy_kernel", False)

        target_index = valid_response_length - 1
        reward_tensor[target_index] = score
        correctness_tensor[0] = float(correctness)
        performance_tensor[0] = speedup
        compilation_tensor[0] = float(compiled)

        reward_extra_info["correctness"] = correctness
        reward_extra_info["performance"] = speedup
        reward_extra_info["is_speedup_positive"] = is_speedup_positive
        reward_extra_info["is_decoy_kernel"] = is_decoy_kernel
        reward_extra_info["compilation"] = compiled
        reward_extra_info["success"] = success
        reward_extra_info["status"] = status
        reward_extra_info["error"] = err_msg
        
        print(f"[DEBUG] num_custom_kernel in reward manager: {num_custom_kernel}")
        print(f"[DEBUG] num_total_kernels in reward manager: {num_total_kernels}")
        print(f"[DEBUG] custom_kernel_cuda_time_in_profiling_us in reward manager: {custom_kernel_cuda_time_in_profiling_us}")
        print(f"[DEBUG] total_kernel_run_time_in_profiling_us in reward manager: {total_kernel_run_time_in_profiling_us}")
        # new features
        reward_extra_info["num_custom_kernel"] = num_custom_kernel
        reward_extra_info["num_total_kernels"] = num_total_kernels
        num_coverage = 0
        if num_total_kernels > 0:
            num_coverage = num_custom_kernel / num_total_kernels
        reward_extra_info["num_coverage"] = float(f"{num_coverage:.2f}")
        reward_extra_info["custom_kernel_cuda_time_in_profiling_us"] = custom_kernel_cuda_time_in_profiling_us
        reward_extra_info["total_kernel_run_time_in_profiling_us"] = total_kernel_run_time_in_profiling_us
        time_coverage = 0
        if total_kernel_run_time_in_profiling_us > 0:
            time_coverage = custom_kernel_cuda_time_in_profiling_us / total_kernel_run_time_in_profiling_us
        reward_extra_info["time_coverage"] = float(f"{time_coverage:.2f}")

        self._write_sample_log(
            response_ids=response_ids,
            response_str=response_str,
            ground_truth=ground_truth,
            entry_point=entry_point,
            uuid=uuid,
            result=results,
            reward_extra_info=reward_extra_info,
            prompt_messages=kwargs.get("prompt_messages"),
            generation_finish_reason=kwargs.get("generation_finish_reason"),
            rollout_finish_reason=kwargs.get("rollout_finish_reason"),
        )

        # reward_extra_info["correctness"].append(correctness)
        # reward_extra_info["performance"].append(speedup)
        # reward_extra_info["is_speedup_positive"].append(is_speedup_positive)
        # reward_extra_info["is_decoy_kernel"].append(is_decoy_kernel)
        # reward_extra_info["compilation"].append(compiled)
        # reward_extra_info["success"].append(success)
        # reward_extra_info.setdefault("status", []).append(status)
        # reward_extra_info.setdefault("error", []).append(err_msg or "")

        if self.print_status:
            self.logger.info(f"[KernelEvalStatus] idx={0} status={status} compiled={compiled} correct={correctness} speedup={speedup} uuid={uuid} entry={entry_point} error={err_msg}")

        if return_dict:
            reward_extra_info["correctness_tensor"] = correctness_tensor
            reward_extra_info["performance_tensor"] = performance_tensor
            reward_extra_info["compilation_tensor"] = compilation_tensor
            

            return_dict = {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }

            if return_full_state:
                return_dict["env_state"] = results

            return return_dict
        else:
            if return_full_state:
                return reward_tensor, reward_extra_info, results
            else:
                return reward_tensor, reward_extra_info
