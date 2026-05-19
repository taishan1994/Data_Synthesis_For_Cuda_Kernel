import json
import multiprocessing
import os
import time
import traceback

from datasets import load_dataset
from openai import OpenAI

from generate_parallel_common import (
    BACK_PROMPT,
    OUTPUT_PROMPT,
    REFERENCE_PROMPT,
    SYSTEM_PROMPT,
    NonDaemonPool,
    evaluate_cuda_agent_example_with_device,
    inference,
    parse_response_to_json,
)


API_PARAMS = {
    "url": "http://192.168.11.18:30055",
    "top_p": 0.95,
    "top_k": 20,
    "use_top_p": False,
    "use_top_k": False,
    "temperature": 1,
    "max_tokens": 16000,
}

API_KEY = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzQzNDQyNzYsImlhdCI6MTc3MzczOTQ3NiwidWlkIjo0M30.2RLeEi0dyvr_AY9kK7_H_fRPYtnt0k_oA09mvT-JJYk"
MODEL_ID = "MiniMax-M2.5"
NUM_ITERATIONS = 5
NUM_GPUS = 2


def process_one_sample(data_idx, data_row, gpu_id, output_dir):
    """Worker function to process a single sample."""
    output_file = os.path.join(output_dir, f"result_{data_idx}.json")

    if os.path.exists(output_file):
        print(f"[GPU {gpu_id}] 跳过数据索引 {data_idx}: 结果文件已存在")
        return

    print(f"[GPU {gpu_id}] 开始处理数据索引: {data_idx}")

    openai_client = OpenAI(base_url=API_PARAMS["url"], api_key=API_KEY)

    original_python_code = data_row.get("original_python_code", "")
    if not original_python_code:
        print(f"[GPU {gpu_id}] 数据索引 {data_idx} 缺少 original_python_code, 跳过")
        return

    iteration_results = []
    messages = []

    for iteration in range(NUM_ITERATIONS):
        print(f"[GPU {gpu_id}] 数据索引 {data_idx} - 迭代 {iteration + 1}/{NUM_ITERATIONS}")

        if iteration == 0:
            user_content = f"{SYSTEM_PROMPT}\n{OUTPUT_PROMPT}\n{REFERENCE_PROMPT.format(original_python_code)}"
        else:
            prev_feedback = iteration_results[iteration - 1]["feedback"]
            feedback_str = (
                json.dumps(prev_feedback, indent=2, ensure_ascii=False)
                if prev_feedback
                else "Feedback not available."
            )
            user_content = f"{BACK_PROMPT}\n{feedback_str}"

        messages.append({"role": "user", "content": user_content})

        assistant_content = inference(messages, openai_client, MODEL_ID, API_PARAMS)
        if not assistant_content:
            print(f"[GPU {gpu_id}] 数据索引 {data_idx} - 推理失败，跳过迭代")
            break

        messages.append({"role": "assistant", "content": assistant_content})

        parsed_json = parse_response_to_json(assistant_content)
        feedback = evaluate_cuda_agent_example_with_device(
            parsed_json["CUDA_KERNELS"],
            parsed_json["APPLY_BINDINGS"],
            parsed_json["MODEL_NEW"],
            original_python_code,
            gpu_id,
            reference_result=None,
        )

        iteration_results.append(
            {
                "iteration": iteration + 1,
                "response": assistant_content,
                "parsed_json": parsed_json,
                "feedback": feedback,
            }
        )

        if feedback and feedback.get("speedup", 0) > 0:
            print(
                f"[GPU {gpu_id}] 数据索引 {data_idx} - 迭代 {iteration + 1} "
                f"加速比: {feedback['speedup']:.2f}x"
            )
        else:
            print(f"[GPU {gpu_id}] 数据索引 {data_idx} - 迭代 {iteration + 1} 评估失败或无加速")

    best_iter_idx = 0
    max_speedup = 0.0
    for i, it in enumerate(iteration_results):
        fb = it.get("feedback")
        if fb and isinstance(fb, dict):
            sp = fb.get("speedup", 0.0)
            if sp is not None and sp > max_speedup:
                max_speedup = sp
                best_iter_idx = i

    final_output = {
        "messages": messages,
        "uuid": data_idx,
        "entry_point": "Model",
        "repo_name": "",
        "module_name": "",
        "final_speedup": max_speedup,
        "num_rounds": len(iteration_results),
        "original_python_code": original_python_code,
        "best_round": best_iter_idx + 1,
        "timestamp": time.time(),
        "conversion_mode": "full_conversation_enhanced",
        "enable_thinking": False,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"[GPU {gpu_id}] 数据索引 {data_idx} 完成，结果保存至 {output_file}")


def init_worker(q):
    global gpu_queue
    gpu_queue = q


def worker_wrapper(args):
    data_idx, data_row, out_dir = args
    gpu_id = gpu_queue.get()
    try:
        process_one_sample(data_idx, data_row, gpu_id, out_dir)
    except Exception as e:
        print(f"Worker failed for idx {data_idx}: {e}")
        traceback.print_exc()
    finally:
        gpu_queue.put(gpu_id)


def main():
    parquet_path = "data/drkernel-coldstart-8k/drkernel-coldstart-8k.parquet"
    print(f"Loading dataset from {parquet_path}...")
    ds = load_dataset("parquet", data_files=parquet_path, split="train")
    print(f"Dataset loaded, total samples: {len(ds)}")
    print(f"Dataset columns: {ds.column_names}")

    ds_list = ds
    output_dir = "data/parallel_drkernel_minimax_results"
    os.makedirs(output_dir, exist_ok=True)

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    for i in range(NUM_GPUS):
        queue.put(i)

    tasks = [(i, ds_list[i], output_dir) for i in range(len(ds_list))]
    print(f"Planning to process {len(tasks)} samples using {NUM_GPUS} GPUs.")

    with NonDaemonPool(processes=NUM_GPUS, initializer=init_worker, initargs=(queue,)) as pool:
        pool.map(worker_wrapper, tasks)

    print("All tasks completed.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
