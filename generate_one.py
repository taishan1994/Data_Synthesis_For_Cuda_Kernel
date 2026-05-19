import json

from datasets import load_dataset
from openai import OpenAI

from generate_parallel_common import (
    BACK_PROMPT,
    OUTPUT_PROMPT,
    REFERENCE_PROMPT,
    SYSTEM_PROMPT,
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
NUM_GPUS = 8
DEVICE_ID = 0
DATASET_PATH = "data/drkernel-coldstart-8k"
SAMPLE_INDEX = 0
SAVE_FILE = "data/generate_one"


def save_parsed_json(parsed_json, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, ensure_ascii=False, indent=2)
    print(f"JSON output saved to {output_file}")


def main():
    openai_client = OpenAI(base_url=API_PARAMS["url"], api_key=API_KEY)

    # models = openai_client.models.list()
    # print(models)
    # model_id = models.data[0].id
    # print(model_id)

    model_id = MODEL_ID

    ds = load_dataset(DATASET_PATH, split="train")
    prompt = ds["original_python_code"][SAMPLE_INDEX]

    messages = []
    iteration_results = []

    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration + 1}/{NUM_ITERATIONS}")
        print(f"{'=' * 60}\n")

        if iteration == 0:
            user_content = f"{SYSTEM_PROMPT}\n{OUTPUT_PROMPT}\n{REFERENCE_PROMPT.format(prompt)}"
        else:
            feedback = iteration_results[iteration - 1]["feedback"]
            feedback_str = (
                json.dumps(feedback, indent=2, ensure_ascii=False)
                if feedback
                else "Feedback not available."
            )
            user_content = f"{BACK_PROMPT}\n{feedback_str}"

        messages.append({"role": "user", "content": user_content})

        assistant_content = inference(messages, openai_client, model_id, API_PARAMS)
        if not assistant_content:
            print(f"Iteration {iteration + 1}: inference failed, stopping.")
            break

        print(f"Response content:\n{assistant_content}\n")
        messages.append({"role": "assistant", "content": assistant_content})

        parsed_json = parse_response_to_json(assistant_content)
        save_parsed_json(parsed_json, f"{SAVE_FILE}/output_iteration_{iteration + 1}.json")
        print(f"Parsed JSON:\n{json.dumps(parsed_json, indent=2, ensure_ascii=False)}\n")

        print(f"Evaluating iteration {iteration + 1}...")
        feedback = evaluate_cuda_agent_example_with_device(
            parsed_json["CUDA_KERNELS"],
            parsed_json["APPLY_BINDINGS"],
            parsed_json["MODEL_NEW"],
            prompt,
            DEVICE_ID,
            reference_result=None,
        )
        print(f"Evaluation result:\n{feedback}\n")

        iteration_results.append(
            {
                "iteration": iteration + 1,
                "response": assistant_content,
                "parsed_json": parsed_json,
                "feedback": feedback,
            }
        )

    print(f"\n{'=' * 60}")
    print("ALL ITERATIONS COMPLETED")
    print(f"{'=' * 60}\n")

    for i, result in enumerate(iteration_results):
        print(f"\n--- Iteration {i + 1} Summary ---")
        feedback = result["feedback"]
        if isinstance(feedback, dict):
            if "speedup" in feedback:
                print(f"Speedup: {feedback['speedup']:.2f}x")
            if "correctness" in feedback:
                print(f"Correctness: {feedback['correctness']}")

    if iteration_results:
        best_iteration = max(
            iteration_results,
            key=lambda x: x["feedback"].get("speedup", 0)
            if isinstance(x["feedback"], dict)
            else 0,
        )
        print(f"\nBest iteration: {best_iteration['iteration']}")

    with open("{SAVE_FILE}/all_iterations_results.json", "w", encoding="utf-8") as f:
        json.dump(iteration_results, f, ensure_ascii=False, indent=2)
    print("All iteration results saved to all_iterations_results.json")


if __name__ == "__main__":
    main()
