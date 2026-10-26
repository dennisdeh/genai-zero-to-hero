"""
In this script we look at how to fine-tune an instruction-tuned LLM with reinforcement learning, using
GRPO (Group Relative Policy Optimisation, Shao et al. 2024) from TRL together with Unsloth (4-bit + LoRA).

Instead of imitating reference answers (supervised fine-tuning, see s00_peft_unsloth), the model samples a
group of answers for each GSM8K maths question and is rewarded by simple, verifiable rules:
    - correctness: does the number inside <answer>...</answer> equal the GSM8K reference answer?
    - format: does the output follow the <reasoning>...</reasoning><answer>...</answer> template?
    - integer: is the answer a plain integer?
    - XML tag counting: partial credit for each correctly placed tag.
The reward functions live in utils/rewards.py (pure Python, unit-tested on CPU in test_rewards.py).

Steps (training=True):
    1. Load a 4-bit base model with Unsloth and add LoRA adapters
    2. Load and format GSM8K with a system prompt describing the required output format
    3. Evaluate the model *before* RL on a handful of test questions (accuracy + format compliance)
    4. Train with GRPOTrainer and the rule-based reward functions
    5. Evaluate *after* RL on the same questions and compare
    6. Save the LoRA adapters (and tokeniser) to outputs/
With training=False the saved adapters are loaded and evaluated (inference only).

Requirements: an NVIDIA GPU (~12-16 GB VRAM for the 3B model with the defaults below) and the conda env
p07_llms/c03_more_fine_tuning/conda_env_gzh_llm3.yml (unsloth==2026.1.4, trl==0.24.0).

Run from the repo root:
    python -m p07_llms.c03_more_fine_tuning.s01_rl_unsloth.main_grpo_unsloth

Other small instruct models that work the same way (from HuggingFace):
    "unsloth/Qwen2.5-1.5B-Instruct"     (less VRAM, weaker)
    "unsloth/Qwen2.5-3B-Instruct"       (default)
    "unsloth/Llama-3.2-3B-Instruct"
"""

import os
import json
import datetime
from typing import Any, Dict, List

from p07_llms.c03_more_fine_tuning.s01_rl_unsloth.utils.rewards import (
    REWARD_FUNCS,
    build_prompt,
    format_gsm8k_example,
    score_responses,
)

# 0: Set global settings
training = True  # True: train with GRPO and save adapters; False: load saved adapters and evaluate
load_pt_model = "unsloth/Qwen2.5-3B-Instruct"  # HuggingFace model
load_ft_model = "grpo_lora_20260101_000000"  # folder in outputs/ with saved adapters (for training=False)
path_outputs = os.path.join(os.path.dirname(__file__), "outputs")

max_seq_length = 1024  # prompt + completion tokens
max_prompt_length = 256
max_completion_length = max_seq_length - max_prompt_length
lora_rank = 16  # rank r of the LoRA matrices (higher = more capacity, more memory)
lora_alpha = 16  # scaling of the LoRA update (alpha / r)
max_steps = 250  # GRPO optimisation steps (~1-2 h on a single consumer GPU without vLLM)
num_generations = 4  # group size G: completions sampled per prompt to compute group-relative advantages
n_eval = 20  # number of GSM8K test questions for the before/after evaluation
seed = 3407


def generate_responses(model: Any, tokenizer: Any, questions: List[str], max_new_tokens: int) -> List[str]:
    """
    Generate one greedy response per question using the chat template and the GRPO system prompt.

    Args:
        model: Unsloth/HF causal LM (already switched to inference mode).
        tokenizer: matching tokeniser with a chat template.
        questions: list of maths questions.
        max_new_tokens: maximum number of generated tokens per answer.

    Returns:
        List of decoded responses (only the newly generated text).
    """
    responses = []
    for question in questions:
        inputs = tokenizer.apply_chat_template(
            build_prompt(question),
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding for a reproducible evaluation
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


def evaluate(model: Any, tokenizer: Any, eval_rows: List[Dict[str, Any]], label: str) -> Dict[str, float]:
    """
    Evaluate accuracy and format compliance on a few GSM8K test rows and print an example.

    Args:
        model: model to evaluate.
        tokenizer: matching tokeniser.
        eval_rows: rows with "question" and "answer" (normalised reference number).
        label: name printed with the results (e.g. "before GRPO").

    Returns:
        dict with "accuracy", "format_rate" and "n" (see utils.rewards.score_responses).
    """
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)  # enable Unsloth's fast generation path
    questions = [row["question"] for row in eval_rows]
    references = [row["answer"] for row in eval_rows]
    responses = generate_responses(model, tokenizer, questions, max_new_tokens=max_completion_length)
    metrics = score_responses(responses, references)

    print(f"\n===== Evaluation {label} =====")
    print(f"Question: {questions[0]}\nReference answer: {references[0]}\nModel response:\n{responses[0]}")
    print(
        f"-> accuracy: {metrics['accuracy']:.2%}, format compliance: {metrics['format_rate']:.2%} "
        f"(n={metrics['n']})"
    )
    return metrics


def load_eval_rows(n: int) -> List[Dict[str, Any]]:
    """Load the first n GSM8K test questions with normalised reference answers."""
    from datasets import load_dataset

    test = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    rows = []
    for example in test:
        formatted = format_gsm8k_example(example)
        rows.append({"question": example["question"], "answer": formatted["answer"]})
    return rows


if __name__ == "__main__":
    os.makedirs(path_outputs, exist_ok=True)

    # %% Training
    if training:
        # import modules (unsloth must be imported before trl/transformers so that its patches apply)
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer

        # 1: Load the pre-trained model in 4-bit and add LoRA adapters
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=load_pt_model,
            max_seq_length=max_seq_length,
            dtype=None,  # None for auto-detection
            load_in_4bit=True,  # QLoRA: frozen 4-bit base weights + trainable LoRA adapters
            fast_inference=False,  # True requires vLLM (faster generation); not part of the conda env
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",  # saves memory for the long sampled completions
            random_state=seed,
        )

        # 2: Load and format the GSM8K training data
        # Each row becomes {"prompt": [system msg, user msg], "answer": "72"}; "answer" is passed to the
        # reward functions as a keyword argument by GRPOTrainer.
        train_dataset = load_dataset("openai/gsm8k", "main", split="train")
        train_dataset = train_dataset.map(format_gsm8k_example, remove_columns=["question"])
        print(train_dataset[0])

        # 3: Evaluate before RL
        # The LoRA matrix B is initialised to zero, so the adapted model is still identical to the base model.
        eval_rows = load_eval_rows(n_eval)
        metrics_before = evaluate(model, tokenizer, eval_rows, label="before GRPO")

        # 4: GRPO training
        FastLanguageModel.for_training(model)
        training_args = GRPOConfig(
            # optimisation
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            max_grad_norm=0.1,
            max_steps=max_steps,
            # batching: each step uses per_device_train_batch_size completions, i.e. 1 prompt x G completions.
            # The generation batch (batch size x gradient accumulation) must be divisible by num_generations.
            per_device_train_batch_size=num_generations,
            gradient_accumulation_steps=1,
            num_generations=num_generations,
            # generation
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            temperature=1.0,  # sampling diversity within a group is what creates a learning signal
            use_vllm=False,
            # GRPO objective
            beta=0.04,  # KL penalty towards the reference policy (LoRA disabled); 0.0 switches it off
            scale_rewards="group",  # advantage = (r - group mean) / group std, as in the GRPO paper
            loss_type="dapo",  # token-level loss normalisation (TRL default); "grpo" = original, length-biased
            # logging / saving
            logging_steps=1,
            save_steps=max_steps,
            report_to="none",
            seed=seed,
            output_dir=os.path.join(path_outputs, "grpo_checkpoints"),
        )
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=REWARD_FUNCS,
            args=training_args,
            train_dataset=train_dataset,
        )
        # Watch the logged "reward" (mean total reward) and "rewards/<func>/mean" columns: the format rewards
        # usually rise within the first ~50-100 steps, correctness more slowly.
        trainer.train()

        # 5: Evaluate after RL
        metrics_after = evaluate(model, tokenizer, eval_rows, label="after GRPO")
        print("\n===== Before vs after GRPO =====")
        for key in ["accuracy", "format_rate"]:
            print(f"{key:12s}: {metrics_before[key]:.2%} -> {metrics_after[key]:.2%}")

        # 6: Save the LoRA adapters, tokeniser and metrics
        path_model = os.path.join(
            path_outputs, f"grpo_lora_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        model.save_pretrained(path_model)  # saves only the (small) LoRA adapters
        tokenizer.save_pretrained(path_model)
        with open(os.path.join(path_model, "eval_metrics.json"), "w") as f:
            json.dump({"before": metrics_before, "after": metrics_after}, f, indent=2)
        print(f"Saved LoRA adapters to {path_model}")

    # %% Inference
    else:
        from unsloth import FastLanguageModel

        # Unsloth reads adapter_config.json, loads the 4-bit base model and attaches the saved adapters
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=os.path.join(path_outputs, load_ft_model),
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        evaluate(model, tokenizer, load_eval_rows(n_eval), label="of the GRPO model")

        question = "A shop sells pencils at 3 for $1. How many dollars do 27 pencils cost?"
        print(f"\nQuestion: {question}")
        print(generate_responses(model, tokenizer, [question], max_new_tokens=max_completion_length)[0])
