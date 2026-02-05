"""
In this script we look at how to fine-tune a pre-trained GPT OSS model from HuggingFace on a new dataset using
PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation) with unsloth.

We will fine-tune the model on the finance QnA dataset prepared in step1_data_collection.py.

GPT OSS models on HuggingFace:
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    "unsloth/gpt-oss-120b-unsloth-bnb-4bit"
    "unsloth/gpt-oss-20b"
    "unsloth/gpt-oss-120b"
"""

import os
import datetime
from typing import Dict, Optional, List, Any


# 0: Set global settings and load pre-trained (and fine-tuned) models
training = False
max_seq_length = 1024
load_pt_model = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"  # HuggingFace model
load_ft_model = "model_ft_20260205_060000"
path_model = "p07_llms/c03_more_fine_tuning/s00_peft_unsloth/trained_models"

# %% Training
if training:
    # import modules
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only
    from transformers import TextStreamer
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from p07_llms.c00_gpt_like_models.s01_minigpt.data_collection import (
        data_collection as dc,
    )
    import torch

    # 1: Load model from HuggingFace
    # Load pre-trained model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_pt_model,
        dtype=None,  # None for auto-detection
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    # Ensure that the model has an explicit pad_token_id
    # If the tokeniser has no pad token, we reuse EOS for padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Prepare the model for LoRA fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # rank of adaption matrix
        target_modules=[  # modules to adapt in the transformer architecture
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=42,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    # Inference with base model
    # Example 1: direct input to the model
    prompt = "How can I make money?"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=model.device).unsqueeze(0)
    response = model.generate(input_ids, max_length=100)
    print(tokenizer.decode(response[0], skip_special_tokens=False))

    # Example 2: Reasoning and chat templates use
    messages = [
        {"role": "user", "content": "How should I best manage my money?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="low",  # low, medium or high
    ).to(model.device)

    _ = model.generate(**inputs, max_new_tokens=512, streamer=TextStreamer(tokenizer))

    # Prepare fine-tuning data
    # use the curated finance QnA dataset as an example, which is prepared in step1_data_collection.data
    d_qna = dc.dc_finance_qna(n_samples=None, threshold_skip_long_text=1000)

    def d_qna_to_sft_dataset(
        d_qna: Dict[str, str],
        *,
        system_prompt: Optional[str] = None,
        add_empty_assistant_prefix: bool = False,
    ) -> Dataset:
        """
        Convert a {"question": "answer", ...} dict into a HuggingFace Dataset
        compatible with the common SFTTrainer + chat-template pipeline.

        Output rows look like:
          {"messages": [{"role": "system", ...?}, {"role": "user", ...}, {"role": "assistant", ...}]}

        Args:
            d_qna: Mapping of question -> answer.
            system_prompt: If provided, prepended as a system message in every sample.
            add_empty_assistant_prefix: If True, inserts an empty assistant message before the
                real assistant answer (sometimes useful for specific chat templates; usually False).

        Returns:
            datasets.Dataset with a "messages" column.
        """
        rows: List[dict[str, Any]] = []

        for q, a in d_qna.items():
            if q is None or a is None:
                continue
            q = str(q).strip()
            a = str(a).strip()
            if not q or not a:
                continue
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": q})
            if add_empty_assistant_prefix:
                messages.append({"role": "assistant", "content": ""})
            messages.append({"role": "assistant", "content": a})
            rows.append({"messages": messages})

        if not rows:
            raise ValueError("No valid QnA pairs found in d_qna after cleaning.")

        return Dataset.from_list(rows)

    dataset = d_qna_to_sft_dataset(
        d_qna,
        system_prompt="You are a helpful assistant who answers questions about finance exclusively.",
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {
            "text": texts,
        }

    # # define dataset
    # dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(
        formatting_prompts_func,
        num_proc=None,
        batched=True,
    )

    # print an example
    print(dataset[0]["text"])

    # Setup supervised fine-tuning trainer object
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,  # Set this for 1 full training run.
            max_steps=1000,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="outputs",
            report_to="none",  # Use TrackIO/WandB etc
        ),
    )

    # train only on responses, ignore input tokens
    gpt_oss_kwargs = dict(
        instruction_part="<|start|>user<|message|>",
        response_part="<|start|>assistant<|message|>",
    )

    trainer = train_on_responses_only(
        trainer,
        **gpt_oss_kwargs,
    )

    # check
    tokenizer.decode(trainer.train_dataset[100]["input_ids"])
    tokenizer.decode(
        [
            tokenizer.pad_token_id if x == -100 else x
            for x in trainer.train_dataset[100]["labels"]
        ]
    ).replace(tokenizer.pad_token, " ")

    # start training loop
    trainer.train()

    # save
    model.save_pretrained(
        os.path.join(
            path_model, f"model_ft_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )


# %% Inference
else:
    from unsloth import FastLanguageModel
    from transformers import TextStreamer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=os.path.join(path_model, load_ft_model),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    messages = [
        {"role": "user", "content": "How should I best manage my money?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="low",
    ).to(model.device)

    _ = model.generate(**inputs, max_new_tokens=256, streamer=TextStreamer(tokenizer))
