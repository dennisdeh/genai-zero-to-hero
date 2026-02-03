from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only
from trl import SFTConfig, SFTTrainer

max_seq_length = 1024
dtype = None

# 1: Load model from HuggingFace
# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit",  # 20B model using bitsandbytes 4bit quantization
    "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    "unsloth/gpt-oss-20b",  # 20B model using MXFP4 format
    "unsloth/gpt-oss-120b",
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",
    dtype=dtype,  # None for auto detection
    max_seq_length=max_seq_length,  # Choose any for long context!
    load_in_4bit=True,  # 4 bit quantization to reduce memory
    full_finetuning=False,  # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

# set LoRA settings
model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
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
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# Inference with base model
# Example 1: of output for the model
prompt = "Once upon a time, in a faraway land, there lived a king who loved adventure and exploration."
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor(tokens, dtype=torch.long, device=model.device).unsqueeze(0)
response = model.generate(input_ids, max_length=100)
print(tokenizer.decode(response[0], skip_special_tokens=False))

# Reasoning and chat templates use
messages = [
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="medium",  # low, medium or high
).to("cuda")

_ = model.generate(**inputs, max_new_tokens=20, streamer=TextStreamer(tokenizer))


# %% Training on new data
# Prepare fine-tuning data
# TODO use NVIDIA NEMO data


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


# define dataset
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(
    formatting_prompts_func,
    num_proc=1,
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
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps=10,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Use TrackIO/WandB etc
    ),
)

# train only on responses, ignore input tokens
gpt_oss_kwargs = dict(
    instruction_part="<|start|>user<|message|>",
    response_part="<|start|>assistant<|channel|>final<|message|>",
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

trainer.train()

# check a simple inference
messages = [
    {
        "role": "system",
        "content": "reasoning language: French\n\nYou are a helpful assistant that can solve mathematical problems.",
    },
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="medium",
).to("cuda")
from transformers import TextStreamer

_ = model.generate(**inputs, max_new_tokens=64, streamer=TextStreamer(tokenizer))


# save
model.save_pretrained("finetuned_model")
# model.push_to_hub("hf_username/finetuned_model", token = "hf_...") # Save to HF

# %% inference

if False:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="finetuned_model",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )

messages = [
    {
        "role": "system",
        "content": "reasoning language: French\n\nYou are a helpful assistant that can solve mathematical problems.",
    },
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="high",
).to("cuda")
from transformers import TextStreamer

_ = model.generate(**inputs, max_new_tokens=64, streamer=TextStreamer(tokenizer))
