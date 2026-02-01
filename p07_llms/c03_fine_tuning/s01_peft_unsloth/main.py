import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# 1. Load Model and Tokenizer
# Unsloth supports optimized versions of Llama, Mistral, Phi-3, etc.
model_name = "unsloth/llama-3-8b-bnb-4bit"  # 4bit quantization for memory efficiency
max_seq_length = 2048  # Supports RoPE Scaling internally

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# 2. Set up LoRA (Parameter Efficient Fine-Tuning)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank: higher = more parameters but better learning
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
    lora_dropout=0,  # Optimized for 0
    bias="none",  # Optimized for "none"
    use_gradient_checkpointing="unsloth",  # Saves 70% VRAM
    random_state=3407,
)

# 3. Load Fine-Tuning Dataset (finance-alpaca)
# Consistent with your data.py source
dataset = load_dataset("gbharti/finance-alpaca", split="train")

# Define Alpaca-style prompt format
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


dataset = dataset.map(formatting_prompts_func, batched=True)

# 4. Set up Trainer and Perform Fine-Tuning
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Small steps for demonstration
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

print(" *** 4. Starting Fine-tuning with Unsloth *** ")
trainer_stats = trainer.train()

# 5. Validate (Inference)
print(" *** 5. Validation / Inference *** ")
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

inputs = tokenizer(
    [
        alpaca_prompt.format(
            "What is a dividend?",  # Test prompt
            "",  # Leave output blank for generation
        )
    ],
    return_tensors="pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
print(tokenizer.batch_decode(outputs))

# Save the adapter (LoRA weights)
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
