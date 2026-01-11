"""
In this script we will take a look at the first two steps in the general workflow of training a language model:
1. Data collection and preparation
2. Model pretraining
"""

# change working directory to p07_llms/c00_gpt_like_models/s01_minigpt if running in interactive mode:
# import os
# os.chdir("p07_llms/c00_gpt_like_models/s01_minigpt")

import random
import tiktoken
from datasets import load_dataset
import torch
from pp_data.dataset_class import TextDataset
from mingpt.model import GPT


# %% 1. Data collection and preparation
# Use the finance-alpaca dataset as an example (this contains only data labelled "train";
# we split it manually into train and validation sets).
# We do not do any further data preprocessing or analysis here
ds = load_dataset("gbharti/finance-alpaca", split="train")
ls_instructions = list(ds["instruction"])
print(f"Text examples: {ls_instructions[:2]}")

# Longest instruction in the dataset
max_len = max(len(instruction) for instruction in ls_instructions)
print(f"Longest instruction: {max_len} characters")

# split into train and validation sets
random.shuffle(ls_instructions)
split_idx = int(len(ls_instructions) * 0.9)
ls_train = ls_instructions[:split_idx]
ls_val = ls_instructions[split_idx:]
print(f"Training samples: {len(ls_train)}, Validation samples: {len(ls_val)}")

# Initialize encoding and get EOT token
encoding = tiktoken.get_encoding("cl100k_base")
eot_token = encoding.encode_ordinary("<|endoftext|>")[0]  # End of text token

# Tokenise all data and concatenate with EOT token
data_train = []
max_tokens = 0
for instruction in ls_train:
    tokens = encoding.encode(instruction)
    max_tokens = max(max_tokens, len(tokens))
    data_train.extend(tokens)
    data_train.append(eot_token)  # Add EOT token after each instruction

data_val = []
for instruction in ls_val:
    tokens = encoding.encode(instruction)
    max_tokens = max(max_tokens, len(tokens))
    data_val.extend(tokens)
    data_val.append(eot_token)
all_tokens = data_train + data_val
# Determine the block size (number of tokens in a context)
block_size = 256
assert (
    max_tokens <= block_size
), "The block size must be larger than the maximum number of tokens in the data"
print(f"Block size: {block_size} tokens > {max_tokens} maximum tokens in data")

# instantiate a custom dataset classes
dataset_train = TextDataset(data_train, block_size)
dataset_val = TextDataset(data_val, block_size)

# Get vocabulary size (number of unique tokens)
vocab_size = encoding.n_vocab
print(
    f"""
Statistics of training and validation data:
    Total number of tokens: {len(data_train) + len(data_val)}
    Unique tokens: {len(set(data_train + data_val))}
    Vocabulary size: {encoding.n_vocab}
    Average number of tokens per sentence (training): {len(data_train)/len(ls_train):.2f}
    Average number of tokens per sentence (validation): {len(data_val)/len(ls_val):.2f}
"""
)


# %% 2: Model pretraining
# get device to train on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = TextDataset(all_tokens, block_size)
# instantiate the model
model_config = GPT.get_default_config()
model_config.model_type = "gpt2"
model_config.vocab_size = vocab_size
model_config.block_size = block_size
model = GPT(model_config)
model = model.to(device)

# create a Trainer object
from mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = (
    5e-4  # the model we're using is so small that we can go a bit faster
)
train_config.max_iters = 3000
train_config.batch_size = 16
train_config.num_workers = 0
trainer = Trainer(train_config, model, dataset_train)


@torch.no_grad()
def estimate_loss(model, dataset, device, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        # Sample a random batch from the dataset
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]
        x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
        logits, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()


def batch_end_callback(trainer):
    str_print = ""
    if trainer.iter_num % 10 == 0:
        str_print = f"iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"

    if trainer.iter_num % 100 == 0:
        val_loss = estimate_loss(model, dataset_val, device)
        str_print += f" -- validation loss {val_loss:.5f}"

    if str_print != "":
        print(str_print)


trainer.set_callback("on_batch_end", batch_end_callback)
trainer.set_callback("on_batch_end", batch_end_callback)

import gc

gc.collect()
torch.cuda.empty_cache()  # releases cached memory back to CUDA driver
torch.cuda.ipc_collect()

trainer.run()

# Final evaluation
final_val_loss = estimate_loss(model, dataset_val, device, eval_iters=100)
print(f"Final validation loss after training: {final_val_loss:.5f}")

# save the model
torch.save(model.state_dict(), "model.bin")


# do some simple inference
prompt = "The account"
tokens = encoding.encode(prompt)
input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
output_ids = model.generate(input_ids, max_new_tokens=50)
print(encoding.decode(output_ids[0][len(tokens) - 1 :].tolist()))


# %% 3: Fine-tuning
