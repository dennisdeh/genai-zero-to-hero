"""
In this script we will take a look at the first two steps in the general workflow of training a language model:
1. Data collection and preparation
2. Model pretraining
"""

# change working directory to p07_llms/c00_gpt_like_models/s01_minigpt if running in interactive mode:
import os
import sys

if hasattr(sys, "ps1") or hasattr(sys, "ps2"):
    os.chdir("p07_llms/c00_gpt_like_models/s01_minigpt")

import torch
from step1_data_collection.data import data_preparation
from mingpt.trainer import Trainer, estimate_loss, batch_end_callback
from mingpt.model import GPT
import gc


# %% 1: Data collection and preparation
# set block size for the model (consistency with data will be validated)
block_size = 256
# Use the finance-alpaca dataset as an example, which is prepared in step1_data_collection.data
dataset_train, dataset_val, encoding = data_preparation(block_size=block_size)
vocab_size = int(encoding.n_vocab)
print(
    f"Data preparation complete:\n"
    f"  encoding object: {encoding.__class__.__name__}\n"
    f"  vocab size: {vocab_size}\n"
    f"  block size: {block_size}\n"
    f"  number of training examples: {len(dataset_train)}\n"
    f"  number of validation examples: {len(dataset_val)}"
)


# %% 2: Model pretraining
print(" *** 2. Model pretraining *** ")
# get the device to train on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use several GPUs if available
use_multi_gpu = False
# instantiate the model
model_config = GPT.get_default_config()
model_config.model_type = "gpt2"
model_config.vocab_size = vocab_size
model_config.block_size = block_size
model = GPT(model_config)
if use_multi_gpu:
    # required if several GPUs are to be used available
    model = torch.nn.DataParallel(model, device_ids=None)
model = model.to(device)

# create a Trainer object
train_config = Trainer.get_default_config()
train_config.learning_rate = 3e-4
train_config.max_iters = 3000
train_config.batch_size = 32 if use_multi_gpu else 16
train_config.num_workers = 0
trainer = Trainer(train_config, model, dataset_train)


# define a callback to evaluate the loss on the validation set
def eval_callback(trainer):
    return batch_end_callback(trainer, dataset_val=dataset_val, device=device)


# set callbacks and clear caches
trainer.set_callback("on_batch_end", eval_callback)
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()  # releases cached memory back to CUDA driver
    torch.cuda.ipc_collect()
# run training loop
trainer.run()

# Final evaluation
final_val_loss = estimate_loss(model, dataset_val, device, eval_iters=100)
print(f"Final validation loss after training: {final_val_loss:.5f}")

# save the model
# Use .module.state_dict() if wrapped to avoid prefixing keys with "module."
raw_model = model.module if hasattr(model, "module") else model
torch.save(raw_model.state_dict(), "model.bin")


# do some simple inference
prompt = "Dividend"
tokens = encoding.encode(prompt)
input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
output_ids = raw_model.generate(input_ids, max_new_tokens=50)
print(encoding.decode(output_ids[0][len(tokens) - 1 :].tolist()))


# %% 3: Fine-tuning
