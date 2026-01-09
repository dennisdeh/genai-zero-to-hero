import tiktoken
from datasets import load_dataset

# Use the finance-alpaca dataset as an example
ds = load_dataset("gbharti/finance-alpaca")

# Get all rows from the training split
train_data = ds["train"]

# Initialize encoding and get EOT token
encoding = tiktoken.get_encoding("cl100k_base")
eot_token = encoding.encode_ordinary("<|endoftext|>")[0]  # End of text token

# Tokenise all instructions and concatenate with EOT token
all_tokens = []
for instruction in train_data["instruction"]:
    tokens = encoding.encode(instruction)
    all_tokens.extend(tokens)
    all_tokens.append(eot_token)  # Add EOT token after each instruction

print(f"Total number of tokens: {len(all_tokens)}")
print(f"Unique tokens: {len(set(all_tokens))}")
print(f"Number of instructions tokenized: {len(train_data)}")
print(f"Average tokens per instruction: {len(all_tokens) / len(train_data):.2f}")

"""
TODO:
    - create a custom tokenizer using BPE 
"""
