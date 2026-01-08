"""
Portions of this file are derived from:
https://github.com/codewithaarohi/Build-a-Mini-GPT-Model-From-Scratch-Using-PyTorch
Original author: Aarohi Singla
Changes: import of packages changed, examples changed, comments added, changes to the generate logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_blocks import Block

print(
    f"""
System info:
   PyTorch version: {torch.__version__}
   CUDA available: {torch.cuda.is_available()}
   GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}   
"""
)

# 50 randomly generated sentences related to european history
sentences = [
    "people in europe lived under king rule",
    "kings in europe led war for land",
    "church power shaped europe life and rule",
    "roman empire ruled much europe land",
    "people in europe paid tax to king",
    "war changed europe city and land",
    "medieval europe had strong church power",
    "trade grew between europe city and land",
    "empire fell and new nation rose in europe",
    "kings fought war to build empire",
    "people in europe worked land for king",
    "church and king shared power in europe",
    "roman road linked europe city",
    "war in europe hurt people and land",
    "medieval king ruled city and land",
    "europe people followed church rule",
    "empire war made new europe border",
    "trade helped europe city grow",
    "king power grew in europe war",
    "people in europe faced long war",
    "church law guided europe people",
    "roman army held europe land",
    "medieval life in europe was hard",
    "nation power rose in europe",
    "war and trade shaped europe history",
    "king and church ruled europe people",
    "city life grew in medieval europe",
    "empire rule brought peace to europe land",
    "europe land changed after war",
    "people in europe built stone city",
    "church power rose in medieval europe",
    "king led people in europe war",
    "trade road crossed europe land",
    "roman law shaped europe rule",
    "medieval war burned europe city",
    "nation state formed in europe",
    "people in europe sought peace after war",
    "empire fall changed europe life",
    "king rule passed to new nation",
    "church school taught europe people",
    "europe history tells of war and king",
    "land and city fed europe people",
    "medieval europe saw empire fall",
    "trade ship linked europe land",
    "king tax burdened europe people",
    "war ended and peace came to europe",
    "roman culture lived in europe",
    "nation pride grew in europe people",
    "church bell rang in europe city",
    "europe people remember long war",
]
# create one long text string separating the sentences with the end-of-text <END> token
sentences = [s + " <END>" for s in sentences]
text = " ".join(sentences)

# unique words in the text
words = list(set(text.split()))
print(f"Unique tokens in the text: {words}")
# determine the vocabulary size (number of unique tokens)
vocab_size = len(words)
print(f"Size of the vocabulary to be embedded: {vocab_size}")
# mapping from words to integer encoding (and vice versa)
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}
print("word2idx: ", word2idx)
# create a tensor with the integer encoding of the text
data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)
print("data.shape: ", data.shape)
print("data : ", data)


# parameters for the model and training batches
block_size = 6
embedding_dim = 32
n_heads = 2
n_layers = 2
lr = 1e-2
epochs = 1500


def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# Define TinyGPT model class, as the combination of the token embedding, position embedding,
# transformer blocks, and output layer.
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim
        )  # (num_embeddings, embedding_dim)

        self.position_embedding = nn.Embedding(
            block_size, embedding_dim
        )  # (num_embeddings, embedding_dim)
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # reshape the index tensor to add a new dimension for the block_size
            idx_cond = idx[:, -block_size:]
            # forward pass
            logits, _ = self(idx_cond)
            # select the logits at the final step
            logits = logits[:, -1, :]
            # select the index with the highest probability as the next generated token index
            next_idx = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_idx), dim=1)
            # stop generating if we predict the end-of-text token
            if next_idx == word2idx["<END>"]:
                break
        return idx


# Training loop
# instantiate the model and optimiser
model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 250 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")


# Generate output predicting the next word after the token "europe"
context = torch.tensor([[word2idx["europe"]]], dtype=torch.long)
out = model.generate(context, max_new_tokens=15)

print("\nGenerated text:")
print(" ".join(idx2word[int(i)] for i in out[0]))
