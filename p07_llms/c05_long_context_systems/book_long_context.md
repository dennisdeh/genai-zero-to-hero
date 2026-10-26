## Long Context Systems
Sometimes we can fit the entire context in the memory, if the models context window is large enough.

This means that in some cases, we can avoid the need for external memory storage and simply keep the
entire conversation history in the model's memory. This can be particularly useful for applications that
require a high degree of context awareness, such as chatbots or virtual assistants.

Upsides:
- Reduced latency
- Simplified architecture

Downsides:
- Still the context window is limited compared to what is needed for many tasks
- Inefficient to do inference on long contexts (there might be much non-important information in the context)

The *context window* is the maximum number of tokens (prompt plus generated output) that a model can process
in one request. It has grown from 2,048 tokens (GPT-3, 2020) to 128k-1M tokens in current models; open models
such as Qwen3 have a native window of 32k tokens that can be extended to about 128k. This chapter looks at
what a long context costs, how well models actually use it, and what to do when the input is still too long.

### Tokens and context budgets

Everything is measured in tokens, not characters or words. The ratio depends on the tokeniser and on the
text: English prose is roughly 4 characters per token for GPT-style BPE tokenisers, while other languages,
numbers, code and JSON need more tokens per character. `main_1_token_budgeting.py` counts tokens with
`tiktoken` (exact for OpenAI models, an estimate for others - each model family has its own tokeniser) and
falls back to the 4-characters rule when the tiktoken vocabulary cannot be downloaded.

In an application the window is shared between several consumers, so we plan a **context budget**:

$$\text{window} \ge \text{system prompt} + \text{history} + \text{retrieved documents} + \text{question} + \text{reserved output}.$$

Forgetting to reserve output tokens is a classic bug: the prompt fits, but the answer is cut off. The script
computes how many retrieved chunks fit for a 4k and a 32k window.

**Ollama detail.** Ollama allocates the KV cache for a fixed context size per model instance, set by the
request option `num_ctx` (or the server variable `OLLAMA_CONTEXT_LENGTH`). The default is only a few
thousand tokens, and a longer prompt is *silently truncated*. `ChatOllama(num_ctx=...)` sets it; the helper
`get_chat_model(num_ctx=...)` passes it through. Always check `response.usage_metadata["input_tokens"]` when
working with long prompts.

### Why long contexts are expensive

**Attention compute** grows quadratically: every token attends to every previous token, so the score
matrix of a sequence of length $n$ has $n^2$ entries per head and layer. Going from 4k to 128k tokens
multiplies attention FLOPs by about 1,000. FlashAttention (Dao et al., 2022) does not change the FLOPs but
avoids materialising the $n \times n$ matrix in GPU memory, which made long contexts practical.

**KV cache memory** grows linearly. During generation the keys and values of all previous tokens are cached
so that each new token needs only one forward step. The cache size is

$$\text{KV bytes} = 2 \times n_\text{layers} \times n_\text{kv heads} \times d_\text{head} \times n_\text{tokens} \times \text{bytes per value}.$$

For Qwen3-8B (36 layers, 8 KV heads, head dimension 128, fp16) this is 144 KiB per token: 4.5 GB for 32k
tokens, 18 GB for 128k tokens - per concurrent request, on top of the weights. Architectural tricks reduce
it: **grouped-query attention** (GQA, Ainslie et al., 2023) shares each key/value head between several query
heads (8 instead of 32 KV heads = 4x less memory), sliding-window attention limits how far back layers look,
and quantising the cache to 8 bits halves it again. Serving engines such as vLLM manage the cache in pages
(PagedAttention) and **prompt caching** reuses the KV cache of a shared prefix (system prompt, documents)
across requests - which is why many APIs charge less for cached input tokens. The **prefill** (processing
the prompt) is compute bound and dominates time-to-first-token for long prompts; generation is memory bound.

### Positional encodings and RoPE scaling

Attention itself is permutation invariant, so the position of each token must be encoded. Most current
models use **rotary position embeddings (RoPE)** (Su et al., 2021): each pair of dimensions $(2i, 2i+1)$ of
the query and key vectors at position $m$ is rotated by the angle $m\theta_i$ with

$$\theta_i = b^{-2i/d}, \quad b = 10{,}000 \text{ (or larger)}.$$

Because rotations compose, the dot product $q_m^\top k_n$ depends only on the *relative* distance $m - n$.
Low dimensions rotate fast (fine, local position information), high dimensions rotate slowly (coarse,
long-range information).

A model trained on sequences up to length $L$ has never seen rotation angles beyond $L\theta_i$; feeding it
longer sequences produces out-of-distribution attention scores and quality collapses. **Context extension**
methods rescale the positions so that a longer sequence maps into the trained range, followed by a short
fine-tuning on long documents:

- **Position interpolation** (Chen et al., 2023): use position $m \cdot L / L'$ instead of $m$ - simple, but
  squeezes the fast-rotating (local) dimensions too.
- **NTK-aware scaling**: increase the base $b$ instead, which stretches mainly the slow dimensions.
- **YaRN** (Peng et al., 2023): interpolate per frequency band (high frequencies untouched, low frequencies
  interpolated) plus an attention temperature correction. Qwen3, for example, uses YaRN to go from 32k to
  ~128k tokens.

Alternatives such as ALiBi (Press et al., 2021) add a distance-dependent penalty to the attention scores
instead of rotating vectors.

### Advertised vs effective context: lost in the middle

A model *accepting* 128k tokens does not mean it *uses* them equally well. Liu et al. (2023) showed that
accuracy on multi-document question answering is highest when the relevant document is at the beginning
or the end of the context and drops markedly when it is in the middle - **lost in the middle**, a U-shaped
curve that mirrors primacy and recency effects. Causes include the training data (important information is
often at the start or end), the attention patterns favouring the first tokens ("attention sinks") and
recent tokens, and the long-range decay of RoPE.

The **needle-in-a-haystack** (NIAH) test makes this visible: insert a single fact at a relative depth of a
long filler text and ask for it, for a grid of context lengths and depths. `main_2_needle_in_haystack.py`
runs this against the local model (filler and needles are generated deterministically, `num_ctx` is set
from the prompt length) and saves a heatmap to `outputs/`. It also sends one long prompt *without* `num_ctx`
to show Ollama's silent truncation. NIAH is an easy, *retrieval-only* test - modern models usually pass it
across their window. Harder benchmarks such as RULER (multiple needles, variable tracking, aggregation) or
LongBench show that the *effective* context, where performance is still close to short-context quality, is
often much shorter than the advertised one, especially for small models and for reasoning over many facts.

### Strategies when the input is too long

When the document (or conversation) does not fit - or fits, but is too expensive or too diluted - we reduce
what the model sees. `main_3_long_document_strategies.py` compares the classic strategies on a synthetic
~8k-token annual report with 12 key figures, given a budget of 2,000 document tokens per call:

| Strategy | How | Calls | Strengths / weaknesses |
|---|---|---|---|
| truncation (head / tail / middle-out) | cut to the budget | 1 | cheap; loses everything outside the kept part |
| map-reduce | summarise chunks independently (in parallel), then combine the summaries, recursively if needed | $k$ + reduce calls | scales to any length, parallel; chunks lack global context |
| refine | running summary updated chunk by chunk | $k$ | keeps context; sequential, early details may be rewritten away |
| stuff | everything in one call with a large window | 1 | simplest and best when it fits; KV cost, lost in the middle |

The chunker (`utils/tokens.py`) is a **sliding window** over sentences with overlap, so facts on a chunk
boundary appear in both chunks. The script reports **fact recall** - the share of the 12 figures that
survive in the final summary - together with the number of LLM calls (counted with a LangChain callback)
and the run time.

For conversations, the same trade-off appears as **history management**
(`main_4_chat_history_trimming.py`):

- `trim_messages(strategy="last", max_tokens=..., include_system=True, start_on="human")` keeps the system
  prompt and the most recent turns within a token budget - cheap and predictable, but early facts are
  silently forgotten,
- **summary memory** replaces old turns by an LLM-written summary and keeps the last turns verbatim,
- **retrieval memory** stores old turns in a vector store and retrieves the relevant ones (chapter c04 and
  the agents-with-memory examples in p08).

### Long context or RAG?

Long context and retrieval-augmented generation (chapter c04) are complementary:

| | Long context | RAG |
|---|---|---|
| setup | none - put the documents in the prompt | chunking, embeddings, vector store |
| cost per query | proportional to the full context (mitigated by prompt caching) | proportional to the retrieved chunks |
| corpus size | bounded by the window | practically unbounded |
| cross-document reasoning | good: the model sees everything | limited to what is retrieved |
| failure mode | lost in the middle, distraction by irrelevant text | retrieval misses the relevant chunk |
| freshness / access control | per request | per index (filters, permissions) |

Rules of thumb: if the relevant material is small (a contract, a report, a code module) and queried a few
times, put it in the context. If the corpus is large, changes often, or is queried many times, retrieve.
Many production systems combine both: retrieve generously (tens of chunks) and let a long-context model
reason over them, ordering the most relevant chunks at the start and the end of the prompt.
