## Running LLMs

In this chapter we look at various ways to run LLMs.

We will explore the different options available for running LLMs, including local deployment, cloud-based services, 
and containerised solutions. 
We will also discuss the trade-offs and considerations when choosing a deployment strategy for LLMs.

The chapter has four sections that send the same kind of prompt to an LLM in four different ways: a commercial
cloud API (`s00_cloud_apis`), a local model server in Docker (`s01_ollama`), the LangChain abstraction that hides
the difference between the two (`s02_langchain`), and the Hugging Face `transformers` library that loads the
model weights directly into the Python process (`s03_hugging_face`). The `get_llm` helper from `s02_langchain`
is reused by almost every later LLM and agent example in this repository.

### Options for running an LLM

| Option | Where the model runs | Typical use |
|---|---|---|
| cloud API (OpenAI, Anthropic, Google, Mistral, ...) | provider's data centre | best models, no hardware, pay per token |
| cloud platforms (Azure OpenAI, AWS Bedrock, Google Vertex AI) | provider, in your cloud tenant/region | enterprise contracts, data residency |
| local model server (Ollama, llama.cpp, LM Studio) | your machine/GPU, behind an HTTP API | privacy, offline use, development |
| high-throughput server (vLLM, Hugging Face TGI, SGLang) | your GPU server | serving open models to many users |
| in-process library (Hugging Face `transformers`) | your Python process | research, fine-tuning, full control |

From the point of view of application code most options look the same: an HTTP endpoint that takes a list of
messages and returns generated text. Many local servers (Ollama, vLLM, llama.cpp) even offer an
**OpenAI-compatible** endpoint, so the same client code can talk to a cloud model or to a local one by changing
the base URL.

### Section s00_cloud_apis: the OpenAI API

`main_openai.py` reads the API key from the environment variable `OPENAI_API_KEY` (never hard-code keys or commit
them), creates an `OpenAI` client and sends one request to the **Chat Completions** endpoint:
`client.chat.completions.create(model=..., messages=[{"role": "user", "content": prompt}])`. The request is a list
of messages with roles (`system` for instructions, `user`, `assistant` for earlier answers - the model itself is
stateless, so a conversation is continued by resending the history), and the answer is read from
`response.choices[0].message.content`. The response also reports the number of input and output tokens, which is
what you pay for. The model name (`gpt-5.2` in the script) has to be one that your account can access; check the
provider's model list.

### Section s01_ollama: a local model server in Docker

**Ollama** packages `llama.cpp` (an efficient C/C++ inference engine) behind a simple REST API and a model
registry (https://ollama.com/search) from which quantised open-weight models (Qwen, Llama, Gemma, Mistral,
DeepSeek, ...) can be pulled by name.

- `docker-compose.yml` starts the official `ollama/ollama` image as container `gzh_ollama`, maps the container port
  11434 to port `07011` on the host (so the API is at `http://localhost:07011`), stores downloaded models in the
  named volume `ollama_data` (they survive container restarts), and reserves all NVIDIA GPUs. The GPU reservation
  requires the NVIDIA Container Toolkit; without a GPU, remove the `deploy` block and Ollama runs on the CPU
  (slowly, but small models work).
- `main.sh` lists the commands to run *inside* the container (`docker exec -it gzh_ollama /bin/bash`):
  `ollama pull qwen3:4b` downloads a model and `ollama run qwen3` starts an interactive chat in the terminal
  (`ollama run` first pulls the model if it is missing; the bare name `qwen3` means `qwen3:latest`, the 8B model).
- `main_ollama.py` calls the REST API with `requests`: `POST /api/generate` with a JSON payload
  `{"model": ..., "prompt": ..., "stream": false}` returns a JSON object whose `response` field holds the
  generated text (with `"stream": true` the answer arrives as a sequence of JSON lines, token by token). The
  script requests `model = "qwen3"`, so `qwen3:latest` must be installed (`ollama pull qwen3`); to use another
  model such as `qwen3:4b`, change `model` to that tag.
- `utils/helpers.py` contains two helpers used throughout the repository: `ollama_get_available_models` queries
  `GET /api/tags` for the installed models (and returns an empty list with a helpful message if the server is not
  reachable), and `normalise_ollama_model` resolves the common naming mismatch between a requested model and the
  installed tags (`qwen3` vs `qwen3:latest` vs `qwen3:4b`). Note that a bare name only matches `name:latest` or the
  bare name itself: after `ollama pull qwen3:4b`, request `qwen3:4b` explicitly.

### Section s02_langchain: one interface for all models

**LangChain** wraps every provider in the same *chat model* interface: `llm.invoke(prompt_or_messages)` returns
an `AIMessage` whose `.content` is the answer, and the same object supports streaming (`llm.stream`), batching
(`llm.batch`), tool calling (`llm.bind_tools`) and structured outputs (`llm.with_structured_output`, chapter c02).
Switching from a local model to a cloud model is then a configuration change instead of a code change.

`utils/helpers.py` implements this switch in `get_llm(model=None, use="ollama", base_url_ollama=...)`:

- `LLM_TO_USE` (environment variable, `ollama` or `openai`) overrides the `use` argument.
- For **OpenAI** it requires `OPENAI_API_KEY` and returns `ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)`.
- For **Ollama** it reads `OLLAMA_BASE_URL` (default `http://localhost:07011`) and, if no model is passed,
  `OLLAMA_MODEL`. It lists the installed models and resolves the requested name with `normalise_ollama_model`;
  if the model is not installed it prints a warning and falls back to the first installed model, and without any
  request it simply uses the first installed model. It returns `ChatOllama(model=..., base_url=..., temperature=0)`.
- `temperature=0` makes the answers as deterministic as possible, which is what we want for examples, tests and
  most applications that extract or transform information.

Because the environment variables are read inside the function, nodes deep inside a LangGraph agent can call
`get_llm()` without passing configuration around. `main_langchain.py` sets `LLM_TO_USE=ollama`, gets the model and
prints the answer to "What is LangChain?".

### Section s03_hugging_face: running the weights in-process

`main_huggingface.py` uses the `transformers` **pipeline** API: `pipeline("text-generation", model="gpt2")`
downloads the model weights and tokeniser from the Hugging Face Hub (cached in `~/.cache/huggingface`), and the
call `generator(prompt, max_length=100, do_sample=True, temperature=0.7)` tokenises the prompt, runs the
autoregressive sampling loop of chapter c00 and decodes the result. GPT-2 (124M parameters, 2019) is a *base*
model: it continues text rather than answering questions, so the output is a plausible but usually unhelpful
continuation. Any other model id from https://huggingface.co/models can be used - e.g. a small instruction-tuned
model such as `Qwen/Qwen2.5-0.5B-Instruct`, ideally prompted through its chat template - and with `device_map="auto"`
and the `accelerate` package the weights are placed on the available GPU(s). This route gives full control (access
to logits, hidden states, fine-tuning, chapter c03) but leaves batching, memory management and serving to you.

### Trade-offs

- **Quality and capability**: the largest proprietary models are still ahead on hard reasoning, coding and long
  agentic tasks; open-weight models in the 7-70B range are good enough for many extraction, classification,
  summarisation and RAG tasks, especially after fine-tuning.
- **Cost**: APIs charge per input and output token (output tokens are several times more expensive), with no
  fixed costs - ideal for prototypes and spiky load. Self-hosting has fixed costs (GPUs, operations) and becomes
  cheaper only at high, steady utilisation.
- **Privacy and compliance**: with a cloud API the prompts leave your infrastructure. Enterprise agreements
  (no training on your data, regional data residency) mitigate this, but for strictly confidential data
  (customer data, banking secrecy) a local model may be the only option.
- **Latency**: two numbers matter - time to first token (dominated by processing the prompt) and tokens per second
  during generation (dominated by memory bandwidth). Local models avoid the network round trip but run on smaller
  hardware.
- **Hardware and memory**: the weights alone need (number of parameters) x (bytes per parameter): an 8B model
  needs about 16 GB in 16-bit precision, about 8 GB in 8-bit and about 4-5 GB in 4-bit precision. On top comes
  the **KV cache**, which grows linearly with the context length and the number of concurrent requests.
- **Quantisation** stores the weights with fewer bits (8, 5, 4 or even fewer per weight instead of 16) at a small
  loss in quality, which is what makes 7-14B models run on consumer GPUs and laptops. Ollama and llama.cpp use the
  **GGUF** file format, a single file containing the quantised weights, the tokeniser and metadata; names like
  `Q4_K_M` or `Q8_0` describe the quantisation scheme. GPU servers typically use formats such as AWQ, GPTQ or FP8.
- **Context length**: models are trained for a maximum context (e.g. 32k or 128k tokens), but servers often use a
  smaller default to save memory - Ollama, for instance, has a default context window of only a few thousand tokens
  that must be increased (`num_ctx`) for long documents (see chapter c05).
- **Throughput**: Ollama is optimised for a single user. For serving many concurrent users, servers such as
  **vLLM** (PagedAttention for KV-cache memory management and continuous batching), Hugging Face TGI or SGLang
  achieve many times the throughput on the same GPU.
- **Control and lock-in**: API models can change or be retired, while open weights are frozen and reproducible;
  abstractions such as LangChain (or an OpenAI-compatible endpoint) keep the switching costs low.

A pragmatic path, used throughout this book, is to develop against a local Ollama model and to keep the option of
switching to a cloud model with one environment variable (`LLM_TO_USE=openai`).
