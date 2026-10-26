### Prompt Engineering

A chat model receives a list of *messages*, each with a role:

- **system**: instructions that set the behaviour for the whole conversation (persona, rules, format),
- **human / user**: the user's input,
- **ai / assistant**: earlier model answers (conversation history or few-shot examples),
- **tool**: results of tool calls (see the agent chapters).

The chat template of the model (chapter c00) turns these messages into a single token sequence with special
tokens. Instruction tuning has taught the model to give the system message high priority, which is why it is
the natural place for instructions.

#### Zero-shot and few-shot prompting

In **zero-shot** prompting we describe the task and let the model answer directly. In **few-shot** (or
*in-context learning*, Brown et al., 2020) we add a few solved examples to the prompt. The model's weights
do not change; the examples steer the output distribution by showing

- the expected **format** (here: a single lower-case label),
- the **decision boundaries** (is a CFO appointment "neutral" or "positive"?),
- the **style** and length of the answer.

`main_1_zero_vs_few_shot.py` classifies financial headlines as positive/negative/neutral with three prompt
variants:

1. a bare question ("What is the sentiment of this headline?"),
2. a system prompt with a role, label definitions and an output format,
3. the same plus four examples, built with LangChain's `FewShotChatMessagePromptTemplate`, which renders
   each example as a human/ai message pair.

```python
example_prompt = ChatPromptTemplate.from_messages([("human", "{headline}"), ("ai", "{label}")])
few_shot_examples = FewShotChatMessagePromptTemplate(examples=FEW_SHOT_EXAMPLES, example_prompt=example_prompt)
few_shot = ChatPromptTemplate.from_messages([("system", system_text), few_shot_examples, ("human", "{headline}")])
```

The script prints the rendered prompts (so you can see exactly what the model receives) and the accuracy of
each variant. The bare prompt typically produces long explanations that cannot be mapped to a label - the
most common failure in practice is not a wrong answer but an answer in the wrong *format*. Practical tips for
few-shot examples: cover all classes, include hard/borderline cases, keep the examples distinct from the test
data, and be aware that the order and the label balance of the examples can bias the model.

#### System prompts and roles

`main_2_roles_and_chain_of_thought.py` (part A) asks "What is an ETF?" under three system prompts: a teacher
for children, a CFA charterholder and a one-sentence constraint. The content of the answer is the same
knowledge, but vocabulary, depth and length change dramatically. A role ("You are a ...") is a compact way
of selecting a register; explicit constraints (length, format, what to do when unsure) are more reliable than
the role alone.

#### Chain-of-thought and reasoning models

An autoregressive model spends a fixed amount of computation per generated token. If we force it to output
the final number immediately, all the "thinking" must happen inside one forward pass. **Chain-of-thought
(CoT)** prompting (Wei et al., 2022) asks the model to write out intermediate steps first; each step becomes
context for the next one. Even the simple instruction "Let's think step by step" (Kojima et al., 2022)
improves multi-step arithmetic and logic.

Part B of the script solves small finance word problems (percentage changes, compound interest, weighted
averages) three ways:

| Prompt | What the model does | Cost |
|---|---|---|
| direct ("answer with the number only") | answers in one step | few tokens, fast |
| CoT ("solve step by step ... `ANSWER: <number>`") | writes intermediate calculations | more tokens |
| built-in thinking (`reasoning=True`) | the model generates hidden reasoning first | most tokens |

A fixed final-answer marker (`ANSWER: ...`) makes the result machine-readable despite the free-form
reasoning. **Reasoning models** (OpenAI o-series, DeepSeek-R1, qwen3 in thinking mode, gpt-oss) have been
trained with reinforcement learning to produce such reasoning on their own (see chapter c03); with Ollama the
reasoning is returned separately in `additional_kwargs["reasoning_content"]`. For these models an explicit
CoT instruction adds little, but switching thinking off (`reasoning=False`) is a useful speed knob for easy
tasks such as classification. Related techniques: *self-consistency* (sample several reasoning paths and take
the majority answer, see the voting workflow in chapter c06) and *decomposition* (prompt chaining, also c06).

#### Prompt templates

Prompts in applications are *templates* filled with data at runtime. `main_3_templates_and_temperature.py`
shows

- `PromptTemplate.from_template("... {doc_type} ... {audience} ... {text}")` - input variables are detected
  from the braces,
- `.partial(today=...)` to fix some variables early (dates, format instructions),
- `ChatPromptTemplate` with `MessagesPlaceholder("history")` to insert a conversation history,
- LCEL composition `prompt | llm | StrOutputParser()` and `chain.batch([...])` to run many inputs
  concurrently.

Templates keep prompts versionable and testable: the same template can be rendered and unit tested without
any model call, and evaluated as a unit (chapter c07).

#### Temperature and sampling

At each step the model outputs logits $z_i$ over the vocabulary. With temperature $T$ the next token is
sampled from

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}.$$

$T \to 0$ approaches greedy decoding (always the most likely token), $T = 1$ samples from the model's
distribution and $T > 1$ flattens it, giving more diverse and more erratic text. Top-$k$ and top-$p$
(nucleus) sampling additionally cut off the unlikely tail. The script samples the same slogan prompt five
times at $T \in \{0, 0.8, 1.5\}$ and reports the number of distinct answers and the mean pairwise Jaccard
similarity of their words. Rules of thumb: $T = 0$ for extraction, classification, code and evaluation;
moderate $T$ for writing; high $T$ for brainstorming. Note that even $T = 0$ is not guaranteed to be
bit-for-bit deterministic on GPUs (batching and floating point non-associativity); set a `seed` where the
backend supports it if reproducibility matters.
