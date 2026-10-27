## LLM Workflows

Most useful LLM applications need more than one model call: an answer is drafted and then checked, a request
is classified and then handled by a specialist, a long task is split up and the parts are combined. This
chapter implements the classic **workflow patterns** with LangGraph.

### Workflows vs agents

Anthropic's essay *Building effective agents* (December 2024) draws a useful line between two kinds of
*agentic systems*:

- **Workflows**: LLMs and tools are orchestrated through **predefined code paths**. The developer decides the
  steps and the control flow; the LLM fills in the content of each step.
- **Agents**: the LLM **dynamically directs its own process** and tool use - it decides in a loop what to do
  next based on feedback from the environment, until it judges the task to be done (see part p08).

Workflows are predictable, testable, cheaper and easier to debug; agents are flexible but harder to control,
and their cost and latency are unbounded unless explicitly limited. The essay's main advice is to start with
the simplest solution (often a single well-prompted call with retrieval and examples), to add workflow
structure when a task decomposes into well-defined steps, and to reach for agents only when the steps cannot
be predicted in advance. The building block of all patterns is the **augmented LLM**: a model with
retrieval, tools and memory - in this repository `get_llm()` plus structured outputs (chapter c02),
retrieval (c04) and tools (p08).

### Why LangGraph

All patterns below could be written as plain Python functions. LangGraph adds a small amount of structure
that pays off as workflows grow:

- **state**: a `TypedDict` that every node reads from and returns *updates* to; **reducers**
  (`Annotated[list, operator.add]`) define how parallel updates to the same key are merged,
- **nodes and edges**: plain functions connected by fixed edges, conditional edges (a routing function
  returns the next node) and the **`Send` API** for dynamic fan-out (a routing function returns
  `[Send("worker", payload), ...]`, one parallel task per payload),
- **parallel execution** of nodes in the same step, **streaming** of intermediate results
  (`graph.stream(..., stream_mode="updates")`), and checkpointing, human-in-the-loop and visualisation
  (`graph.get_graph().draw_mermaid()`) for free - the same machinery that the agents in p08 use.

Every script has a `build_graph(llm)` function that receives the model. This dependency injection makes the
workflows testable without a running LLM server - with one caveat: four of the five graphs call
`llm.with_structured_output(...)`, which LangChain's `GenericFakeChatModel` (canned text replies) does not
implement. Those tests need a small fake model that supports structured output, or a mock server that speaks
the Ollama or OpenAI API. The Mermaid
diagrams are written to `outputs/` (and rendered to PNG when the mermaid.ink service is reachable).

### Pattern 1: prompt chaining

*Decompose a task into a fixed sequence of steps; each LLM call processes the previous output. Add
programmatic checks ("gates") between steps to verify that the process is on track.*

`main_1_prompt_chaining.py`: facts about a company -> **outline** -> **gate** -> **draft** -> **translate**
(to Danish). The gate is plain Python: it requires 3-6 bullet points containing figures and otherwise routes
to `END`, so no draft is written from a bad outline. Each step has a simpler prompt than one monolithic
"write a Danish newsletter item from these facts" prompt, which trades latency for accuracy and makes each step
inspectable. Use it when the task decomposes cleanly into fixed subtasks.

### Pattern 2: routing

*Classify an input and direct it to a specialised follow-up.*

`main_2_routing.py`: a classifier with structured output (`Route` with a `Literal` category and a short
reasoning field, generated *before* the category so the model "thinks" first) routes bank customer messages
to card support, investment information, complaints or a general fallback. Each handler has a focused system
prompt - optimising one prompt for one category does not degrade the others. Routing is also used to send
easy requests to a small, cheap model and hard ones to a large model. Classification errors propagate, so
evaluate the router separately (a confusion matrix on labelled examples, see c07) and keep a safe fallback.

### Pattern 3: parallelisation

*Run LLM calls simultaneously and aggregate the outputs programmatically.* Two variants
(`main_3_parallelisation.py`):

- **Sectioning**: independent subtasks in parallel - three analysts look at a news article from the
  financial, risk and ESG angle. Static edges `START -> financial/risks/esg` fan out; `add_edge([...],
  "aggregate")` fans in and waits for all three. Each call gets a focused prompt, and the wall-clock time is
  that of the slowest branch rather than the sum. Guardrails are a common application: one call answers the
  user while another screens the input for inappropriate content.
- **Voting**: the same task several times for diverse outputs - three reviewers with different perspectives
  judge whether an e-mail is phishing (structured `Vote`), and a majority vote decides. The voters are
  created dynamically with `Send`, and the `operator.add` reducer collects their votes. Voting increases
  robustness (compare *self-consistency* in c02) and lets you trade off precision and recall via the
  threshold (e.g. flag if *any* voter says phishing).

### Pattern 4: orchestrator-workers

*A central LLM dynamically breaks down a task, delegates subtasks to worker LLMs, and synthesises their
results.*

`main_4_orchestrator_workers.py`: the orchestrator plans the sections of a briefing (structured `Plan` with a
list of `Section`s), `assign_workers` returns one `Send("worker", {...})` per planned section, workers write
their sections in parallel, and the synthesiser assembles them in the planned order (workers finish in
arbitrary order, so each result carries its index). The difference to sectioning is that the number and
nature of the subtasks are decided by the model at runtime - typical uses are coding changes across an
unknown number of files or research over several sources. Bound the fan-out (here at most 5 sections).

### Pattern 5: evaluator-optimiser

*One LLM call generates a response while another provides evaluation and feedback in a loop.*

`main_5_evaluator_optimiser.py`: a generator drafts a reply to an angry customer complaint; the evaluator
combines **deterministic checks** (word limit, forbidden promises such as "we will refund", matched as
whole phrases with word boundaries so that "we cannot guarantee" passes) with an **LLM judge**
(structured `Evaluation` with grade and feedback) against a written policy; the conditional edge loops back
to the generator with the feedback until the reply passes or `MAX_ITERATIONS` is reached. The pattern works
when there are clear evaluation criteria and when feedback demonstrably improves the result - like a human
editor. Always bound the loop, keep the evaluator's criteria explicit, and log every round (the `history`
list) to understand where the generator struggles.

### Choosing and combining patterns

| Pattern | Control flow | Typical use | Cost/latency |
|---|---|---|---|
| prompt chaining | fixed sequence + gates | multi-step transformation | sum of steps |
| routing | classifier + branches | heterogeneous requests, model tiering | one extra (small) call |
| parallelisation | static fan-out/fan-in | independent aspects, guardrails, voting | max of branches, N x tokens |
| orchestrator-workers | dynamic fan-out (`Send`) | subtasks unknown in advance | plan + workers + synthesis |
| evaluator-optimiser | bounded loop | quality-critical text with clear criteria | up to 2 x iterations calls |

The patterns compose: a router can send requests to different chains, an orchestrator's workers can be
evaluator-optimiser loops. When the number and order of steps can no longer be written down in advance, the
system becomes an agent - the subject of part p08.
