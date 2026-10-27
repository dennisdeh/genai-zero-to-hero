## Agents with LangChain and LangGraph

This chapter puts the concepts of the introduction into code. The sections build up step by step: graphs without
any LLM (s00), single agents with tools, memory and human approval (s01), an agent that decides itself when to
retrieve documents (s02), two multi-agent architectures - supervisor (s03) and swarm (s04) - and an agent with
long-term memory (s05). All agents run against a local Ollama model through the `get_llm` helper of chapter
p07/c01: from `main_6` of s01 onwards the scripts request `qwen3:8b` explicitly, while `main_1` to `main_5` of s01
call `get_llm()` without a model name and therefore use `OLLAMA_MODEL` if it is set, otherwise the first model
installed on the Ollama server. The retrieval agents reuse the FINMA document collection in Qdrant from chapter
p07/c04.

### LangGraph in a nutshell

LangGraph models an application as a **state machine**:

- the **state** is a `TypedDict` (or `MessagesState`) that is passed from node to node. Nodes return the state, or
  only the keys they want to update; a **reducer** declared with `Annotated[..., reducer]` defines how an update is
  merged - `add_messages` appends new messages to the conversation instead of replacing it;
- **nodes** are plain Python functions (or runnables such as an LLM call or the prebuilt `ToolNode`);
- **edges** connect nodes: fixed edges (`add_edge`) and **conditional edges** (`add_conditional_edges`), where a
  routing function inspects the state and returns the name of the next node. Loops are simply edges back to an
  earlier node, which is what distinguishes an agent graph from a fixed pipeline (DAG);
- `compile()` turns the builder into a runnable graph with `invoke`, `stream` (with `stream_mode="values"` for the
  full state after each step or `"updates"` for the changes) and `get_graph()` for visualisation; compiled with a
  **checkpointer**, the state is saved after every step per `thread_id`, which enables conversation memory,
  interrupts and resuming.

LangChain's `create_agent` builds the most common graph - the ReAct loop *model -> tools -> model* - in one call
and adds **middleware** hooks (e.g. human-in-the-loop, call limits) around the model and tool calls; internally it
is a LangGraph graph as well.

### Section s00: graphs without an LLM

The two scripts use a state with an `operation` (`"sum"` or `"multiplication"`), a list of `values` and a
`result`. In `main_very_simple_langgraph.py` the graph is `START -> initial -> (tool_sum | tool_multiplication) ->
final -> END`; the routing function `determine_branch` reads `state["operation"]` and returns the name of the
branch, exactly as an LLM-based router would - only deterministically. `main_very_simple_langgraph_looping.py`
adds a **loop**: the node `add_random_number` appends a random integer and increments a counter, and the
conditional edge `determine_loop` sends the graph back to the same node until the counter exceeds `threshold`.
An `identity` node (it returns the state unchanged) separates the loop decision from the branch decision. Both
scripts save a diagram of the graph with `save_graph_image` from `utils/graphs.py`: `get_graph().draw_mermaid_png()`
renders a PNG via the mermaid.ink web service, and without internet access the helper falls back to the Mermaid
text (`draw_mermaid()`), which can be viewed at https://mermaid.live.

### Section s01: simple agents

- `main_1_simple_chat.py`: a one-node graph that calls the LLM. The chat loop passes only the latest message, so
  the bot has **no memory** - the LLM is stateless.
- `main_2_simple_chat_memory.py`: the same graph, but the loop keeps appending to the `messages` list and sends the
  whole history every time - the simplest form of short-term memory (and the reason why long conversations
  eventually exceed the context window).
- `main_3_simple_tools.py`: **tool calling** built by hand. Two `@tool` functions (sum and multiplication of a
  list) are bound to the model with `llm.bind_tools`; the model either answers or returns `tool_calls`. The
  conditional edge `determine_loop` routes to a `ToolNode`, which executes the calls and appends `ToolMessage`s -
  but the graph has no edge from the `ToolNode` back to the model, so it ends right after this single round of
  tool calls: the printed stream shows the model's tool call(s) and the tool results, and the model never sees
  the results or writes a final answer. The question "multiply 2, 3, 4, add 1, multiply by 4" needs several
  rounds (each step depends on the previous result), which makes this limitation visible.
- `main_4_simple_stock_analyser.py` and `main_5_simple_stock_analyser_chat.py`: the complete **ReAct loop** with a
  mock stock-analysis tool. The additional edge `add_edge("tools", "chatbot")` sends the tool results back to the
  model, which can then call further tools or answer; the loop ends when the model replies without tool calls
  (adding the same edge to `main_3` lets it solve the multi-step calculation). The chat version compiles the graph
  with an `InMemorySaver` checkpointer, so the history is stored per `thread_id` by LangGraph instead of by the
  calling code.
- `main_6_math_agent_hitl.py`: `create_agent` with `HumanInTheLoopMiddleware`. Calls of `tool_multiplication` are
  **interrupted** before execution; the script shows the requested action and asks the user to approve, edit (new
  values) or reject it, and resumes the graph with `Command(resume={"decisions": [...]})`. Interrupts require a
  checkpointer, because the graph has to be paused and continued from its saved state.

### Section s02: a RAG agent

`main_rag_agent.py` turns the RAG chain of p07/c04 into **agentic RAG**: retrieval becomes a tool
(`retrieve_context`, which returns the serialised chunks with their metadata as content and the `Document`
objects as artefact), and the model decides whether, how often and with which query to call it - next to a mock
stock-analysis tool. The question "analyse the Apple stock and tell me what FINMA says about risks of stock
trading" requires both tools. Compared with a fixed chain, the agent can skip retrieval for questions that do not
need it and can reformulate and repeat searches, at the price of more LLM calls and less predictable behaviour.

### Section s03: multi-agent system with a supervisor

`main_0_multi_agent_supervisor.py` implements the supervisor architecture with the "agents as tools" pattern:
a math agent and a RAG agent are created with `create_agent`, each wrapped in a `@tool` function (`math`,
`finma_rag`) that invokes the sub-agent and returns its final answer, and a supervisor agent receives these two
tools. The supervisor decomposes the request ("how often does FINMA mention climate risks ... and add the numbers
[2,3,4,5]"), delegates the parts and combines the answers. Each sub-agent has a short, focused system prompt and
only the tools it needs, and the supervisor's context only contains the sub-agents' final answers.

`main_1_multi_agent_supervisor_hitl.py` adds a **query-rewriting** step with human oversight: before retrieval, an
`agent_rag_rewriter` calls a `query_rewrite` tool (an LLM that only normalises spelling and grammar of the search
query); `HumanInTheLoopMiddleware` interrupts this call so that the user can approve or edit the rewritten query,
and `ToolCallLimitMiddleware` ensures that the rewrite happens only once. Because the interrupt happens inside a
sub-agent that is called from a tool of the supervisor, both use checkpointers, and the script resumes the
supervisor with `Command(resume=...)` until no interrupts are left.

### Section s04: multi-agent swarm

`main.py` builds the same two agents as a **swarm** with the `langgraph-swarm` package: there is no supervisor;
instead each agent has a **handoff tool** (`create_handoff_tool`) that transfers the conversation to the other
agent. `create_swarm([agent_math, agent_rag], default_active_agent="agent_math")` creates the graph and keeps
track of the active agent; compiled with a checkpointer, the next user message of the same thread goes directly
to the agent that was active last. Swarms save the supervisor's extra LLM calls and let specialists talk to the
user directly, but control flow and debugging are harder.

### Section s05: an agent with long-term memory

`utils/graph.py` builds a graph `START -> retrieve_memories -> call_model -> (store_memory -> call_model) -> END`
around a LangGraph **store** (`InMemoryStore`), which, unlike the checkpointer, persists across threads. Memories
are stored per user in the namespace `("memories", user_id)`:

- `retrieve_memories` (in `utils/memory.py`) searches the user's namespace with the latest message as query and
  puts the hits into a `memory_context` that `llm_chain` appends to the system prompt. Without an embedding index
  the `InMemoryStore` ignores the query and simply returns the stored memories (up to its default limit of 10);
- the model has an `upsert_memory(content, context, memory_id=None)` tool and is instructed to store new facts
  about the user (preferences, personal information) and to update existing ones;
- `store_memory` executes these tool calls: it reuses the id of an identical memory or creates a new UUID, writes
  `{"context", "content"}` to the store and answers with a `ToolMessage`.

`main_0_longterm_in_memory.py` simulates a conversation with the user `carl42` ("I like musicals ...", "What do I
like?", "I also like to read the news", ...) to show that the facts are remembered and updated. In production the
in-memory store and checkpointer would be replaced by persistent backends (e.g. PostgreSQL), and the store would be
configured with an embedding index so that `store.search(namespace, query=...)` returns the semantically most
relevant memories instead of all of them.
