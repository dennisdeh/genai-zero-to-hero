# Agents and Agentic AI
## Introduction
In this part we will look at AI agents and how they can be used to create Agentic AI.

### Types of agents
Modern agentic AI systems extend beyond single prompt–response interactions. 
They combine reasoning models with memory, external tools, retrieval systems, orchestration layers, 
and safety controls to perform multi-step, goal-directed tasks.

Agents can be classified into two main types:

- **Zero-Shot Agents**: These agents are designed to work with a single prompt and do not require any memory or context management.
- **ReAct Agents**: These agents use a reasoning process to break down tasks into smaller subtasks and use the model's responses to guide the next step.

### Frameworks
As LLMs are stateless, the entire chain of messages must be passed to the model – or at least the relevant messages.

This can create problems when we are approaching the context window limit of the model

Thus, special care needs to be taken to ensure that the agents are able to access the relevant
context without getting overwhelmed.

Within the LangChain ecosystem there are two main frameworks that can be used to help with building agents:

- **LangChain**: LangChain provides a set of tools and a high level of abstraction for building agents. It includes a wide range of components such as prompts, memory, and tools that can be used to build agents.
- **LangGraph**: When finer control and customisation over the way the agents interact with the environment, user, other agents is needed.

The two frameworks are complementary and can be used together or separately. Behind the scenes, LangChain uses LangGraph to manage the interactions between agents and the environment.


### Main concepts in Agentic AI
Agentic systems operate through iterative reasoning loops in which a model evaluates state, decides actions, and updates context.

A typical loop:
**observe → reason → act → update state → repeat**

Unlike stateless LLM applications, agents require persistent state, external integrations, and orchestration middleware.

#### Memory
Memory allows agents to maintain context beyond the token window and across interactions.
Three practical layers are commonly used and are explored in more detail below.

##### Short-Term Memory (Working Memory)
Short-term memory contains the current task context. This usually includes:

- conversation history
- intermediate reasoning steps
- tool outputs
- current plan

The following implementation patterns:

- message buffers (i.e. saved in the context chain)
- planning trees

Working memory is typically stored in the prompt context and trimmed via summarisation or compression.

##### Long-Term Memory
Long-term memory persists knowledge across sessions.

Common storage formats:

| Type             | Purpose            |
|------------------|--------------------|
| Vector stores    | Semantic recall    |
| Key–value stores | Structured facts   |
| Document stores  | Logs and artifacts |

Typical retrieval pipeline:
**experience → embedding → vector DB → semantic search → injected into prompt**

Agents write to memory through reflection or summarisation.

An example: Observation: user prefers Python for data pipelines. Memory write to the database: {"user_preference_language": "python"}

##### Episodic vs Semantic Memory

Borrowing from cognitive architectures, there are two main types of memory:

| Memory Type         | Description                                                                               | Examples                                  |
|---------------------|-------------------------------------------------------------------------------------------|-------------------------------------------|
| **Episodic Memory** | Records of past experiences and interactions used for historical context and traceability | past interactions, logs, execution traces |
| **Semantic Memory** | Distilled knowledge and persistent facts extracted from experiences                       | distilled knowledge, persistent facts     |

Many production agents periodically run memory consolidation jobs to convert episodic logs into semantic summaries.


#### Interrupts and Human-in-the-Loop

Agentic systems often require human oversight for safety, correctness, or governance.
Interrupt mechanisms allow an agent to pause execution and request input.

Typical interrupt triggers:

- low confidence
- policy-sensitive actions
- irreversible operations
- ambiguous instructions

Example workflow:
**Agent proposes action → interrupt gate → human approves / edits / rejects → execution resumes**


Human-in-the-loop systems commonly appear in:

- autonomous coding systems
- enterprise workflow automation
- financial operations
- regulated domains

Implementation strategies include:

- approval queues
- structured feedback APIs
- UI checkpoints

#### Retrieval

Retrieval systems provide agents with external knowledge beyond the model weights.
This is usually implemented as Retrieval-Augmented Generation (RAG).

The general RAG pipeline:
**query → embedding → vector search → reranking → context injection → model reasoning**

Advanced systems may add:

- hybrid search (vector + keyword)
- contextual compression
- query rewriting
- multi-hop retrieval

Agents may perform retrieval iteratively during reasoning rather than once at the beginning.

#### External Tool Calling
Agents interact with the outside world via tools.

Tools expose deterministic capabilities such as:

- APIs
- databases
- calculators
- code execution
- file systems
- web search

Example tool schema:

```json
{
  "name": "search_docs",
  "description": "Search internal documentation",
  "parameters": {
    "query": "string"
  }
}
```

Agent workflow:
**model reasoning → tool selection → tool execution → result returned to context → next reasoning step**

Tool use allows agents to offload computation and access e.g. real-time information into the context window.

#### Middleware and Orchestration
Agentic systems typically rely on orchestration layers that manage:
 
- prompt templates
- tool routing
- memory storage
- execution loops
- logging

Examples of middleware components:

| Component         | Role                               |
|-------------------|------------------------------------|
| Agent loop engine | Manages reasoning iterations       |
| Tool router       | Dispatches external calls          |
| Memory manager    | Reads/writes memory                |
| Event bus         | Coordinates asynchronous workflows |
| Trace system      | Records execution                  |

This layer effectively acts as the operating system for agents.

#### Persistence

Persistence ensures that the agent state survives process restarts and supports observability.
Persistent artifacts often include:

- execution traces
- intermediate plans
- tool outputs
- memory stores
- audit logs

Persistence enables:

- reproducibility
- debugging
- offline evaluation
- compliance

Many modern stacks store traces in structured JSON or event logs to allow replay of agent decisions.


### Multi-Agent Architectures

Single agents struggle with complex tasks due to limited planning depth and context constraints. Multi-agent systems decompose problems across specialized agents.

Two dominant architectures are widely and described further below.

#### Supervisor (Manager–Worker) Architecture
A supervisor agent decomposes tasks and delegates work to specialised agents, which
might in turn delegate work to other sub-agents.

The general workflow looks as follows:
**user task → supervisor planning → subtask assignment → worker execution 
→ results aggregation → final synthesis**

The supervisor agent is responsible for:

- task decomposition
- subtask assignment
- coordination of worker agents
- aggregation of results
- synthesis of final output

This architecture allows for efficient task decomposition and parallel execution, 
enabling agents to tackle complex problems more effectively.

Advantages are:

- modular specialisation
- clear control hierarchy
- easier debugging

Challenges are:

- supervisor bottleneck
- context coordination


#### Swarm Architecture
Swarm architectures remove centralised control.
Agents communicate through shared memory or message passing.
This architecture is useful for more open-ended problems, i.e. 
research prototypes, complex reasoning tasks. Agents independently:

1. read the shared state
2. decide actions
3. update the state

The advantages are:

- emergent collaboration
- scalability
- fault tolerance

Challenges are:

- coordination overhead
- redundant work
- convergence issues


### Model Context Protocol (MCP)

The Model Context Protocol (MCP) standardises how models access external capabilities.
Conceptually, MCP acts as a universal interface between models and tools.
Instead of custom tool integrations, MCP exposes resources through a consistent protocol.

The core MCP abstractions are:

- **Resources**: Static or dynamic data accessible to the model (e.g. documents, database records, logs).
- **Tools**: Executable actions the model can invoke (e.g. run SQL query, call REST API, execute code).
- **Prompts**: Reusable prompt templates that define interaction patterns.


The MCP architecture follows the following layers:
**LLM → MCP Client → MCP Server → Tools / Resources**

The key benefits are standardised tool discovery, portable integrations, and vendor-neutral ecosystem,
meaning the same tools can be used across different models and platforms in a "plug-and-play" fashion.

### Guardrails and Testing
Agentic systems introduce new safety and reliability challenges due to their autonomy and ability to act.
Robust systems implement guardrails across multiple layers.

#### Input Guardrails

Input validation prevents malicious or malformed prompts.

Common techniques include:

- prompt injection detection
- schema validation
- policy filtering
- removing sensitive information/personal identifiers

For example, one would probably like to reject the following prompting situations:

- system prompt override
- data exfiltration attempt
- jailbreak patterns



#### Output Guardrails
Outputs must be validated before execution.

Common techniques include:

- structured output schemas
- policy classifiers
- rule-based filters

This ensures that the model is not overly permissive and that it is not misused.
One would also have a human review sensitive outputs in a Human-in-the-Loop (HIL) workflow 
and implement checks that tools are not abused.

#### Action Guardrails
Actions with real-world consequences require stronger controls.
One would include Human-in-the-Loop (HIL) mechanisms to ensure that the agent is not misused, where appropriate,
and to approve or reject critical actions, e.g.:

- financial transactions
- infrastructure changes
- code deployment

Common strategies include to build into the agentic system:

- human-in-the-loop approval mechanisms
- approval workflows
- sandboxed execution
- rate limiting


### Testing and validation of agentic systems
Traditional unit testing is insufficient for agent behaviour because of the asynchronous nature of 
the system and non-deterministic execution through dynamic workflows. 
Effective evaluation combines multiple and complementary approaches.

The following table summarises the most common testing methods:

| Testing Method           | Purpose                                                           | What It Validates                                                              | Example                                                                 |
|--------------------------|-------------------------------------------------------------------|--------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **Deterministic Tests**  | Validate deterministic system components and integrations         | Tool interfaces, schema parsing, memory reads/writes, API behavior             | Verify that a SQL query tool returns correctly parsed structured output |
| **Simulation Tests**     | Evaluate agent performance in controlled task environments        | End-to-end workflows, decision-making, task completion                         | Simulate customer support conversations and measure resolution accuracy |
| **Trace Evaluation**     | Analyze execution logs to assess reasoning and agent behavior     | Planning quality, tool usage efficiency, hallucinations, reasoning correctness | Inspect agent traces to verify correct sequence of tool calls           |
| **Red-Teaming**          | Stress-test system robustness against adversarial inputs          | Prompt injection resistance, policy violations, unsafe actions                 | Run adversarial prompts attempting to exfiltrate system prompts         |
| **Human Evaluation**     | Assess qualitative behavior that automated metrics cannot capture | Helpfulness, correctness, safety, user satisfaction                            | Human reviewers score agent responses to complex tasks                  |
| **Regression Testing**   | Ensure system changes do not degrade performance                  | Stability across versions, prompt updates, or tool changes                     | Re-run evaluation suite after updating agent prompt templates           |
| **Sandbox Testing**      | Safely test real-world actions without consequences               | External actions such as infrastructure changes or transactions                | Execute deployment commands in a sandbox environment                    |
| **Offline Benchmarking** | Compare agent performance across standardised datasets            | Model reasoning ability, retrieval effectiveness, planning accuracy            | Run an agent on a benchmark set of multi-step tasks                     |

A useful framework for evaluating agentic systems is the open-source MLflow that we will cover in the next chapters.