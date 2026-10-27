"""
Keeping a growing chat history inside the context window: trimming vs summarising.

A long conversation with a (fictional) banking assistant mentions important user facts early on and then
drifts into small talk. We compare three ways to fit the history into a token budget:
  1. `trim_messages(strategy="last")`: keep the system message and the most recent messages that fit
     (start_on="human" so the kept history never starts with an orphaned AI answer),
  2. `trim_messages(strategy="first")`: keep the oldest messages (rarely what you want in chat),
  3. summary memory: replace old messages by an LLM-written summary and keep the last few turns verbatim.
Part 1 (trimming) runs offline. Part 2 asks the LLM a question whose answer is only in the early messages,
using each history variant, and is skipped if no LLM is reachable.

How to run (from the repo root):
    python -m p07_llms.c05_long_context_systems.main_4_chat_history_trimming
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
    is_llm_available,
    strip_think,
)
from p07_llms.c05_long_context_systems.utils.tokens import message_token_counter

MAX_HISTORY_TOKENS = 150
KEEP_LAST_MESSAGES = 4
QUESTION = "Remind me: how many NWIND shares do I hold, and what is my risk profile?"


def build_conversation() -> list[BaseMessage]:
    """
    A synthetic conversation: important facts first, then many unrelated turns.
    """
    messages: list[BaseMessage] = [
        SystemMessage("You are a helpful banking assistant. Answer briefly."),
        HumanMessage(
            "Hi, I'm Dennis. My risk profile is conservative and I hold 120 shares of NWIND."
        ),
        AIMessage(
            "Thanks Dennis, noted: conservative risk profile and 120 NWIND shares."
        ),
    ]
    small_talk = [
        ("What are your opening hours?", "Our branches are open 9-16 on weekdays."),
        (
            "Can I order a new debit card online?",
            "Yes, in the app under Cards > Order new card.",
        ),
        ("Is there a fee for that?", "The first replacement card each year is free."),
        (
            "How do I change my PIN?",
            "In the app under Cards > Change PIN, or at any ATM.",
        ),
        (
            "Do you support Apple Pay?",
            "Yes, all our debit and credit cards support Apple Pay.",
        ),
        ("What is the daily ATM limit?", "The default daily ATM limit is DKK 10,000."),
        (
            "Can I raise it temporarily?",
            "Yes, for up to 7 days in the app under Limits.",
        ),
        (
            "Thanks! Also, how do I add a beneficiary?",
            "Go to Payments > Beneficiaries > Add new.",
        ),
    ]
    for question, answer in small_talk:
        messages += [HumanMessage(question), AIMessage(answer)]
    return messages


def show(title: str, messages: list[BaseMessage]) -> None:
    """
    Print a history with its token count.
    """
    print(
        f"\n--- {title}: {len(messages)} messages, ~{message_token_counter(messages)} tokens ---"
    )
    for m in messages:
        print(f"  [{m.type:>6}] {str(m.content)[:90]}")


def summary_memory(
    llm: BaseChatModel, messages: list[BaseMessage], keep_last: int
) -> list[BaseMessage]:
    """
    Replace all but the last `keep_last` non-system messages by a summary written by the LLM.
    """
    system = [m for m in messages if isinstance(m, SystemMessage)]
    dialogue = [m for m in messages if not isinstance(m, SystemMessage)]
    old, recent = dialogue[:-keep_last], dialogue[-keep_last:]
    transcript = "\n".join(f"{m.type}: {m.content}" for m in old)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarise the conversation in at most 3 sentences. Keep all facts about the user "
                "(name, holdings, preferences, numbers).",
            ),
            ("human", "{transcript}"),
        ]
    )
    summary = strip_think(
        (prompt | llm | StrOutputParser()).invoke({"transcript": transcript})
    )
    return (
        system
        + [SystemMessage(f"Summary of the earlier conversation: {summary}")]
        + recent
    )


if __name__ == "__main__":
    # 1. The full history
    history = build_conversation()
    show("full history", history)

    # 2. Trimming strategies (no LLM needed - only a token counter)
    trimmed_last = trim_messages(
        history,
        max_tokens=MAX_HISTORY_TOKENS,
        token_counter=message_token_counter,
        strategy="last",
        include_system=True,
        start_on="human",
    )
    show(
        f"trim_messages(strategy='last', max_tokens={MAX_HISTORY_TOKENS})", trimmed_last
    )
    trimmed_first = trim_messages(
        history,
        max_tokens=MAX_HISTORY_TOKENS,
        token_counter=message_token_counter,
        strategy="first",
    )
    show(
        f"trim_messages(strategy='first', max_tokens={MAX_HISTORY_TOKENS})",
        trimmed_first,
    )
    print(
        "\nNote: 'last' dropped the early messages with the user's holdings - trimming forgets silently."
    )

    # 3. Ask a question that needs the early facts, with each history variant
    if not is_llm_available():
        print(
            "\nNo LLM reachable - skipping the LLM part (set OLLAMA_BASE_URL or LLM_TO_USE=openai)."
        )
    else:
        llm = get_chat_model(temperature=0)
        summarised = summary_memory(llm, history, keep_last=KEEP_LAST_MESSAGES)
        show("summary memory", summarised)
        for name, variant in [
            ("full", history),
            ("trimmed (last)", trimmed_last),
            ("summary memory", summarised),
        ]:
            answer = strip_think(
                str(llm.invoke(variant + [HumanMessage(QUESTION)]).content)
            )
            print(
                f"\n[{name}] ({message_token_counter(variant)} tokens of history)\n  {answer[:300]}"
            )
