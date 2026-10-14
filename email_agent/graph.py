"""Graph assembly: nodes, edges, conditional routing, and the HITL breakpoint.

There are two complementary ways to pause a LangGraph run, and this module uses
both deliberately:

``interrupt()`` (inside ``ask_approval_node``)
    The modern, payload-carrying interrupt. The pause point is defined by the
    node that *wants* a decision, and the request travels with it. Resuming
    with ``Command(resume=<decision>)`` replays the node and returns the value.

``interrupt_before=["ask_approval"]`` (on the checkpointer-aware compile)
    A static breakpoint in the graph topology. Useful for debugging, for
    non-interactive front-ends that drive the run via
    ``graph.get_state``/``graph.update_state``, and as a defence in depth: even
    if the node-level interrupt were removed, the edge into ``ask_approval``
    would still halt.

Note the safety ordering — ``ask_approval`` is entered *before* any send-capable
node, and ``send_node`` independently re-verifies ``approval_status``, so
pausing is what *gates* the send rather than what merely warns about it.
"""

from __future__ import annotations

import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from .state import EmailState


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #


def route_after_auth(state: EmailState) -> str:
    """The only gate in the graph. Nothing mailbox-related runs unless this
    returns ``"read"``."""
    return "read" if state.get("authenticated") is True else END


def route_after_draft(state: EmailState) -> str:
    """Skip the approval step entirely when nothing needs a reply — there is
    no draft for a human to approve."""
    if not (state.get("drafts") or []) or state.get("errors"):
        return "finished"
    return "ask_approval"


def route_after_approval(state: EmailState) -> str:
    """Approve -> send. Revise -> loop back to the drafting node. Quit -> end."""
    status = state.get("approval_status")
    if status == "approved":
        return "send"
    if status == "rejected":
        return "finished"
    return "draft"


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build_graph(static_breakpoint: bool = False):
    """Compile the Email Assistant graph.

    Args:
        static_breakpoint: add ``interrupt_before=["ask_approval"]`` in
            addition to the in-node ``interrupt()``. Left off in normal CLI
            operation, since two pauses on the same step is confusing.
    """
    # Imported here rather than at module scope so that patching a node
    # function before build_graph() is picked up by the compiled graph.
    from . import nodes

    builder = StateGraph(EmailState)

    builder.add_node("authenticate", nodes.authenticate_node)
    builder.add_node("read", nodes.read_node)
    builder.add_node("draft", nodes.draft_node)
    builder.add_node("ask_approval", nodes.ask_approval_node)
    builder.add_node("send", nodes.send_node)

    builder.add_edge(START, "authenticate")

    # Authenticate -> Read only when authenticated; otherwise straight to END.
    builder.add_conditional_edges(
        "authenticate",
        route_after_auth,
        {"read": "read", END: END},
    )

    builder.add_edge("read", "draft")

    # Draft -> approval gate when there is something to approve.
    builder.add_conditional_edges(
        "draft",
        route_after_draft,
        {"ask_approval": "ask_approval", "finished": END},
    )

    # The HITL step, and the only edge that can re-enter the drafting node.
    builder.add_conditional_edges(
        "ask_approval",
        route_after_approval,
        {"send": "send", "draft": "draft", "finished": END},
    )

    builder.add_edge("send", END)

    # The state carries Pydantic models, so the checkpointer must be told they
    # are trusted for (de)serialisation rather than relying on pickle fallback.
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("email_agent.state", "Email"),
            ("email_agent.state", "DraftReply"),
            ("email_agent.state", "HumanDecision"),
        ]
    )
    checkpointer = MemorySaver(serde=serializer)

    if static_breakpoint:
        return builder.compile(checkpointer=checkpointer, interrupt_before=["ask_approval"])
    return builder.compile(checkpointer=checkpointer)


def new_thread_id() -> str:
    return uuid.uuid4().hex


def initial_state(run_id: str | None = None, *, fetch_limit: int = 10, unread_only: bool = True) -> EmailState:
    """Seed state. ``authenticated`` is absent so it defaults to falsy."""
    return {
        "run_id": run_id or new_thread_id(),
        "authenticated": False,
        "unread_only": unread_only,
        "fetch_limit": fetch_limit,
        "revision_count": 0,
        "approval_status": "pending",
        "drafts": [],
        "emails": [],
        "errors": [],
        "sent_message_ids": [],
    }