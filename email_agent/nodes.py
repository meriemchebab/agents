"""Node implementations for the Email Assistant graph.

Node contract: each returns a partial state dict of channel updates. Nodes are
side-effect free before any ``interrupt()`` call, because LangGraph replays a
node from the top when it resumes.
"""

from __future__ import annotations

import logging
import smtplib
from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt

from . import tools
from .auth import authenticate, clear_credentials
from .state import AuthError, DraftReply, EmailState, HumanDecision

logger = logging.getLogger(__name__)

MAX_REVISIONS = 5
"""Backstop against a revise/approve feedback loop."""

@lru_cache(maxsize=1)
def _model():
    """Lazily constructed so importing this module never requires API keys."""
    return init_chat_model(model="claude-sonnet-4-5", model_provider="anthropic")

SYSTEM_PROMPT = """\
You are an email drafting assistant. For each incoming message decide whether a \
reply is genuinely required by the recipient.

Reply is NOT needed for: newsletters, marketing, automated notifications, \
no-reply senders, spam, or messages that only need archiving.
Reply IS needed for: direct questions, requests for action, and threads where \
the recipient is clearly the expected responder.

Draft requirements:
- Match the formality of the original message.
- Keep it concise: three sentences or fewer unless the source demands detail.
- Never invent facts, dates, commitments, or numbers that are not in the thread.
- If the message is ambiguous, draft a short clarifying reply.
- Set needs_reply=false and leave the body empty when no reply is warranted.
"""

USER_PROMPT = """\
Inbox message to assess:

From: {sender}
Subject: {subject}
Date: {date}

{body}

{drafting_note}
"""

REVISION_NOTE = """\
An earlier draft was returned for revision. The operator's feedback is the \
authoritative instruction — apply it directly.

Operator feedback:
{feedback}

Do not discard parts of the earlier draft that the feedback did not question.
"""


def _drafting_note(state: EmailState) -> str:
    feedback = (state.get("feedback") or "").strip()
    if not feedback:
        return "This is the first draft."
    return REVISION_NOTE.format(feedback=feedback)


# --------------------------------------------------------------------------- #
# 1. Authentication gate
# --------------------------------------------------------------------------- #


def authenticate_node(state: EmailState) -> dict[str, Any]:
    """Block the graph until mailbox credentials are proven valid.

    This is the only node that can flip ``authenticated`` to True, and the
    router reads nothing else before permitting an edge toward the read node.
    """
    run_id = state["run_id"]
    authenticated, user, _, error = authenticate(run_id)

    if not authenticated:
        logger.warning("Authentication gate closed for run %s: %s", run_id, error)
        clear_credentials(run_id)
        return {
            "authenticated": False,
            "user_email": user or None,
            "auth_error": error,
            "approval_status": "rejected",
        }

    logger.info("Authentication gate passed for %s", user)
    return {"authenticated": True, "user_email": user, "auth_error": None}


# --------------------------------------------------------------------------- #
# 2. Read
# --------------------------------------------------------------------------- #


def read_node(state: EmailState) -> dict[str, Any]:
    """Fetch the inbox snapshot. Refuses unless the auth gate has passed."""
    try:
        messages = tools.read_emails(state)
    except AuthError as exc:
        # Defence in depth: the router should already have prevented this.
        return {"authenticated": False, "errors": [str(exc)]}
    return {"emails": messages, "errors": []}


# --------------------------------------------------------------------------- #
# 3. Draft
# --------------------------------------------------------------------------- #


def draft_node(state: EmailState) -> dict[str, Any]:
    """LLM node: decide whether a reply is needed and draft one.

    Runs on the first pass and again on every revision, using ``feedback`` to
    refine the previous draft.
    """
    messages = state.get("emails") or []
    if not messages:
        return {"drafts": [], "target_message_id": None, "approval_status": "not_required"}

    revisions = int(state.get("revision_count", 0))
    if revisions > MAX_REVISIONS:
        logger.warning("Revision cap of %d reached; abandoning draft.", MAX_REVISIONS)
        return {
            "drafts": [],
            "approval_status": "rejected",
            "errors": [f"Revision limit ({MAX_REVISIONS}) reached without approval."],
        }

    previous = (state.get("drafts") or [None])[0]
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])
    chain = prompt | _model().with_structured_output(DraftReply)

    draft: DraftReply | None = None
    errors: list[str] = []
    for message in messages:
        note = _drafting_note(state)
        if previous is not None and revisions > 0:
            note += (
                "\n\nPrevious draft (subject: "
                f"{previous.subject!r}):\n{previous.body}\n"
            )
        payload = {
            "sender": f"{message.sender_name or ''} <{message.sender_email}>".strip(),
            "subject": message.subject,
            "date": message.date or "unknown",
            "body": message.body or "(empty body)",
            "drafting_note": note,
        }
        try:
            result = chain.invoke(payload)
        except Exception as exc:  # pragma: no cover - provider failures
            logger.exception("LLM drafting failed for %s", message.message_id)
            errors.append(f"Drafting failed for {message.message_id}: {exc}")
            continue

        if result.needs_reply:
            if not result.in_reply_to:
                result.in_reply_to = message.in_reply_to or message.message_id or None
            draft = result
            break

    if draft is None:
        return {
            "drafts": [],
            "target_message_id": None,
            "approval_status": "not_required",
            "revision_count": revisions + 1,
            "errors": errors,
        }

    target = next((m.message_id for m in messages if m.message_id == draft.in_reply_to), None)
    logger.info("Draft ready for approval (revision %d).", revisions + 1)
    return {
        "drafts": [draft],
        "target_message_id": target,
        # A fresh draft invalidates any prior approval.
        "approval_status": "pending",
        "feedback": None,
        "revision_count": revisions + 1,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# 4. Human approval (HITL interrupt)
# --------------------------------------------------------------------------- #


def ask_approval_node(state: EmailState) -> dict[str, Any]:
    """Pause the graph and block until a human decides.

    ``interrupt()`` suspends execution and persists the checkpoint. The
    operator's decision arrives via ``Command(resume=...)`` on the next invoke,
    which replays this node and returns the parsed :class:`HumanDecision`.

    Everything above the ``interrupt`` call must stay free of side effects —
    it re-runs on replay. Formatting and I/O belong below it.
    """
    draft = (state.get("drafts") or [None])[0]

    # -- suspend here until a decision is supplied ------------------------- #
    decision_payload = interrupt(
        {
            "event": "approval_required",
            "subject": draft.subject if draft else None,
            "to": draft.to if draft else [],
            "cc": draft.cc if draft else [],
            "body": draft.body if draft else "",
            "in_reply_to": draft.in_reply_to if draft else None,
            "revision": int(state.get("revision_count", 0)),
            "allowed_actions": ["approve", "revise", "quit"],
        }
    )

    # -- resumes here on the next invoke ----------------------------------- #
    try:
        decision = HumanDecision.model_validate(decision_payload)
    except Exception as exc:
        logger.warning("Unparsable approval payload %r: %s", decision_payload, exc)
        return {
            "approval_status": "pending",
            "feedback": "The approval response could not be parsed. Please try again.",
        }

    logger.info("Human decision: %s", decision.action)
    if decision.action == "approve":
        return {"approval_status": "approved", "feedback": None}
    if decision.action == "quit":
        clear_credentials(state["run_id"])
        return {"approval_status": "rejected", "drafts": [], "feedback": None}
    return {
        "approval_status": "pending",
        "feedback": decision.feedback or "Please revise the draft.",
    }


# --------------------------------------------------------------------------- #
# 5. Send
# --------------------------------------------------------------------------- #


def send_node(state: EmailState) -> dict[str, Any]:
    """Dispatch approved drafts. Raises rather than sending if the gate is open."""
    drafts = [d for d in (state.get("drafts") or []) if d.needs_reply]
    if state.get("approval_status") != "approved":
        return {
            "approval_status": "rejected",
            "errors": ["Send blocked: the draft was never approved."],
        }
    if not drafts:
        return {"errors": ["Nothing to send."]}

    sent: list[str] = []
    errors: list[str] = []
    for draft in drafts:
        try:
            message_id = tools.send_email(state, draft)
            if message_id:
                sent.append(message_id)
        except (AuthError, smtplib.SMTPException, OSError) as exc:
            errors.append(f"Send failed: {exc}")

    clear_credentials(state["run_id"])
    return {"sent_message_ids": sent, "errors": errors}