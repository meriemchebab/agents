"""Typed state and helper models for the Email Assistant graph.

Security note: `Credentials` deliberately lives *outside* the checkpointed graph
state (see `auth.py`). Only non-secret booleans such as `authenticated` cross the
node boundary, so a checkpointer dump can never leak an app password.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, EmailStr, Field


# --------------------------------------------------------------------------- #
# Domain models
# --------------------------------------------------------------------------- #


class Email(BaseModel):
    """A single fetched message, normalised to the fields the agent needs."""

    message_id: str
    thread_id: str | None = None
    sender_name: str | None = None
    sender_email: EmailStr
    subject: str = ""
    body: str = ""
    date: str | None = None
    is_unread: bool = False
    in_reply_to: str | None = None
    references: list[str] = Field(default_factory=list)


class DraftReply(BaseModel):
    """The LLM's structured proposal for a response to one message."""

    needs_reply: bool
    reason: str = ""
    to: list[EmailStr] = Field(default_factory=list)
    cc: list[EmailStr] = Field(default_factory=list)
    subject: str = ""
    body: str = ""
    in_reply_to: str | None = None


class HumanDecision(BaseModel):
    """Shape of the value the operator submits to resume the paused graph."""

    action: Literal["approve", "revise", "quit"] = "revise"
    feedback: str = ""


# --------------------------------------------------------------------------- #
# Internal accumulators (not part of the persisted surface)
# --------------------------------------------------------------------------- #


class ReadBuffer(TypedDict, total=False):
    """Collects drafts across the read node without exposing email bodies
    to every later node. Kept in state for checkpoint fidelity but never
    serialised into an LLM prompt on its own."""

    drafts: list[DraftReply]


def replace(_current, new) -> list:
    """Reducer that overwrites rather than accumulates.

    Used for fields a single node rewrites wholesale on each pass — notably
    ``emails``, so a revision loop never duplicates the inbox snapshot.
    """
    return new or []


def append(_current, new) -> list:
    """Reducer that accumulates values across nodes (e.g. an error log)."""
    return (_current or []) + (new or [])


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #


class EmailState(TypedDict, total=False):
    """Channel schema for the state graph.

    Every field is optional in the sense that a node only reads what it needs,
    but the `authenticated` invariant is enforced at runtime: `read_emails`,
    `draft_replies` and `send_email` all raise `AuthError` unless
    `authenticated is True`.
    """

    # -- identity / auth ----------------------------------------------------
    user_email: str | None
    """Authenticated mailbox address. Set only by the authenticate node."""
    authenticated: bool
    """The single source of truth for tool access. Defaults to False."""
    auth_error: str | None
    """Human-readable reason authentication failed, surfaced by the CLI."""

    # -- run configuration --------------------------------------------------
    run_id: str
    """Correlation id for logs and sent-message receipts."""
    revision_count: int
    """How many times the drafting node has run. Guards revision loops."""

    # -- inbox --------------------------------------------------------------
    unread_only: bool
    fetch_limit: int
    emails: Annotated[list[Email], replace]
    """Inbox snapshot. The reducer *overwrites* on each read to avoid
    duplicating messages when a draft is revised and the graph loops."""

    # -- drafting -----------------------------------------------------------
    drafts: list[DraftReply]
    """Drafts produced by the LLM. Cleared and rebuilt on each revision."""
    target_message_id: str | None
    """Which inbox message the current draft answers."""

    # -- human-in-the-loop --------------------------------------------------
    feedback: str | None
    """Latest operator feedback fed back into the drafting node."""
    approval_status: Literal["pending", "approved", "rejected", "not_required"]
    """Gate for the send node. Only `approved` permits dispatch."""
    sent_message_ids: list[str]
    """Receipts, populated by the send node for audit."""

    # -- errors -------------------------------------------------------------
    errors: Annotated[list[str], append]


class AuthError(PermissionError):
    """Raised when a node or tool attempts mailbox access before the
    authentication gate has been passed."""