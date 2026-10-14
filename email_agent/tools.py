"""Mailbox tools: read and send.

Two invariants are enforced here rather than trusted to the graph:

1. ``_require_auth`` raises :class:`AuthError` unless the state carries
   ``authenticated=True`` *and* the run holds credentials in the vault.
2. ``send_email`` additionally requires ``approval_status == "approved"``.
   Even a mis-wired edge cannot dispatch an unapproved message.

Neither tool is decorated with LangChain's ``@tool`` on purpose: they are
called directly from node functions so that mailbox access cannot be triggered
by model-chosen tool calls. Binding them to the LLM is deliberately out of
scope for an agent with send capability.
"""

from __future__ import annotations

import email
import email.header
import email.utils
import imaplib
import logging
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

from .auth import load_credentials
from .state import AuthError, DraftReply, Email, EmailState

logger = logging.getLogger(__name__)

MAX_BODY_CHARS = 4000
"""Truncated before the body ever reaches the prompt to bound context cost."""


def _require_auth(state: EmailState) -> str:
    """Return the authenticated mailbox, or refuse. The auth gate."""
    if not state.get("authenticated"):
        raise AuthError(
            "Authentication gate not passed: refusing mailbox access. "
            "The graph must route through the `authenticate` node first."
        )
    run_id = state.get("run_id")
    if not run_id:
        raise AuthError("State is missing `run_id`; cannot resolve credentials.")
    load_credentials(run_id)  # raises if the vault entry is gone
    return state.get("user_email") or ""


def _decode(value: str | None) -> str:
    if not value:
        return ""
    parts = email.header.decode_header(value)
    return "".join(
        part.decode(charset or "utf-8", errors="replace") if isinstance(part, bytes) else part
        for part in parts
    )


def _to_email(msg: email.message.Message) -> Email:
    sender_name, sender_addr = email.utils.parseaddr(msg.get("From", ""))
    subject = _decode(msg.get("Subject"))

    plain: str | None = None
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                plain = payload.decode(part.get_content_charset() or "utf-8", "replace")
                break
    else:
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            plain = payload.decode(msg.get_content_charset() or "utf-8", "replace")
        elif isinstance(payload, str):
            plain = payload

    date = msg.get("Date")
    return Email(
        message_id=msg.get("Message-ID", ""),
        thread_id=(msg.get("X-Gmail-Thread-ID") or None),
        sender_name=sender_name or None,
        sender_email=sender_addr or "unknown@invalid",
        subject=subject,
        body=(plain or "")[:MAX_BODY_CHARS],
        date=_decode(date) or None,
        is_unread="\\Seen" not in (msg.get("Flags", "") or ""),
        in_reply_to=msg.get("In-Reply-To"),
        references=msg.get("References", "").split(),
    )


def read_emails(state: EmailState) -> list[Email]:
    """Fetch recent or unread messages from the authenticated INBOX.

    The connection is opened in ``readonly=True`` mode, so inspecting a
    message can never mark it as seen as a side effect of reading it.
    """
    run_id = state["run_id"]
    _require_auth(state)
    creds = load_credentials(run_id)

    limit = int(state.get("fetch_limit", 10))
    unread_only = bool(state.get("unread_only", False))
    criteria = "UNSEEN" if unread_only else "ALL"

    context = ssl.create_default_context()
    client = (
        imaplib.IMAP4_SSL(creds.imap_host, creds.imap_port, ssl_context=context)
        if creds.use_ssl
        else imaplib.IMAP4(creds.imap_host, creds.imap_port)
    )
    messages: list[Email] = []
    try:
        client.login(creds.user, creds.password)
        client.select("INBOX", readonly=True)
        status, data = client.search(None, criteria)
        if status != "OK":
            raise OSError(f"IMAP search failed: {status}")

        ids = data[0].split() if data and data[0] else []
        # Newest first, then cap — so `fetch_limit` means "most recent N".
        for uid in reversed(ids[-limit:]):
            status, payload = client.fetch(uid, "(RFC822)")
            if status != "OK" or not payload or not isinstance(payload[0], tuple):
                continue
            messages.append(_to_email(payload[0][1]))
    finally:
        try:
            client.logout()
        except Exception:  # pragma: no cover - best effort
            pass

    logger.info("Read %d message(s) for %s", len(messages), creds.user)
    return messages


def send_email(state: EmailState, draft: DraftReply) -> str | None:
    """Dispatch one approved draft over SMTP.

    Returns the provider-assigned ``Message-ID`` on success, ``None`` on
    failure (the caller records the failure; it does not raise, so the graph
    can present a retry choice instead of dying).
    """
    run_id = state["run_id"]
    _require_auth(state)

    if state.get("approval_status") != "approved":
        raise AuthError(
            f"Refusing to send: approval_status is {state.get('approval_status')!r}, "
            "not 'approved'. Sending requires explicit human approval."
        )
    if not draft.needs_reply:
        raise AuthError("Refusing to send: the draft does not require a reply.")
    if not draft.to or not draft.body.strip():
        raise AuthError("Refusing to send: draft has no recipients or an empty body.")

    creds = load_credentials(run_id)

    message = EmailMessage()
    message["From"] = creds.user
    message["To"] = ", ".join(draft.to)
    if draft.cc:
        message["Cc"] = ", ".join(draft.cc)
    message["Subject"] = draft.subject
    if draft.in_reply_to:
        message["In-Reply-To"] = draft.in_reply_to
        message["References"] = draft.in_reply_to
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain=creds.user.split("@")[-1])
    message.set_content(draft.body)

    context = ssl.create_default_context()
    with smtplib.SMTP(creds.smtp_host, creds.smtp_port, timeout=30) as smtp:
        if creds.smtp_starttls:
            smtp.starttls(context=context)
        smtp.login(creds.user, creds.password)
        smtp.send_message(message)

    message_id = message["Message-ID"]
    logger.info("Sent draft to %s (message-id=%s)", ", ".join(draft.to), message_id)
    return message_id