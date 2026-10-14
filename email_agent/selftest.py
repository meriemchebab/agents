"""End-to-end checks with the network and the LLM stubbed out.

Run: python -m email_agent.selftest
"""

from __future__ import annotations

import sys
from unittest.mock import patch

from langgraph.types import Command

from . import tools as tools_mod
from .graph import build_graph, initial_state
from .state import AuthError, DraftReply, Email, HumanDecision

PASS, FAIL = "  PASS", "  FAIL"
results: list[tuple[bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    results.append((condition, label))
    print(f"{PASS if condition else FAIL}  {label}{(' -> ' + detail) if detail and not condition else ''}")


# --------------------------------------------------------------------------- #
# 1. Security: the auth gate blocks mailbox access
# --------------------------------------------------------------------------- #


def test_auth_gate_blocks_tools() -> None:
    print("\n[1] Authentication gate")
    unauth = {"run_id": "r1", "authenticated": False}

    try:
        tools_mod.read_emails(unauth)  # type: ignore[arg-type]
        check("read_emails refuses when unauthenticated", False, "no exception raised")
    except AuthError:
        check("read_emails refuses when unauthenticated", True)

    draft = DraftReply(
        needs_reply=True, to=["someone@example.com"], subject="Hi", body="Hello there"
    )
    try:
        tools_mod.send_email({**unauth, "approval_status": "approved"}, draft)  # type: ignore[arg-type]
        check("send_email refuses when unauthenticated", False, "no exception raised")
    except AuthError:
        check("send_email refuses when unauthenticated", True)


def test_send_requires_approval() -> None:
    print("\n[2] Send gate")
    from .auth import Credentials, clear_credentials, store_credentials

    state = {"run_id": "r1", "authenticated": True, "approval_status": "pending"}
    draft = DraftReply(needs_reply=True, to=["a@b.com"], subject="s", body="b")

    store_credentials(
        "r1",
        Credentials(
            user="me@example.com", password="pw", imap_host="imap.gmail.com",
            smtp_host="smtp.gmail.com", smtp_port=587,
        ),
    )
    sent: list = []

    class FakeSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, context=None):
            pass

        def login(self, user, password):
            pass

        def send_message(self, msg):
            sent.append(msg)

    with patch("smtplib.SMTP", FakeSMTP):
        try:
            tools_mod.send_email(state, draft)  # type: ignore[arg-type]
            check("send_email refuses when approval_status != approved", False, "no exception")
        except AuthError as exc:
            check("send_email refuses when approval_status != approved", True, str(exc))

        message_id = tools_mod.send_email(
            {**state, "approval_status": "approved"}, draft  # type: ignore[arg-type]
        )
        check(
            "send_email dispatches once approved (mocked SMTP)",
            len(sent) == 1 and bool(message_id),
            f"sent={len(sent)} id={message_id}",
        )
        if sent:
            msg = sent[0]
            check(
                "SMTP message has From/To/Subject/Body",
                msg["From"] == "me@example.com"
                and msg["To"] == "a@b.com"
                and msg["Subject"] == "s"
                and "b" in msg.get_content(),
                str(dict(msg.items())),
            )

        # A draft that does not require a reply must never be dispatched.
        before = len(sent)
        try:
            tools_mod.send_email(
                {**state, "approval_status": "approved"},
                DraftReply(needs_reply=False, to=["a@b.com"], subject="s", body=""),  # type: ignore[arg-type]
            )
            check("send_email refuses a no-reply draft", False, "no exception")
        except AuthError:
            check("send_email refuses a no-reply draft", len(sent) == before)

    clear_credentials("r1")


def test_secrets_not_in_state() -> None:
    print("\n[3] Secret hygiene")
    import json

    from .auth import Credentials, clear_credentials, store_credentials

    clear_credentials("r1")
    store_credentials(
        "r1",
        Credentials(user="me@example.com", password="super-secret", imap_host="imap.gmail.com"),
    )
    blob = json.dumps(initial_state("r1"), default=str)
    check("app password absent from checkpointed state", "super-secret" not in blob)
    clear_credentials("r1")


# --------------------------------------------------------------------------- #
# 2. HITL: the interrupt/resume/revise loop
# --------------------------------------------------------------------------- #


INBOX = [
    Email(
        message_id="<m1@example.com>",
        sender_name="Dana Client",
        sender_email="dana@client.com",
        subject="Question about the proposal",
        body="Hi, could you send the updated pricing by Friday?",
        is_unread=True,
    )
]


def stubs_send(send_calls: list[str], draft: DraftReply) -> str:
    """Stand-in for the SMTP tool that records what would have gone out."""
    send_calls.append(draft.body)
    return f"<sent-{len(send_calls)}@example.com>"


def _make_stubs(send_calls: list[str]) -> dict:
    """Deterministic replacements for the auth and LLM-drafting nodes.

    The first draft is plain; the second (post-feedback) is deliberately
    different so the test can prove the revision loop actually re-drafts.
    """
    revisions = {"n": 0}

    def passing_auth(state):
        return {"authenticated": True, "user_email": "me@example.com", "auth_error": None}

    def fake_draft(state):
        n = revisions["n"] = revisions["n"] + 1
        body = (
            "Thanks for reaching out."
            if n == 1
            else "Thank you for your inquiry. I will gladly assist."
        )
        return {
            "drafts": [
                DraftReply(
                    needs_reply=True,
                    to=["dana@client.com"],
                    subject="Re: Question about the proposal",
                    body=body,
                    in_reply_to="<m1@example.com>",
                )
            ],
            "target_message_id": "<m1@example.com>",
            "approval_status": "pending",
            "feedback": None,
            "revision_count": n,
            "errors": [],
        }

    return {"authenticate_node": passing_auth, "draft_node": fake_draft}


def test_interrupt_resume_loop() -> None:
    print("\n[4] HITL interrupt loop")
    from . import nodes as nodes_mod
    from .state import EmailState

    send_calls: list[str] = []
    stubs = _make_stubs(send_calls)

    with patch.multiple(nodes_mod, **stubs), patch.multiple(
        tools_mod, read_emails=lambda state: INBOX, send_email=lambda s, d: stubs_send(send_calls, d)
    ):
        graph = build_graph()
        config = {"configurable": {"thread_id": "hitl"}}
        state: EmailState = initial_state("hitl")

        # -- pass 1: graph pauses at the approval interrupt ---------------- #
        result = graph.invoke(state, config=config)
        interrupts = result.get("__interrupt__") or []
        check("graph pauses at approval", bool(interrupts), str(result)[:300])
        if interrupts:
            payload = interrupts[-1].value
            check(
                "interrupt payload carries the draft",
                payload.get("event") == "approval_required"
                and payload.get("subject") == "Re: Question about the proposal",
                str(payload)[:200],
            )
            check("nothing sent before approval", not send_calls)

            # -- pass 2: revise -> loop back to drafting ------------------ #
            result = graph.invoke(
                Command(resume={"action": "revise", "feedback": "Make it more professional"}),
                config=config,
            )
            interrupts = result.get("__interrupt__") or []
            check("revise pauses again for approval", bool(interrupts), str(result)[:200])
            check("still nothing sent after revise", not send_calls)
            if interrupts:
                revised = interrupts[-1].value
                check(
                    "revised body reflects the second drafting pass",
                    "gladly assist" in revised.get("body", ""),
                    revised.get("body", ""),
                )
                check("revision counter incremented", revised.get("revision") == 2,
                      str(revised.get("revision")))

                # -- pass 3: approve -> send ---------------------------- #
                result = graph.invoke(Command(resume={"action": "approve", "feedback": ""}), config=config)
                check("approve reaches the send node", bool(send_calls), "send not called")
                check("exactly one message sent", len(send_calls) == 1, str(send_calls))
                check(
                    "the approved (revised) body was the one sent",
                    bool(send_calls) and "gladly assist" in send_calls[0],
                    str(send_calls),
                )
                final = result if isinstance(result, dict) else graph.get_state(config).values
                check("approval_status recorded as approved",
                      final.get("approval_status") == "approved", str(final.get("approval_status")))
                check("message-id receipt recorded", bool(final.get("sent_message_ids")))


def test_quit_blocks_send() -> None:
    print("\n[5] Quit path")
    from . import nodes as nodes_mod

    send_calls: list[str] = []
    stubs = _make_stubs(send_calls)
    with patch.multiple(nodes_mod, **stubs), patch.multiple(
        tools_mod, read_emails=lambda state: INBOX, send_email=lambda s, d: stubs_send(send_calls, d)
    ):
        graph = build_graph()
        config = {"configurable": {"thread_id": "quit"}}
        graph.invoke(initial_state("quit"), config=config)
        result = graph.invoke(Command(resume={"action": "quit", "feedback": ""}), config=config)
        check("quit sends nothing", not send_calls)
        final = result if isinstance(result, dict) else graph.get_state(config).values
        check("approval_status is rejected", final.get("approval_status") == "rejected",
              str(final.get("approval_status")))


def test_auth_failure_ends_early() -> None:
    print("\n[6] Auth failure routing")
    from . import nodes as nodes_mod

    read_calls: list[int] = []

    def failing_auth(state):
        return {
            "authenticated": False,
            "user_email": None,
            "auth_error": "IMAP rejected the credentials: [AUTHENTICATIONFAILED]",
            "approval_status": "rejected",
        }

    with patch.multiple(nodes_mod, authenticate_node=failing_auth), patch.multiple(
        tools_mod, read_emails=lambda state: read_calls.append(1)  # type: ignore[func-returns-value]
    ):
        graph = build_graph()
        config = {"configurable": {"thread_id": "noauth"}}
        result = graph.invoke(initial_state("noauth"), config=config)
        check("read node never runs without auth", not read_calls)
        check("no interrupt raised without auth", not (result.get("__interrupt__") or []))
        check("auth_error surfaced", bool(result.get("auth_error")), str(result.get("auth_error")))
        check("route ended at END without auth", not (result.get("drafts") or []))


def main() -> int:
    print("Email Assistant - self test (network and LLM stubbed)")
    test_auth_gate_blocks_tools()
    test_send_requires_approval()
    test_secrets_not_in_state()
    test_interrupt_resume_loop()
    test_quit_blocks_send()
    test_auth_failure_ends_early()

    passed = sum(1 for ok, _ in results if ok)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        print("FAILED:")
        for ok, label in results:
            if not ok:
                print(f"  - {label}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())