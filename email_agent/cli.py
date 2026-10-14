"""Terminal front-end: drives the graph and serves the approval prompt.

The approval interaction is the interesting part. ``graph.invoke`` returns a
normal result whose ``__interrupt__`` key holds whatever was passed to
``interrupt()``. The run is then resumed with ``Command(resume=<decision>)``,
which replays ``ask_approval`` and hands the decision back to it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from langgraph.types import Command

from .graph import build_graph, initial_state, new_thread_id
from .state import EmailState

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Terminal rendering
# --------------------------------------------------------------------------- #

RULE = "=" * 72
THIN = "-" * 72


def _ascii(text: str) -> str:
    """Keep terminal output safe on legacy Windows code pages (cp1252)."""
    return text.encode("ascii", "replace").decode("ascii")


def _wrap(text: str, width: int = 76, indent: str = "  ") -> str:
    import textwrap

    return textwrap.fill(text or "", width=width, initial_indent=indent, subsequent_indent=indent)


def render_interrupt(payload: dict[str, Any]) -> None:
    """Present a pending draft and the approval options."""
    print(f"\n{RULE}")
    print("  APPROVAL REQUIRED - nothing has been sent")
    print(RULE)
    print(f"  To:      {', '.join(payload.get('to') or ['(none)'])}")
    if payload.get("cc"):
        print(f"  Cc:      {', '.join(payload['cc'])}")
    print(f"  Subject: {payload.get('subject') or '(no subject)'}")
    print(f"  Draft:   revision {payload.get('revision', 1)}")
    print(THIN)
    print(_ascii(_wrap(payload.get("body", ""))))
    print(THIN)
    print("  [a] approve and send")
    print("  [r] revise - then type feedback, e.g. 'make it more professional'")
    print("  [q] quit without sending")
    print(RULE)


def collect_decision() -> dict[str, str]:
    """Read the operator's choice. Returns a resume payload."""
    while True:
        try:
            raw = input("  Your choice [a/r/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            return {"action": "quit", "feedback": ""}

        if raw in {"a", "approve", "y", "yes"}:
            return {"action": "approve", "feedback": ""}
        if raw in {"q", "quit", "n", "no"}:
            return {"action": "quit", "feedback": ""}
        if raw in {"r", "revise"}:
            try:
                feedback = input("  Feedback for the assistant: ").strip()
            except (EOFError, KeyboardInterrupt):
                return {"action": "quit", "feedback": ""}
            return {"action": "revise", "feedback": feedback or "Please rewrite the draft."}
        print("  Unrecognised choice. Enter a, r, or q.")


def render_summary(state: EmailState) -> None:
    print(f"\n{RULE}")
    print("  RUN SUMMARY")
    print(RULE)
    print(f"  Authenticated : {state.get('authenticated')}")
    if state.get("auth_error"):
        print(f"  Auth error    : {state['auth_error']}")
    print(f"  Messages read : {len(state.get('emails') or [])}")
    print(f"  Revisions     : {state.get('revision_count', 0)}")
    print(f"  Approval      : {state.get('approval_status')}")
    sent = state.get("sent_message_ids") or []
    print(f"  Sent          : {len(sent)}")
    for message_id in sent:
        print(f"    - {message_id}")
    for error in state.get("errors") or []:
        print(f"  Error         : {error}")
    print(RULE)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run(
    fetch_limit: int,
    unread_only: bool,
    provider: str | None,
    static_breakpoint: bool = False,
) -> int:
    graph = build_graph(static_breakpoint=static_breakpoint)
    run_id = new_thread_id()
    config = {"configurable": {"thread_id": run_id}}

    state: EmailState = initial_state(
        run_id, fetch_limit=fetch_limit, unread_only=unread_only
    )

    if provider:
        import os

        os.environ.setdefault("EMAIL_PROVIDER", provider)

    print(f"Run {run_id}. Authenticate to begin.")
    result = graph.invoke(state, config=config)

    # ---- approval loop --------------------------------------------------- #
    # Each resume may produce another interrupt (revision) or may finish.
    while True:
        interrupts = result.get("__interrupt__") or []
        if not interrupts:
            break

        raw = interrupts[-1].value if hasattr(interrupts[-1], "value") else interrupts[-1]

        if isinstance(raw, dict) and raw.get("event") == "approval_required":
            render_interrupt(raw)
            decision = collect_decision()
        else:
            # A static interrupt_before breakpoint halts with no payload; read
            # the pending draft straight out of the checkpoint instead.
            snapshot = graph.get_state(config).values
            pending = (snapshot.get("drafts") or [None])[0]
            if pending is None:
                print("\n  Paused with no draft to approve.")
                decision = {"action": "quit", "feedback": ""}
            else:
                render_interrupt(
                    {
                        "event": "approval_required",
                        "to": pending.to,
                        "cc": pending.cc,
                        "subject": pending.subject,
                        "body": pending.body,
                        "revision": snapshot.get("revision_count", 1),
                    }
                )
                decision = collect_decision()

        print(f"  -> {decision['action']}", flush=True)
        result = graph.invoke(Command(resume=decision), config=config)

    final = result if isinstance(result, dict) else graph.get_state(config).values
    render_summary(final)

    return 0 if (final.get("approval_status") == "approved" or final.get("sent_message_ids")) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Human-in-the-loop Email Assistant Agent")
    parser.add_argument("--limit", type=int, default=10, help="Max messages to fetch (default: 10)")
    parser.add_argument("--all-mail", action="store_true", help="Fetch all mail, not just unread")
    parser.add_argument("--provider", choices=["gmail", "outlook"], help="Credential endpoints")
    parser.add_argument("--static-breakpoint", action="store_true",
                        help="Also register a static breakpoint before ask_approval")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.static_breakpoint:
        logger.warning("static_breakpoint is a debugging mode; the in-node interrupt already pauses.")

    try:
        return run(args.limit, not args.all_mail, args.provider, args.static_breakpoint)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())