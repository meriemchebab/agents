# Email Assistant Agent (Human-in-the-Loop)

A LangGraph state graph that reads unread mail, drafts a reply, and **cannot send
anything without an explicit human approval at a real execution pause**.

Built on LangGraph 1.2 / Python 3.13, matching the tooling already in this repo.

---


Two edge types do the routing:

- **Conditional edges** gate the two decision points — post-auth (`route_after_auth`)
  and post-approval (`route_after_approval`), which owns the revise loop.
- **A pause** in the middle of the approval node, which is the whole point.

---

## The HITL pause: two complementary mechanisms

### 1. `interrupt()` inside `ask_approval_node` — the production path

`interrupt()` suspends execution and writes a checkpoint. The node then
**replays from the top** on resume and receives the value passed to
`Command(resume=...)`:

```python
def ask_approval_node(state):
    draft = (state.get("drafts") or [None])[0]

    # Everything ABOVE this line must be side-effect free — it re-runs on replay.
    decision_payload = interrupt({
        "event": "approval_required",
        "subject": draft.subject,
        "to": draft.to,
        "body": draft.body,
        "revision": state.get("revision_count", 0),
        "allowed_actions": ["approve", "revise", "quit"],
    })

    # Everything BELOW this line only runs once a decision exists.
    decision = HumanDecision.model_validate(decision_payload)
    if decision.action == "approve":
        return {"approval_status": "approved"}
    ...
```

The CLI renders that payload, collects a choice, and resumes:

```python
result = graph.invoke(state, config)                       # runs -> pauses
interrupts = result["__interrupt__"]                       # the draft payload
render_interrupt(interrupts[-1].value)
decision = collect_decision()                              # a / r / q
result = graph.invoke(Command(resume=decision), config)    # resumes
```

**Feedback loop:** `revise` writes `feedback` back into state and routes to
`draft`. That node reads `feedback`, injects it into the prompt alongside the
previous draft, clears `approval_status` back to `pending`, and the graph
pauses again. `MAX_REVISIONS = 5` caps the loop.

### 2. `interrupt_before=["ask_approval"]` — the debugging / UI path

Available via `build_graph(static_breakpoint=True)`. It halts *before* the node
with no payload, so a non-interactive front-end drives it through
`get_state()` / `update_state()` instead. Kept off by default because two pauses
on one step is confusing in the terminal.

The interruption is what **gates** the send, not merely what warns about it —
`send_node` independently refuses unless `approval_status == "approved"`, so a
mis-wired edge still cannot dispatch an unapproved message.

---

## Security model

| Control | Implementation |
|---|---|
| Auth gate | Only `authenticate_node` sets `authenticated=True`. `route_after_auth` is the single edge into `read`. |
| Tool-level refusal | `_require_auth()` raises `AuthError` from `read_emails` and `send_email` regardless of routing. |
| Send-level refusal | `send_email` raises unless `approval_status == "approved"`. |
| Secrets out of state | Passwords live in a process-local `_VAULT` keyed by `run_id`, never in the checkpoint. `Credentials.__repr__` redacts. |
| Read-only fetch | IMAP opens with `readonly=True`, so reading never marks a message as seen. |
| No auto-tool-calling | Tools are plain functions called by nodes, **not** `@tool`-bound to the LLM. A model cannot invoke `send_email`. |
| Prompt-injection posture | Email bodies are data inside a bounded prompt, never instructions. |
| Revision cap | `MAX_REVISIONS` bounds the revise loop. |
| Readonly vault | `.env` holds an **app password**, never the account password. |

Verified by `python -m email_agent.selftest` — 25 checks, all passing, covering
the auth gate, the send gate, secret hygiene, the full approve/revise/approve
cycle, the quit path, and auth-failure routing.

---

## Files

| File | Contents |
|---|---|
| `state.py` | `EmailState` schema, `Email` / `DraftReply` / `HumanDecision` models, `AuthError`, reducers |
| `auth.py` | Credential resolution (args → env → hidden prompt), IMAP verification, the vault |
| `tools.py` | `read_emails()` and `send_email()` with the auth + approval guards |
| `nodes.py` | `authenticate_node`, `read_node`, `draft_node`, `ask_approval_node`, `send_node` |
| `graph.py` | Graph assembly, conditional routers, checkpointer, `initial_state()` |
| `cli.py` | Terminal front-end and the approval/resume loop |
| `selftest.py` | Stubbed end-to-end tests |

---

## Setup

Add to `.env`:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...

EMAIL_PROVIDER=gmail      # or outlook
EMAIL_USER=you@example.com
EMAIL_PASSWORD=your-16-char-app-password
```

Gmail: Google Account → Security → 2-Step Verification → **App passwords**.
Outlook: enable SMTP AUTH for the mailbox.

> Use an app password, never your primary account password. Gmail and Outlook
> both refuse IMAP/SMTP login with the account password.

## Usage

```bash
python -m email_agent                 # unread only, up to 10
python -m email_agent --limit 25
python -m email_agent --all-mail
python -m email_agent --provider outlook --verbose
```

At the prompt:

```
  [a] approve and send
  [r] revise - then type feedback, e.g. 'make it more professional'
  [q] quit without sending
```

---

## Swapping in a different LLM

`nodes._model()` is the single point of change:

```python
init_chat_model(model="gemini-3.6-flash", model_provider="google_genai")
init_chat_model(model="gpt-4o", model_provider="openai")
```

The `with_structured_output(DraftReply)` contract is provider-agnostic.

## Known limitations

- IMAP app-password flow only. `auth.py` exposes an `AuthMethod` hook for
  adding an OAuth2 callback handler.
- Messages are assessed independently; there is no cross-message thread reasoning.
- `MemorySaver` is per-process, so a paused run cannot survive a restart.
  Swap in `SqliteSaver` and move the `_VAULT` to a keyring to fix both.
- One draft per run — the drafting node stops at the first message needing a reply.