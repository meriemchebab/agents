"""Authentication gate: IMAP/SMTP credential acquisition and storage.

Design constraints:

* Secrets never enter the checkpointed graph state. They are held in a
  process-local `_VAULT` keyed by ``run_id`` and handed to the tool layer
  directly.
* A failed authentication returns ``authenticated=False`` rather than
  raising, so the graph can route to a clean terminal state instead of
  crashing mid-run.
"""

from __future__ import annotations

import getpass
import imaplib
import logging
import os
import smtplib
import ssl
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Credentials:
    """Validated IMAP/SMTP credentials for one mailbox."""

    user: str
    password: str = field(repr=False)
    imap_host: str
    imap_port: int = 993
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_ssl: bool = True
    smtp_starttls: bool = True

    def __repr__(self) -> str:  # pragma: no cover - defensive
        return f"Credentials(user={self.user!r}, imap={self.imap_host!r}, password=<redacted>)"


# Process-local secret store. Deliberately NOT part of the graph state: a
# MemorySaver/SqliteSaver dump must never contain an app password.
_VAULT: dict[str, Credentials] = {}


def store_credentials(run_id: str, creds: Credentials) -> None:
    _VAULT[run_id] = creds


def load_credentials(run_id: str) -> Credentials:
    try:
        return _VAULT[run_id]
    except KeyError as exc:
        raise PermissionError(f"No credentials held for run {run_id!r}; authenticate first.") from exc


def clear_credentials(run_id: str) -> None:
    _VAULT.pop(run_id, None)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #

PROVIDERS: dict[str, dict[str, str | int]] = {
    "gmail": {
        "imap_host": "imap.gmail.com",
        "imap_port": 993,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
    },
    "outlook": {
        "imap_host": "outlook.office365.com",
        "imap_port": 993,
        "smtp_host": "smtp.office365.com",
        "smtp_port": 587,
    },
}


def _verify(creds: Credentials) -> None:
    """Prove the credentials work by opening and closing an IMAP session.

    This is the security-critical step: until it returns without raising, no
    tool is permitted to touch the mailbox.
    """
    context = ssl.create_default_context()
    if creds.use_ssl:
        imap = imaplib.IMAP4_SSL(creds.imap_host, creds.imap_port, ssl_context=context)
    else:
        imap = imaplib.IMAP4(creds.imap_host, creds.imap_port)
    try:
        imap.login(creds.user, creds.password)
        imap.select("INBOX", readonly=True)
    finally:
        try:
            imap.logout()
        except Exception:  # pragma: no cover - best effort
            pass


def authenticate(
    run_id: str,
    *,
    provider: str | None = None,
    user: str | None = None,
    password: str | None = None,
    interactive: bool = True,
) -> tuple[bool, str, str | None, str | None]:
    """Resolve credentials for ``run_id`` and verify them.

    Resolution order, most explicit first:

    1. Explicit ``user``/``password`` arguments.
    2. Environment variables (``EMAIL_USER`` / ``EMAIL_PASSWORD``).
    3. A hidden terminal prompt (``getpass``), only if ``interactive``.

    Returns ``(authenticated, user_email, error)``.
    """
    load_dotenv(override=False)

    resolved_user = user or os.getenv("EMAIL_USER")
    resolved_password = password or os.getenv("EMAIL_PASSWORD")

    # A Ctrl-D / non-interactive stdin must close the gate cleanly rather than
    # propagating EOFError out of the node and tearing down the graph.
    if not resolved_user and interactive:
        try:
            resolved_user = input("Mailbox address: ").strip()
        except (EOFError, KeyboardInterrupt):
            return False, "", None, "Authentication cancelled at the prompt."

    if not resolved_user:
        return False, "", None, "No mailbox address supplied."

    if not resolved_password and interactive:
        try:
            resolved_password = getpass.getpass(f"App password for {resolved_user}: ")
        except (EOFError, KeyboardInterrupt):
            return False, resolved_user, None, "Authentication cancelled at the password prompt."

    if not resolved_password:
        return (
            False,
            resolved_user,
            None,
            "No password or app-password supplied. Use an OAuth2 token or an IMAP "
            "app password — never your primary account password.",
        )

    key = (provider or os.getenv("EMAIL_PROVIDER") or "").lower()
    endpoint = PROVIDERS.get(key, {})

    creds = Credentials(
        user=resolved_user,
        password=resolved_password,
        imap_host=str(endpoint.get("imap_host", os.getenv("IMAP_HOST", "imap.gmail.com"))),
        imap_port=int(endpoint.get("imap_port", int(os.getenv("IMAP_PORT", 993)))),  # type: ignore[arg-type]
        smtp_host=str(endpoint.get("smtp_host", os.getenv("SMTP_HOST", "smtp.gmail.com"))),
        smtp_port=int(endpoint.get("smtp_port", int(os.getenv("SMTP_PORT", 587)))),  # type: ignore[arg-type]
        use_ssl=os.getenv("IMAP_USE_SSL", "1") not in {"0", "false", "False"},
    )

    try:
        _verify(creds)
    except imaplib.IMAP4.error as exc:
        logger.warning("IMAP authentication failed: %s", exc)
        return False, resolved_user, None, f"IMAP rejected the credentials: {exc}"
    except (OSError, ssl.SSLError) as exc:
        logger.warning("IMAP connection failed: %s", exc)
        return False, resolved_user, None, f"Could not reach {creds.imap_host}: {exc}"

    store_credentials(run_id, creds)
    return True, resolved_user, None, None


AuthMethod = Literal["app_password", "oauth2", "cli_prompt"]
"""Documented extension point: an OAuth2 callback handler can populate the
vault here in place of the app-password flow."""