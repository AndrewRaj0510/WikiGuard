"""Who edited: logged-out detection, hashing, and keeping identifiers out of text and logs.

Since November 2025, logged-out enwiki editors get a generated name like ~2026-12345-67 instead
of showing an IP address. Only the API flags these accounts, and the live stream doesn't, so the
name is the one test that works on both (F-051, DEC-034).
"""

import hashlib
import hmac
import ipaddress
import logging
import re

TEMPORARY_ACCOUNT = re.compile(r"~\d{4}-\d{1,12}-\d{1,12}")
# Bounded repeats, so a crafted edit can't make these slow (section 8 of the plan). A dot after
# an address only rules it out when a digit follows, since "at 8.8.8.8." ends in one (F-066).
IPV4_IN_TEXT = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d|\.\d)")
# A run of hex digits and colons is only a candidate. ipaddress decides, because a pattern alone
# can't tell an address from a time like 12:34:56 (F-066).
IPV6_CANDIDATE = re.compile(r"(?<![0-9A-Fa-f:])[0-9A-Fa-f:]{6,64}(?![0-9A-Fa-f:])")
# Below this, too much ordinary text parses as an address, and C++'s "d::c" is one example.
MIN_IPV6_COLONS = 3
REDACTED_ACCOUNT = "[temporary account]"
REDACTED_IP = "[ip address]"


def is_temporary_account(name: str) -> bool:
    return TEMPORARY_ACCOUNT.fullmatch(name) is not None


def is_ip_address(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def is_logged_out(name: str, anon_flag: bool = False) -> bool:
    """True for temporary accounts, and for older edits that still show an IP address."""
    return anon_flag or is_temporary_account(name) or is_ip_address(name)


def hash_user(name: str, key: str) -> str:
    """HMAC rather than a plain hash, so a database leak can't be reversed from known usernames."""
    normalized = name.replace("_", " ").strip()
    return hmac.new(key.encode(), normalized.encode(), hashlib.sha256).hexdigest()


def _is_ipv6(text: str) -> bool:
    try:
        ipaddress.IPv6Address(text)
    except ValueError:
        return False
    return True


def _redact_ipv6(match: re.Match[str]) -> str:
    run = match.group()
    if run.count(":") < MIN_IPV6_COLONS:
        return run
    # Comments write "User talk:2001:...: reverted", so a colon can touch either end.
    lead = ":" if run.startswith(":") and not run.startswith("::") else ""
    trail = ":" if run.endswith(":") and not run.endswith("::") else ""
    core = run[len(lead) : len(run) - len(trail)]
    if _is_ipv6(core):
        return f"{lead}{REDACTED_IP}{trail}"
    # A word made only of hex letters can touch it too, as in "Added:2001:...".
    word, _, rest = core.partition(":")
    if _is_ipv6(rest):
        return f"{lead}{word}:{REDACTED_IP}{trail}"
    return run


def redact_identifiers(text: str) -> str:
    """Remove temporary account names and IP addresses, keeping everything else."""
    text = TEMPORARY_ACCOUNT.sub(REDACTED_ACCOUNT, text)
    text = IPV4_IN_TEXT.sub(REDACTED_IP, text)
    return IPV6_CANDIDATE.sub(_redact_ipv6, text)


# httpx and httpcore log each request's full URL at INFO. A URL can hold a secret (a Discord
# webhook, a Healthchecks.io ping) or an editor's name (list=users), so only their warnings and
# errors get through (F-143).
QUIET_BELOW_WARNING = ("httpx", "httpcore")


class RedactingFilter(logging.Filter):
    """Attach to a log handler so an identifier in any message never reaches the logs, and
    request URLs from the HTTP client are never written at all."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING and record.name.startswith(QUIET_BELOW_WARNING):
            return False
        record.msg = redact_identifiers(record.getMessage())
        record.args = None
        return True
