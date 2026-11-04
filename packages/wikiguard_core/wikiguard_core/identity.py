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
# Bounded repeats, so a crafted edit can't make these slow (section 8 of the plan).
IPV4_IN_TEXT = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6_IN_TEXT = re.compile(
    r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){2,7}(?::?[0-9A-Fa-f]{1,4}){1,7}(?![\w:])"
)
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


def redact_identifiers(text: str) -> str:
    """Remove temporary account names and IP addresses, keeping everything else."""
    text = TEMPORARY_ACCOUNT.sub(REDACTED_ACCOUNT, text)
    text = IPV4_IN_TEXT.sub(REDACTED_IP, text)
    return IPV6_IN_TEXT.sub(REDACTED_IP, text)


class RedactingFilter(logging.Filter):
    """Attach to a log handler so an identifier in any message never reaches the logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_identifiers(record.getMessage())
        record.args = None
        return True
