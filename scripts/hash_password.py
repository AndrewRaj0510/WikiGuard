"""Print an Argon2id hash for USER_PASSWORD_HASH. The password is typed, never an argument.

python scripts/hash_password.py
"""

import getpass
import sys

from argon2 import PasswordHasher

MIN_LENGTH = 16


def make_hash(password: str) -> str:
    if len(password) < MIN_LENGTH:
        raise ValueError(f"use at least {MIN_LENGTH} characters, ideally a generated password")
    return PasswordHasher().hash(password)


def main() -> int:
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Again: "):
        print("They don't match.", file=sys.stderr)
        return 1
    try:
        print(make_hash(first))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
