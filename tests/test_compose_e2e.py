"""The Compose end-to-end script's throwaway credentials: on disk, readable only by the owner."""

import os
import stat
import sys
from pathlib import Path

import compose_e2e
import pytest

POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="Windows ignores the file mode")


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_the_env_file_holds_fresh_credentials(tmp_path: Path) -> None:
    path, user_password = compose_e2e.write_env(tmp_path)
    lines = dict(line.split("=", 1) for line in path.read_text("utf-8").splitlines())
    assert len(lines["POSTGRES_PASSWORD"]) >= 32
    # The login password itself isn't in the file, only its hash.
    assert user_password not in path.read_text("utf-8")
    assert lines["USER_PASSWORD_HASH"].startswith("'$argon2")


@POSIX_ONLY
def test_the_env_file_is_readable_only_by_its_owner(tmp_path: Path) -> None:
    path, _ = compose_e2e.write_env(tmp_path)
    assert mode(path) == 0o600


@POSIX_ONLY
def test_a_file_already_there_is_narrowed_too(tmp_path: Path) -> None:
    path = tmp_path / "user_password.txt"
    path.write_text("old", "utf-8")
    os.chmod(path, 0o644)
    compose_e2e.write_private(path, "new")
    assert mode(path) == 0o600
    assert path.read_text("utf-8") == "new"
