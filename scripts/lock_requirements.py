"""Compile every requirements .in file into a hash-locked .txt next to it.

A .in file holds the packages a service asks for, pinned by hand. Its .txt holds every package
that actually gets installed, each with the SHA-256 of its files, so pip refuses anything that
differs from what was locked (`pip install --require-hashes`). The .txt files are generated: edit
the .in, then run this.

    python scripts/lock_requirements.py              # lock every file, keeping current versions
    python scripts/lock_requirements.py --upgrade    # also move indirect packages to the newest
    python scripts/lock_requirements.py --check      # no network: is every lock complete?

Locks are universal (one file works on Linux amd64, Linux arm64, and Windows), because the laptop
is Windows, CI and Render are amd64, and the cluster is arm64. A .in file can change that with a
first line such as `# lock: --python-platform x86_64-manylinux_2_28`.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = (
    "requirements-*.in",
    "services/*/requirements*.in",
    "jobs/*/requirements*.in",
    ".github/requirements-*.in",
)
PYTHON_VERSION = "3.11"
OPTIONS_PREFIX = "# lock:"
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?==([^\s;\\]+)")


def inputs(root: Path = ROOT) -> list[Path]:
    return sorted(path for pattern in PATTERNS for path in root.glob(pattern))


def normalize(name: str) -> str:
    """PyPI treats -, _, and . as the same character, and ignores case."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pins(path: Path) -> dict[str, str]:
    """Each `name==version` in a file, following `-r other-file` lines."""
    found: dict[str, str] = {}
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if line.startswith("-r "):
            found |= pins(path.parent / line.removeprefix("-r ").strip())
        elif match := PIN.match(line):
            found[normalize(match.group(1))] = match.group(3)
    return found


def extra_options(path: Path) -> list[str]:
    lines = path.read_text("utf-8").splitlines()
    if lines and lines[0].startswith(OPTIONS_PREFIX):
        return lines[0].removeprefix(OPTIONS_PREFIX).split()
    return []


def problems(source: Path) -> list[str]:
    """What's wrong with one .in file's lock. Reads files only, so CI needs no resolver."""
    lock = source.with_suffix(".txt")
    name = source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else source.name
    if not lock.is_file():
        return [f"{name}: no lock file; run scripts/lock_requirements.py"]
    found = []
    locked = pins(lock)
    for package, version in pins(source).items():
        if locked.get(package) != version:
            found.append(
                f"{name}: {package}=={version} is asked for, but the lock has "
                f"{locked.get(package, 'nothing')}; run scripts/lock_requirements.py"
            )
    # Every locked package needs a hash, or pip would accept whatever the index serves.
    entries = re.split(r"\n(?=\S)", lock.read_text("utf-8"))
    for entry in entries:
        if PIN.match(entry) and "--hash=sha256:" not in entry:
            found.append(f"{name}: {entry.split()[0]} has no hash in the lock")
    if not locked:
        found.append(f"{name}: the lock pins nothing")
    return found


def compile_one(source: Path, upgrade: bool, constraints: Path | None) -> None:
    options = extra_options(source)
    command = [
        sys.executable,
        "-m",
        "uv",
        "pip",
        "compile",
        source.name,
        "--output-file",
        source.with_suffix(".txt").name,
        "--python-version",
        PYTHON_VERSION,
        "--generate-hashes",
        "--emit-index-url",
        "--custom-compile-command",
        "python scripts/lock_requirements.py",
        "--quiet",
    ]
    if not any(option.startswith("--python-platform") for option in options):
        command.append("--universal")
    if upgrade:
        command.append("--upgrade")
    if constraints is not None:
        command += ["--constraints", str(constraints.resolve())]
    # Run in the file's own folder, so the lock's "via -r" comments are the same on any machine.
    subprocess.run([*command, *options], check=True, cwd=source.parent)  # noqa: S603 - fixed command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify the locks, change nothing")
    parser.add_argument("--upgrade", action="store_true", help="move indirect packages forward")
    parser.add_argument(
        "--constraints", type=Path, help="a pip freeze to prefer, for the first lock of a file"
    )
    parser.add_argument("only", nargs="*", type=Path, help="lock just these .in files")
    args = parser.parse_args()
    sources = [path.resolve() for path in args.only] or inputs()
    if args.check:
        found = [problem for source in sources for problem in problems(source)]
        for problem in found:
            print(problem, file=sys.stderr)  # noqa: T201
        print(f"{len(sources)} lock files checked, {len(found)} problems")  # noqa: T201
        return 1 if found else 0
    for source in sources:
        print(f"locking {source.relative_to(ROOT).as_posix()}")  # noqa: T201
        compile_one(source, args.upgrade, args.constraints)
    return 0


if __name__ == "__main__":
    sys.exit(main())
