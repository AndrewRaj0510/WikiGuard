"""Give back the disk space a local build or test run used. Shared by the scripts that build images.

Docker keeps three things after a build, and on Docker Desktop all of them live in one virtual disk
that only grows: the image itself, BuildKit's layer cache, and any volume a container left behind.
On the laptop, WikiGuard's builds and kind cluster came to tens of GB this way (2026-11-13). Each
script that builds calls these when it's done, unless it's told to keep things (`--keep-images`).

The build-cache prune is Docker-wide: it drops every cache entry no image or running build uses,
other projects' too. That costs them a slower next build, never data.
"""

import subprocess
from collections.abc import Iterable


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed tool, arguments from these scripts
        ["docker", *args],  # noqa: S607
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def remove_images(tags: Iterable[str]) -> list[str]:
    """Delete these images from the host. A missing one is skipped. Returns the ones removed."""
    removed = []
    for tag in tags:
        if _docker("image", "rm", "--force", tag).returncode == 0:
            removed.append(tag)
    return removed


def images_matching(reference: str) -> list[str]:
    """Tags matching a pattern such as `kind.local/wikiguard-*`."""
    listed = _docker(
        "image", "ls", "--filter", f"reference={reference}", "--format", "{{.Repository}}:{{.Tag}}"
    )
    return [line for line in listed.stdout.split() if line and not line.endswith(":<none>")]


def container_volumes(container: str) -> list[str]:
    """Named and anonymous volumes a container mounts, so they can be removed with it."""
    found = _docker(
        "inspect",
        "--format",
        '{{range .Mounts}}{{if eq .Type "volume"}}{{.Name}} {{end}}{{end}}',
        container,
    )
    return found.stdout.split() if found.returncode == 0 else []


def remove_volumes(names: Iterable[str]) -> list[str]:
    return [name for name in names if _docker("volume", "rm", name).returncode == 0]


def prune_build_cache() -> str:
    """Drop BuildKit's unused cache. Returns Docker's 'Total reclaimed space' line, if any."""
    done = _docker("builder", "prune", "--all", "--force")
    lines = [line for line in done.stdout.splitlines() if "reclaimed" in line.lower()]
    return lines[-1].strip() if lines else ""


def report(what: str, items: list[str], cache: str = "") -> None:
    parts = [f"{len(items)} {what}"] if items else []
    if cache:
        parts.append(f"build cache: {cache}")
    if parts:
        print("Freed disk space: " + "; ".join(parts))  # noqa: T201
