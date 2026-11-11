"""The DVC and git steps the purge and retention jobs share.

Commands run as argument lists with no shell, from the repo's root. The one dangerous step is
`dvc gc --cloud`: it deletes every version the checkout doesn't point at, from the remote and from
the local cache. Two guards keep it safe (DEC-070):

- It runs only when the job is told it may (SNAPSHOT_GC_ENABLED), which is meant for the job's own
  clone and never for a laptop holding data the models trained on (F-093).
- It runs only when every pointer file matches what GitHub's main branch holds. Otherwise gc
  would delete a version that a published commit names, and anyone checking it out couldn't
  download it.
"""

import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

Runner = Callable[[Sequence[str], Path], str]
TIMEOUT_SECONDS = 60 * 60


class DvcError(RuntimeError):
    pass


def run_command(args: Sequence[str], cwd: Path) -> str:
    executable = shutil.which(args[0])
    if executable is None:
        raise DvcError(f"{args[0]} isn't installed")
    completed = subprocess.run(  # noqa: S603 - fixed commands, no shell, no request input
        [executable, *args[1:]],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        # The command and the end of stderr only: DVC's output names files, never row content.
        raise DvcError(f"{' '.join(args[:3])} failed: {completed.stderr.strip()[-300:]}")
    return completed.stdout


@dataclass
class DvcRepo:
    root: Path
    runner: Runner = run_command

    def _run(self, *args: str) -> str:
        return self.runner(list(args), self.root)

    def pointers(self, folder: str = "snapshots") -> list[Path]:
        return sorted((self.root / folder).glob("*.dvc"))

    def pull(self, pointer: Path) -> None:
        self._run("dvc", "pull", str(pointer.relative_to(self.root)))

    def add(self, target: Path) -> None:
        self._run("dvc", "add", str(target.relative_to(self.root)))

    def push(self, pointer: Path) -> None:
        self._run("dvc", "push", str(pointer.relative_to(self.root)))

    def remove(self, pointer: Path) -> None:
        """Stop tracking a folder. Its data stays in the cache until gc."""
        self._run("dvc", "remove", str(pointer.relative_to(self.root)))

    def pointers_published(self, remote_ref: str = "origin/main") -> bool:
        """True when no pointer file differs from the published branch, or from the checkout."""
        local = self._run("git", "status", "--porcelain", "--", "snapshots")
        if any(line.strip().endswith(".dvc") for line in local.splitlines()):
            return False
        self._run("git", "fetch", "--quiet", remote_ref.split("/", 1)[0])
        published = self._run("git", "diff", "--name-only", remote_ref, "--", "snapshots")
        return not any(line.strip().endswith(".dvc") for line in published.splitlines())

    def gc_cloud(self, enabled: bool, remote_ref: str = "origin/main") -> str:
        """Delete every version the checkout doesn't use, locally and on the remote."""
        if not enabled:
            return "skipped: SNAPSHOT_GC_ENABLED is off"
        if not self.pointers_published(remote_ref):
            return "skipped: pointer files differ from the published branch, so publish them first"
        self._run("dvc", "gc", "--workspace", "--cloud", "--force")
        return "done"
