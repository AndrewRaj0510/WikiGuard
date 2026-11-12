"""The delivery pipeline is an attack path too, so its rules are tested like code.

Each test names one rule from docs/PLAN.md (sections 8 and 10) and fails when a workflow, a
Dockerfile, a Compose file, or a lock file breaks it. A rule that's only written down gets
forgotten the first time someone adds a workflow in a hurry.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from lock_requirements import inputs, problems

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
ACTIONS = sorted((ROOT / ".github" / "actions").glob("*/action.yml"))
DOCKERFILES = sorted(
    path
    for pattern in ("services/*/Dockerfile*", "jobs/*/Dockerfile*")
    for path in ROOT.glob(pattern)
)
COMPOSE_FILES = sorted(ROOT.glob("docker-compose*.yml"))
# A measuring tool run by hand on the laptop. It's never published or deployed.
NOT_DEPLOYED = {"jobs/trainer/Dockerfile.benchmark"}
DEPLOYED = [path for path in DOCKERFILES if path.relative_to(ROOT).as_posix() not in NOT_DEPLOYED]

PINNED_ACTION = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")
EXPRESSION = re.compile(r"\$\{\{\s*([^}]+?)\s*\}\}")
# Each job that may write, and exactly what. Everything else gets a read-only token.
ALLOWED_WRITES = {
    ("cd.yml", "build"): {"packages"},
    ("cd.yml", "publish"): {"packages"},
    ("codeql.yml", "analyze"): {"security-events"},
    ("release.yml", "release"): {"contents", "packages"},
}
ALLOWED_IN_AN_IMAGE = {
    "!requirements-build.txt",
    "!packages/wikiguard_core",
    "!services",
    "!jobs",
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text("utf-8"))
    return loaded


def triggers(workflow: dict[str, Any]) -> Any:
    # YAML 1.1 reads the bare word `on` as the boolean true.
    return workflow.get("on", workflow.get(True))


def steps_of(document: dict[str, Any]) -> list[dict[str, Any]]:
    if "runs" in document:
        return list(document["runs"]["steps"])
    return [step for job in document["jobs"].values() for step in job.get("steps", [])]


def test_the_files_under_test_were_found() -> None:
    assert len(WORKFLOWS) >= 5
    assert len(ACTIONS) >= 1
    assert len(DEPLOYED) >= 11
    assert len(COMPOSE_FILES) >= 2


# --- GitHub Actions ---


@pytest.mark.parametrize("path", WORKFLOWS + ACTIONS, ids=relative)
def test_every_action_is_pinned_to_a_commit(path: Path) -> None:
    """A tag can be moved to different code. A commit SHA can't."""
    document = load(path)
    used = [step["uses"] for step in steps_of(document) if "uses" in step]
    used += [job["uses"] for job in document.get("jobs", {}).values() if "uses" in job]
    for name in used:
        # ./ is a file in this repo, at the same commit as the workflow.
        assert name.startswith("./") or PINNED_ACTION.match(name), name


@pytest.mark.parametrize("path", WORKFLOWS, ids=relative)
def test_no_workflow_runs_fork_code_with_secrets(path: Path) -> None:
    """pull_request_target runs with secrets and a write token, on code a stranger can send."""
    assert "pull_request_target" not in str(triggers(load(path)))


@pytest.mark.parametrize("path", WORKFLOWS, ids=relative)
def test_the_default_token_is_read_only(path: Path) -> None:
    assert load(path)["permissions"] == {"contents": "read"}


@pytest.mark.parametrize("path", WORKFLOWS, ids=relative)
def test_only_listed_jobs_may_write_and_only_what_is_listed(path: Path) -> None:
    for name, job in load(path)["jobs"].items():
        writes = {scope for scope, level in job.get("permissions", {}).items() if level == "write"}
        assert writes == ALLOWED_WRITES.get((path.name, name), set()), (path.name, name)


@pytest.mark.parametrize("path", WORKFLOWS, ids=relative)
def test_a_workflow_with_secrets_never_runs_on_a_pull_request(path: Path) -> None:
    """Secrets reach a job only after the code is on main, or when the owner starts it."""
    text = path.read_text("utf-8")
    # GITHUB_TOKEN is the run's own short-lived token, read-only unless a job says otherwise.
    named = set(re.findall(r"secrets\.(\w+)", text)) - {"GITHUB_TOKEN"}
    uses_environment = any("environment" in job for job in load(path)["jobs"].values())
    if named or uses_environment:
        assert "pull_request" not in str(triggers(load(path)))


@pytest.mark.parametrize("path", WORKFLOWS + ACTIONS, ids=relative)
def test_scripts_never_paste_in_text_an_outsider_controls(path: Path) -> None:
    """A branch name or title pasted into a script runs as code. Values go through env instead."""
    for step in steps_of(load(path)):
        for expression in EXPRESSION.findall(step.get("run", "")):
            # A matrix value is a constant written in the workflow file itself.
            assert expression.startswith("matrix."), (path.name, expression)


@pytest.mark.parametrize("path", WORKFLOWS, ids=relative)
def test_checkouts_dont_leave_the_token_on_disk(path: Path) -> None:
    for step in steps_of(load(path)):
        if step.get("uses", "").startswith("actions/checkout@"):
            assert step.get("with", {}).get("persist-credentials") is False, path.name


def test_downloaded_tools_are_checked_against_a_pinned_checksum() -> None:
    for path in WORKFLOWS:
        for step in steps_of(load(path)):
            script = step.get("run", "")
            if "releases/download" in script:
                assert "sha256sum --check" in script, (path.name, step.get("name"))


# --- Images ---


@pytest.mark.parametrize("path", DOCKERFILES, ids=relative)
def test_every_base_image_is_pinned_by_digest(path: Path) -> None:
    bases = re.findall(r"^FROM\s+(\S+)", path.read_text("utf-8"), re.MULTILINE)
    assert bases
    for base in bases:
        assert DIGEST.search(base), base


@pytest.mark.parametrize("path", DEPLOYED, ids=relative)
def test_deployed_images_install_only_from_hash_locked_files(path: Path) -> None:
    text = path.read_text("utf-8")
    installs = [line for line in text.splitlines() if re.search(r"\bpip install\b", line)]
    assert installs
    for line in installs:
        # The one install without hashes is this repo's own wheel, built in the stage before.
        assert "--require-hashes" in line or "--no-deps /wheels/" in line, line
    for copied in re.findall(r"^COPY\s+(\S*requirements\S*\.txt)\s", text, re.MULTILINE):
        assert (ROOT / copied).with_suffix(".in").is_file(), f"{copied} isn't a lock file"


@pytest.mark.parametrize("path", DEPLOYED, ids=relative)
def test_deployed_images_build_in_one_stage_and_run_in_another(path: Path) -> None:
    text = path.read_text("utf-8")
    assert len(re.findall(r"^FROM\s", text, re.MULTILINE)) == 2
    runtime = text[text.rindex("\nFROM ") :]
    assert "COPY --from=build /venv /venv" in runtime
    assert "pip install" not in runtime


@pytest.mark.parametrize("path", DOCKERFILES, ids=relative)
def test_every_image_runs_as_a_user_that_isnt_root(path: Path) -> None:
    users = re.findall(r"^USER\s+(\S+)", path.read_text("utf-8"), re.MULTILINE)
    assert users
    assert users[-1] not in {"root", "0"}


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=relative)
def test_compose_images_are_pinned_by_digest(path: Path) -> None:
    # Read as text: Compose's own tags, such as !override, aren't plain YAML.
    images = re.findall(r"^\s*image:\s*(\S+)", path.read_text("utf-8"), re.MULTILINE)
    assert images
    for image in images:
        assert DIGEST.search(image), image


def test_env_files_and_data_never_reach_an_image_build() -> None:
    """The ignore file starts by excluding everything, then lets only the code back in."""
    lines = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text("utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines[0] == "*"
    assert {line for line in lines if line.startswith("!")} == ALLOWED_IN_AN_IMAGE


def image_names() -> dict[str, str]:
    """Every deployed Dockerfile, under the name CI and CD give its image."""
    names = {}
    for path in DEPLOYED:
        name = path.parent.name.replace("_", "-")
        if path.suffix:
            name = f"{name}-{path.suffix.removeprefix('.')}"
        names[name] = relative(path)
    return names


def test_ci_builds_and_scans_every_image() -> None:
    matrix = load(ROOT / ".github/workflows/ci.yml")["jobs"]["images"]["strategy"]["matrix"]
    assert {row["name"]: row["dockerfile"] for row in matrix["include"]} == image_names()


def test_cd_publishes_every_image_and_a_release_names_every_image() -> None:
    jobs = load(ROOT / ".github/workflows/cd.yml")["jobs"]
    built = jobs["build"]["strategy"]["matrix"]["image"]
    assert {row["name"]: row["dockerfile"] for row in built} == image_names()
    assert set(jobs["publish"]["strategy"]["matrix"]["name"]) == set(image_names())
    release = (ROOT / ".github/workflows/release.yml").read_text("utf-8")
    listed = re.search(r"for name in (.*?); do", release, re.DOTALL)
    assert listed is not None
    assert set(listed.group(1).replace("\\", " ").split()) == set(image_names())


# --- Dependencies ---


@pytest.mark.parametrize("source", inputs(), ids=relative)
def test_every_lock_file_is_complete_and_matches_its_input(source: Path) -> None:
    assert problems(source) == []


def test_every_service_and_job_has_a_lock_file() -> None:
    locked = {source.parent for source in inputs()}
    for path in DEPLOYED:
        assert path.parent in locked, path


def folder_of(path: Path) -> str:
    """A folder the way dependabot.yml writes it: "/" for the root, "/jobs/purge" below it."""
    return "/" if path == ROOT else f"/{relative(path)}"


def test_dependabot_watches_every_place_a_dependency_is_pinned() -> None:
    watched: dict[str, set[str]] = {}
    for update in load(ROOT / ".github/dependabot.yml")["updates"]:
        folders = update.get("directories", [update.get("directory")])
        watched.setdefault(update["package-ecosystem"], set()).update(folders)
    assert {"github-actions", "docker", "docker-compose", "pip", "npm"} <= set(watched)
    assert {folder_of(path.parent) for path in DOCKERFILES} <= watched["docker"]
    assert {folder_of(source.parent) for source in inputs()} <= watched["pip"]
    assert {folder_of(path.parent) for path in ACTIONS} <= watched["github-actions"]
