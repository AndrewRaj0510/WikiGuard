"""The disk-space helper: what it asks Docker to delete, checked without touching Docker."""

import subprocess
from typing import Any

import docker_space
import pytest


class FakeDocker:
    def __init__(self, answers: dict[str, tuple[int, str]] | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.answers = answers or {}

    def __call__(self, *args: str) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        code, out = self.answers.get(" ".join(args[:2]), (0, ""))
        return subprocess.CompletedProcess(["docker", *args], code, out, "")


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    fake = FakeDocker()
    monkeypatch.setattr(docker_space, "_docker", fake)
    return fake


def test_images_are_removed_by_tag_and_a_missing_one_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(*args: str) -> subprocess.CompletedProcess[str]:
        code = 1 if args[-1] == "gone:dev" else 0
        return subprocess.CompletedProcess(["docker", *args], code, "", "")

    monkeypatch.setattr(docker_space, "_docker", fake)
    assert docker_space.remove_images(["a:dev", "gone:dev"]) == ["a:dev"]


def test_only_named_tags_match_and_untagged_layers_are_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeDocker({"image ls": (0, "kind.local/wikiguard-scorer:dev\nkind.local/x:<none>\n")})
    monkeypatch.setattr(docker_space, "_docker", fake)
    assert docker_space.images_matching("kind.local/wikiguard-*") == [
        "kind.local/wikiguard-scorer:dev"
    ]
    assert fake.calls[0][:4] == ("image", "ls", "--filter", "reference=kind.local/wikiguard-*")


def test_a_kind_nodes_volumes_are_found_for_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeDocker({"inspect --format": (0, "435cc585 other-volume \n")})
    monkeypatch.setattr(docker_space, "_docker", fake)
    assert docker_space.container_volumes("wikiguard-control-plane") == ["435cc585", "other-volume"]


def test_a_missing_container_has_no_volumes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_space, "_docker", FakeDocker({"inspect --format": (1, "")}))
    assert docker_space.container_volumes("nothing") == []


def test_the_build_cache_prune_drops_all_unused_cache(docker: FakeDocker) -> None:
    docker.answers["builder prune"] = (0, "ID\nabc\nTotal:\t15.66GB reclaimed\n")
    assert docker_space.prune_build_cache() == "Total:\t15.66GB reclaimed"
    assert docker.calls == [("builder", "prune", "--all", "--force")]


def test_the_report_says_what_was_freed(capsys: Any) -> None:
    docker_space.report("images", ["a", "b"], "Total: 1GB reclaimed")
    assert (
        capsys.readouterr().out == "Freed disk space: 2 images; build cache: Total: 1GB reclaimed\n"
    )
    docker_space.report("images", [])
    assert capsys.readouterr().out == ""
