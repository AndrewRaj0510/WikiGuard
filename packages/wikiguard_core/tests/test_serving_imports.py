"""Serving images must never need PyTorch, Transformers, or MLflow. CI fails if one creeps in."""

import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
FORBIDDEN = ("torch", "transformers", "mlflow", "safetensors", "onnx", "pandas", "evidently")
SERVING_REQUIREMENTS = ["services/scorer/requirements.txt", "services/api/requirements.txt"]
EXTRAS = tomllib.loads((ROOT / "packages/wikiguard_core/pyproject.toml").read_text("utf-8"))[
    "project"
]["optional-dependencies"]


def package_names(lines: list[str]) -> set[str]:
    names = set()
    for line in lines:
        text = line.split("#")[0].strip()
        if text:
            names.add(re.split(r"[\[<>=!~ ]", text)[0].lower())
    return names


@pytest.mark.parametrize("file", SERVING_REQUIREMENTS)
def test_serving_requirements_have_no_training_tools(file: str) -> None:
    names = package_names((ROOT / file).read_text("utf-8").splitlines())
    assert not names & set(FORBIDDEN)
    assert "onnxruntime" in names


def test_the_serve_extra_has_no_training_tools() -> None:
    assert not package_names(EXTRAS["serve"]) & set(FORBIDDEN)


def test_the_train_extra_holds_the_training_tools() -> None:
    assert {"torch", "transformers", "mlflow-skinny"} <= package_names(EXTRAS["train"])


def test_serving_modules_import_with_training_tools_blocked() -> None:
    """Run in a subprocess that refuses the imports, so it fails even where they're installed."""
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {FORBIDDEN!r}:
                    raise ImportError("serving must not import " + name)
                return None

        sys.meta_path.insert(0, Block())
        import wikiguard_core.chain
        import wikiguard_core.config
        import wikiguard_core.features
        import wikiguard_core.predictor
        import wikiguard_core.schemas
        print("ok")
        """
    )
    done = subprocess.run(  # noqa: S603 - this interpreter, running the script above
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.stdout.strip() == "ok", done.stderr


def test_the_inference_module_avoids_training_tools() -> None:
    """Checked by reading it, because inference.py needs ONNX Runtime, which CI doesn't install."""
    source = (ROOT / "packages/wikiguard_core/wikiguard_core/inference.py").read_text("utf-8")
    imported = re.findall(r"^(?:from|import) +([\w.]+)", source, flags=re.MULTILINE)
    assert not {name.split(".")[0] for name in imported} & set(FORBIDDEN)
