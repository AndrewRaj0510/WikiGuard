from pathlib import Path

import pytest
from wikiguard_core.artifacts import (
    CHECKSUMS_FILE,
    CONFIG_FILE,
    ArtifactError,
    verify_artifact,
    write_checksums,
    write_json,
)
from wikiguard_core.features import FEATURE_VERSION


def model_folder(tmp_path: Path, feature_version: int = FEATURE_VERSION) -> Path:
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.onnx").write_bytes(b"model bytes")
    write_json(folder / CONFIG_FILE, {"feature_version": feature_version, "kind": "mlp"})
    return folder


def test_an_untouched_folder_verifies(tmp_path: Path) -> None:
    folder = model_folder(tmp_path)
    digest = write_checksums(folder)

    assert verify_artifact(folder, digest)["kind"] == "mlp"


def test_changing_a_file_is_caught(tmp_path: Path) -> None:
    folder = model_folder(tmp_path)
    write_checksums(folder)
    (folder / "model.onnx").write_bytes(b"swapped")

    with pytest.raises(ArtifactError, match="checksum"):
        verify_artifact(folder)


def test_adding_a_file_is_caught(tmp_path: Path) -> None:
    folder = model_folder(tmp_path)
    write_checksums(folder)
    (folder / "extra.json").write_text("{}", "utf-8")

    with pytest.raises(ArtifactError, match="extra.json"):
        verify_artifact(folder)


def test_removing_a_file_is_caught(tmp_path: Path) -> None:
    folder = model_folder(tmp_path)
    write_checksums(folder)
    (folder / "model.onnx").unlink()

    with pytest.raises(ArtifactError, match="model.onnx"):
        verify_artifact(folder)


@pytest.mark.parametrize("name", ["evil.pkl", "weights.pt", "model.joblib", "weights.bin"])
def test_a_pickle_can_not_be_recorded(tmp_path: Path, name: str) -> None:
    folder = model_folder(tmp_path)
    (folder / name).write_bytes(b"pickle")

    with pytest.raises(ArtifactError, match="allowed"):
        write_checksums(folder)


def test_a_pickle_added_later_is_refused_by_its_format(tmp_path: Path) -> None:
    folder = model_folder(tmp_path)
    write_checksums(folder)
    (folder / "evil.pkl").write_bytes(b"pickle")

    with pytest.raises(ArtifactError, match="allowed"):
        verify_artifact(folder)


def test_a_rewritten_checksum_list_fails_the_registered_digest(tmp_path: Path) -> None:
    folder = model_folder(tmp_path)
    digest = write_checksums(folder)
    (folder / "model.onnx").write_bytes(b"swapped")
    write_checksums(folder)

    with pytest.raises(ArtifactError, match="registered digest"):
        verify_artifact(folder, digest)


def test_another_feature_version_is_refused(tmp_path: Path) -> None:
    folder = model_folder(tmp_path, FEATURE_VERSION + 1)
    write_checksums(folder)

    with pytest.raises(ArtifactError, match="feature_version"):
        verify_artifact(folder)


def test_a_folder_with_no_list_is_refused(tmp_path: Path) -> None:
    folder = model_folder(tmp_path)

    with pytest.raises(ArtifactError, match=CHECKSUMS_FILE):
        verify_artifact(folder)
