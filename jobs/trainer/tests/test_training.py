"""Training end to end on tiny data: model, export, quantize, and scoring without PyTorch."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from tokenizers.processors import TemplateProcessing  # noqa: E402
from trainer.data import (  # noqa: E402
    SequenceEncoded,
    encode_tfidf,
    encode_tokens,
    load_dataset,
    sample_rows,
)
from trainer.export import save_weights, write_served_folder  # noqa: E402
from trainer.loop import (  # noqa: E402
    FitConfig,
    _parameter_groups,
    expected_initial_loss,
    fit,
    positive_weight,
    predict_logits,
    train_step_losses,
)
from trainer.recipes import Prepared, prepare  # noqa: E402
from trainer.report import evaluate_served, parity, training_warnings  # noqa: E402
from trainer.sanity import METADATA_LEAK_PR_AUC, metadata_only  # noqa: E402
from wikiguard_core.artifacts import ArtifactError, verify_artifact  # noqa: E402
from wikiguard_core.evaluation import pr_auc, sigmoid  # noqa: E402
from wikiguard_core.inference import TRANSFORMER, ServedModel  # noqa: E402
from wikiguard_core.inputs import (  # noqa: E402
    MetadataSpec,
    TfidfEncoder,
    TokenVocabulary,
    transformer_text,
)
from wikiguard_core.models import (  # noqa: E402
    BiLSTMClassifier,
    MLPClassifier,
    TransformerClassifier,
    count_parameters,
    freeze_lower_layers,
)
from wikiguard_core.predictor import ChainEntry, Predictor  # noqa: E402
from wikiguard_core.validation import DataValidationError  # noqa: E402

CPU = torch.device("cpu")
MLP_PARAMS = {"hidden": 32, "dropout": 0.1, "learning_rate": 3e-3, "use_comment": True}
BILSTM_PARAMS = {"hidden": 16, "dropout": 0.1, "learning_rate": 3e-3, "use_comment": True}


@pytest.fixture(scope="module")
def data(snapshot_folder: Path) -> Any:
    return load_dataset(snapshot_folder)


def test_a_snapshot_that_fails_validation_can_not_be_loaded(
    snapshot_folder: Path, tmp_path: Path
) -> None:
    damaged = tmp_path / "damaged"
    damaged.mkdir()
    manifest = json.loads((snapshot_folder / "manifest.json").read_text("utf-8"))
    manifest["content_sha256"] = "0" * 64
    (damaged / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    (damaged / "edits.parquet").write_bytes((snapshot_folder / "edits.parquet").read_bytes())

    with pytest.raises(DataValidationError, match="content_hash"):
        load_dataset(damaged)


def test_a_capped_sample_keeps_the_positive_rate_and_time_order(data: Any) -> None:
    rows = sample_rows(data.test, 300, seed=1)
    rate = sum(row["label"] for row in data.test) / len(data.test)

    assert len(rows) == 300
    assert sum(row["label"] for row in rows) / 300 == pytest.approx(rate, abs=0.01)
    stamps = [row["timestamp"] for row in rows]
    assert stamps == sorted(stamps)


def test_metadata_check_catches_a_planted_leak(data: Any) -> None:
    """If a metadata field gives away the label, this is the check that has to notice."""
    leaky = [dict(row, is_minor=bool(row["label"])) for row in [*data.train, *data.validation]]
    leaked = replace(data, train=leaky[: len(data.train)], validation=leaky[len(data.train) :])

    assert metadata_only(leaked, 0, CPU) > METADATA_LEAK_PR_AUC


def test_positive_weight_balances_the_classes() -> None:
    assert positive_weight(np.array([0, 0, 0, 1])) == 3.0


def test_expected_initial_loss_matches_a_network_that_outputs_zero() -> None:
    labels = np.array([0] * 9 + [1])
    weight = positive_weight(labels)

    actual = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))(
        torch.zeros(10), torch.tensor(labels, dtype=torch.float32)
    )

    assert float(actual) == pytest.approx(expected_initial_loss(labels, weight))


def test_mlp_learns_a_clear_signal_and_logs_epoch_zero(data: Any) -> None:
    prepared = prepare("mlp", data, MLP_PARAMS, CPU)
    config = FitConfig(epochs=4, batch_size=32, learning_rate=3e-3, patience=4)

    result = fit(
        prepared.model, prepared.train, prepared.train_sample, prepared.validation, config, CPU
    )

    assert result.history[0]["epoch"] == 0
    assert result.best_validation_pr_auc > 0.95
    assert result.best_validation_pr_auc > result.history[0]["validation_pr_auc"]
    assert result.best_epoch > 0


def test_training_is_repeatable_with_one_seed(data: Any) -> None:
    def once() -> float:
        prepared = prepare("mlp", data, MLP_PARAMS, CPU, seed=5)
        config = FitConfig(epochs=2, batch_size=32, learning_rate=1e-3, seed=5)
        return fit(
            prepared.model, prepared.train, prepared.train_sample, prepared.validation, config, CPU
        ).history[-1]["validation_loss"]

    assert once() == pytest.approx(once())


def test_early_stopping_keeps_the_best_state_not_the_last(data: Any) -> None:
    prepared = prepare("mlp", data, MLP_PARAMS, CPU)
    # Far too high a rate, so a later epoch is worse than an early one.
    config = FitConfig(epochs=6, batch_size=32, learning_rate=0.5, patience=1)

    result = fit(
        prepared.model, prepared.train, prepared.train_sample, prepared.validation, config, CPU
    )

    logits = predict_logits(prepared.model, prepared.validation, 64, CPU)
    assert pr_auc(prepared.validation.labels, logits) == pytest.approx(
        result.best_validation_pr_auc
    )


def test_bilstm_batches_pad_and_score_the_same_as_alone(data: Any) -> None:
    vocabulary = TokenVocabulary.fit(data.train, 500, 1, True)
    metadata = MetadataSpec.fit(data.train)
    model = BiLSTMClassifier(vocabulary.size, 8, 8, len(metadata.columns), 0.0).eval()
    rows = [
        dict(data.train[0], added_text="a very much longer addition than the other"),
        data.train[1],
    ]
    encoded = encode_tokens(rows, vocabulary, metadata)

    batched = predict_logits(model, encoded, 2, CPU)
    alone = [
        predict_logits(model, encode_tokens([row], vocabulary, metadata), 1, CPU)[0] for row in rows
    ]

    assert batched == pytest.approx(alone, abs=1e-5)


def tiny_tokenizer(path: Path) -> Path:
    words = "comment added removed : a the of older words copyedit garbage nonsense spam".split()
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    vocabulary |= {word: index + 4 for index, word in enumerate(words)}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer.save(str(path))
    return path


def tiny_transformer(metadata_size: int, vocabulary: int) -> Any:
    config = transformers.DistilBertConfig(
        vocab_size=vocabulary,
        dim=16,
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        max_position_embeddings=256,
    )
    return TransformerClassifier(transformers.DistilBertModel(config), metadata_size, 0.0)


def test_freezing_leaves_only_the_top_blocks_training() -> None:
    model = tiny_transformer(3, 50)
    before = count_parameters(model)

    frozen = freeze_lower_layers(model, 1)

    assert frozen == 1
    assert count_parameters(model)["trainable"] < before["trainable"]
    assert count_parameters(model)["total"] == before["total"]
    assert all(not p.requires_grad for p in model.encoder.transformer.layer[0].parameters())
    assert all(p.requires_grad for p in model.encoder.transformer.layer[1].parameters())
    assert all(p.requires_grad for p in model.head.parameters())


@pytest.mark.parametrize(
    "steps,rate",
    [(120, 3e-3)],
)
def test_every_model_type_can_overfit_one_batch(data: Any, steps: int, rate: float) -> None:
    rows = data.train[:16]
    labels = np.array([row["label"] for row in rows])
    metadata = MetadataSpec.fit(data.train)
    tfidf = TfidfEncoder.fit(data.train, 200, 1, True)
    vocabulary = TokenVocabulary.fit(data.train, 500, 1, True)
    tfidf_batch = next(iter(encode_tfidf(rows, tfidf, metadata).batches(16, None)))[0]
    token_batch, token_members = next(
        iter(encode_tokens(rows, vocabulary, metadata).batches(16, None))
    )
    mlp = MLPClassifier(tfidf.size + len(metadata.columns), 32, 0.0)
    lstm = BiLSTMClassifier(vocabulary.size, 16, 16, len(metadata.columns), 0.0)

    assert train_step_losses(mlp, tfidf_batch, labels, steps, rate, CPU)[-1] < 0.1
    assert train_step_losses(lstm, token_batch, labels[token_members], steps, rate, CPU)[-1] < 0.1


def _served(prepared: Prepared, tmp_path: Path, data: Any) -> Path:
    config = FitConfig(epochs=2, batch_size=32, learning_rate=3e-3, patience=2)
    result = fit(
        prepared.model, prepared.train, prepared.train_sample, prepared.validation, config, CPU
    )
    folder = tmp_path / "model"
    write_served_folder(prepared, folder, data.manifest, result.summary())
    return folder


@pytest.mark.parametrize("name,params", [("mlp", MLP_PARAMS), ("bilstm", BILSTM_PARAMS)])
def test_export_scores_match_pytorch_and_load_with_no_pytorch_code(
    name: str, params: dict[str, Any], data: Any, tmp_path: Path
) -> None:
    prepared = prepare(name, data, params, CPU, pretrained=False)
    folder = _served(prepared, tmp_path, data)
    rows = data.validation[:60]

    served = ServedModel(folder).logits(rows)
    from trainer.recipes import encode_like

    original = predict_logits(prepared.model, encode_like(prepared, rows, "v", data), 16, CPU)
    # Int8 moves scores a little, and the ranking should barely change.
    gap = np.abs(sigmoid(served) - sigmoid(original)).max()
    assert gap < 0.2
    assert verify_artifact(folder)["kind"] == name


def test_bilstm_batched_scores_stay_close_to_single_edit_scores(data: Any, tmp_path: Path) -> None:
    """Dynamic quantization scales each batch by its own range, so scores shift slightly (F-073)."""
    prepared = prepare("bilstm", data, BILSTM_PARAMS, CPU, pretrained=False)
    served = ServedModel(_served(prepared, tmp_path, data))
    rows = data.validation[:12]

    assert served.logits(rows, 8) == pytest.approx(
        [served.logits([row])[0] for row in rows], abs=0.02
    )


def _tiny_transformer_prepared(data: Any, tmp_path: Path) -> Prepared:
    tokenizer = tiny_tokenizer(tmp_path / "source_tokenizer.json")
    metadata = MetadataSpec.fit(data.train)
    loaded = Tokenizer.from_file(str(tokenizer))
    loaded.enable_truncation(32)

    def encoded(rows: list[dict[str, Any]]) -> SequenceEncoded:
        pieces = loaded.encode_batch_fast([transformer_text(row) for row in rows])
        ids = [np.array(piece.ids, dtype=np.int64) for piece in pieces]
        labels = np.array([row["label"] for row in rows], dtype=np.int64)
        return SequenceEncoded(ids, metadata.transform(rows), labels, 0, True)

    model = tiny_transformer(len(metadata.columns), loaded.get_vocab_size())
    return Prepared(
        "distilbert",
        {"learning_rate": 1e-3, "dropout": 0.0},
        model,
        metadata,
        FitConfig(epochs=1, batch_size=16, learning_rate=1e-3),
        encoded(data.train),
        encoded(data.train[:50]),
        encoded(data.validation),
        {
            "tokenizer": tokenizer,
            "pad_token_id": 0,
            "repo": "tiny",
            "revision": "tiny",
            "frozen_blocks": 0,
        },
        len(data.train),
    )


def test_transformer_exports_and_scores_any_length(data: Any, tmp_path: Path) -> None:
    prepared = _tiny_transformer_prepared(data, tmp_path)
    folder = _served(prepared, tmp_path, data)
    served = ServedModel(folder)

    assert served.kind == TRANSFORMER
    rows = data.validation[:6]
    batched = served.logits(rows, 6)
    assert batched == pytest.approx([served.logits([row])[0] for row in rows], abs=0.02)
    original = predict_logits(prepared.model, prepared.validation, 16, CPU)
    assert np.abs(sigmoid(served.logits(data.validation)) - sigmoid(original)).max() < 0.3


def test_report_writes_calibration_into_the_artifact(data: Any, tmp_path: Path) -> None:
    prepared = prepare("mlp", data, MLP_PARAMS, CPU)
    folder = _served(prepared, tmp_path, data)
    validation = predict_logits(prepared.model, prepared.validation, 64, CPU)
    from trainer.recipes import encode_like

    test = predict_logits(prepared.model, encode_like(prepared, data.test, "t", data), 64, CPU)

    report, digest = evaluate_served(folder, data, validation, test, tmp_path)

    config = verify_artifact(folder, digest)
    assert config["temperature"] == report["temperature"]
    assert config["threshold_logit"] == report["threshold"]
    assert report["parity"]["test"]["passed"]
    assert report["pr_auc"] > 0.5
    assert report["batch_vs_single"]["rows"] > 0
    assert (tmp_path / "scores.npz").exists()
    # Calibration is applied at serving, and it never reorders edits.
    served = ServedModel(folder)
    probabilities = served.probabilities(data.test[:50])
    assert ((0 <= probabilities) & (probabilities <= 1)).all()
    assert served.threshold == config["threshold_probability"]
    assert 0 < served.threshold < 1

    # A real model goes through the chain, including when a broken champion sits in front of it.
    class BrokenChampion:
        threshold = None

        def probabilities(self, rows: Any) -> Any:
            raise RuntimeError("broken")

    rows = data.test[:5]
    chain = [
        ChainEntry("champion", "roberta", "1", BrokenChampion()),
        ChainEntry("fallback-4", "mlp", "5", served),
    ]
    result = Predictor(chain, 5000).score(rows)
    assert result.model_name == "mlp"
    assert result.used_fallback is True
    assert result.threshold == served.threshold
    np.testing.assert_array_equal(result.scores, served.probabilities(rows))


def test_tampering_after_calibration_stops_the_model_loading(data: Any, tmp_path: Path) -> None:
    prepared = prepare("mlp", data, MLP_PARAMS, CPU)
    folder = _served(prepared, tmp_path, data)
    (folder / "model.onnx").write_bytes(b"not a model")

    with pytest.raises(ArtifactError):
        ServedModel(folder)


def test_parity_fails_when_quantizing_changes_the_ranking() -> None:
    labels = np.array([1, 1, 0, 0, 0, 0])
    good = np.array([5.0, 4.0, 1.0, 0.0, -1.0, -2.0])
    bad = good[::-1]

    assert parity(labels, good, good)["passed"] is True
    assert parity(labels, good, bad)["passed"] is False


def test_saved_weights_are_safetensors_and_load_back(data: Any, tmp_path: Path) -> None:
    from safetensors.torch import load_file

    model = MLPClassifier(4, 8, 0.0)

    path = save_weights(model, tmp_path)

    assert path.suffix == ".safetensors"
    fresh = MLPClassifier(4, 8, 0.0)
    fresh.load_state_dict(load_file(path))
    inputs = torch.ones(1, 4)
    assert model.eval()(inputs).item() == pytest.approx(fresh.eval()(inputs).item())


def test_training_is_plain_bce_unless_a_trial_asks_for_weight() -> None:
    config = FitConfig(epochs=1, batch_size=2, learning_rate=1e-3)

    assert config.weight_power == 0.0
    assert positive_weight(np.array([0, 0, 0, 1])) ** config.weight_power == 1.0


def test_a_head_rate_gives_the_head_its_own_rate() -> None:
    model = MLPClassifier(4, 8, 0.0)
    model.head = torch.nn.Linear(4, 1)
    config = FitConfig(epochs=1, batch_size=2, learning_rate=1e-5, head_learning_rate=1e-3)

    encoder_group, head_group = _parameter_groups(model, config)

    assert head_group["lr"] == 1e-3
    assert {id(value) for value in head_group["params"]} == {
        id(value) for value in model.head.parameters()
    }
    assert "lr" not in encoder_group
    assert len(encoder_group["params"]) == len(list(model.layers.parameters()))


def test_without_a_head_rate_everything_shares_one_group() -> None:
    model = MLPClassifier(4, 8, 0.0)

    groups = _parameter_groups(model, FitConfig(epochs=1, batch_size=2, learning_rate=1e-5))

    assert len(groups) == 1


HEALTHY_HISTORY = [{"train_loss": 0.69}, {"train_loss": 0.25}]


def test_a_healthy_run_has_no_training_warnings() -> None:
    assert training_warnings(HEALTHY_HISTORY, 0.30, 0.207, True) == []


def test_loss_that_ends_above_the_untrained_network_is_flagged() -> None:
    history = [{"train_loss": 1.30}, {"train_loss": 1.42}]

    assert "above the untrained" in training_warnings(history, 0.30, 0.207, True)[0]


def test_a_model_no_better_than_its_metadata_is_flagged() -> None:
    # The first RoBERTa run: 0.2072 on validation, against 0.2067 for metadata alone (F-075).
    assert "metadata alone" in training_warnings(HEALTHY_HISTORY, 0.2072, 0.2067, True)[0]


def test_failed_parity_is_flagged() -> None:
    assert "differs from PyTorch" in training_warnings(HEALTHY_HISTORY, 0.30, None, False)[0]
