"""The one training loop every model goes through.

Plain BCE, with no class weight: PR-AUC is about ranking, which weighting didn't help, and plain BCE
leaves scores close to real probabilities (DEC-047). Early stopping on validation PR-AUC, the metric
that decides everything else. The state with the best validation score is what's kept,
not the last one.
"""

import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from wikiguard_core.evaluation import pr_auc

from trainer.data import Batch, Encoded

GRADIENT_CLIP = 1.0
# Scoring doesn't need gradients, so it fits this many times the training batch in memory.
PREDICT_BATCH_MULTIPLE = 2
TRAIN_METRIC_SAMPLE = 5000


@dataclass(frozen=True)
class FitConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.01
    patience: int = 2
    warmup_share: float = 0.0
    # Lets a small GPU reach the same effective batch.
    accumulate: int = 1
    mixed_precision: bool = False
    seed: int = 0
    # The positive class counts (negatives / positives) ** weight_power: 0 is plain BCE, and 1 is
    # fully balanced. Kept so the search can keep testing the choice (DEC-047).
    weight_power: float = 0.0
    gradient_clip: float | None = GRADIENT_CLIP
    # A new head starts from nothing, so it can learn faster than a pretrained encoder should.
    head_learning_rate: float | None = None


@dataclass
class FitResult:
    best_epoch: int
    best_validation_pr_auc: float
    epochs_run: int
    seconds: float
    # One entry per epoch, starting with epoch 0, the untrained network (PLAN.md, section 4).
    history: list[dict[str, float]] = field(default_factory=list)
    state: dict[str, Tensor] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        result = asdict(self)
        del result["state"]
        return result


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - some libraries read the global generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def positive_weight(labels: NDArray[np.int64]) -> float:
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("training data has no positives")
    return (labels.size - positives) / positives


def expected_initial_loss(labels: NDArray[np.int64], weight: float) -> float:
    """Weighted BCE when every logit is 0, which is roughly where an untrained network starts."""
    share = float(labels.mean())
    return math.log(2) * (weight * share + (1 - share))


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _autocast(device: torch.device, enabled: bool) -> torch.autocast:
    return torch.autocast(device.type, dtype=torch.float16, enabled=enabled)


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    data: Encoded,
    batch_size: int,
    device: torch.device,
    mixed_precision: bool = False,
) -> NDArray[np.float64]:
    model.eval()
    logits = np.zeros(data.labels.size, dtype=np.float64)
    for batch, members in data.batches(batch_size, None):
        with _autocast(device, mixed_precision):
            output = model(**_to_device(batch, device))
        logits[members] = output.float().cpu().numpy()
    return logits


def _schedule(optimizer: torch.optim.Optimizer, total: int, warmup_share: float) -> Any:
    """Linear warmup and decay when warmup is set, which is the standard transformer recipe."""
    warmup = int(total * warmup_share)

    def factor(step: int) -> float:
        if warmup_share == 0:
            return 1.0
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        return max(0.0, (total - step) / max(total - warmup, 1))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _bce(weight: Tensor) -> nn.BCEWithLogitsLoss:
    return nn.BCEWithLogitsLoss(pos_weight=weight)


def _epoch_metrics(
    model: nn.Module,
    train_sample: Encoded,
    validation: Encoded,
    config: FitConfig,
    device: torch.device,
    loss_function: nn.Module,
) -> dict[str, float]:
    batch_size = config.batch_size * PREDICT_BATCH_MULTIPLE
    amp = config.mixed_precision
    train_logits = predict_logits(model, train_sample, batch_size, device, amp)
    validation_logits = predict_logits(model, validation, batch_size, device, amp)

    def loss_of(logits: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
        with torch.no_grad():
            return float(
                loss_function(
                    torch.tensor(logits, dtype=torch.float32, device=device),
                    torch.tensor(labels, dtype=torch.float32, device=device),
                )
            )

    return {
        "train_sample_pr_auc": pr_auc(train_sample.labels, train_logits),
        "train_sample_loss": loss_of(train_logits, train_sample.labels),
        "validation_pr_auc": pr_auc(validation.labels, validation_logits),
        "validation_loss": loss_of(validation_logits, validation.labels),
    }


def _copy_state(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _parameter_groups(model: nn.Module, config: FitConfig) -> list[dict[str, Any]]:
    trainable = [(key, value) for key, value in model.named_parameters() if value.requires_grad]
    if config.head_learning_rate is None:
        return [{"params": [value for _, value in trainable]}]
    head = [value for key, value in trainable if key.startswith("head.")]
    rest = [value for key, value in trainable if not key.startswith("head.")]
    return [{"params": rest}, {"params": head, "lr": config.head_learning_rate}]


def fit(
    model: nn.Module,
    train: Encoded,
    train_sample: Encoded,
    validation: Encoded,
    config: FitConfig,
    device: torch.device,
) -> FitResult:
    set_seeds(config.seed)
    rng = np.random.default_rng(config.seed)
    model.to(device)
    weight = torch.tensor(
        positive_weight(train.labels) ** config.weight_power, dtype=torch.float32, device=device
    )
    loss_function = _bce(weight)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, config), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    batches_per_epoch = math.ceil(train.labels.size / config.batch_size)
    steps = math.ceil(batches_per_epoch * config.epochs / config.accumulate)
    scheduler = _schedule(optimizer, steps, config.warmup_share)
    scaler = torch.amp.GradScaler(device.type, enabled=config.mixed_precision)

    started = time.perf_counter()
    first = _epoch_metrics(model, train_sample, validation, config, device, loss_function)
    history = [{"epoch": 0.0, "train_loss": first["train_sample_loss"], "seconds": 0.0} | first]
    best_score = first["validation_pr_auc"]
    best_epoch = 0
    best_state = _copy_state(model)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        optimizer.zero_grad(set_to_none=True)
        for batch, members in train.batches(config.batch_size, rng):
            targets = torch.tensor(train.labels[members], dtype=torch.float32, device=device)
            with _autocast(device, config.mixed_precision):
                logits = model(**_to_device(batch, device))
                loss = loss_function(logits, targets) / config.accumulate
            scaler.scale(loss).backward()
            batches += 1
            total_loss += float(loss.detach()) * config.accumulate
            if batches % config.accumulate == 0:
                scaler.unscale_(optimizer)
                if config.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        metrics = _epoch_metrics(model, train_sample, validation, config, device, loss_function)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(batches, 1),
                "seconds": time.perf_counter() - started,
            }
            | metrics
        )
        if not math.isfinite(total_loss):
            raise FloatingPointError(f"loss became {total_loss} in epoch {epoch}")
        if metrics["validation_pr_auc"] > best_score:
            best_score = metrics["validation_pr_auc"]
            best_epoch = epoch
            best_state = _copy_state(model)
        elif epoch - best_epoch >= config.patience:
            break

    model.load_state_dict(best_state)
    return FitResult(
        best_epoch,
        best_score,
        len(history) - 1,
        time.perf_counter() - started,
        history,
        best_state,
    )


def train_step_losses(
    model: nn.Module,
    batch: Batch,
    labels: NDArray[np.int64],
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> list[float]:
    """Repeatedly train on one batch, for the overfitting sanity check."""
    model.to(device)
    model.train()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    moved = _to_device(batch, device)
    targets = torch.tensor(labels, dtype=torch.float32, device=device)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(**moved), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses
