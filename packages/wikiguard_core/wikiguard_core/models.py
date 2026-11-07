"""The five models, as three PyTorch classes.

MLP and BiLSTM each have a class. DistilBERT, BERT, and RoBERTa differ only in their pretrained
encoder, so they share TransformerClassifier. Every model returns one raw logit per edit, which is
what the loss, calibration, and evaluation all expect. Only training imports this file.
"""

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoConfig, AutoModel, PreTrainedModel

MLP = "mlp"
BILSTM = "bilstm"
DISTILBERT = "distilbert"
BERT = "bert"
ROBERTA = "roberta"
MODEL_NAMES = (MLP, BILSTM, DISTILBERT, BERT, ROBERTA)
TRANSFORMERS = (DISTILBERT, BERT, ROBERTA)
# Pinned by commit, so a change on Hugging Face can't change a run (PLAN.md, section 8).
PRETRAINED = {
    DISTILBERT: ("distilbert/distilbert-base-uncased", "12040accade4e8a0f71eabdb258fecc2e7e948be"),
    BERT: ("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594"),
    ROBERTA: ("FacebookAI/roberta-base", "e2da8e2f811d1448a5b465c236feacd80ffbac7b"),
}


class MLPClassifier(nn.Module):
    """Baseline and last fallback: a few MB, and no tokenizer."""

    def __init__(self, inputs: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(inputs, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.layers(features)).squeeze(-1)


class BiLSTMClassifier(nn.Module):
    """Reads the diff in order, both ways. Max pooling catches one bad phrase in a long addition,
    which the final hidden state alone would mostly have forgotten by the end."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        hidden: int,
        metadata_size: int,
        dropout: float,
        pretrained: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)
        self.lstm = nn.LSTM(embedding_size, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 4 + metadata_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids: Tensor, metadata: Tensor, lengths: Tensor | None = None) -> Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        if lengths is None:
            # Export and serving: every sequence in a batch has the same length (inference.py).
            outputs, (final, _) = self.lstm(embedded)
            pooled = outputs.max(dim=1).values
        else:
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_outputs, (final, _) = self.lstm(packed)
            outputs, _ = pad_packed_sequence(
                packed_outputs, batch_first=True, total_length=input_ids.size(1)
            )
            steps = torch.arange(input_ids.size(1), device=input_ids.device)
            padding = steps.unsqueeze(0) >= lengths.unsqueeze(1)
            pooled = outputs.masked_fill(padding.unsqueeze(-1), float("-inf")).max(dim=1).values
        last = torch.cat([final[-2], final[-1]], dim=1)
        combined = torch.cat([self.dropout(pooled), self.dropout(last), metadata], dim=1)
        return cast(Tensor, self.head(combined)).squeeze(-1)


class TransformerClassifier(nn.Module):
    """A pretrained encoder, with metadata joined to its first token's vector."""

    def __init__(self, encoder: PreTrainedModel, metadata_size: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        size = encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(size + metadata_size, size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(size // 2, 1),
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor, metadata: Tensor) -> Tensor:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state[:, 0]
        return cast(Tensor, self.head(torch.cat([hidden, metadata], dim=1))).squeeze(-1)


def _encoder_blocks(encoder: PreTrainedModel) -> nn.ModuleList:
    if hasattr(encoder, "transformer"):  # DistilBERT
        return cast(nn.ModuleList, encoder.transformer.layer)
    return cast(nn.ModuleList, encoder.encoder.layer)


def _embedding_module(encoder: PreTrainedModel) -> nn.Module:
    return cast(nn.Module, encoder.embeddings)


def freeze_lower_layers(model: TransformerClassifier, trainable_blocks: int) -> int:
    """Freeze the embeddings and all but the top blocks. Returns the blocks frozen.

    This is what makes a CPU run affordable: frozen layers skip the backward pass (PLAN.md, 13).
    """
    blocks = _encoder_blocks(model.encoder)
    frozen = max(len(blocks) - trainable_blocks, 0)
    for parameter in _embedding_module(model.encoder).parameters():
        parameter.requires_grad = False
    for block in list(blocks)[:frozen]:
        for parameter in block.parameters():
            parameter.requires_grad = False
    return frozen


def load_encoder(name: str, pretrained: bool = True) -> PreTrainedModel:
    """Pretrained weights at the pinned commit, as safetensors, and never remote code."""
    repo, revision = PRETRAINED[name]
    if pretrained:
        return AutoModel.from_pretrained(
            repo, revision=revision, use_safetensors=True, trust_remote_code=False
        )
    config = AutoConfig.from_pretrained(repo, revision=revision, trust_remote_code=False)
    return AutoModel.from_config(config, trust_remote_code=False)


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
