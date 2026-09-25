"""Train and cross-validate the neural challenger.

Evaluation uses the identity-clustered folds from :mod:`validation.clustering`, not random
ones. That matters more for this model than for any other in the bake-off: a network with
149,807 trainable parameters against 129 labels will memorise, and random folds leave near
-identical relatives of every held-out protein in training, so a memorising model scores
well. Clustered folds remove that route.

Every run reports the trainable-parameter count beside the training-set size, because the
ratio is the single most important caveat on any number this model produces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from domain_layout.progress import progress
from experimental.neural.architecture import (
    DEFAULT_ESM_MODEL,
    MAX_SEQUENCE_LENGTH,
    ModelConfig,
    TorchUnavailableError,
    TrainingConfig,
    build_language_model,
    build_model,
    count_trainable_parameters,
    torch_available,
    transformers_available,
)
from experimental.neural.data import label_index
from validation.metrics import evaluate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from experimental.neural.data import ProteinExample

logger = logging.getLogger(__name__)

NEURAL_MODEL_NAME = "neural_hierarchical_cv"


@dataclass(frozen=True, slots=True)
class LabelSpace:
    """The label vocabulary both heads predict over.

    Bundled rather than passed as loose mappings so the two heads cannot be given
    inconsistent vocabularies, which would silently mislabel every prediction.
    """

    class_to_index: Mapping[str, int]
    subclass_to_index: Mapping[str, int]
    n_syntax_tokens: int

    @property
    def index_to_class(self) -> dict[int, str]:
        """Reverse mapping, for turning logits back into labels."""
        return {index: label for label, index in self.class_to_index.items()}


@dataclass(frozen=True, slots=True)
class FoldResult:
    """One fold's held-out predictions."""

    fold: int
    predictions: dict[str, str]
    n_train: int
    n_test: int


def _pad_profiles(examples: Sequence[ProteinExample], torch: Any) -> tuple[Any, Any]:
    """Stack variable-length profiles into (batch, channels, length) with a mask."""
    lengths = [len(item.profile[0]) for item in examples]
    width = max(lengths) if lengths else 1
    channels = len(examples[0].profile) if examples else 1
    padded = torch.zeros(len(examples), channels, width)
    mask = torch.zeros(len(examples), width)
    for row, item in enumerate(examples):
        for channel_index, channel in enumerate(item.profile):
            if channel:
                padded[row, channel_index, : len(channel)] = torch.tensor(channel, dtype=torch.float32)
        mask[row, : len(item.profile[0])] = 1.0
    return padded, mask


def _pad_syntax(examples: Sequence[ProteinExample], torch: Any) -> Any:
    """Stack token sequences into (batch, length), padded with zeros."""
    width = max((len(item.syntax) for item in examples), default=1)
    padded = torch.zeros(len(examples), width, dtype=torch.long)
    for row, item in enumerate(examples):
        padded[row, : len(item.syntax)] = torch.tensor(item.syntax, dtype=torch.long)
    return padded


def _tokenize(examples: Sequence[ProteinExample], tokenizer: Any) -> dict[str, Any]:
    """Tokenize raw sequences for the language model.

    ESM-2 expects residues space-separated in some tokenizer configurations and raw in
    others; the fast tokenizer handles both, so the sequence is passed through unmodified
    and padding is left to the tokenizer rather than reimplemented here.
    """
    encoded = tokenizer(
        [item.sequence for item in examples],
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt",
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def _static_matrix(examples: Sequence[ProteinExample], torch: Any) -> Any:
    """Stack the static feature vectors, standardised per column."""
    matrix = torch.tensor([list(item.static) for item in examples], dtype=torch.float32)
    mean = matrix.mean(dim=0, keepdim=True)
    std = matrix.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (matrix - mean) / std


def _fit(
    model: Any,
    tensors: dict[str, Any],
    optimisation: dict[str, Any],
    *,
    n_examples: int,
    settings: TrainingConfig,
) -> None:
    """Run the training loop in place, keeping the best-scoring weights.

    Early stopping on training loss rather than a validation split: at this sample size,
    holding out a further slice of an already-small fold costs more than the early-stopping
    signal is worth, and the honest evaluation is the clustered outer fold regardless.
    """
    torch = optimisation["torch"]
    criterion = optimisation["criterion"]
    subclass_criterion = optimisation["subclass_criterion"]
    optimizer = optimisation["optimizer"]

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    since_improved = 0
    order = torch.arange(n_examples)

    model.train()
    for _epoch in range(settings.epochs):
        permutation = order[torch.randperm(len(order))]
        epoch_loss = 0.0
        for start in range(0, len(permutation), settings.batch_size):
            batch = permutation[start : start + settings.batch_size]
            optimizer.zero_grad()
            class_logits, subclass_logits = model(
                tensors["profile"][batch],
                tensors["mask"][batch],
                tensors["syntax"][batch],
                static_features=tensors["static"][batch],
                input_ids=tensors["input_ids"][batch] if "input_ids" in tensors else None,
                attention_mask=tensors["attention_mask"][batch] if "attention_mask" in tensors else None,
            )
            # The subclass term is down-weighted: it is the finer, noisier label, and at
            # this sample size letting it dominate costs accuracy on the coarse one.
            loss = criterion(class_logits, tensors["class"][batch]) + 0.3 * subclass_criterion(
                subclass_logits,
                tensors["subclass"][batch],
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= settings.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def train_fold(
    train_examples: Sequence[ProteinExample],
    test_examples: Sequence[ProteinExample],
    *,
    labels: LabelSpace,
    settings: TrainingConfig,
) -> dict[str, str]:
    """Train on one fold and return held-out class predictions."""
    if not torch_available():
        message = "torch is not installed; the neural challenger cannot train"
        raise TorchUnavailableError(message)

    import torch
    from torch import nn

    torch.manual_seed(settings.seed)
    if settings.device != "auto":
        device = torch.device(settings.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ModelConfig(
        n_classes=len(labels.class_to_index),
        n_subclasses=len(labels.subclass_to_index),
        n_syntax_tokens=labels.n_syntax_tokens,
        n_static_features=len(train_examples[0].static) if train_examples else 0,
        use_language_model=settings.use_language_model,
    )

    # The language model is rebuilt per fold rather than shared. Sharing it would carry
    # adapter weights fitted on one fold's training set into the next fold's held-out
    # proteins, which is precisely the leak the clustered folds exist to prevent.
    tokenizer = None
    encoder = None
    if settings.use_language_model and transformers_available():
        tokenizer, encoder = build_language_model(config)
        config = replace(config, esm_hidden_size=int(encoder.config.hidden_size))
    elif settings.use_language_model:
        logger.warning("transformers/peft unavailable; training grammar and syntax encoders only")

    model = build_model(config, language_model=encoder).to(device)

    profile, mask = _pad_profiles(train_examples, torch)
    tensors = {
        "profile": profile.to(device),
        "mask": mask.to(device),
        "syntax": _pad_syntax(train_examples, torch).to(device),
        "static": _static_matrix(train_examples, torch).to(device),
        "class": torch.tensor(
            [labels.class_to_index[item.label_class] for item in train_examples],
            dtype=torch.long,
        ).to(device),
        "subclass": torch.tensor(
            [labels.subclass_to_index[item.label_subclass] for item in train_examples],
            dtype=torch.long,
        ).to(device),
    }
    if tokenizer is not None:
        encoded = _tokenize(train_examples, tokenizer)
        tensors["input_ids"] = encoded["input_ids"].to(device)
        tensors["attention_mask"] = encoded["attention_mask"].to(device)

    # Class weights: the largest labelled class is several times the smallest, and without
    # them the model learns the prior and stops.
    weights = None
    if settings.balance_classes:
        counts = torch.bincount(tensors["class"], minlength=len(labels.class_to_index)).float().clamp(min=1.0)
        weights = (counts.sum() / (len(labels.class_to_index) * counts)).to(device)

    _fit(
        model,
        tensors,
        {
            "torch": torch,
            "criterion": nn.CrossEntropyLoss(weight=weights),
            "subclass_criterion": nn.CrossEntropyLoss(),
            # Two parameter groups. The adapters move at a far smaller rate than the
            # heads: the pretrained representation is the asset, and 129 labels are not
            # enough to justify moving it far. One rate for both would either leave the
            # heads undertrained or drag ESM-2 off its pretrained solution.
            "optimizer": torch.optim.AdamW(
                [
                    {
                        "params": [
                            parameter
                            for name, parameter in model.named_parameters()
                            if parameter.requires_grad and name.startswith("language_model.")
                        ],
                        "lr": settings.language_model_learning_rate,
                    },
                    {
                        "params": [
                            parameter
                            for name, parameter in model.named_parameters()
                            if parameter.requires_grad and not name.startswith("language_model.")
                        ],
                        "lr": settings.learning_rate,
                    },
                ],
                weight_decay=settings.weight_decay,
            ),
        },
        n_examples=len(train_examples),
        settings=settings,
    )

    model.eval()
    with torch.no_grad():
        test_profile, test_mask = _pad_profiles(test_examples, torch)
        test_tokens = _tokenize(test_examples, tokenizer) if tokenizer is not None else {}
        class_logits, _ = model(
            test_profile.to(device),
            test_mask.to(device),
            _pad_syntax(test_examples, torch).to(device),
            static_features=_static_matrix(test_examples, torch).to(device),
            input_ids=test_tokens["input_ids"].to(device) if test_tokens else None,
            attention_mask=test_tokens["attention_mask"].to(device) if test_tokens else None,
        )
        predicted = class_logits.argmax(dim=-1).cpu().tolist()
    index_to_class = labels.index_to_class
    return {item.accession: index_to_class[index] for item, index in zip(test_examples, predicted, strict=True)}


def cross_validate(
    examples: Sequence[ProteinExample],
    folds: Sequence[Sequence[str]],
    *,
    n_syntax_tokens: int,
    settings: TrainingConfig | None = None,
) -> dict[str, str]:
    """Out-of-fold predictions across identity-clustered folds."""
    config = settings or TrainingConfig()
    by_accession = {item.accession: item for item in examples}
    label_space = LabelSpace(
        class_to_index=label_index([item.label_class for item in examples]),
        subclass_to_index=label_index([item.label_subclass for item in examples]),
        n_syntax_tokens=n_syntax_tokens,
    )

    usable = sum(
        1
        for fold in folds
        if any(accession in by_accession for accession in fold)
        and len([item for item in examples if item.accession not in set(fold)]) >= len(label_space.class_to_index)
    )
    if usable == 0:
        # Every fold is unusable, which happens when redundancy clustering collapses the
        # whole set into one group: the single fold then holds every protein, leaving no
        # training set. Silently returning nothing produced a report reading "mean accuracy
        # 0.0 over 0 seeds", which looks like a model that failed rather than an evaluation
        # that could not be constructed.
        message = (
            f"no usable folds: {len(folds)} fold(s) over {len(examples)} examples leave no "
            f"training set. This usually means identity clustering produced a single "
            f"cluster, so there is no held-out set to evaluate on."
        )
        raise ValueError(message)

    predictions: dict[str, str] = {}
    for index, fold in enumerate(progress(list(folds), description="neural folds", unit="fold")):
        test = [by_accession[accession] for accession in fold if accession in by_accession]
        train = [item for item in examples if item.accession not in set(fold)]
        if not test or len(train) < len(label_space.class_to_index):
            continue
        predictions.update(
            train_fold(train, test, labels=label_space, settings=config),
        )
        logger.info("fold %s: trained on %s, predicted %s", index + 1, len(train), len(test))
    return predictions


def neural_report(
    examples: Sequence[ProteinExample],
    folds: Sequence[Sequence[str]],
    *,
    n_syntax_tokens: int,
    settings: TrainingConfig | None = None,
) -> dict[str, object]:
    """Cross-validate over several seeds and report the spread.

    Several seeds, because a model this large on a set this small is seed-sensitive, and a
    single run's accuracy would be an unreproducible number rather than a measurement.
    """
    config = settings or TrainingConfig()
    truth = {item.accession: item.label_class for item in examples}

    per_seed: list[dict[str, object]] = []
    per_seed_predictions: list[dict[str, str]] = []
    for seed in config.seeds:
        # replace() rather than reconstructing field by field: the manual version silently
        # dropped every setting added after it was written, which would have disabled the
        # language model on every seed while still reporting a number.
        seeded = replace(config, seed=seed)
        predictions = cross_validate(examples, folds, n_syntax_tokens=n_syntax_tokens, settings=seeded)
        if not predictions:
            continue
        per_seed_predictions.append(dict(predictions))
        scored = {key: value for key, value in truth.items() if key in predictions}
        per_seed.append(
            {
                "seed": seed,
                **evaluate(f"{NEURAL_MODEL_NAME}_seed{seed}", scored, predictions).to_json_dict(),
            },
        )

    accuracies = [float(item["accuracy"]) for item in per_seed]
    # Predictions from the first seed are carried out so the bake-off can score this model
    # on exactly the proteins every other model was scored on. Without them the neural
    # challenger would only ever have a report of its own, which cannot be compared.
    out_of_fold = per_seed_predictions[0] if per_seed_predictions else {}
    model_config = ModelConfig(
        n_classes=len({item.label_class for item in examples}),
        n_subclasses=len({item.label_subclass for item in examples}),
        n_syntax_tokens=n_syntax_tokens,
        n_static_features=len(examples[0].static) if examples else 0,
        use_language_model=config.use_language_model,
    )
    trainable = 0
    if torch_available():
        encoder = None
        if config.use_language_model and transformers_available():
            _, encoder = build_language_model(model_config)
        trainable = count_trainable_parameters(build_model(model_config, language_model=encoder))

    return {
        "model": NEURAL_MODEL_NAME,
        "n_examples": len(examples),
        "n_folds": len(folds),
        "n_seeds": len(per_seed),
        "language_model": config.use_language_model and transformers_available(),
        "language_model_name": DEFAULT_ESM_MODEL if config.use_language_model else None,
        "trainable_parameters": trainable,
        "parameters_per_labelled_example": round(trainable / len(examples), 1) if examples else 0.0,
        "accuracy_by_seed": accuracies,
        "accuracy_mean": round(sum(accuracies) / len(accuracies), 4) if accuracies else 0.0,
        "accuracy_spread": round(max(accuracies) - min(accuracies), 4) if accuracies else 0.0,
        "per_seed": per_seed,
        "out_of_fold_predictions": out_of_fold,
        "caveat": (
            "Folds are identity-clustered, not random, so a memorising model cannot score "
            "by recognising near-copies of held-out proteins. The parameters-per-example "
            "ratio is reported because it, not the architecture, is the binding constraint "
            "here: see validation.capacity for how fine a distinction this label set can "
            "support at all."
        ),
    }
