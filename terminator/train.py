from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from terminator.dataset import TerminatorHiddenStateDataset
from terminator.hf_runtime import load_model_and_tokenizer
from terminator.probe import build_probe_from_loaded_model

IGNORE_INDEX = -100


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    max_len = max(item["hidden_states"].shape[0] for item in batch)
    hidden_size = batch[0]["hidden_states"].shape[1]

    hidden_states = torch.zeros(len(batch), max_len, hidden_size, dtype=torch.float32)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    position_ids = torch.zeros(len(batch), max_len, dtype=torch.long)

    for row, item in enumerate(batch):
        seq_len = item["hidden_states"].shape[0]
        hidden_states[row, :seq_len] = item["hidden_states"].to(torch.float32)
        labels[row, :seq_len] = item["labels"]
        attention_mask[row, :seq_len] = 1
        positions = item.get("positions")
        position_ids[row, :seq_len] = (
            positions.to(torch.long)
            if positions is not None
            else torch.arange(seq_len, dtype=torch.long)
        )

    return {
        "hidden_states": hidden_states,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def compute_class_weights(dataset: TerminatorHiddenStateDataset) -> tuple[float, float]:
    n0 = 0
    n1 = 0
    for index in range(len(dataset)):
        labels = dataset[index]["labels"]
        n1 += int(labels.sum().item())
        n0 += int(labels.numel() - labels.sum().item())
    total = n0 + n1
    w0 = total / (2 * n0) if n0 else 1.0
    w1 = total / (2 * n1) if n1 else 1.0
    return w0, w1


def masked_weighted_bce_loss(logits, labels, w0: float, w1: float):
    import torch
    import torch.nn.functional as F

    valid = labels != IGNORE_INDEX
    logits_valid = logits[valid]
    labels_valid = labels[valid].float()
    per_token_loss = F.binary_cross_entropy_with_logits(logits_valid, labels_valid, reduction="none")
    weights = torch.where(labels_valid == 1, torch.as_tensor(w1), torch.as_tensor(w0))
    return (per_token_loss * weights).sum() / weights.sum().clamp(min=1.0)


def cosine_lr_with_floor(step: int, *, warmup_steps: int, total_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (lr - min_lr) * cosine


def _predicted_exit_index(preds_1d) -> int:
    """First position predicted positive; sequence length if never predicted
    positive (i.e. the probe never signals it's safe to stop)."""
    for index, value in enumerate(preds_1d.tolist()):
        if value == 1:
            return index
    return len(preds_1d)


def evaluate_probe(probe, dataset: TerminatorHiddenStateDataset, batch_size: int = 8) -> dict[str, float]:
    import torch

    probe.eval()
    tp1 = fp1 = fn1 = 0  # class 1 (post-answer)
    tp0 = fp0 = fn0 = 0  # class 0 (pre-answer)
    exit_errors = []
    premature = 0
    total_sequences = 0

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            collated = collate_fn(batch)
            logits = probe(
                collated["hidden_states"],
                attention_mask=collated["attention_mask"],
                position_ids=collated["position_ids"],
            )
            preds = (torch.sigmoid(logits) >= 0.5).long()

            for row in range(len(batch)):
                valid = collated["labels"][row] != IGNORE_INDEX
                row_labels = collated["labels"][row][valid]
                row_preds = preds[row][valid]

                tp1 += int(((row_preds == 1) & (row_labels == 1)).sum())
                fp1 += int(((row_preds == 1) & (row_labels == 0)).sum())
                fn1 += int(((row_preds == 0) & (row_labels == 1)).sum())
                tp0 += int(((row_preds == 0) & (row_labels == 0)).sum())
                fp0 += int(((row_preds == 0) & (row_labels == 1)).sum())
                fn0 += int(((row_preds == 1) & (row_labels == 0)).sum())

                true_exit = int(row_labels.tolist().count(0))
                predicted_exit = _predicted_exit_index(row_preds)
                exit_errors.append(abs(predicted_exit - true_exit))
                if predicted_exit < true_exit:
                    premature += 1
                total_sequences += 1

    def f1(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    f1_class0 = f1(tp0, fp0, fn0)
    f1_class1 = f1(tp1, fp1, fn1)
    return {
        "f1_class0": f1_class0,
        "f1_class1": f1_class1,
        "macro_f1": (f1_class0 + f1_class1) / 2,
        "mean_exit_index_error": sum(exit_errors) / len(exit_errors) if exit_errors else 0.0,
        "premature_exit_rate": premature / total_sequences if total_sequences else 0.0,
    }


def train(
    base_model: str,
    train_shards: Path,
    val_shards: Path,
    output_dir: Path,
    *,
    device: str = "auto",
    epochs: int = 2,
    micro_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2.0e-4,
    min_learning_rate: float = 1.0e-6,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-8,
    warmup_steps: int = 100,
    seed: int = 1337,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)

    loaded = load_model_and_tokenizer(base_model, device=device)
    probe = build_probe_from_loaded_model(loaded).to(loaded.device)

    train_dataset = TerminatorHiddenStateDataset(train_shards)
    val_dataset = TerminatorHiddenStateDataset(val_shards)
    w0, w1 = compute_class_weights(train_dataset)

    loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    steps_per_epoch = math.ceil(len(loader) / gradient_accumulation_steps)
    total_steps = steps_per_epoch * epochs

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    best_metrics: dict[str, Any] | None = None
    optimizer_step = 0

    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        for micro_step, batch in enumerate(loader):
            hidden_states = batch["hidden_states"].to(loaded.device)
            labels = batch["labels"].to(loaded.device)
            attention_mask = batch["attention_mask"].to(loaded.device)
            position_ids = batch["position_ids"].to(loaded.device)

            logits = probe(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            loss = masked_weighted_bce_loss(logits, labels, w0, w1) / gradient_accumulation_steps
            loss.backward()

            if (micro_step + 1) % gradient_accumulation_steps == 0:
                lr = cosine_lr_with_floor(
                    optimizer_step,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                    lr=learning_rate,
                    min_lr=min_learning_rate,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

        metrics = evaluate_probe(probe, val_dataset)
        metrics["epoch"] = epoch
        print(f"epoch {epoch}: {metrics}")

        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            torch.save(probe.state_dict(), output_dir / "probe.pt")

    (output_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    (output_dir / "class_weights.json").write_text(
        json.dumps({"w0": w0, "w1": w1}, indent=2), encoding="utf-8"
    )
    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TERMINATOR probe.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-shards", type=Path, required=True)
    parser.add_argument("--val-shards", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        args.base_model,
        args.train_shards,
        args.val_shards,
        args.output_dir,
        device=args.device,
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
