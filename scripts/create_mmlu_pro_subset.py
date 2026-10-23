import argparse
from decimal import Decimal, ROUND_HALF_UP

from datasets import load_dataset
from huggingface_hub import DatasetCard


SOURCE_DATASET = "TIGER-Lab/MMLU-Pro"
SOURCE_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
DESTINATION_DATASET = "hung20gg/mmlu-pro-subset"
DEFAULT_SPLIT = "test"
DEFAULT_FRACTION = Decimal("0.20")
DEFAULT_SEED = 42


def publish_dataset_card(
    repo_id: str,
    source_dataset: str,
    source_revision: str,
    split: str,
    fraction: Decimal,
    seed: int,
    source_size: int,
    subset_size: int,
    license_name: str,
) -> None:
    card = DatasetCard.load(repo_id)
    card.data.language = ["en"]
    card.data.license = license_name
    card.data.pretty_name = "MMLU-Pro 20% Subset"
    card.data.task_categories = ["question-answering"]
    card.data.tags = ["evaluation", "multiple-choice"]
    card.text = f"""# MMLU-Pro 20% Subset

This dataset contains a uniform random sample without replacement from the
[`{source_dataset}`](https://huggingface.co/datasets/{source_dataset}) `{split}`
split. It preserves every source column and its original feature type.

## Sampling details

- Source revision: `{source_revision}`
- Source rows: {source_size:,}
- Selected rows: {subset_size:,} ({subset_size / source_size:.4%})
- Requested fraction: {fraction:.0%}
- Random seed: `{seed}`
- Procedure: shuffle the full `{split}` split with Hugging Face Datasets using
  the stated seed, then select the first {subset_size:,} rows

The selection is a simple uniform random sample, not a category-stratified sample.
"""
    card.push_to_hub(
        repo_id,
        commit_message="Document subset provenance and sampling procedure",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and optionally publish a uniform random MMLU-Pro subset."
    )
    parser.add_argument("--source", default=SOURCE_DATASET)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--destination", default=DESTINATION_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--fraction", type=Decimal, default=DEFAULT_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the generated split to --destination on Hugging Face.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Decimal("0") < args.fraction <= Decimal("1"):
        raise ValueError("fraction must be in the interval (0, 1]")

    source = load_dataset(
        args.source,
        split=args.split,
        revision=args.source_revision,
    )
    sample_size = int(
        (Decimal(len(source)) * args.fraction).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    subset = source.shuffle(seed=args.seed).select(range(sample_size))

    if subset.features != source.features:
        raise RuntimeError("Subset schema differs from the source schema")
    if len(set(subset["question_id"])) != len(subset):
        raise RuntimeError("Subset contains duplicate question_id values")

    subset.info.description = (
        f"A {args.fraction:.0%} uniform random sample without replacement of "
        f"the {args.source} {args.split} split. Seed: {args.seed}. "
        f"Source revision: {args.source_revision}."
    )
    subset.info.homepage = f"https://huggingface.co/datasets/{args.source}"
    license_name = source.info.license or "mit"
    subset.info.license = license_name

    print(
        f"Selected {len(subset)} of {len(source)} rows "
        f"({len(subset) / len(source):.4%}) with seed {args.seed}."
    )
    print(subset)

    if args.push:
        result = subset.push_to_hub(
            args.destination,
            split=args.split,
            commit_message=(
                f"Add seeded {args.fraction:.0%} subset of {args.source} "
                f"{args.split}"
            ),
        )
        print(result)
        publish_dataset_card(
            repo_id=args.destination,
            source_dataset=args.source,
            source_revision=args.source_revision,
            split=args.split,
            fraction=args.fraction,
            seed=args.seed,
            source_size=len(source),
            subset_size=len(subset),
            license_name=license_name,
        )


if __name__ == "__main__":
    main()
