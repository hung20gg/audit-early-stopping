from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_archive(source_dir: Path, archive_path: Path) -> int:
    source_dir = source_dir.resolve()
    archive_path = archive_path.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Trace directory does not exist: {source_dir}")

    entries = sorted(source_dir.rglob("*"))
    file_count = 0
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for path in entries:
            if path.resolve() == archive_path:
                continue
            archive_name = Path(source_dir.name) / path.relative_to(source_dir)
            archive.write(path, archive_name.as_posix())
            if path.is_file():
                file_count += 1
    return file_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zip the output trace directory and upload it to Hugging Face."
    )
    parser.add_argument("repo_id", help="Hugging Face repository, e.g. user/traces")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--filename", default="output.zip", help="Archive path in the repository")
    parser.add_argument("--repo-type", choices=("dataset", "model", "space"), default="dataset")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None, help="Defaults to HF_TOKEN or saved HF login")
    parser.add_argument("--private", action="store_true", help="Create a private repo if it does not exist")
    parser.add_argument("--commit-message", default="Upload replay traces")
    return parser.parse_args()


def main() -> None:
    from huggingface_hub import HfApi

    args = parse_args()
    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(prefix="early-exit-traces-") as temp_dir:
        archive_path = Path(temp_dir) / "output.zip"
        file_count = create_archive(args.source, archive_path)
        print(
            f"Created {archive_path} with {file_count} files "
            f"({archive_path.stat().st_size / 2**30:.2f} GiB)"
        )
        url = api.upload_file(
            path_or_fileobj=archive_path,
            path_in_repo=args.filename,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=args.commit_message,
        )
    print(f"Uploaded to {url}")


if __name__ == "__main__":
    main()
