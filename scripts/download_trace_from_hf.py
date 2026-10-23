from __future__ import annotations

import argparse
import os
import stat
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_archive(archive_path: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    file_count = 0

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if os.path.commonpath((str(destination), str(target))) != str(destination):
                raise ValueError(f"Unsafe archive member path: {member.filename!r}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Refusing archive symlink: {member.filename!r}")
        archive.extractall(destination)
        file_count = sum(not member.is_dir() for member in archive.infolist())
    return file_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a trace archive from Hugging Face and restore its folder structure."
    )
    parser.add_argument("repo_id", help="Hugging Face repository, e.g. user/traces")
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--filename", default="output.zip", help="Archive path in the repository")
    parser.add_argument("--repo-type", choices=("dataset", "model", "space"), default="dataset")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None, help="Defaults to HF_TOKEN or saved HF login")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    from huggingface_hub import hf_hub_download

    args = parse_args()
    archive_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type=args.repo_type,
            revision=args.revision,
            token=args.token,
            force_download=args.force_download,
        )
    )
    file_count = extract_archive(archive_path, args.destination)
    print(f"Extracted {file_count} files into {args.destination.resolve()}")


if __name__ == "__main__":
    main()
