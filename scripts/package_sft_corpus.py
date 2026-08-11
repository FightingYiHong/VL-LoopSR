#!/usr/bin/env python3
"""Package, validate, or unpack the paper's 10k Proposer SFT corpus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path


CORPUS_ARCHIVE = "proposer_mllm_sft.json.gz"
CORPUS_JSON = "proposer_mllm_sft.json"
ASSET_ARCHIVES = {"images": "images.tar", "csv": "csv.tar"}
PROPOSAL_SUFFIX = "__proposer__mm_form"
REPAIR_SUFFIX = "__proposer__refine"


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            records = json.load(handle)
    else:
        with path.open("rt", encoding="utf-8") as handle:
            records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"corpus must be a JSON list: {path}")
    return records


def normalized_conversation_hash(record: dict) -> str:
    payload = json.dumps(
        record.get("conversations"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_id(record_id: str) -> str:
    for suffix in (PROPOSAL_SUFFIX, REPAIR_SUFFIX):
        if record_id.endswith(suffix):
            return record_id[: -len(suffix)]
    raise ValueError(f"unexpected example id suffix: {record_id}")


def require_materialized(path: Path) -> None:
    with path.open("rb") as handle:
        prefix = handle.read(64)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(f"Git LFS asset is not downloaded: {path}; run `git lfs pull`")


def validate_records(records: list[dict], root: Path) -> dict:
    if len(records) != 10_000:
        raise ValueError(f"expected 10,000 examples, found {len(records)}")

    ids = [str(item.get("id", "")) for item in records]
    if len(set(ids)) != len(ids):
        raise ValueError("example ids are not unique")

    modes = Counter(
        "proposal" if item_id.endswith(PROPOSAL_SUFFIX) else "repair"
        if item_id.endswith(REPAIR_SUFFIX)
        else "unknown"
        for item_id in ids
    )
    if modes != Counter({"proposal": 5_000, "repair": 5_000}):
        raise ValueError(f"unexpected proposal/repair counts: {dict(modes)}")

    base_counts = Counter(task_id(item_id) for item_id in ids)
    if len(base_counts) != 5_000 or set(base_counts.values()) != {2}:
        raise ValueError("expected 5,000 tasks with one proposal and one repair each")

    image_refs: list[str] = []
    for record in records:
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or len(conversations) != 2:
            raise ValueError(f"invalid conversations for {record['id']}")
        if [turn.get("from") for turn in conversations] != ["human", "gpt"]:
            raise ValueError(f"invalid roles for {record['id']}")
        images = record.get("images")
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(f"expected one image for {record['id']}")
        image_ref = str(images[0])
        if Path(image_ref).is_absolute() or ".." in Path(image_ref).parts:
            raise ValueError(f"non-portable image path for {record['id']}: {image_ref}")
        image_path = root / image_ref
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image for {record['id']}: {image_path}")
        require_materialized(image_path)
        image_refs.append(image_ref)

    if len(set(image_refs)) != 10_000:
        raise ValueError("expected 10,000 unique image references")

    csv_paths = [root / "csv" / f"{base_id}.csv" for base_id in base_counts]
    missing_csv = [str(path) for path in csv_paths if not path.is_file()]
    if missing_csv:
        raise FileNotFoundError(f"missing task CSV files, first: {missing_csv[0]}")
    for path in csv_paths:
        require_materialized(path)
    actual_csv_count = sum(1 for path in (root / "csv").glob("*.csv") if path.is_file())
    actual_image_count = sum(1 for path in (root / "images").glob("*.png") if path.is_file())
    if actual_csv_count != 5_000 or actual_image_count != 10_000:
        raise ValueError(
            f"asset counts differ: csv={actual_csv_count}, images={actual_image_count}"
        )

    hashes = [normalized_conversation_hash(record) for record in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError("exact duplicate normalized conversation examples detected")

    return {
        "examples": len(records),
        "unique_tasks": len(base_counts),
        "proposal_examples": modes["proposal"],
        "repair_examples": modes["repair"],
        "images": actual_image_count,
        "csv_files": actual_csv_count,
        "exact_duplicate_examples": 0,
    }


def write_deterministic_gzip(records: list[dict], output: Path) -> None:
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(payload)


def write_deterministic_tar(source: Path, output: Path, arcname: str) -> None:
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
        paths = [source, *sorted(source.rglob("*"))]
        for path in paths:
            relative = Path(arcname) if path == source else Path(arcname) / path.relative_to(source)
            info = archive.gettarinfo(str(path), arcname=str(relative))
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if path.is_file():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)


def archive_assets(root: Path, destination: Path | None = None) -> None:
    root = root.resolve()
    destination = (destination or root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for directory, archive_name in ASSET_ARCHIVES.items():
        source = root / directory
        output = destination / archive_name
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing archive: {output}")
        if not source.is_dir():
            raise SystemExit(f"missing asset directory: {source}")
        write_deterministic_tar(source, output, directory)
        print(output)


def package(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise SystemExit(f"source contains symbolic links: {source}")

    source_json = source / CORPUS_JSON
    records = load_records(source_json)
    portable_records = []
    for record in records:
        copied = dict(record)
        images = copied.get("images") or []
        if len(images) != 1:
            raise ValueError(f"expected one image for {record.get('id')}")
        copied["images"] = [f"images/{Path(str(images[0])).name}"]
        portable_records.append(copied)

    output.mkdir(parents=True, exist_ok=True)
    write_deterministic_gzip(portable_records, output / CORPUS_ARCHIVE)
    summary = validate_records(portable_records, source)
    archive_assets(source, output)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate(root: Path) -> None:
    root = root.resolve()
    records = load_records(root / CORPUS_ARCHIVE)
    summary = validate_records(records, root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def unpack(root: Path) -> None:
    root = root.resolve()
    source = root / CORPUS_ARCHIVE
    target = root / CORPUS_JSON
    if not target.exists():
        with gzip.open(source, "rb") as compressed, target.open("wb") as raw:
            shutil.copyfileobj(compressed, raw)
        print(target)

    for directory, archive_name in ASSET_ARCHIVES.items():
        target_dir = root / directory
        if target_dir.is_dir() and any(target_dir.iterdir()):
            continue
        archive_path = root / archive_name
        require_materialized(archive_path)
        with tarfile.open(archive_path, "r") as archive:
            for member in archive.getmembers():
                member_path = Path(member.name)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or not member_path.parts
                    or member_path.parts[0] != directory
                ):
                    raise ValueError(f"unsafe archive member: {member.name}")
            archive.extractall(root, filter="data")
        print(target_dir)
    validate(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    package_parser = subparsers.add_parser("package", help="create a portable package")
    package_parser.add_argument("source", type=Path)
    package_parser.add_argument("output", type=Path)

    validate_parser = subparsers.add_parser("validate", help="validate the tracked package")
    validate_parser.add_argument("root", type=Path, nargs="?", default=Path("data/proposer_sft"))

    archive_parser = subparsers.add_parser("archive", help="archive materialized image and CSV assets")
    archive_parser.add_argument("root", type=Path, nargs="?", default=Path("data/proposer_sft"))

    unpack_parser = subparsers.add_parser("unpack", help="unpack JSON for training")
    unpack_parser.add_argument("root", type=Path, nargs="?", default=Path("data/proposer_sft"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "package":
        package(args.source, args.output)
    elif args.command == "validate":
        validate(args.root)
    elif args.command == "archive":
        archive_assets(args.root)
    else:
        unpack(args.root)


if __name__ == "__main__":
    main()
