"""Extract one frozen projected-token embedding per annotated Sem-Lex video."""

import argparse
import csv
import json
from pathlib import Path

from src.graph.data import normalize_value
from src.graph.fields import FEATURE_FIELDS
from src.qwen.run import extract_embedding, load_model
from src.video import resolve_video


def load_tokens(metadata):
    """Keep partial per-field labels; a missing field must not discard a token."""
    grouped = {}
    with Path(metadata).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for field in FEATURE_FIELDS + ["video_id", "split", "label", "signer_id", "label_type"]:
            if field not in (reader.fieldnames or []):
                raise ValueError("Metadata is missing column: " + field)
        for row in reader:
            if row["label_type"].strip().lower() != "asllex":
                continue
            if row["split"].strip() not in ["train", "test"]:
                continue
            video_id = normalize_value(row["video_id"])
            if video_id is not None:
                grouped.setdefault(video_id, []).append(row)
    records = []
    rejected = []
    for video_id in sorted(grouped):
        rows = grouped[video_id]
        splits = set()
        for row in rows:
            splits.add(row["split"].strip())
        if len(splits) > 1:
            raise ValueError("Train/test video overlap: " + video_id)
        record = {"video_id": video_id, "features": {}, "ambiguous_fields": []}
        bad_identity = False
        for field in ["label", "split", "signer_id"]:
            values = []
            for row in rows:
                value = normalize_value(row[field])
                if value is not None and value not in values:
                    values.append(value)
            if len(values) != 1:
                bad_identity = True
            else:
                record[field] = values[0]
        if bad_identity:
            rejected.append(video_id)
            continue
        for field in FEATURE_FIELDS:
            values = []
            for row in rows:
                value = normalize_value(row[field])
                if value is not None and value not in values:
                    values.append(value)
            if len(values) == 1:
                record["features"][field] = values[0]
            elif len(values) > 1:
                record["ambiguous_fields"].append(field)
        if record["features"]:
            records.append(record)
    return records, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="resources/semlex_metadata.csv")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", default="experiments/results/embeddings")
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not Path(args.video_root).is_dir():
        parser.error("--video-root must be an existing directory")

    records, rejected = load_tokens(args.metadata)
    available = []
    missing = []
    for record in records:
        spec = {"metadata": record, "role": "Target sign", "num_frames": 32,
                "sampling": "uniform", "crop": "none",
                "min_pixels": 16384, "max_pixels": 307200}
        try:
            spec["path"] = str(resolve_video(spec, args.video_root))
        except FileNotFoundError:
            missing.append(record["video_id"])
            continue
        available.append((record, spec))
    if args.limit is not None:
        available = available[:args.limit]
    if not available:
        raise ValueError("No annotated videos were found under --video-root.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for record, spec in available:
        if (output / (record["video_id"] + ".npy")).exists() and not args.resume:
            raise FileExistsError("Embedding files already exist; use --resume or a new output directory.")
    bundle = load_model(args.checkpoint, args.revision, args.dtype, args.device, args.attention)
    environment = bundle["environment"]
    run_path = output / "_run.json"
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous["environment"] != environment:
            raise ValueError("Cached embeddings use a different model/software configuration.")
    run_path.write_text(json.dumps({"environment": environment,
        "rejected_identity_conflicts": rejected, "missing_video_ids": missing,
        "requested_available_tokens": len(available)}, indent=2) + "\n", encoding="utf-8")

    import numpy as np

    errors = 0
    for index in range(len(available)):
        record, spec = available[index]
        video_id = record["video_id"]
        matrix_path = output / (video_id + ".npy")
        metadata_path = output / (video_id + ".json")
        if args.resume and matrix_path.exists() and metadata_path.exists():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if "error" not in previous:
                for key in ["features", "split", "label", "signer_id"]:
                    if previous.get(key) != record[key]:
                        raise ValueError(f"Metadata changed for cached video {video_id}: {key}.")
                continue
        try:
            embedding, audit = extract_embedding(bundle, spec)
            np.save(matrix_path, embedding.astype(np.float32))
            payload = dict(record)
            payload["embedding_dim"] = int(embedding.shape[0])
            payload["input_audit"] = audit
            metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except Exception as error:
            errors += 1
            payload = {"video_id": video_id, "error": type(error).__name__, "message": str(error)}
            metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"{index + 1}/{len(available)} {video_id}", flush=True)
    print(f"Completed with {errors} errors; {len(missing)} missing videos recorded in _run.json.")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
