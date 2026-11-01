"""Load complete, consistent test annotations from the Sem-Lex CSV."""

import csv
import math
from pathlib import Path

from .fields import FEATURE_FIELDS


def normalize_value(value):
    """Preserve the category 'None'; empty and NaN-like values are missing."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text or text.lower() in ["nan", "null", "<na>", "nat"]:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def resolve_video(video_id, root):
    """Support either the test directory or the parent containing test/."""
    if Path(video_id).name != video_id:
        raise ValueError("video_id must be a single file or directory name")
    root = Path(root).expanduser()
    names = [
        Path(video_id) / "videos/rgb_clean_480p.mp4",
        Path(video_id) / "videos/rgb_480.mp4",
        Path(video_id) / "rgb_clean_480p.mp4",
        Path(video_id) / "rgb_480.mp4",
        Path(video_id if video_id.endswith(".mp4") else video_id + ".mp4"),
    ]
    for base in [root, root / "test"]:
        for name in names:
            path = base / name
            if path.is_file():
                return str(path.resolve())
    return None


def load_metadata(path, video_root=None, metadata_only=False):
    """Return (nodes, summary), using only ASL-LEX-labelled test rows.

    Missing media are removed before annotations are grouped. Ambiguous video
    identities are excluded. A label is retained only when its observed values
    agree and all selected fields are known; train/val rows never fill gaps.
    """
    if not metadata_only and video_root is None:
        raise ValueError("Supply video_root, or explicitly request metadata_only=True")
    rows_by_video = {}
    available_paths = {}
    missing = []
    video_status = {}
    row_count = 0
    kept_rows = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = FEATURE_FIELDS + ["video_id", "signer_id", "label", "label_type", "split"]
        for column in required:
            if column not in (reader.fieldnames or []):
                raise ValueError("Metadata is missing column: " + column)
        for row in reader:
            if row["label_type"].strip().lower() != "asllex" or row["split"].strip() != "test":
                continue
            label = normalize_value(row["label"])
            video_id = normalize_value(row["video_id"])
            if label is None or video_id is None:
                continue
            row_count += 1
            if video_id not in video_status:
                video_path = None
                if not metadata_only:
                    video_path = resolve_video(video_id, video_root)
                video_status[video_id] = metadata_only or video_path is not None
                if video_path is not None:
                    available_paths[video_id] = video_path
                elif not metadata_only:
                    missing.append(video_id)
            if not video_status[video_id]:
                continue
            kept_rows += 1
            rows_by_video.setdefault(video_id, []).append(row)

    rows_by_label = {}
    ambiguous = []
    for video_id, rows in rows_by_video.items():
        labels = []
        signers = []
        for row in rows:
            label = normalize_value(row["label"])
            signer = normalize_value(row["signer_id"])
            if label not in labels:
                labels.append(label)
            if signer is not None and signer not in signers:
                signers.append(signer)
        if len(labels) != 1 or len(signers) > 1:
            ambiguous.append({"video_id": video_id, "labels": labels, "signers": signers})
            available_paths.pop(video_id, None)
            continue
        rows_by_label.setdefault(labels[0], []).extend(rows)

    nodes = {}
    conflicts = []
    incomplete = []
    for label in sorted(rows_by_label):
        rows = rows_by_label[label]
        vector = {}
        bad_fields = {}
        for field in FEATURE_FIELDS:
            observed = []
            for row in rows:
                value = normalize_value(row[field])
                if value is not None and value not in observed:
                    observed.append(value)
            if len(observed) > 1:
                bad_fields[field] = sorted(observed)
            elif len(observed) == 1:
                vector[field] = observed[0]
        if bad_fields:
            conflicts.append({"label": label, "conflicting_fields": bad_fields})
            continue
        if len(vector) != len(FEATURE_FIELDS):
            missing_fields = []
            for field in FEATURE_FIELDS:
                if field not in vector:
                    missing_fields.append(field)
            incomplete.append({"label": label, "missing_fields": missing_fields})
            continue
        video_ids = []
        signers = {}
        for row in rows:
            video_id = normalize_value(row["video_id"])
            if video_id not in video_ids:
                video_ids.append(video_id)
            signers[video_id] = normalize_value(row["signer_id"])
        nodes[label] = {
            "label": label, "vector": vector,
            "video_ids": sorted(video_ids), "signers": signers,
        }
    summary = {
        "split": "test", "metadata_only": metadata_only,
        "metadata_rows": row_count, "metadata_videos": len(video_status),
        "retained_rows_after_media_filter": kept_rows,
        "labels_after_identity_filter": len(rows_by_label),
        "complete_consistent_nodes": len(nodes),
        "missing_video_ids": sorted(missing), "available_paths": available_paths,
        "ambiguous_video_identities": ambiguous,
        "conflicting_labels": conflicts, "incomplete_labels": incomplete,
    }
    return nodes, summary
