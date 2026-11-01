"""Single-video feature data and prompts shared by prediction and adaptation."""

import csv
import json
from pathlib import Path
import re

from src.graph.data import normalize_value
from src.graph.fields import FEATURE_FIELDS
from src.video import resolve_video


def field_name(value):
    key = value.strip().lower().replace("_", " ")
    for field in FEATURE_FIELDS:
        if field.lower() == key:
            return field
    raise ValueError("Unknown annotation field: " + value)


def load_feature_data(metadata, video_root, field):
    """Return train/test examples, the metadata ontology, and selection counts.

    The allowed vocabulary comes from ASL-LEX-labelled metadata across splits.
    Example selection uses one requested field, so unrelated missing fields
    do not discard a video. Duplicate video annotations must agree.
    """
    field = field_name(field)
    vocabulary = set()
    grouped = {}
    with Path(metadata).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for column in ["video_id", "signer_id", "split", "label_type", field]:
            if column not in (reader.fieldnames or []):
                raise ValueError("Metadata is missing column: " + column)
        for row in reader:
            if row["label_type"].strip().lower() != "asllex":
                continue
            value = normalize_value(row[field])
            if value is not None:
                vocabulary.add(value)
            split = row["split"].strip()
            video_id = normalize_value(row["video_id"])
            if split not in ["train", "test"] or video_id is None:
                continue
            grouped.setdefault((split, video_id), []).append(row)
    if not vocabulary:
        raise ValueError("No observed labels for " + field)

    splits = {"train": [], "test": []}
    counts = {"field": field, "unlabelled": 0, "ambiguous": 0, "missing_videos": 0}
    for (split, video_id), rows in sorted(grouped.items()):
        values = set()
        signers = set()
        for row in rows:
            value = normalize_value(row[field])
            signer = normalize_value(row["signer_id"])
            if value is not None:
                values.add(value)
            if signer is not None:
                signers.add(signer)
        if not values:
            counts["unlabelled"] += 1
            continue
        if len(values) != 1 or len(signers) != 1:
            counts["ambiguous"] += 1
            continue
        record = {"video_id": video_id, "split": split,
                  "signer_id": list(signers)[0], "value": list(values)[0]}
        spec = {"metadata": record, "path": f"{split}/{video_id}.mp4"}
        if video_root is not None:
            try:
                record["path"] = str(resolve_video(spec, video_root))
            except FileNotFoundError:
                counts["missing_videos"] += 1
                continue
        else:
            record["path"] = spec["path"]
        splits[split].append(record)
    train_videos = set()
    train_signers = set()
    for row in splits["train"]:
        train_videos.add(row["video_id"])
        train_signers.add(row["signer_id"])
    for row in splits["test"]:
        if row["video_id"] in train_videos:
            raise ValueError("Train/test video overlap: " + row["video_id"])
        if row["signer_id"] in train_signers:
            raise ValueError("Train/test signer overlap: " + row["signer_id"])
    counts["train_examples"] = len(splits["train"])
    counts["test_examples"] = len(splits["test"])
    counts["media_checked"] = video_root is not None
    return splits, sorted(vocabulary), counts


def feature_prompt(field, labels):
    with (Path(__file__).parents[1] / "prompts/feature_specs.json").open() as handle:
        specifications = json.load(handle)
    specification = specifications[field_name(field)]
    lines = ["You are evaluating one isolated American Sign Language (ASL) sign.",
             "Use the visible signing form across the full video sequence.",
             "Ignore filenames, captions, glosses and English meaning.",
             "Choose exactly one allowed label and preserve its spelling.",
             "Treat allowed negative labels such as None or 0 as valid categories.",
             "", "Feature: " + field_name(field),
             "Definition: " + specification["definition"], "", "Allowed labels:"]
    for label in labels:
        hint = specification["label_hints"].get(label)
        lines.append("- " + label + (": " + hint if hint else ""))
    lines.append('\nReturn only JSON: {"answer": "<one allowed label>"}')
    return "\n".join(lines)


def feature_instance(example, prompt, frames=32):
    return {"prompt": prompt, "videos": [{
        "path": example["path"], "role": "Target sign", "num_frames": frames,
        "sampling": "uniform", "crop": "none", "min_pixels": 16384,
        "max_pixels": 307200, "metadata": {"video_id": example["video_id"],
        "split": example["split"]},
    }]}


def parse_label(raw_output, labels):
    """Recover one allowed label; conflicting answers and unknown labels fail."""
    values = []
    saw_answer = False
    for match in re.finditer(r"\{[^{}]*\}", raw_output, re.DOTALL):
        try:
            pairs = json.loads(match.group(), object_pairs_hook=list)
        except json.JSONDecodeError:
            continue
        for key, value in pairs:
            if key == "answer":
                saw_answer = True
                if value is None or isinstance(value, (bool, list, dict)):
                    return None
                values.append(str(value).strip())
    if not saw_answer:
        values.append(raw_output.strip().strip('"'))
    recovered = set()
    for value in values:
        canonical = normalize_value(value)
        if canonical in labels:
            recovered.add(canonical)
            continue
        matches = []
        for label in labels:
            if value.casefold() == label.casefold():
                matches.append(label)
        if len(matches) != 1:
            return None
        recovered.add(matches[0])
    return list(recovered)[0] if len(recovered) == 1 else None


def feature_metrics(examples, results, labels):
    """Every requested example is scored, including missing/invalid answers."""
    correct = valid = 0
    classes = set(labels)
    for row in examples:
        classes.add(row["value"])
    counts = {}
    for label in sorted(classes):
        counts[label] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}
    for example in examples:
        result = results.get(example["video_id"], {})
        prediction = result.get("prediction") if result.get("status") == "ok" else None
        if prediction not in labels:
            prediction = None
        gold = example["value"]
        counts[gold]["support"] += 1
        if prediction is not None:
            valid += 1
        if prediction == gold:
            correct += 1
            counts[gold]["tp"] += 1
        else:
            counts[gold]["fn"] += 1
            if prediction is not None:
                counts[prediction]["fp"] += 1
    recall = f1 = weighted_f1 = 0.0
    supported = 0
    for count in counts.values():
        if count["support"]:
            recall += count["tp"] / count["support"]
            supported += 1
        denominator = 2 * count["tp"] + count["fp"] + count["fn"]
        if denominator:
            score = 2 * count["tp"] / denominator
            f1 += score
            weighted_f1 += score * count["support"]
    total = len(examples)
    return {"examples": total, "valid": valid, "correct": correct,
            "accuracy": correct / total if total else None,
            "valid_rate": valid / total if total else None,
            "invalid_rate": (total - valid) / total if total else None,
            "balanced_accuracy": recall / supported if supported else None,
            "macro_f1": f1 / len(classes) if classes else None,
            "weighted_f1": weighted_f1 / total if total else None,
            "per_label": counts}
