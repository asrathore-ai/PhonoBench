"""Train one class-weighted linear probe per field from frozen video embeddings."""

import argparse
import json
import random
from pathlib import Path

from src.files import write_json, write_jsonl
from src.graph.data import normalize_value
from src.graph.fields import FEATURE_FIELDS


def load_embeddings(directory):
    """Read paired video_id.npy/video_id.json files, including split folders."""
    import numpy as np
    import torch

    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Embedding directory does not exist: {directory}")
    rows = []
    embeddings = []
    identities = set()
    dimension = None
    skipped = 0
    for metadata_path in sorted(directory.rglob("*.json")):
        matrix_path = metadata_path.with_suffix(".npy")
        if not matrix_path.is_file():
            continue
        with metadata_path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        if "error" in row or row.get("split") not in ["train", "test"]:
            skipped += 1
            continue
        for key in ["video_id", "label", "signer_id"]:
            value = normalize_value(row.get(key))
            if value is None:
                raise ValueError(f"Missing {key} in {metadata_path}")
            row[key] = value
        if not isinstance(row.get("features"), dict):
            raise ValueError(f"Missing features dictionary in {metadata_path}")
        identity = (row["split"], row["video_id"])
        if identity in identities:
            raise ValueError(f"Duplicate embedding for {identity}")
        identities.add(identity)
        vector = np.load(matrix_path, allow_pickle=False).astype(np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError(f"Expected a finite nonempty embedding vector: {matrix_path}")
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            raise ValueError(f"Embedding dimension differs in {matrix_path}")
        rows.append(row)
        embeddings.append(torch.from_numpy(vector))
    if not rows:
        raise ValueError("No paired train/test embeddings were found")
    return rows, torch.stack(embeddings), skipped


def split_indices(rows):
    train = []
    test = []
    train_signers = set()
    test_signers = set()
    train_videos = set()
    test_videos = set()
    train_labels = set()
    test_labels = set()
    for index, row in enumerate(rows):
        if row["split"] == "train":
            train.append(index)
            train_signers.add(row["signer_id"])
            train_videos.add(row["video_id"])
            train_labels.add(row["label"])
        elif row["split"] == "test":
            test.append(index)
            test_signers.add(row["signer_id"])
            test_videos.add(row["video_id"])
            test_labels.add(row["label"])
    if not train or not test:
        raise ValueError("The cache must contain both train and test videos")
    if train_signers & test_signers:
        raise ValueError(f"Train/test signer overlap: {sorted(train_signers & test_signers)}")
    if train_videos & test_videos:
        raise ValueError(f"Train/test video overlap: {sorted(train_videos & test_videos)}")
    return train, test, {
        "train_videos": len(train), "test_videos": len(test),
        "train_signers": sorted(train_signers), "test_signers": sorted(test_signers),
        "lexical_type_overlap": len(train_labels & test_labels),
        "video_overlap": 0, "signer_overlap": 0,
    }


def classification_metrics(gold, prediction, train_labels):
    """Evaluate every labeled test row, including labels absent from training."""
    if not gold or len(gold) != len(prediction):
        raise ValueError("Gold and prediction must be nonempty lists of equal length")
    labels = set()
    gold_labels = set()
    correct = known = 0
    for actual, predicted in zip(gold, prediction):
        labels.add(actual)
        labels.add(predicted)
        gold_labels.add(actual)
        correct += actual == predicted
        known += actual in train_labels
    recall_sum = f1_sum = weighted_f1 = 0.0
    for label in sorted(labels):
        true_positive = false_positive = false_negative = 0
        for actual, predicted in zip(gold, prediction):
            true_positive += actual == label and predicted == label
            false_positive += actual != label and predicted == label
            false_negative += actual == label and predicted != label
        support = true_positive + false_negative
        if support:
            recall_sum += true_positive / support
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        f1_sum += f1
        weighted_f1 += support * f1
    return {
        "n": len(gold), "accuracy": correct / len(gold),
        "balanced_accuracy": recall_sum / len(gold_labels),
        "macro_f1": f1_sum / len(labels), "weighted_f1": weighted_f1 / len(gold),
        "known_gold_label_fraction": known / len(gold),
        "test_labels": sorted(gold_labels), "macro_f1_labels": sorted(labels),
    }


def fit_probe(train_x, train_y, num_classes, epochs=60, batch_size=512,
              learning_rate=1e-3, weight_decay=1e-4, device="cpu", seed=42):
    """Fit a standardized torch.nn.Linear classifier using ordinary batch loops."""
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    mean = train_x.mean(dim=0, keepdim=True)
    correction = 1 if train_x.shape[0] > 1 else 0
    std = train_x.std(dim=0, keepdim=True, correction=correction).clamp_min(1e-6)
    matrix = ((train_x - mean) / std).to(device)
    targets = train_y.to(device)
    counts = torch.bincount(train_y, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    weights = (weights / weights.mean()).to(device)
    classifier = torch.nn.Linear(matrix.shape[1], num_classes).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(epochs):
        classifier.train()
        order = torch.randperm(len(matrix), device=device)
        for start in range(0, len(matrix), batch_size):
            batch = order[start:start + batch_size]
            loss = loss_function(classifier(matrix[batch]), targets[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    classifier.eval()
    return classifier, mean, std


def train_probes(embeddings_dir, output_dir, fields=None, epochs=60, batch_size=512,
                 learning_rate=1e-3, weight_decay=1e-4, device="auto", seed=42):
    import torch

    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Use positive epochs, batch size and learning rate, and nonnegative weight decay")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ["cpu", "cuda"]:
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    fields = FEATURE_FIELDS if fields is None else fields
    for field in fields:
        if field not in FEATURE_FIELDS:
            raise ValueError(f"Unknown feature field: {field}")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose an empty output directory")
    rows, embeddings, skipped = load_embeddings(embeddings_dir)
    train, test, summary = split_indices(rows)
    summary.update({"embedding_dimension": int(embeddings.shape[1]), "skipped_cache_records": skipped,
                    "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                    "weight_decay": weight_decay, "device": device, "seed": seed,
                    "torch": torch.__version__, "class_weight": "inverse training frequency"})
    write_json(output_dir / "config.json", summary)
    metrics = {}
    for field in fields:
        train_indices = []
        test_indices = []
        train_values = []
        test_values = []
        for index in train:
            value = normalize_value(rows[index]["features"].get(field))
            if value is not None:
                train_indices.append(index)
                train_values.append(value)
        for index in test:
            value = normalize_value(rows[index]["features"].get(field))
            if value is not None:
                test_indices.append(index)
                test_values.append(value)
        if not train_indices or not test_indices:
            metrics[field] = {"status": "skipped", "reason": "No labeled train or test rows",
                              "train_rows": len(train_indices), "test_rows": len(test_indices)}
            continue
        train_labels = sorted(set(train_values))
        labels = sorted(set(train_values) | set(test_values))
        label_to_id = {}
        for index, label in enumerate(labels):
            label_to_id[label] = index
        train_ids = []
        for value in train_values:
            train_ids.append(label_to_id[value])
        train_y = torch.tensor(train_ids, dtype=torch.long)
        classifier, mean, std = fit_probe(
            embeddings[train_indices], train_y, len(labels), epochs, batch_size,
            learning_rate, weight_decay, device, seed,
        )
        predicted_values = []
        with torch.no_grad():
            for start in range(0, len(test_indices), batch_size):
                indices = test_indices[start:start + batch_size]
                batch = ((embeddings[indices] - mean) / std).to(device)
                predicted_ids = classifier(batch).argmax(dim=-1).cpu().tolist()
                for index in predicted_ids:
                    predicted_values.append(labels[index])
        result = classification_metrics(test_values, predicted_values, train_labels)
        result.update({"status": "ok", "train_rows": len(train_indices), "labels": labels,
                       "train_labels": train_labels})
        metrics[field] = result
        directory = output_dir / field.lower().replace(" ", "_")
        directory.mkdir(parents=True, exist_ok=True)
        state = {}
        for key, value in classifier.state_dict().items():
            state[key] = value.detach().cpu()
        torch.save({"field": field, "labels": labels, "mean": mean, "std": std,
                    "state_dict": state}, directory / "probe.pt")
        predictions = []
        for position, index in enumerate(test_indices):
            predictions.append({
                "video_id": rows[index]["video_id"], "label": rows[index]["label"],
                "signer_id": rows[index]["signer_id"], "split": "test", "field": field,
                "gold": test_values[position], "prediction": predicted_values[position],
                "known_gold_label": test_values[position] in train_labels,
                "correct": test_values[position] == predicted_values[position],
            })
        write_json(directory / "metrics.json", result)
        write_jsonl(directory / "predictions.jsonl", predictions)
        print(f"{field}: train={len(train_indices)} test={len(test_indices)} accuracy={result['accuracy']:.4f}")
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, help="Paired .npy/.json embedding cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--field", action="append", help="Repeat to select fields; default is all fields")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_probes(args.embeddings, args.output, args.field, args.epochs, args.batch_size,
                 args.learning_rate, args.weight_decay, args.device, args.seed)


if __name__ == "__main__":
    main()
