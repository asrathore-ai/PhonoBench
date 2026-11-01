"""Small metadata and scoring checks; no model or video dependencies are needed."""

import csv
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.fields import FEATURE_FIELDS
from src.qwen.extract import load_tokens
from src.qwen.features import feature_metrics, load_feature_data, parse_label


def row(video_id, signer, split, location, handshape="", label=None):
    result = {"video_id": video_id, "signer_id": signer, "split": split,
              "label_type": "asllex", "label": label or video_id}
    for field in FEATURE_FIELDS:
        result[field] = ""
    result["Major Location"] = location
    result["Handshape"] = handshape
    return result


def write_metadata(path, rows):
    columns = ["video_id", "signer_id", "split", "label_type", "label"] + FEATURE_FIELDS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def check_data(directory):
    rows = [row("a", "train1", "train", "Head"),
            row("a", "train1", "train", "Head", "A"),
            row("a", "val1", "val", "Torso", "B", "another_label"),
            row("b", "test1", "test", "None"),
            row("c", "test2", "test", "Arm", "A"),
            row("c", "test2", "test", "Torso", "A"),
            row("d", "train2", "train", "", "B"),
            row("e", "val2", "val", "Neutral", "C"),
            row("missing", "test3", "test", "Arm"),
            row("f", "train3", "train", "Head", label="first"),
            row("f", "train3", "train", "Head", label="second")]
    metadata = directory / "metadata.csv"
    write_metadata(metadata, rows)
    for split, video_id in [("train", "a"), ("test", "b"), ("test", "c")]:
        path = directory / split / (video_id + ".mp4")
        path.parent.mkdir(exist_ok=True)
        path.touch()
    splits, labels, counts = load_feature_data(metadata, directory, "major_location")
    assert labels == ["Arm", "Head", "Neutral", "None", "Torso"]
    assert len(splits["train"]) == 1 and len(splits["test"]) == 1
    assert splits["train"][0]["video_id"] == "a"
    assert splits["test"][0]["value"] == "None"
    assert counts["ambiguous"] == 1 and counts["unlabelled"] == 1
    assert counts["missing_videos"] == 2

    records, rejected = load_tokens(metadata)
    by_id = {}
    for record in records:
        by_id[record["video_id"]] = record
    assert "e" not in by_id and "f" in rejected
    assert by_id["a"]["features"]["Major Location"] == "Head"
    assert by_id["a"]["split"] == "train" and by_id["a"]["signer_id"] == "train1"
    assert by_id["d"]["features"] == {"Handshape": "B"}
    assert by_id["c"]["features"] == {"Handshape": "A"}
    assert "Major Location" in by_id["c"]["ambiguous_fields"]

    overlap = list(rows)
    overlap.append(row("a", "new_signer", "test", "Head"))
    write_metadata(metadata, overlap)
    for loader in [load_tokens, load_feature_data]:
        try:
            if loader is load_tokens:
                loader(metadata)
            else:
                loader(metadata, None, "Major Location")
        except ValueError as error:
            assert "video overlap" in str(error)
        else:
            raise AssertionError("Train/test video overlap was accepted")
    overlap = list(rows)
    overlap.append(row("new_video", "train1", "test", "Head"))
    write_metadata(metadata, overlap)
    try:
        load_feature_data(metadata, None, "Major Location")
    except ValueError as error:
        assert "signer overlap" in str(error)
    else:
        raise AssertionError("Train/test signer overlap was accepted")


def check_predictions():
    labels = ["Head", "None"]
    assert parse_label('{"answer": "head"}', labels) == "Head"
    assert parse_label('{"answer": "None"}', labels) == "None"
    assert parse_label('{"answer": null}', labels) is None
    assert parse_label('{"answer": "Head", "answer": "None"}', labels) is None
    assert parse_label('{"answer": "Head"} {"answer": "Head"}', labels) == "Head"
    assert parse_label('{"answer": 1.0}', ["0", "1"]) == "1"
    assert parse_label('{"answer": false}', ["0", "1"]) is None
    assert parse_label("unknown", labels) is None
    examples = [{"video_id": "1", "value": "Head"}, {"video_id": "2", "value": "Head"},
                {"video_id": "3", "value": "None"}, {"video_id": "4", "value": "None"}]
    results = {"1": {"status": "ok", "prediction": "Head"},
               "2": {"status": "error", "prediction": None},
               "3": {"status": "ok", "prediction": "None"}}
    metrics = feature_metrics(examples, results, labels)
    assert metrics["examples"] == 4 and metrics["accuracy"] == .5
    assert metrics["balanced_accuracy"] == .5 and metrics["valid_rate"] == .5
    assert abs(metrics["macro_f1"] - 2 / 3) < 1e-8


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        check_data(Path(directory))
    check_predictions()
    print("Feature metadata, split separation, parsing, and scoring checks passed.")
