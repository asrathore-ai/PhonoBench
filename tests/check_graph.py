"""Run CPU checks for metadata filtering, formal programs, and input generation."""

import csv
import random
import tempfile
from pathlib import Path

from experiments.build import build_tasks
from experiments.diagnostics import NAMES
from src.files import file_hash, read_json, read_jsonl
from src.graph.data import load_metadata, normalize_value
from src.graph.fields import FEATURE_FIELDS, TASKS
from src.graph.graph import build_graph, graph_audit
from src.graph.programs import generate_programs, set_record, validate_record
from src.graph.symbolic import execute_program
from src.prompts.render import program_instance


def vector(contact="0", repeated="0", wrist="0"):
    result = {}
    for field in FEATURE_FIELDS:
        result[field] = "shared"
    result.update({"Contact": contact, "Repeated Movement": repeated,
                   "Wrist Twist": wrist, "Nondominant Handshape": "None"})
    return result


def metadata_row(label, token, values, split="test"):
    row = dict(values)
    row.update({"label": label, "video_id": token, "signer_id": "signer-" + token,
                "label_type": "asllex", "split": split})
    return row


def write_metadata(path, rows):
    columns = ["label", "video_id", "signer_id", "label_type", "split"] + FEATURE_FIELDS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def video_file(root, token):
    path = root / "test" / token / "videos/rgb_clean_480p.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"File-presence fixture; no video decoding is used.")
    return path


def fixture(directory):
    rows = []
    for contact in ["0", "1"]:
        for repeated in ["0", "1"]:
            for wrist in ["0", "1"]:
                label = "sign-" + contact + repeated + wrist
                for suffix in ["a", "b"]:
                    token = label + "-" + suffix
                    rows.append(metadata_row(label, token, vector(contact, repeated, wrist)))
                    video_file(directory, token)
    for suffix in ["a", "b"]:
        token = "isolated-" + suffix
        rows.append(metadata_row("isolated", token, vector("2", "2", "2")))
        video_file(directory, token)
    metadata = directory / "metadata.csv"
    write_metadata(metadata, rows)
    return metadata


def check_metadata(directory):
    directory.mkdir()
    complete = vector()
    incomplete = dict(complete)
    incomplete["Contact"] = ""
    conflict = dict(complete)
    conflict["Contact"] = "9"
    missing_shape = dict(complete)
    missing_shape["Handshape"] = ""
    rows = [
        metadata_row("alpha", "a1", incomplete),
        metadata_row("alpha", "a2", complete),
        metadata_row("alpha", "train-a", complete, "train"),
        metadata_row("beta", "b1", complete),
        metadata_row("beta", "b2", conflict),
        metadata_row("gamma", "ambiguous", complete),
        metadata_row("delta", "ambiguous", complete),
        metadata_row("missing", "m1", missing_shape),
        metadata_row("missing", "train-m", complete, "train"),
        metadata_row("conflict", "k1", complete),
        metadata_row("conflict", "k2", conflict),
        metadata_row("validation", "v1", complete, "val"),
    ]
    metadata = directory / "metadata.csv"
    write_metadata(metadata, rows)
    for token in ["a1", "a2", "b1", "ambiguous", "m1", "k1", "k2"]:
        video_file(directory, token)
    nodes, summary = load_metadata(metadata, directory)
    assert sorted(nodes) == ["alpha", "beta"]
    assert nodes["beta"]["vector"]["Nondominant Handshape"] == "None"
    assert summary["missing_video_ids"] == ["b2"]
    assert summary["ambiguous_video_identities"][0]["video_id"] == "ambiguous"
    assert summary["conflicting_labels"][0]["label"] == "conflict"
    assert summary["incomplete_labels"][0]["label"] == "missing"
    (directory / "test/a2/videos/rgb_clean_480p.mp4").unlink()
    nodes, summary = load_metadata(metadata, directory)
    assert sorted(nodes) == ["beta"]
    assert summary["incomplete_labels"][0]["label"] == "alpha"
    nodes, summary = load_metadata(metadata, metadata_only=True)
    assert "alpha" in nodes and "beta" not in nodes
    assert summary["metadata_only"] is True
    assert normalize_value("None") == "None"
    assert normalize_value("1.0") == "1"
    assert normalize_value("NaN") is None


def check_symbolic():
    vectors = {
        "a": vector(), "a_copy": vector(), "b": vector("1"),
        "c": vector("0", "1"), "d": vector("1", "1"),
        "e": vector("2", "2", "2"), "f": vector("1", "1", "1"),
        "g": vector("0", "0", "1"),
    }
    labels = {}
    for key in vectors:
        labels[key] = key
    labels["a_copy"] = "a"
    cases = [
        ({"op": "same_sign", "items": ["a", "a_copy"]}, True),
        ({"op": "same_sign", "items": ["a", "b"]}, False),
        ({"op": "one_feature_pair", "items": ["a", "b"]}, True),
        ({"op": "one_feature_pair", "items": ["a", "a_copy"]}, False),
        ({"op": "distinguishing_feature", "items": ["a", "b"]}, "Contact"),
        ({"op": "exists_one_feature_pair", "items": ["a", "b", "c", "d", "e"]}, True),
        ({"op": "universal_shared_feature", "items": ["a", "c", "g"], "field": "Contact"}, True),
        ({"op": "universal_shared_feature", "items": ["a", "b"], "field": "Contact"}, False),
        ({"op": "feature_value_counting", "items": ["a", "b", "c", "d", "e"],
          "field": "Contact", "value": "1"}, 2),
        ({"op": "constraint_satisfaction", "anchor_a": "b", "anchor_b": "c",
          "field_a": "Contact", "field_b": "Repeated Movement", "candidates": ["a", "d", "e", "f"]}, "a"),
        ({"op": "feature_transformation", "base": "c", "field": "Contact",
          "source_value": "0", "target_value": "1", "candidates": ["d", "a", "e", "f"]}, "d"),
        ({"op": "common_neighbor", "anchor_left": "b", "anchor_right": "c",
          "candidates": ["a", "e", "f", "g"]}, "a"),
        ({"op": "isolate_detection", "items": ["a", "b", "c", "d", "e"]}, "e"),
    ]
    covered = set()
    for program, expected in cases:
        answer = execute_program(program, vectors, labels)
        assert type(answer) is type(expected) and answer == expected
        covered.add(program["op"])
    assert covered == set(TASKS)
    try:
        execute_program({"op": "common_neighbor", "anchor_left": "b", "anchor_right": "c",
                         "candidates": ["a", "d", "e", "f"]}, vectors, labels)
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous candidate selections must be rejected")


def check_programs(metadata, directory):
    nodes, summary = load_metadata(metadata, directory)
    graph = build_graph(nodes)
    assert set(graph_audit(graph)["per_field"]) == set(FEATURE_FIELDS)
    labels = sorted(nodes)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            left, right = labels[i], labels[j]
            changed = []
            for field in FEATURE_FIELDS:
                if nodes[left]["vector"][field] != nodes[right]["vector"][field]:
                    changed.append(field)
            assert (right in graph["neighbors"][left]) == (len(changed) == 1)
            if len(changed) == 1:
                assert graph["neighbors"][left][right] == changed[0]
    count = 12
    records = generate_programs(graph, count=count, seed=17)
    assert records == generate_programs(graph, count=count, seed=17)
    counts = {}
    identifiers = set()
    for record in records:
        option = validate_record(record, nodes)
        instance = program_instance(record)
        assert instance["gold_option"] == option
        assert len(instance["videos"]) == len(record["items"])
        identifiers.add(record["program_id"])
        counts[record["task"]] = counts.get(record["task"], 0) + 1
    assert len(identifiers) == len(records)
    assert set(counts) == set(TASKS)
    for actual in counts.values():
        assert 0 < actual <= count
    assert generate_programs(build_graph({}), count=2) == []
    try:
        generate_programs(graph, count=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero count must be rejected")
    pair_nodes = {}
    for label in labels[:2]:
        pair_nodes[label] = dict(nodes[label])
        pair_nodes[label]["video_ids"] = nodes[label]["video_ids"][:1]
    pair_graph = build_graph(pair_nodes)
    identifiers = set()
    orders = set()
    for seed in range(10):
        record = set_record(pair_graph, {"op": "same_sign"}, labels[:2], random.Random(seed))
        identifiers.add(record["program_id"])
        order = []
        for choice in record["choices"]:
            order.append(choice["semantic"])
        orders.add(tuple(order))
    assert len(identifiers) == 1 and len(orders) > 1


def check_build(metadata, directory):
    output = directory / "generated"
    output.mkdir()
    (output / ".gitkeep").touch()
    summary = build_tasks(metadata, directory, experiment="all", count=6, output=output)
    assert summary["metadata_sha256"] == file_hash(metadata)
    assert set(summary["experiments"]) == set(["benchmark"] + NAMES)
    programs = read_jsonl(output / "programs.jsonl")
    graph = read_json(output / "graph.json")
    for record in programs:
        validate_record(record, graph["nodes"])
    for name, details in summary["experiments"].items():
        rows = read_jsonl(output / (name + ".jsonl"))
        assert details["instances"] == len(rows)
        for row in rows:
            matches = 0
            for choice in row["choices"]:
                matches += choice["option_id"] == row["gold_option"]
            assert matches == 1
    notes = output / "notes.txt"
    notes.write_text("Keep user files.", encoding="utf-8")
    try:
        build_tasks(metadata, directory, count=2, output=output)
    except ValueError:
        pass
    else:
        raise AssertionError("Existing outputs require explicit overwrite")
    build_tasks(metadata, directory, count=2, output=output, overwrite=True)
    assert notes.read_text(encoding="utf-8") == "Keep user files."
    assert (output / "benchmark.jsonl").is_file()
    for name in NAMES:
        assert not (output / (name + ".jsonl")).exists()
    try:
        build_tasks(metadata, directory, count=0, output=directory / "zero")
    except ValueError:
        pass
    else:
        raise AssertionError("Zero count must be rejected consistently")
    for suffix in ["a", "b"]:
        (directory / ("test/isolated-" + suffix + "/videos/rgb_clean_480p.mp4")).unlink()
    reduced = build_tasks(metadata, directory, count=2, output=directory / "reduced")
    assert reduced["graph"]["nodes"] == summary["graph"]["nodes"] - 1
    assert len(reduced["data"]["missing_video_ids"]) == 2


def main():
    check_symbolic()
    with tempfile.TemporaryDirectory(prefix="phonobench-graph-") as temporary:
        directory = Path(temporary)
        check_metadata(directory / "filtering")
        metadata = fixture(directory)
        check_programs(metadata, directory)
        check_build(metadata, directory)
    print("Metadata, graph, all ten formal tasks, and experiment generation passed.")


if __name__ == "__main__":
    main()
