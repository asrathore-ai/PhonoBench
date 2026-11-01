"""Check selection, resuming, and scoring with small local fake predictions."""

import argparse
import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import features, run
from src.files import read_json, read_jsonl, write_json, write_jsonl
from src.metrics import join_results, summarize
from src.qwen import run as qwen


def invoke(main, arguments):
    with patch.object(sys, "argv", ["check"] + arguments):
        with contextlib.redirect_stdout(io.StringIO()):
            main()


def must_fail(main, arguments, text):
    try:
        invoke(main, arguments)
    except (ValueError, FileExistsError) as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError("Expected failure: " + text)


def check_runner(directory):
    rows = []
    for index in range(5):
        row = {"instance_id": "i" + str(index), "experiment": "test",
               "task": "atomic_value", "condition": "base", "prompt": "Choose A or B.",
               "metadata": {"queried_field": "Major Location"}, "videos": [],
               "choices": [{"option_id": "A", "semantic": "Head", "text": "Head"},
                           {"option_id": "B", "semantic": "None", "text": "None"}],
               "gold_option": "A"}
        rows.append(row)
    rows[2]["condition"] = "another"
    rows[3]["metadata"]["queried_field"] = "Handshape"
    rows[4]["gold_option"] = "B"
    manifest = directory / "tasks.jsonl"
    output = directory / "predictions.jsonl"
    write_jsonl(manifest, rows)
    arguments = ["--manifest", str(manifest), "--field", "major_location",
                 "--condition", "base", "--limit", "3", "--output", str(output)]
    state = {"errors": {"i1"}, "environment": {"revision": "one"}, "loads": 0, "calls": []}

    def load_model(*args, **kwargs):
        state["loads"] += 1
        return {"environment": state["environment"].copy()}

    def generate(bundle, row, video_root=None):
        state["calls"].append(row["instance_id"])
        if row["instance_id"] in state["errors"]:
            raise RuntimeError("test failure")
        return {"raw_output": '{"answer": "A"}'}

    with patch.object(qwen, "load_model", load_model), patch.object(qwen, "generate", generate):
        invoke(run.main, arguments + ["--dry-run"])
        assert state["loads"] == 0 and not output.exists()
        assert "torch" not in sys.modules and "transformers" not in sys.modules
        invoke(run.main, arguments)
        selected = [rows[0], rows[1], rows[4]]
        assert read_jsonl(output.with_suffix(".inputs.jsonl")) == selected
        joined, audit = join_results(selected, read_jsonl(output))
        assert summarize(joined)["accuracy"] == 1 / 3 and audit["technical_failures"] == 1
        invoke(run.main, arguments + ["--resume"])
        assert state["loads"] == 1 and state["calls"] == ["i0", "i1", "i4"]
        before = output.read_bytes()
        state["environment"] = {"revision": "two"}
        must_fail(run.main, arguments + ["--resume", "--retry-errors"], "environment changed")
        assert output.read_bytes() == before
        state["environment"] = {"revision": "one"}
        state["errors"] = set()
        invoke(run.main, arguments + ["--resume", "--retry-errors"])
        assert state["calls"] == ["i0", "i1", "i4", "i1"]
        joined, audit = join_results(selected, read_jsonl(output))
        assert len(joined) == 3 and summarize(joined)["accuracy"] == 2 / 3
        assert audit["duplicate_result_ids"] == {"i1": 1}
        must_fail(run.main, arguments + ["--resume", "--max-new-tokens", "33"], "settings or inputs changed")
        must_fail(run.main, arguments, "Output already exists")

    duplicate = list(rows)
    duplicate.append(rows[4])
    write_jsonl(manifest, duplicate)
    must_fail(run.main, arguments + ["--dry-run"], "Repeated instance_id")
    evidence = dict(rows[0])
    evidence["experiment"] = "evidence"
    write_jsonl(manifest, [evidence])
    must_fail(run.main, arguments + ["--model", "internvl", "--dry-run"], "fixed 28-frame")

    adapter = directory / "adapter"
    adapter.mkdir()
    write_json(adapter / "feature_config.json", {"checkpoint": "base", "revision": "main",
               "environment": {"resolved_revision": "saved-commit"}})
    write_json(adapter / "adapter_config.json", {"r": 8})
    (adapter / "adapter_model.safetensors").write_bytes(b"test")
    args = argparse.Namespace(adapter=str(adapter), model="qwen", checkpoint=None, revision=None)
    hashes = run.adapter_settings(args)
    assert args.checkpoint == "base" and args.revision == "saved-commit" and len(hashes) == 3


def check_feature_resume(directory):
    metadata = directory / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "signer_id", "split", "label_type", "Major Location"])
        for index in range(2):
            identity = "feature" + str(index)
            writer.writerow([identity, "test" + str(index), "test", "asllex", "Head"])
            (directory / (identity + ".mp4")).touch()
    output = directory / "features"
    arguments = ["--metadata", str(metadata), "--video-root", str(directory),
                 "--field", "major_location", "--output", str(output)]
    state = {"fail": True, "environment": {"revision": "one"}, "calls": 0}

    def load_model(*args, **kwargs):
        return {"environment": state["environment"].copy()}

    def generate(bundle, instance):
        state["calls"] += 1
        identity = instance["videos"][0]["metadata"]["video_id"]
        if state["fail"] and identity == "feature0":
            raise RuntimeError("test failure")
        return {"raw_output": '{"answer": "Head"}'}

    with patch.object(features, "load_model", load_model), patch.object(features, "generate", generate):
        invoke(features.main, arguments)
        assert read_json(output / "metrics.json")["accuracy"] == .5
        before = (output / "predictions.jsonl").read_bytes()
        state["environment"] = {"revision": "two"}
        must_fail(features.main, arguments + ["--resume"], "environment changed")
        assert (output / "predictions.jsonl").read_bytes() == before
        state["environment"] = {"revision": "one"}
        state["fail"] = False
        invoke(features.main, arguments + ["--resume"])
        assert read_json(output / "metrics.json")["accuracy"] == 1
        assert state["calls"] == 3


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        check_runner(Path(directory))
        check_feature_resume(Path(directory))
    print("Runner selection, dry runs, resume checks, retries, and scoring passed.")
