"""Synthetic scoring checks; no videos, model downloads, or GPU are needed."""

import copy
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.files import read_json, write_jsonl
from src.graph.fields import FEATURE_FIELDS
from src.metrics import analyze, compare_results, join_results, semantic_key
from src.parsing import parse_answer


def example(identity, task="relation", condition="original"):
    return {"instance_id": identity, "source_id": identity, "cluster_id": identity,
            "experiment": "synthetic", "task": task, "condition": condition,
            "prompt": "Do these signs differ in exactly one field?", "videos": [],
            "choices": [{"option_id": "A", "semantic": True, "text": "Yes"},
                        {"option_id": "B", "semantic": False, "text": "No"}],
            "gold_option": "A", "metadata": {}}


def answer(identity, option="A"):
    return {"instance_id": identity, "status": "ok", "result": {"raw_output": option}}


def expect_error(function, arguments, message):
    try:
        function(*arguments)
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("Expected rejection: " + message)


def check_parser():
    choices = example("x")["choices"]
    for text in ["A", "(A)", '{"answer": "A"}', 'Answer: A', '{"answer": "A", "answer": "A"}']:
        parsed = parse_answer(text, choices)
        assert parsed["valid"] and parsed["option_id"] == "A", text
    for text in ['{"answer": "A", "answer": "B"}', '{"answer": "A"} Answer: B',
                 'Answer: A or B', '{"answer": "A"} {"answer": "Z"}',
                 "A and B", "A signer makes a movement.", "I cannot choose."]:
        assert not parse_answer(text, choices)["valid"], text
    expanded = [{"option_id": "A", "semantic": "c", "text": "c"},
                {"option_id": "C", "semantic": "f", "text": "f"},
                {"option_id": "AA", "semantic": "other", "text": "other"}]
    assert parse_answer('{"answer": "C"}', expanded)["option_id"] == "C"
    assert parse_answer("AA", expanded)["option_id"] == "AA"
    assert semantic_key(True) != semantic_key(1) != semantic_key("1")


def check_denominators():
    manifest = []
    for identity in ["correct", "invalid", "error", "missing", "retry"]:
        manifest.append(example(identity))
    first = answer("correct")
    first["correct"] = False  # Recompute correctness instead of trusting this flag.
    results = [first, answer("invalid", 'Answer: A or B'),
               {"instance_id": "error", "status": "error"},
               {"instance_id": "retry", "status": "error"}, answer("retry")]
    report = analyze(manifest, results, bootstrap=50, seed=3)
    assert report == analyze(manifest, results, bootstrap=50, seed=3)
    metrics = report["summary"][0]
    assert metrics["n"] == 5 and metrics["correct"] == 2
    assert metrics["valid"] == 2 and metrics["invalid"] == 3
    assert metrics["accuracy"] == .4 and metrics["valid_only_accuracy"] == 1
    assert abs(metrics["chance_normalized_accuracy"] + .2) < 1e-9
    assert report["audit"]["missing_results"] == 1
    assert report["audit"]["technical_failures"] == 1
    assert report["audit"]["duplicate_result_ids"] == {"retry": 1}
    assert report["audit"]["stored_correct_disagreements"] == 1
    expect_error(join_results, [manifest + [manifest[0]], []], "Duplicate manifest ID")
    expect_error(join_results, [manifest, [answer("unknown")]], "absent from manifest")

    saved = answer("correct")
    saved["result"]["parsed"] = {"valid": False, "option_id": None}
    assert analyze([manifest[0]], [saved])["summary"][0]["correct"] == 0
    assert analyze([manifest[0]], [saved], reparse=True)["summary"][0]["correct"] == 1
    failed_retry = [answer("correct"), {"instance_id": "correct", "status": "error"}]
    assert analyze([manifest[0]], failed_retry)["summary"][0]["correct"] == 0
    saved["result"]["parsed"] = {"valid": True, "option_id": "Z"}
    assert analyze([manifest[0]], [saved])["summary"][0]["valid"] == 0

    singleton = example("singleton")
    singleton["choices"] = [{"option_id": "A", "semantic": "None", "text": "None"}]
    metrics = analyze([singleton], [answer("singleton")])["summary"][0]
    assert metrics["accuracy"] == 1 and metrics["uniform_chance"] == 1
    assert metrics["chance_normalized_accuracy"] is None


def check_atomic_metrics():
    manifest = []
    for identity, gold in [("a", "A"), ("b", "A"), ("c", "B"), ("d", "B")]:
        row = example(identity, "atomic_value", "atomic_gold_field")
        row["gold_option"] = gold
        row["metadata"] = {"queried_field": "Handshape", "matched_pair_id": identity, "side": "left"}
        manifest.append(row)
    report = analyze(manifest, [answer("a"), answer("b", "unclear"), answer("c", "B")])
    metrics = report["atomic_fields"][0]
    assert metrics["n"] == 4 and metrics["accuracy"] == .5
    assert metrics["balanced_accuracy"] == .5 and abs(metrics["macro_f1"] - 2 / 3) < 1e-9


def candidate_row(identity, task, swapped):
    condition = "candidate_order_swapped" if swapped else "candidate_order_original"
    row = example(identity, task, condition)
    row["choices"] = [{"option_id": "A", "semantic": "candidate_0", "text": "Candidate 0"},
                      {"option_id": "B", "semantic": "candidate_1", "text": "Candidate 1"}]
    tokens = ["other", "match"] if swapped else ["match", "other"]
    row["videos"] = [{"role": "Anchor", "metadata": {"video_id": "anchor"}}]
    for index, token in enumerate(tokens):
        row["videos"].append({"role": "Candidate " + str(index), "metadata": {"video_id": token}})
    row["metadata"] = {"matched_visual_context_id": "same-context"}
    row["gold_option"] = "B" if swapped else "A"
    return row


def check_candidate_swaps():
    manifest = []
    results = []
    for task in ["follows_video", "follows_position", "missing_swap"]:
        manifest.append(candidate_row(task + "0", task, False))
        manifest.append(candidate_row(task + "1", task, True))
        results.append(answer(task + "0"))
        if task != "missing_swap":
            results.append(answer(task + "1", "B" if task == "follows_video" else "A"))
    by_task = {}
    for metric in analyze(manifest, results)["diagnostics"]["candidate_swaps"]:
        by_task[metric["task"]] = metric
    assert by_task["follows_video"]["same_video_rate"] == 1
    assert by_task["follows_position"]["same_video_rate"] == 0
    assert by_task["missing_swap"]["pairs"] == 1
    assert by_task["missing_swap"]["invalid_or_missing_pairs"] == 1
    assert by_task["missing_swap"]["same_video_rate"] == 0


def check_decomposition():
    pair = {"pair_id": "p", "left_video": "left", "right_video": "right", "gold_field": FEATURE_FIELDS[0]}
    manifest = []
    results = []
    for token in ["left", "right"]:
        for index, field in enumerate(FEATURE_FIELDS):
            identity = token + str(index)
            row = example(identity, "atomic_value", "atomic_full_vector")
            # 'None' is a category, so it must still complete the predicted vector.
            value = "None" if index == 1 else "0"
            row["choices"] = [{"option_id": "A", "semantic": value, "text": value},
                              {"option_id": "B", "semantic": "1", "text": "1"}]
            row["gold_option"] = "B" if token == "right" and index == 0 else "A"
            row["metadata"] = {"video_id": token, "queried_field": field, "vector_pairs": [dict(pair)]}
            manifest.append(row)
            results.append(answer(identity, row["gold_option"]))
    metric = analyze(manifest, results)["diagnostics"]["atomic_decomposition"]
    assert metric["pairs"] == 1 and metric["complete_pair_vectors"] == 1
    assert metric["one_feature_pair_rate"] == 1 and metric["correct_distinguishing_field_rate"] == 1
    metric = analyze(manifest, results[:-1])["diagnostics"]["atomic_decomposition"]
    assert metric["pairs"] == 1 and metric["incomplete_pair_vectors"] == 1
    assert metric["correct_distinguishing_field_rate"] == 0
    wrong = copy.deepcopy(results)
    for result in wrong:
        if result["instance_id"] == "right2":
            result["result"]["raw_output"] = "B"
    metric = analyze(manifest, wrong)["diagnostics"]["atomic_decomposition"]
    assert metric["median_delta_size_complete_pairs"] == 2
    assert metric["one_feature_pair_rate"] == 0 and metric["correct_distinguishing_field_rate"] == 0
    conflicting = copy.deepcopy(manifest)
    conflicting[0]["metadata"]["vector_pairs"][0]["gold_field"] = FEATURE_FIELDS[1]
    expect_error(analyze, [conflicting, results], "Conflicting definitions")


def transfer_examples():
    manifest = []
    for identity, task in [("a", "first"), ("b", "first"), ("c", "first"),
                           ("d", "second"), ("e", "second"), ("f", "second")]:
        manifest.append(example(identity, task))
    current = [answer("a"), answer("b", "B"), answer("c"), answer("d", "B"), answer("e")]
    baseline = [{"instance_id": "a", "status": "error"}, answer("a"), answer("b", "B"),
                answer("c", "B"), answer("d"), {"instance_id": "e", "status": "error"}, answer("extra")]
    baseline_manifest = manifest + [example("extra", "another_task")]
    return manifest, current, baseline, baseline_manifest


def check_transfer():
    manifest, current, baseline, baseline_manifest = transfer_examples()
    report = compare_results(manifest, current, baseline, baseline_manifest)
    metrics = report["overall"]
    assert metrics["n"] == 6 and metrics["both_correct"] == 1 and metrics["both_wrong"] == 2
    assert metrics["improved"] == 2 and metrics["harmed"] == 1
    assert metrics["baseline_accuracy"] == 1 / 3 and metrics["current_accuracy"] == .5
    assert metrics["accuracy_difference"] == 1 / 6
    by_task = {}
    for task in report["by_task"]:
        by_task[task["task"]] = task
    assert by_task["first"]["n"] == 3 and by_task["first"]["accuracy_difference"] == 1 / 3
    assert by_task["second"]["n"] == 3 and by_task["second"]["accuracy_difference"] == 0
    audit = report["audit"]
    assert audit["input_verification"] == "verified" and audit["baseline_extra_result_rows_ignored"] == 1
    assert audit["baseline"]["missing_results"] == 1 and audit["baseline"]["technical_failures"] == 1
    assert audit["current"]["missing_results"] == 1
    assert audit["baseline"]["duplicate_result_ids"] == {"a": 1}
    assert compare_results(manifest, current, baseline)["audit"]["input_verification"] == "unavailable"
    changed = copy.deepcopy(baseline_manifest)
    changed[0]["prompt"] = "A different question"
    expect_error(compare_results, [manifest, current, baseline, changed], "Changed prompt")
    expect_error(compare_results, [manifest, current, baseline, baseline_manifest[1:]], "absent from baseline inputs")
    assert compare_results([], [], [answer("extra")], [example("extra")])["overall"]["accuracy_difference"] is None


def check_cli(directory):
    manifest, current, baseline, baseline_manifest = transfer_examples()
    current_path = directory / "adapter.jsonl"
    baseline_path = directory / "base.jsonl"
    write_jsonl(current_path, current)
    write_jsonl(current_path.with_suffix(".inputs.jsonl"), manifest)
    write_jsonl(baseline_path, baseline)
    write_jsonl(baseline_path.with_suffix(".inputs.jsonl"), baseline_manifest)
    command = [sys.executable, "-m", "experiments.analyze", "--results", str(current_path),
               "--compare", str(baseline_path), "--reparse", "--bootstrap", "20", "--seed", "5"]
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = read_json(current_path.with_suffix(".metrics.json"))
    assert report["audit"]["manifest_rows"] == 6 and report["audit"]["missing_results"] == 1
    assert report["comparison"]["overall"]["accuracy_difference"] == 1 / 6
    assert report["summary"][0]["accuracy_cluster_bootstrap_95"]["repetitions"] == 20
    rejected = subprocess.run(command + ["--output", str(current_path)], cwd=root, capture_output=True, text=True)
    assert rejected.returncode != 0 and "must not overwrite" in rejected.stderr


if __name__ == "__main__":
    check_parser()
    check_denominators()
    check_atomic_metrics()
    check_candidate_swaps()
    check_decomposition()
    check_transfer()
    with tempfile.TemporaryDirectory() as directory:
        check_cli(Path(directory))
    print("Answer parsing, full denominators, paired controls, decomposition, and comparison checks passed.")
