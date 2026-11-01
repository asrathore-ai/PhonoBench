"""Plain-loop metrics. Every manifest row remains in the denominator."""

import json
import random
import statistics

from .graph.data import normalize_value
from .graph.fields import FEATURE_FIELDS
from .parsing import parse_answer, semantic_for_option


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def semantic_key(value):
    """Keep boolean True, integer 1, and string '1' distinct in count tables."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def join_results(manifest, results, reparse=False):
    expected = {}
    for row in manifest:
        identity = row["instance_id"]
        if identity in expected:
            raise ValueError(f"Duplicate manifest ID: {identity}")
        expected[identity] = row
    by_id = {}
    duplicates = {}
    for result in results:
        identity = result["instance_id"]
        if identity not in expected:
            raise ValueError(f"Result ID absent from manifest: {identity}")
        if identity in by_id:
            duplicates[identity] = duplicates.get(identity, 0) + 1
        by_id[identity] = result  # An appended retry supersedes an earlier record.
    joined = []
    audit = {"manifest_rows": len(manifest), "result_rows": len(results),
             "duplicate_result_ids": duplicates, "missing_results": 0,
             "technical_failures": 0, "stored_correct_disagreements": 0,
             "parsing": "reparsed" if reparse else "saved_parser_when_available"}
    for identity, source in expected.items():
        result = by_id.get(identity)
        missing = result is None
        failed = result is not None and result.get("status", "ok") != "ok"
        audit["missing_results"] += missing
        audit["technical_failures"] += failed
        parsed = {"valid": False, "option_id": None, "reason": "missing_result" if missing else "technical_failure"}
        if not missing and not failed:
            payload = result.get("result") or result
            parsed = payload.get("parsed")
            if reparse or parsed is None:
                parsed = parse_answer(payload.get("raw_output", ""), source["choices"])
        option = parsed.get("option_id")
        allowed = []
        for choice in source["choices"]:
            allowed.append(choice["option_id"])
        if not allowed or source["gold_option"] not in allowed:
            raise ValueError(f"Invalid answer choices/gold in manifest: {identity}")
        valid = bool(parsed.get("valid") and option in allowed)
        prediction = semantic_for_option(source["choices"], option) if valid else None
        row = dict(source)
        row.update({"valid": valid, "predicted_option": option if valid else None,
                    "prediction": prediction, "correct": valid and option == source["gold_option"],
                    "gold_semantic": semantic_for_option(source["choices"], source["gold_option"]),
                    "missing_result": missing, "technical_failure": failed,
                    "parse_reason": parsed.get("reason", "unspecified")})
        if result is not None and "correct" in result and bool(result["correct"]) != row["correct"]:
            audit["stored_correct_disagreements"] += 1
        joined.append(row)
    return joined, audit


def summarize(rows):
    valid = correct = 0
    chance = 0.0
    gold_counts = {}
    option_counts = {}
    prediction_counts = {}
    for row in rows:
        correct += row["correct"]
        valid += row["valid"]
        chance += 1 / len(row["choices"])
        gold = semantic_key(row["gold_semantic"])
        gold_counts[gold] = gold_counts.get(gold, 0) + 1
        if row["valid"]:
            option = row["predicted_option"]
            option_counts[option] = option_counts.get(option, 0) + 1
            semantic = semantic_key(row["prediction"])
            prediction_counts[semantic] = prediction_counts.get(semantic, 0) + 1
    n = len(rows)
    accuracy = correct / n
    chance /= n
    return {"n": n, "correct": correct, "valid": valid, "invalid": n - valid,
            "accuracy": accuracy, "valid_rate": valid / n,
            "valid_only_accuracy": fraction(correct, valid), "uniform_chance": chance,
            "majority_baseline": max(gold_counts.values()) / n,
            "chance_normalized_accuracy": fraction(accuracy - chance, 1 - chance),
            "dominant_option_share": fraction(max(option_counts.values(), default=0), valid),
            "dominant_semantic_share": fraction(max(prediction_counts.values(), default=0), valid),
            "dominant_share_denominator": "valid responses",
            "gold_semantic_counts": gold_counts, "predicted_option_counts": option_counts,
            "predicted_semantic_counts": prediction_counts}


def clustered_interval(rows, repetitions, seed=41):
    clusters = {}
    for row in rows:
        key = row.get("cluster_id") or row["instance_id"]
        clusters.setdefault(key, [0, 0])
        clusters[key][0] += row["correct"]
        clusters[key][1] += 1
    values = list(clusters.values())
    randomizer = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        correct = n = 0
        for _ in values:
            selected = randomizer.choice(values)
            correct += selected[0]
            n += selected[1]
        estimates.append(correct / n)
    estimates.sort()
    return {"low": estimates[int(0.025 * (repetitions - 1))],
            "high": estimates[int(0.975 * (repetitions - 1))],
            "clusters": len(values), "repetitions": repetitions, "seed": seed}


def grouped_metrics(rows, bootstrap=0, seed=41):
    groups = {}
    for row in rows:
        key = (row["experiment"], row["task"], row["condition"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        summary = summarize(group)
        summary.update({"experiment": key[0], "task": key[1], "condition": key[2]})
        if bootstrap:
            summary["accuracy_cluster_bootstrap_95"] = clustered_interval(group, bootstrap, seed)
        output.append(summary)
    return output


def atomic_metrics(rows):
    """Macro-F1 uses all displayed labels; balanced accuracy uses supported gold labels.

    Invalid predictions contribute false negatives. No test gold label is dropped.
    """
    groups = {}
    for row in rows:
        if row["task"] not in ["atomic_value", "target_feature_under_distractors"]:
            continue
        meta = row.get("metadata", {})
        field = meta.get("queried_field", meta.get("field"))
        groups.setdefault((row["experiment"], row["condition"], field), []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        labels = {}
        for row in group:
            for choice in row["choices"]:
                labels[semantic_key(choice["semantic"])] = choice["semantic"]
        counts = {}
        for label in labels:
            counts[label] = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "support": 0}
        for row in group:
            actual = semantic_key(row["gold_semantic"])
            predicted = semantic_key(row["prediction"]) if row["valid"] else None
            counts[actual]["support"] += 1
            if predicted == actual:
                counts[actual]["true_positive"] += 1
            else:
                counts[actual]["false_negative"] += 1
                if predicted is not None:
                    counts[predicted]["false_positive"] += 1
        recall_sum = f1_sum = 0.0
        supported = 0
        per_label = []
        for label, count in counts.items():
            tp, fp, fn = count["true_positive"], count["false_positive"], count["false_negative"]
            if count["support"]:
                recall_sum += tp / count["support"]
                supported += 1
            f1_sum += fraction(2 * tp, 2 * tp + fp + fn) or 0.0
            per_label.append({"label": labels[label], **count})
        result = summarize(group)
        result.update({"experiment": key[0], "condition": key[1], "field": key[2],
                       "macro_f1": f1_sum / len(labels), "balanced_accuracy": recall_sum / supported,
                       "inventory_labels": len(labels), "supported_gold_labels": supported,
                       "per_label": per_label})
        output.append(result)
    return output


def condition_pairs(rows, metadata_key, conditions):
    groups = {}
    for row in rows:
        if row["condition"] not in conditions or metadata_key not in row.get("metadata", {}):
            continue
        key = (row["experiment"], row["task"], row["metadata"][metadata_key])
        pair = groups.setdefault(key, {})
        if row["condition"] in pair:
            raise ValueError(f"Repeated condition within matched group: {key}")
        pair[row["condition"]] = row
    return groups


def selected_video(row):
    if row is None or not row["valid"]:
        return None
    semantic = str(row["prediction"])
    if not semantic.startswith("candidate_"):
        return None
    role = "Candidate " + semantic.removeprefix("candidate_")
    for video in row["videos"]:
        if video.get("role") == role:
            return video.get("metadata", {}).get("video_id") or video.get("path")
    return None


def candidate_swaps(rows):
    groups = condition_pairs(rows, "matched_visual_context_id", ["candidate_order_original", "candidate_order_swapped"])
    totals = {}
    for (experiment, task, _), pair in groups.items():
        total = totals.setdefault((experiment, task), {"experiment": experiment, "task": task,
                                  "pairs": 0, "both_valid": 0, "same_video": 0,
                                  "original_correct": 0, "swapped_correct": 0})
        left = pair.get("candidate_order_original")
        right = pair.get("candidate_order_swapped")
        a, b = selected_video(left), selected_video(right)
        valid = a is not None and b is not None
        total["pairs"] += 1
        total["both_valid"] += valid
        total["same_video"] += valid and a == b
        total["original_correct"] += bool(left and left["correct"])
        total["swapped_correct"] += bool(right and right["correct"])
    for total in totals.values():
        total["invalid_or_missing_pairs"] = total["pairs"] - total["both_valid"]
        total["same_video_rate"] = total["same_video"] / total["pairs"]
        total["same_video_rate_when_both_valid"] = fraction(total["same_video"], total["both_valid"])
        total["original_accuracy"] = total["original_correct"] / total["pairs"]
        total["swapped_accuracy"] = total["swapped_correct"] / total["pairs"]
        total["pooled_accuracy"] = (total["original_correct"] + total["swapped_correct"]) / (2 * total["pairs"])
    return list(totals.values())


def counterfactual_twins(rows):
    groups = condition_pairs(rows, "twin_id", ["counterfactual_twin_positive", "counterfactual_twin_negative"])
    valid = both = changed = yes_both = 0
    for pair in groups.values():
        a = pair.get("counterfactual_twin_positive")
        b = pair.get("counterfactual_twin_negative")
        both_valid = bool(a and b and a["valid"] and b["valid"])
        valid += both_valid
        if both_valid:
            both += a["correct"] and b["correct"]
            changed += semantic_key(a["prediction"]) != semantic_key(b["prediction"])
            yes_both += a["prediction"] is True and b["prediction"] is True
    return {"pairs": len(groups), "both_valid": valid, "invalid_or_missing_pairs": len(groups) - valid,
            "both_correct_rate": fraction(both, len(groups)),
            "prediction_change_rate": fraction(changed, len(groups)),
            "yes_on_both_rate": fraction(yes_both, len(groups))}


def conditional_relations(rows):
    atomic = {}
    for row in rows:
        if row["condition"] == "atomic_gold_field":
            meta = row["metadata"]
            atomic.setdefault((row["experiment"], meta["matched_pair_id"]), {})[meta["side"]] = row["correct"]
    selected = set()
    for identity, sides in atomic.items():
        if sides.get("left") is True and sides.get("right") is True:
            selected.add(identity)
    groups = {}
    conditions = ["named_equality_gold", "sole_difference_gold", "two_way_hard_distractor", "seventeen_way"]
    for row in rows:
        pair_id = (row["experiment"], row.get("metadata", {}).get("matched_pair_id"))
        if row["condition"] in conditions and pair_id in selected:
            groups.setdefault((row["experiment"], row["task"], row["condition"]), []).append(row)
    output = []
    for (experiment, task, condition), group in sorted(groups.items()):
        correct = 0
        for row in group:
            correct += row["correct"]
        output.append({"experiment": experiment, "task": task, "condition": condition, "n": len(group),
                       "accuracy_given_both_atomic_correct": correct / len(group)})
    return {"atomic_pairs": len(atomic), "both_atomic_correct_pairs": len(selected),
            "both_atomic_correct_fraction": fraction(len(selected), len(atomic)), "relations": output}


def decomposition(rows, pairs):
    vectors = {}
    for row in rows:
        if row["condition"] == "atomic_full_vector" and row["valid"]:
            meta = row["metadata"]
            value = normalize_value(row["prediction"])
            if value is not None:
                vectors.setdefault(meta["video_id"], {})[meta["queried_field"]] = value
    complete = one_feature = correct = 0
    sizes = []
    distribution = {}
    for pair in pairs:
        left = vectors.get(pair["left_video"], {})
        right = vectors.get(pair["right_video"], {})
        changed = []
        available = True
        for field in FEATURE_FIELDS:
            if field not in left or field not in right:
                available = False
            elif left[field] != right[field]:
                changed.append(field)
        if available:
            complete += 1
            one_feature += len(changed) == 1
            correct += changed == [pair["gold_field"]]
            sizes.append(len(changed))
            distribution[str(len(changed))] = distribution.get(str(len(changed)), 0) + 1
    return {"pairs": len(pairs), "complete_pair_vectors": complete,
            "incomplete_pair_vectors": len(pairs) - complete, "predicted_one_feature_pairs": one_feature,
            "one_feature_pair_rate": fraction(one_feature, len(pairs)),
            "correct_distinguishing_field_rate": fraction(correct, len(pairs)),
            "delta_size_distribution_complete_pairs": distribution,
            "median_delta_size_complete_pairs": statistics.median(sizes) if sizes else None}


def option_rotations(rows):
    groups = {}
    expected = {}
    for row in rows:
        if row["condition"].startswith("option_rotation_"):
            meta = row["metadata"]
            group = meta.get("circular_group_id", meta.get("base_instance_id"))
            task_key = (row["experiment"], row["task"])
            groups.setdefault((row["experiment"], row["task"], group), []).append(row)
            expected.setdefault(task_key, set()).add(row["condition"])
    totals = {}
    for (experiment, task, _), group in groups.items():
        task_key = (experiment, task)
        total = totals.setdefault(task_key, {"experiment": experiment, "task": task,
                                            "groups": 0, "strict_consistent": 0, "all_correct": 0})
        answers = set()
        all_valid = len(group) == len(expected[task_key])
        all_correct = all_valid
        for row in group:
            all_valid = all_valid and row["valid"]
            all_correct = all_correct and row["correct"]
            if row["valid"]:
                answers.add(semantic_key(row["prediction"]))
        total["groups"] += 1
        total["strict_consistent"] += all_valid and len(answers) == 1
        total["all_correct"] += all_correct
    for total in totals.values():
        total["strict_consistency_rate"] = total["strict_consistent"] / total["groups"]
        total["all_correct_rate"] = total["all_correct"] / total["groups"]
    return list(totals.values())


def vector_pairs(manifest):
    """Recover each pair once from the atomic questions that describe it."""
    pairs = {}
    for row in manifest:
        if row["condition"] != "atomic_full_vector":
            continue
        for pair in row.get("metadata", {}).get("vector_pairs", []):
            identity = pair["pair_id"]
            for key in ["left_video", "right_video", "gold_field"]:
                if not pair.get(key):
                    raise ValueError(f"Missing {key} in vector pair: {identity}")
            if pair["gold_field"] not in FEATURE_FIELDS:
                raise ValueError("Unknown distinguishing field in vector pair: " + identity)
            if identity in pairs and pairs[identity] != pair:
                raise ValueError("Conflicting definitions of vector pair: " + identity)
            pairs[identity] = pair
    return list(pairs.values())


def paired_summary(rows):
    """Count improvements and harms on the same requested examples."""
    both_correct = both_wrong = improved = harmed = 0
    for baseline, current in rows:
        if baseline["correct"] and current["correct"]:
            both_correct += 1
        elif not baseline["correct"] and not current["correct"]:
            both_wrong += 1
        elif current["correct"]:
            improved += 1
        else:
            harmed += 1
    n = len(rows)
    return {"n": n, "both_correct": both_correct, "both_wrong": both_wrong,
            "improved": improved, "harmed": harmed,
            "baseline_accuracy": fraction(both_correct + harmed, n),
            "current_accuracy": fraction(both_correct + improved, n),
            "accuracy_difference": fraction(improved - harmed, n)}


def compare_results(manifest, results, baseline_results, baseline_manifest=None, reparse=False):
    """Compare runs on all current inputs, including failed or missing outputs.

    The baseline can include additional tasks. When its inputs are available,
    every current ID must be declared there with the same model inputs and gold.
    """
    current, current_audit = join_results(manifest, results, reparse)
    selected_ids = set()
    for row in manifest:
        selected_ids.add(row["instance_id"])
    checked_fields = ["experiment", "task", "condition", "prompt", "videos", "choices", "gold_option"]
    if baseline_manifest is not None:
        baseline_inputs = {}
        for row in baseline_manifest:
            identity = row["instance_id"]
            if identity in baseline_inputs:
                raise ValueError("Duplicate baseline input ID: " + identity)
            baseline_inputs[identity] = row
        for row in manifest:
            identity = row["instance_id"]
            if identity not in baseline_inputs:
                raise ValueError("Selected ID absent from baseline inputs: " + identity)
            for key in checked_fields:
                if semantic_key(row.get(key)) != semantic_key(baseline_inputs[identity].get(key)):
                    raise ValueError(f"Changed {key} between runs for instance: {identity}")
    selected_baseline = []
    ignored = 0
    for result in baseline_results:
        if result["instance_id"] in selected_ids:
            selected_baseline.append(result)
        else:
            ignored += 1
    baseline, baseline_audit = join_results(manifest, selected_baseline, reparse)
    pairs = []
    groups = {}
    for before, after in zip(baseline, current):
        pair = (before, after)
        pairs.append(pair)
        groups.setdefault(after["task"], []).append(pair)
    by_task = []
    for task, group in sorted(groups.items()):
        summary = paired_summary(group)
        summary["task"] = task
        by_task.append(summary)
    return {"overall": paired_summary(pairs), "by_task": by_task,
            "audit": {"baseline": baseline_audit, "current": current_audit,
                      "baseline_extra_result_rows_ignored": ignored,
                      "input_verification": "verified" if baseline_manifest is not None else "unavailable",
                      "verified_fields": checked_fields if baseline_manifest is not None else []}}


def analyze(manifest, results, reparse=False, bootstrap=0, seed=41, pairs=None):
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    rows, audit = join_results(manifest, results, reparse)
    diagnostics = {"candidate_swaps": candidate_swaps(rows), "counterfactual_twins": counterfactual_twins(rows),
                   "conditional_relations": conditional_relations(rows), "option_rotations": option_rotations(rows)}
    if pairs is None:
        pairs = vector_pairs(manifest)
    if pairs:
        diagnostics["atomic_decomposition"] = decomposition(rows, pairs)
    return {"audit": audit, "summary": grouped_metrics(rows, bootstrap, seed),
            "atomic_fields": atomic_metrics(rows), "diagnostics": diagnostics}
