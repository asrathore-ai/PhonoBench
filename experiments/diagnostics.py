"""Matched relation tests and symbolic controls built from available sign videos."""

import copy
import random

from src.graph.fields import FEATURE_FIELDS
from src.graph.programs import validate_record
from src.prompts.render import program_instance
from .common import (choices, edge_id, identity, instance, inventory, pair_metadata,
                     question, selected_edges, choose_pair, value_text, video, videos_for_labels)


NAMES = ["matched_pairs", "grounding", "counterfactuals", "visual_controls", "evidence",
         "binding", "token_robustness", "interface", "gold_features", "atomic_vectors"]


def relation_instances(graph, edge, left, right, rng, atomic=True):
    """Use the same pair for atomic recovery and six relational questions."""
    metadata = pair_metadata(edge, left, right)
    field = edge["field"]
    distractor = metadata["hard_distractor_field"]
    source = metadata["matched_pair_id"]
    videos = [video(graph, edge["left"], left, "Sign 1"), video(graph, edge["right"], right, "Sign 2")]
    rows = []
    if atomic:
        values = inventory(graph, field)
        if len(values) >= 2:
            display = []
            for value in values:
                display.append(value_text(field, value))
            for side, label, token in [("left", edge["left"], left), ("right", edge["right"], right)]:
                options = choices(values, rng, display)
                target = [video(graph, label, token, "Target sign")]
                prompt = question("atomic_value", {"field": field, "index": 1}, options, target, [field])
                meta = dict(metadata)
                meta.update({"side": side, "video_id": token, "gold_value": graph["nodes"][label]["vector"][field]})
                rows.append(instance("grounding", "atomic_gold_field", "atomic_value", prompt,
                                     target, options, meta["gold_value"], source + ":atomic:" + side,
                                     edge_id(edge), meta))

    for named, role in [(field, "gold"), (distractor, "distractor")]:
        meta = dict(metadata)
        meta.update({"queried_field": named, "queried_field_role": role})
        for task, condition, gold in [
            ("named_field_equality", "named_equality_" + role, role == "distractor"),
            ("sole_difference_verification", "sole_difference_" + role, role == "gold"),
        ]:
            options = choices([True, False], rng, ["Yes", "No"])
            prompt = question(task, {"field": named}, options, videos, [named])
            rows.append(instance("grounding", condition, task, prompt, videos, options, gold,
                                 source + ":" + condition, edge_id(edge), meta))

    for task, condition, values, parameters in [
        ("two_way_field_attribution", "two_way_hard_distractor", [field, distractor],
         {"left_field": field, "right_field": distractor}),
        ("seventeen_way_field_attribution", "seventeen_way", FEATURE_FIELDS, {}),
    ]:
        options = choices(values, rng, values)
        prompt = question(task, parameters, options, videos, values)
        rows.append(instance("grounding", condition, task, prompt, videos, options, field,
                             source + ":" + condition, edge_id(edge), metadata))
    return rows


def build_grounding(graph, count, seed):
    rng = random.Random(seed)
    rows = []
    for edge in selected_edges(graph, count, rng):
        left, right = choose_pair(graph, edge, rng)
        rows.extend(relation_instances(graph, edge, left, right, rng))
    return rows


def relation_base(graph, count, seed):
    """One representative question per relation on each sampled lexical edge."""
    rows = []
    for row in build_grounding(graph, count, seed):
        if row["condition"] in ["named_equality_gold", "sole_difference_gold", "two_way_hard_distractor", "seventeen_way"]:
            rows.append(row)
    return rows


def build_matched_pairs(graph, count, seed):
    rng = random.Random(seed)
    rows = []
    completed = 0
    for edge in selected_edges(graph, len(graph["edges"]), rng):
        selected = None
        orientations = [(edge["left"], edge["right"]), (edge["right"], edge["left"])]
        rng.shuffle(orientations)
        for anchor_label, neighbor_label in orientations:
            tokens = list(graph["nodes"][anchor_label]["video_ids"])
            rng.shuffle(tokens)
            if len(tokens) < 2:
                continue
            for anchor in tokens:
                other_tokens = []
                cross_signer = []
                signers = graph["nodes"][anchor_label]["signers"]
                for token in tokens:
                    if token != anchor:
                        other_tokens.append(token)
                        if signers.get(anchor) and signers.get(token) and signers[anchor] != signers[token]:
                            cross_signer.append(token)
                same = rng.choice(cross_signer or other_tokens)
                neighbors = list(graph["nodes"][neighbor_label]["video_ids"])
                rng.shuffle(neighbors)
                for neighbor in neighbors:
                    if neighbor not in [anchor, same]:
                        selected = (anchor_label, neighbor_label, anchor, same, neighbor)
                        break
                if selected:
                    break
            if selected:
                break
        if selected is None:
            continue
        anchor_label, neighbor_label, anchor, same, neighbor = selected
        source = "triad-" + identity(edge_id(edge), anchor, same, neighbor)
        candidates = [(anchor_label, same, "same"), (neighbor_label, neighbor, "neighbor")]
        rng.shuffle(candidates)
        options = choices(["candidate_0", "candidate_1"], rng, ["Candidate 0", "Candidate 1"])
        for swapped in [False, True]:
            ordered = list(reversed(candidates)) if swapped else candidates
            videos = [video(graph, anchor_label, anchor, "Anchor")]
            kinds = []
            indices = {}
            for index, (label, token, kind) in enumerate(ordered):
                videos.append(video(graph, label, token, f"Candidate {index}"))
                kinds.append(kind)
                indices[kind] = "candidate_" + str(index)
            condition = "candidate_order_swapped" if swapped else "candidate_order_original"
            meta = {"edge_id": edge_id(edge), "edge_field": edge["field"],
                    "matched_visual_context_id": source, "candidate_order": kinds,
                    "anchor_label": anchor_label, "neighbor_label": neighbor_label}
            for task, kind in [("same_sign_retrieval", "same"),
                               ("one_feature_neighbor_retrieval", "neighbor"),
                               ("named_feature_neighbor_retrieval", "neighbor")]:
                fields = [edge["field"]] if task == "named_feature_neighbor_retrieval" else []
                prompt = question(task, {"field": edge["field"]}, options, videos, fields)
                rows.append(instance("matched_pairs", condition, task, prompt, videos, options,
                                     indices[kind], source + ":" + task, edge_id(edge), meta))
        completed += 1
        if completed >= count:
            break
    return rows


def build_counterfactuals(graph, count, seed):
    rng = random.Random(seed)
    motifs = []
    for anchor, neighbors in graph["neighbors"].items():
        groups = {}
        for neighbor, field in neighbors.items():
            groups.setdefault(field, []).append(neighbor)
        fields = sorted(groups)
        for field in fields:
            alternatives = []
            for other in fields:
                if other != field:
                    alternatives.extend(groups[other])
            for positive in groups[field]:
                if alternatives:
                    motifs.append((anchor, positive, rng.choice(alternatives), field))
    rng.shuffle(motifs)
    rows = []
    completed = 0
    for anchor_label, positive, negative, field in motifs:
        selected = videos_for_labels(graph, [anchor_label, positive, negative], rng)
        if selected is None:
            continue
        source = "twin-" + identity(anchor_label, positive, negative, field)
        options = choices([True, False], rng, ["Yes", "No"])
        for polarity, second, gold in [("positive", 1, True), ("negative", 2, False)]:
            videos = [copy.deepcopy(selected[0]), copy.deepcopy(selected[second])]
            videos[0]["role"], videos[1]["role"] = "Anchor sign", "Comparison sign"
            prompt = question("counterfactual_twin", {"field": field}, options, videos, [field])
            meta = {"twin_id": source, "polarity": polarity, "queried_field": field,
                    "anchor_label": anchor_label, "positive_neighbor": positive, "negative_neighbor": negative}
            rows.append(instance("counterfactuals", "counterfactual_twin_" + polarity,
                                 "counterfactual_twin", prompt, videos, options, gold,
                                 source + ":" + polarity, source, meta))
        completed += 1
        if completed >= count:
            break
    return rows


def build_gold_features(programs, count, seed):
    rng = random.Random(seed)
    groups = {}
    for program in programs or []:
        if program["task"] != "same_sign":
            groups.setdefault(program["task"], []).append(program)
    rows = []
    for group in groups.values():
        rng.shuffle(group)
        for record in group[:count]:
            validate_record(record)
            base = program_instance(record)
            modes = ["full"]
            if record["task"] in ["universal_shared_feature", "feature_value_counting"]:
                modes.append("minimal")
            for mode in modes:
                fields = FEATURE_FIELDS if mode == "full" else [record["program"]["field"]]
                blocks = ["Use the following correct annotation values to answer the question."]
                for key in record["program"]["item_order"]:
                    item = record["items"][key]
                    lines = [item["role"] + ":"]
                    for field in fields:
                        lines.append(field + " = " + item["vector"][field])
                    blocks.append("\n".join(lines))
                prompt = "\n\n".join(blocks) + "\n\n" + base["prompt"]
                rows.append(instance("gold_features", "gold_features_" + mode, record["task"],
                                     prompt, [], record["choices"], record["gold_semantic"],
                                     record["program_id"], record["cluster_id"],
                                     {"program": record["program"], "program_id": record["program_id"],
                                      "displayed_fields": list(fields)}))
    return rows


def build_atomic_vectors(graph, count, seed):
    rng = random.Random(seed)
    tokens = {}
    for edge in selected_edges(graph, count, rng):
        left, right = choose_pair(graph, edge, rng)
        pair = {"pair_id": "pair-" + identity(edge_id(edge), left, right), "edge_id": edge_id(edge),
                "gold_field": edge["field"], "left_label": edge["left"], "right_label": edge["right"],
                "left_video": left, "right_video": right}
        for label, token in [(edge["left"], left), (edge["right"], right)]:
            if token in tokens and tokens[token]["label"] != label:
                raise ValueError("A video maps to multiple graph labels: " + token)
            tokens.setdefault(token, {"label": label, "pairs": []})["pairs"].append(pair)
    rows = []
    for token, information in tokens.items():
        label = information["label"]
        videos = [video(graph, label, token, "Target sign")]
        for field in FEATURE_FIELDS:
            values = inventory(graph, field)
            display = []
            for value in values:
                display.append(value_text(field, value))
            options = choices(values, rng, display)
            prompt = question("atomic_value", {"field": field, "index": 1}, options, videos, [field])
            gold = graph["nodes"][label]["vector"][field]
            rows.append(instance("atomic_vectors", "atomic_full_vector", "atomic_value", prompt,
                                 videos, options, gold, token + ":" + field, token,
                                 {"video_id": token, "queried_field": field, "gold_value": gold,
                                  "lexical_type": label, "vector_pairs": information["pairs"]}))
    return rows


def build(name, graph, count=100, seed=17, programs=None):
    """Count limits base examples; conditions expand each retained base example."""
    if count < 1:
        raise ValueError("count must be positive")
    if name == "matched_pairs":
        return build_matched_pairs(graph, count, seed)
    if name == "grounding":
        return build_grounding(graph, count, seed)
    if name == "counterfactuals":
        return build_counterfactuals(graph, count, seed)
    if name == "gold_features":
        return build_gold_features(programs, count, seed)
    if name == "atomic_vectors":
        return build_atomic_vectors(graph, count, seed)
    from .controls import build_controls
    return build_controls(name, graph, count, seed)
