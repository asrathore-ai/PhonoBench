"""Visual evidence, context load, token production and interface interventions."""

import copy
import random

from src.graph.fields import FEATURE_FIELDS
from .common import (choices, clone, definition_text, edge_id, gold_option, identity, instance,
                     inventory, option_ids, question, selected_edges,
                     value_text, video, videos_for_labels)
from .diagnostics import relation_base, relation_instances


def build_visual_controls(graph, count, seed):
    base = relation_base(graph, count, seed)
    rows = []
    for original in base:
        for condition in ["original", "black", "no_video", "single_frame"]:
            row = clone(original, "visual_controls", condition)
            if condition == "no_video":
                row["videos"] = []
            for spec in row["videos"]:
                spec["black"] = condition == "black"
                spec["repeat_middle"] = condition == "single_frame"
            rows.append(row)
    groups = {}
    for row in base:
        groups.setdefault((row["task"], len(row["videos"])), []).append(row)
    for group in groups.values():
        for index, original in enumerate(group):
            original_ids = []
            for spec in original["videos"]:
                original_ids.append(spec["metadata"]["video_id"])
            donor = None
            for offset in range(1, len(group)):
                candidate = group[(index + offset) % len(group)]
                disjoint = True
                for spec in candidate["videos"]:
                    if spec["metadata"]["video_id"] in original_ids:
                        disjoint = False
                if disjoint:
                    donor = candidate
                    break
            if donor is None:
                continue
            row = clone(original, "visual_controls", "deranged")
            for position in range(len(row["videos"])):
                replacement = copy.deepcopy(donor["videos"][position])
                replacement["role"] = row["videos"][position]["role"]
                row["videos"][position] = replacement
            row["metadata"]["deranged_from_instance"] = donor["instance_id"]
            rows.append(row)
    return rows


def build_evidence(graph, count, seed):
    conditions = [
        ("full_32", 32, "uniform", "none"), ("crop_32", 32, "uniform", "signing_space"),
        ("full_64", 64, "uniform", "none"), ("crop_64", 64, "uniform", "signing_space"),
        ("event_32", 32, "event", "none"), ("crop_event_32", 32, "event", "signing_space"),
        ("shuffled_32", 32, "shuffle", "none"), ("reversed_32", 32, "reverse", "none"),
    ]
    rows = []
    for original in relation_base(graph, count, seed):
        for name, frames, sampling, crop in conditions:
            row = clone(original, "evidence", name)
            for spec in row["videos"]:
                spec.update({"num_frames": frames, "sampling": sampling, "crop": crop, "seed": seed})
            rows.append(row)
    return rows


def build_token_robustness(graph, count, seed, pairs_per_edge=4):
    rng = random.Random(seed)
    rows = []
    for edge in selected_edges(graph, count, rng):
        pairs = {"same_signer": [], "cross_signer": []}
        left = graph["nodes"][edge["left"]]
        right = graph["nodes"][edge["right"]]
        for a in left["video_ids"]:
            for b in right["video_ids"]:
                signer_a, signer_b = left["signers"].get(a), right["signers"].get(b)
                if a == b or not signer_a or not signer_b:
                    continue
                kind = "same_signer" if signer_a == signer_b else "cross_signer"
                pairs[kind].append((a, b))
        for group in pairs.values():
            rng.shuffle(group)
        selected = []
        while len(selected) < pairs_per_edge:
            found = False
            for kind, group in pairs.items():
                if group and len(selected) < pairs_per_edge:
                    a, b = group.pop()
                    selected.append((a, b, kind))
                    found = True
            if not found:
                break
        choice_seed = int(identity(seed, edge_id(edge)), 16)
        for pair_index, (a, b, kind) in enumerate(selected):
            # Reuse the choice order across token realizations of this edge.
            for original in relation_instances(graph, edge, a, b, random.Random(choice_seed), atomic=False):
                if original["condition"] not in ["named_equality_gold", "sole_difference_gold", "two_way_hard_distractor", "seventeen_way"]:
                    continue
                row = clone(original, "token_robustness", kind)
                row["metadata"].update({"token_pair_index": pair_index, "signer_condition": kind,
                                         "left_signer": left["signers"][a], "right_signer": right["signers"][b]})
                rows.append(row)
    return rows


def interface_prompt(row, mode, rng):
    meta = row["metadata"]
    task = row["task"]
    field = meta["queried_field"]
    distractor = meta["hard_distractor_field"]
    options = copy.deepcopy(row["choices"])
    if task == "two_way_field_attribution":
        fields = [field, distractor]
        parameters = {"left_field": field, "right_field": distractor}
    elif task == "seventeen_way_field_attribution":
        fields = list(FEATURE_FIELDS)
        parameters = {}
    else:
        fields = [field]
        parameters = {"field": field}
    definitions = [] if mode == "name_only" else definition_text(fields)
    if mode == "anonymous_definition":
        aliases = ["Property Q", "Property R"]
        rng.shuffle(aliases)
        mapping = {}
        definitions = []
        for index, name in enumerate(fields):
            mapping[name] = aliases[index]
            description = definition_text([name])[0].split(": ", 1)[1]
            definitions.append(aliases[index] + ": " + description)
        for key, value in parameters.items():
            parameters[key] = mapping.get(value, value)
        for choice in options:
            choice["text"] = mapping.get(choice["semantic"], choice["text"])
    prompt = question(task, parameters, options, row["videos"], definitions=definitions)
    return prompt, options, parameters, definitions


def build_interface(graph, count, seed, rotations=4):
    rng = random.Random(seed)
    rows = []
    for original in relation_base(graph, count, seed):
        gold = None
        for option in original["choices"]:
            if option["option_id"] == original["gold_option"]:
                gold = option["semantic"]
        modes = ["name_only", "name_definition"]
        if original["task"] != "seventeen_way_field_attribution":
            modes.append("anonymous_definition")
        for mode in modes:
            row = clone(original, "interface", mode)
            row["prompt"], row["choices"], parameters, definitions = interface_prompt(original, mode, rng)
            row["gold_option"] = gold_option(row["choices"], gold)
            rows.append(row)
        prompt, options, parameters, definitions = interface_prompt(original, "name_definition", rng)
        identifiers = option_ids(len(options))
        for rotation in range(min(rotations, len(options))):
            ordered = options[rotation:] + options[:rotation]
            rotated = []
            for index, option in enumerate(ordered):
                changed = dict(option)
                changed["option_id"] = identifiers[index]
                rotated.append(changed)
            row = clone(original, "interface", "option_rotation_" + str(rotation))
            row["choices"] = rotated
            row["gold_option"] = gold_option(rotated, gold)
            row["prompt"] = question(row["task"], parameters, rotated, row["videos"], definitions=definitions)
            row["metadata"].update({"option_rotation": rotation, "circular_group_id": original["instance_id"]})
            rows.append(row)
        swapped = clone(original, "interface", "video_order_swapped")
        swapped["videos"].reverse()
        for position, spec in enumerate(swapped["videos"]):
            spec["role"] = original["videos"][position]["role"]
        swapped["metadata"]["order_group_id"] = original["instance_id"]
        rows.append(swapped)
    return rows


def target_binding(graph, count, rng):
    bases = []
    for field in FEATURE_FIELDS:
        if len(inventory(graph, field)) < 2:
            continue
        for label in graph["nodes"]:
            bases.append((field, label))
    rng.shuffle(bases)
    rows = []
    for field, target in bases[:count]:
        actual = graph["nodes"][target]["vector"][field]
        same, different = [], []
        for label, node in graph["nodes"].items():
            if label == target:
                continue
            if node["vector"][field] == actual:
                same.append(label)
            else:
                different.append(label)
        rng.shuffle(same)
        rng.shuffle(different)
        target_token = rng.choice(graph["nodes"][target]["video_ids"])
        source = "target-" + identity(target, target_token, field)
        values = inventory(graph, field)
        display = []
        for value in values:
            display.append(value_text(field, value))
        options = choices(values, rng, display)
        scenarios = [("target_only", [])]
        for size in [2, 3, 5, 7]:
            if len(same) >= size - 1:
                scenarios.append((f"congruent_n{size}", same[:size - 1]))
            if len(different) >= size - 1:
                scenarios.append((f"conflicting_n{size}", different[:size - 1]))
        if len(same) >= 3 and len(different) >= 3:
            scenarios.append(("mixed_n7", same[:3] + different[:3]))
        for condition, labels in scenarios:
            distractors = videos_for_labels(graph, labels, rng)
            if distractors is None:
                continue
            target_spec = video(graph, target, target_token, "Target sign")
            duplicate = False
            for spec in distractors:
                if spec["metadata"]["video_id"] == target_token:
                    duplicate = True
            if duplicate:
                continue
            size = len(distractors) + 1
            positions = []
            for position in [0, (size - 1) // 2, size - 1]:
                if position not in positions:
                    positions.append(position)
            for position in positions:
                videos = copy.deepcopy(distractors)
                videos.insert(position, copy.deepcopy(target_spec))
                for index, spec in enumerate(videos, 1):
                    spec["role"] = f"Sign {index}"
                prompt = question("target_feature_under_distractors",
                                  {"field": field, "target_index": position + 1}, options, videos, [field])
                name = condition + "_position_" + str(position)
                rows.append(instance("binding", name, "target_feature_under_distractors", prompt,
                                     videos, options, actual, source, source,
                                     {"field": field, "queried_field": field, "target_label": target,
                                      "target_video_id": target_token, "target_position": position,
                                      "num_videos": size, "context_id": source, "target_value": actual}))
    return rows


def universal_binding(graph, count, rng):
    groups = {}
    for field in FEATURE_FIELDS:
        for label, node in graph["nodes"].items():
            groups.setdefault((field, node["vector"][field]), []).append(label)
    bases = []
    for (field, value), labels in groups.items():
        different = []
        for label, node in graph["nodes"].items():
            if node["vector"][field] != value:
                different.append(label)
        if len(labels) >= 7 and different:
            bases.append((field, value, labels, different))
    rng.shuffle(bases)
    rows = []
    for field, value, labels, different in bases[:count]:
        selected = rng.sample(labels, 7)
        outlier_label = rng.choice(different)
        tokens = videos_for_labels(graph, selected + [outlier_label], rng)
        if tokens is None:
            continue
        shared, outlier = tokens[:7], tokens[7]
        source = "universal-" + identity(field, value, selected, outlier_label)
        options = choices([True, False], rng, ["Yes", "No"])
        for size in [2, 3, 5, 7]:
            for polarity in ["positive", "negative"]:
                videos = copy.deepcopy(shared[:size])
                if polarity == "negative":
                    videos[-1] = copy.deepcopy(outlier)
                for index, spec in enumerate(videos, 1):
                    spec["role"] = f"Sign {index}"
                condition = f"universal_{polarity}_n{size}"
                prompt = question("controlled_universal", {"field": field}, options, videos, [field])
                meta = {"field": field, "num_videos": size, "polarity": polarity,
                        "shared_value": value, "context_id": source}
                row = instance("binding", condition, "controlled_universal", prompt, videos, options,
                               polarity == "positive", source, source, meta)
                rows.append(row)
                if size == 7:
                    for position in range(size):
                        masked = clone(row, "binding", condition + "_mask_" + str(position))
                        masked["videos"][position]["black"] = True
                        masked["metadata"]["masked_position"] = position
                        rows.append(masked)
        for position in [0, 3, 6]:
            videos = copy.deepcopy(shared[:6])
            videos.insert(position, copy.deepcopy(outlier))
            for index, spec in enumerate(videos, 1):
                spec["role"] = f"Sign {index}"
            prompt = question("universal_outlier_position", {"field": field}, options, videos, [field])
            rows.append(instance("binding", "outlier_position_" + str(position),
                                 "universal_outlier_position", prompt, videos, options, False,
                                 source, source, {"field": field, "num_videos": 7,
                                                  "outlier_position": position, "shared_value": value,
                                                  "context_id": source}))
    return rows


def build_controls(name, graph, count, seed):
    if name == "visual_controls":
        return build_visual_controls(graph, count, seed)
    if name == "evidence":
        return build_evidence(graph, count, seed)
    if name == "token_robustness":
        return build_token_robustness(graph, count, seed)
    if name == "interface":
        return build_interface(graph, count, seed)
    if name == "binding":
        rng = random.Random(seed)
        return target_binding(graph, count, rng) + universal_binding(graph, count, rng)
    raise ValueError("Unknown diagnostic experiment: " + name)
