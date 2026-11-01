"""Sample formal programs with complete annotations and unique correct answers."""

import hashlib
import json
import random

from .fields import FEATURE_FIELDS, TASKS
from .data import normalize_value
from .graph import delta
from .symbolic import execute_program


def same_value(left, right):
    """Python equates True with 1; answer semantics must also have the same type."""
    return type(left) is type(right) and left == right


def validate_record(record, nodes=None):
    """Recompute the answer and optionally check each token against its node."""
    program = record["program"]
    items = record["items"]
    if record["task"] != program["op"] or program["op"] not in TASKS:
        raise ValueError("Task and formal operation do not agree")
    order = program.get("item_order")
    if order is None or len(order) != len(items) or sorted(order) != sorted(items):
        raise ValueError("Missing or invalid explicit item_order")
    expected_items = {
        "same_sign": 2, "one_feature_pair": 2, "distinguishing_feature": 2,
        "exists_one_feature_pair": 5, "universal_shared_feature": 5,
        "feature_value_counting": 5, "constraint_satisfaction": 6,
        "feature_transformation": 7, "common_neighbor": 6, "isolate_detection": 5,
    }
    if len(items) != expected_items[program["op"]]:
        raise ValueError("Unexpected number of video roles for " + program["op"])
    vectors = {}
    labels = {}
    videos = []
    for key in order:
        item = items[key]
        if item["video_id"] in videos:
            raise ValueError("Repeated video token within one formal program")
        videos.append(item["video_id"])
        vector = item["vector"]
        for field in FEATURE_FIELDS:
            if field not in vector or vector[field] is None:
                raise ValueError("Incomplete vector: " + key + " / " + field)
            if normalize_value(vector[field]) is None:
                raise ValueError("Missing annotation sentinel: " + key + " / " + field)
        vectors[key] = vector
        labels[key] = item["lexical_type"]
        if nodes is not None:
            label = item["lexical_type"]
            if label not in nodes:
                raise ValueError("Label is ineligible under current metadata: " + label)
            if item["video_id"] not in nodes[label]["video_ids"]:
                raise ValueError("Video is absent from the eligible test tokens: " + item["video_id"])
            if vector != nodes[label]["vector"]:
                raise ValueError("Stored vector disagrees with metadata: " + label)
    if program["op"] == "feature_transformation" and "example_from" in program:
        left, right = program["example_from"], program["example_to"]
        field = program["field"]
        if delta(vectors[left], vectors[right]) != [field]:
            raise ValueError("Transformation example does not change only the supplied field")
        if vectors[right][field] != str(program["target_value"]):
            raise ValueError("Transformation example has the wrong target value")
        if "source_value" in program and vectors[left][field] != str(program["source_value"]):
            raise ValueError("Transformation example has the wrong source value")
    answer = execute_program(program, vectors, labels)
    if not same_value(answer, record["gold_semantic"]):
        raise ValueError("Stored gold answer disagrees with symbolic execution")
    if program["op"] == "distinguishing_feature":
        expected_choices = FEATURE_FIELDS
    elif program["op"] == "feature_value_counting":
        expected_choices = list(range(6))
    elif program["op"] == "isolate_detection":
        expected_choices = program["items"]
    elif "candidates" in program:
        expected_choices = program["candidates"]
        if len(expected_choices) != 4:
            raise ValueError("Candidate tasks require four candidates")
    else:
        expected_choices = [True, False]
    if len(record["choices"]) != len(expected_choices):
        raise ValueError("Unexpected number of displayed choices")
    for semantic in expected_choices:
        count = 0
        for choice in record["choices"]:
            if same_value(choice["semantic"], semantic):
                count += 1
        if count != 1:
            raise ValueError("Choice space is incomplete or contains duplicate semantics")
    ids = []
    matches = []
    for choice in record["choices"]:
        if choice["option_id"] in ids:
            raise ValueError("Duplicate option identifier")
        ids.append(choice["option_id"])
        if same_value(choice["semantic"], answer):
            matches.append(choice["option_id"])
    if len(matches) != 1:
        raise ValueError("Gold answer does not map to exactly one displayed option")
    return matches[0]


def make_record(graph, program, specs, semantics, display, rng):
    """Choose distinct token IDs, execute the program, then shuffle answer options."""
    items = {}
    order = []
    used_videos = []
    vectors = {}
    labels = {}
    for key, label, role in specs:
        videos = []
        for video_id in graph["nodes"][label]["video_ids"]:
            if video_id not in used_videos:
                videos.append(video_id)
        if not videos:
            return None
        selected = rng.choice(videos)
        used_videos.append(selected)
        vector = dict(graph["nodes"][label]["vector"])
        items[key] = {"video_id": selected, "lexical_type": label, "role": role, "vector": vector}
        vectors[key], labels[key] = vector, label
        order.append(key)
    program["item_order"] = order
    gold = execute_program(program, vectors, labels)
    option_order = list(range(len(semantics)))
    rng.shuffle(option_order)
    choices = []
    for position, index in enumerate(option_order):
        choices.append({"option_id": chr(65 + position), "text": display[index], "semantic": semantics[index]})
    structure = json.dumps([program, labels], sort_keys=True, separators=(",", ":"))
    cluster_id = "structure-" + hashlib.sha256(structure.encode()).hexdigest()[:20]
    record = {
        "task": program["op"], "program": program, "items": items,
        "choices": choices, "gold_semantic": gold, "cluster_id": cluster_id,
        "metadata": {"evaluation_split": "test"},
    }
    # Changing only the answer-option order does not create a new example.
    signature = json.dumps([program, items], sort_keys=True, separators=(",", ":"))
    record["program_id"] = "program-" + hashlib.sha256(signature.encode()).hexdigest()[:20]
    validate_record(record)
    return record


def set_record(graph, program, labels, rng):
    specs = []
    keys = []
    for i, label in enumerate(labels):
        key = "v" + str(i + 1)
        keys.append(key)
        specs.append((key, label, "Video " + str(i + 1)))
    program["items"] = keys
    if program["op"] == "feature_value_counting":
        semantics, display = list(range(6)), ["0", "1", "2", "3", "4", "5"]
    elif program["op"] == "isolate_detection":
        semantics, display = keys, []
        for i in range(5):
            display.append("Video " + str(i + 1))
    elif program["op"] == "distinguishing_feature":
        semantics, display = FEATURE_FIELDS, FEATURE_FIELDS
    else:
        semantics, display = [True, False], ["Yes", "No"]
    return make_record(graph, program, specs, semantics, display, rng)


def candidate_record(graph, program, anchors, correct, rng):
    """Choose three distractors that fail the predicate, preserving a unique gold."""
    excluded = [correct]
    for key, label, role in anchors:
        excluded.append(label)
    distractors = []
    for label in graph["nodes"]:
        if label in excluded:
            continue
        if program["op"] == "constraint_satisfaction":
            a = delta(graph["nodes"][anchors[0][1]]["vector"], graph["nodes"][label]["vector"])
            b = delta(graph["nodes"][anchors[1][1]]["vector"], graph["nodes"][label]["vector"])
            valid = a == [program["field_a"]] and b == [program["field_b"]]
        elif program["op"] == "common_neighbor":
            valid = label in graph["neighbors"][anchors[0][1]] and label in graph["neighbors"][anchors[1][1]]
        else:
            base = anchors[2][1]
            vector = graph["nodes"][label]["vector"]
            valid = delta(graph["nodes"][base]["vector"], vector) == [program["field"]]
            valid = valid and vector[program["field"]] == program["target_value"]
        if not valid:
            distractors.append(label)
    if len(distractors) < 3:
        return None
    candidates = [correct] + rng.sample(distractors, 3)
    rng.shuffle(candidates)
    specs = list(anchors)
    keys = []
    display = []
    for i, label in enumerate(candidates):
        key = "candidate_" + str(i)
        keys.append(key)
        display.append("Candidate " + str(i))
        specs.append((key, label, "Candidate " + str(i)))
    program["candidates"] = keys
    return make_record(graph, program, specs, keys, display, rng)


def independent_fill(graph, labels, selected, rng):
    """Extend a seed to five nodes without adding an edge to the seed or new nodes."""
    pool = list(labels)
    rng.shuffle(pool)
    selected = list(selected)
    for label in pool:
        if label in selected:
            continue
        independent = True
        for other in selected:
            if other in graph["neighbors"][label]:
                independent = False
                break
        if independent:
            selected.append(label)
        if len(selected) == 5:
            return selected
    return None


def prepare_sampling(graph):
    """Small candidate pools reused across the ten straightforward sampling loops."""
    labels = sorted(graph["nodes"])
    same_labels = []
    groups = {}
    motifs = []
    constraints = []
    transitions = {}
    for label in labels:
        if len(graph["nodes"][label]["video_ids"]) >= 2:
            same_labels.append(label)
        vector = graph["nodes"][label]["vector"]
        for field in FEATURE_FIELDS:
            groups.setdefault((field, vector[field]), []).append(label)
        neighbors = sorted(graph["neighbors"][label])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                left, right = neighbors[i], neighbors[j]
                motif = (left, right, label)
                motifs.append(motif)
                if graph["neighbors"][label][left] != graph["neighbors"][label][right]:
                    constraints.append(motif)
    group_pools = []
    for (field, value), matching in groups.items():
        other = []
        for label in labels:
            if graph["nodes"][label]["vector"][field] != value:
                other.append(label)
        group_pools.append((field, value, matching, other))
    for edge in graph["edges"]:
        for left, right in [(edge["left"], edge["right"]), (edge["right"], edge["left"])]:
            field = edge["field"]
            key = (field, graph["nodes"][left]["vector"][field], graph["nodes"][right]["vector"][field])
            transitions.setdefault(key, []).append((left, right))
    return labels, same_labels, group_pools, motifs, constraints, list(transitions.items())


def sample_program(task, graph, pools, index, rng):
    """Each branch implements one task; unavailable motifs return None."""
    labels, same_labels, groups, motifs, constraints, transitions = pools
    program = {"op": task}
    positive = index % 2 == 0

    if task == "same_sign":
        if positive:
            if not same_labels:
                return None
            left = rng.choice(same_labels)
            chosen = [left, left]
        else:
            if len(labels) < 2:
                return None
            chosen = rng.sample(labels, 2)
        return set_record(graph, program, chosen, rng)

    if task in ["one_feature_pair", "distinguishing_feature"]:
        if positive or task == "distinguishing_feature":
            if not graph["edges"]:
                return None
            edge = rng.choice(graph["edges"])
            chosen = [edge["left"], edge["right"]]
        else:
            if len(labels) < 2:
                return None
            chosen = rng.sample(labels, 2)
            if chosen[1] in graph["neighbors"][chosen[0]]:
                return None
        rng.shuffle(chosen)
        return set_record(graph, program, chosen, rng)

    if task == "exists_one_feature_pair":
        if len(labels) < 5:
            return None
        seed = []
        if positive:
            if not graph["edges"]:
                return None
            edge = rng.choice(graph["edges"])
            seed = [edge["left"], edge["right"]]
        chosen = independent_fill(graph, labels, seed, rng)
        if chosen is None:
            return None
        rng.shuffle(chosen)
        return set_record(graph, program, chosen, rng)

    if task in ["universal_shared_feature", "feature_value_counting"]:
        if len(labels) < 5 or not groups:
            return None
        field, value, matching, other = rng.choice(groups)
        if task == "universal_shared_feature":
            count = 5 if positive else 4
        else:
            count = index % 6
            program["value"] = value
        if len(matching) < count or len(other) < 5 - count:
            return None
        chosen = rng.sample(matching, count) + rng.sample(other, 5 - count)
        rng.shuffle(chosen)
        program["field"] = field
        return set_record(graph, program, chosen, rng)

    if task in ["constraint_satisfaction", "common_neighbor"]:
        if len(labels) < 6:
            return None
        pool = constraints if task == "constraint_satisfaction" else motifs
        if not pool:
            return None
        left, right, correct = rng.choice(pool)
        if task == "constraint_satisfaction":
            program.update({"anchor_a": "anchor_a", "anchor_b": "anchor_b",
                            "field_a": graph["neighbors"][correct][left],
                            "field_b": graph["neighbors"][correct][right]})
            anchors = [("anchor_a", left, "Anchor A"), ("anchor_b", right, "Anchor B")]
        else:
            program.update({"anchor_left": "anchor_left", "anchor_right": "anchor_right"})
            anchors = [("anchor_left", left, "Anchor X"), ("anchor_right", right, "Anchor Y")]
        return candidate_record(graph, program, anchors, correct, rng)

    if task == "feature_transformation":
        if len(labels) < 7 or not transitions:
            return None
        (field, source, target), edges = rng.choice(transitions)
        if len(edges) < 2:
            return None
        first, second = rng.sample(edges, 2)
        unique = []
        for label in first + second:
            if label not in unique:
                unique.append(label)
        if len(unique) != 4:
            return None
        program.update({"example_from": "example_from", "example_to": "example_to", "base": "base",
                        "field": field, "source_value": source, "target_value": target})
        anchors = [("example_from", first[0], "Anchor A: example before change"),
                   ("example_to", first[1], "Anchor B: example after change"),
                   ("base", second[0], "Anchor C: base sign")]
        return candidate_record(graph, program, anchors, second[1], rng)

    if task == "isolate_detection":
        if len(labels) < 5 or not motifs:
            return None
        a, c, b = rng.choice(motifs)
        fourth = []
        for label in graph["neighbors"][c]:
            if label not in [a, b, c]:
                fourth.append(label)
        if not fourth:
            return None
        chosen = independent_fill(graph, labels, [a, b, c, rng.choice(fourth)], rng)
        if chosen is None:
            return None
        rng.shuffle(chosen)
        return set_record(graph, program, chosen, rng)

    raise ValueError("Unknown task: " + task)


def generate_programs(graph, count=1000, seed=17):
    """Generate up to count examples per task from the available graph.

    Binary answers and counting answers receive equal sampling quotas, with
    the remainder assigned in answer order. A finite search may return fewer
    examples when tokens or graph motifs are insufficient. Option permutations
    alone never create additional examples.
    """
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    pools = prepare_sampling(graph)
    records = []
    seen = {}
    for task in TASKS:
        slots = 1
        if task in ["same_sign", "one_feature_pair", "exists_one_feature_pair", "universal_shared_feature"]:
            slots = 2
        elif task == "feature_value_counting":
            slots = 6
        for slot in range(slots):
            target = count // slots + int(slot < count % slots)
            completed = 0
            if target == 0:
                continue
            for attempt in range(max(200, target * 100)):
                record = sample_program(task, graph, pools, slot, rng)
                if record is None or record["program_id"] in seen:
                    continue
                seen[record["program_id"]] = True
                records.append(record)
                completed += 1
                if completed == target:
                    break
    return records
