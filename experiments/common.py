"""Small helpers for constructing experiment inputs from the current graph."""

import copy
import hashlib
import json
from pathlib import Path

from src.graph.fields import FEATURE_FIELDS
from src.prompts.render import render_diagnostic


HARD_DISTRACTORS = {
    "Major Location": ["Minor Location", "Second Minor Location"],
    "Minor Location": ["Major Location", "Second Minor Location"],
    "Second Minor Location": ["Minor Location", "Major Location"],
    "Contact": ["Thumb Contact"], "Thumb Contact": ["Contact", "Thumb Position"],
    "Handshape": ["Selected Fingers", "Nondominant Handshape", "Flexion"],
    "Selected Fingers": ["Handshape", "Flexion"],
    "Flexion": ["Flexion Change", "Selected Fingers"],
    "Flexion Change": ["Flexion", "Spread Change"],
    "Spread": ["Spread Change"], "Spread Change": ["Spread", "Flexion Change"],
    "Path Movement": ["Repeated Movement", "Wrist Twist"],
    "Repeated Movement": ["Path Movement", "Wrist Twist"],
    "Wrist Twist": ["Path Movement", "Repeated Movement"],
    "Thumb Position": ["Thumb Contact"],
    "Sign Type": ["Nondominant Handshape", "Handshape"],
    "Nondominant Handshape": ["Handshape", "Sign Type"],
}


def identity(*values):
    encoded = json.dumps(values, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()[:20]


def option_ids(count):
    output = []
    for index in range(count):
        number = index + 1
        token = ""
        while number:
            number, remainder = divmod(number - 1, 26)
            token = chr(65 + remainder) + token
        output.append(token)
    return output


def choices(values, rng, display=None):
    order = list(range(len(values)))
    rng.shuffle(order)
    identifiers = option_ids(len(values))
    output = []
    for position, index in enumerate(order):
        text = str(values[index]) if display is None else display[index]
        output.append({"option_id": identifiers[position], "semantic": values[index], "text": text})
    return output


def gold_option(options, gold):
    found = []
    for option in options:
        if type(option["semantic"]) is type(gold) and option["semantic"] == gold:
            found.append(option["option_id"])
    if len(found) != 1:
        raise ValueError("An experiment must have one correct displayed choice")
    return found[0]


def video(graph, label, video_id, role):
    node = graph["nodes"][label]
    return {
        "path": video_id + "/videos/rgb_clean_480p.mp4", "role": role,
        "num_frames": 32, "sampling": "uniform", "crop": "none", "seed": 0,
        "black": False, "repeat_middle": False, "min_pixels": 16384, "max_pixels": 307200,
        "metadata": {"video_id": video_id, "lexical_type": label,
                     "signer_id": node["signers"].get(video_id), "split": "test"},
    }


def definition_text(fields):
    path = Path(__file__).resolve().parents[1] / "src/prompts/feature_definitions.json"
    with path.open(encoding="utf-8") as handle:
        definitions = json.load(handle)
    output = []
    for field in fields:
        output.append(field + ": " + definitions[field])
    return output


def value_text(field, value):
    path = Path(__file__).resolve().parents[1] / "src/prompts/value_glosses.json"
    with path.open(encoding="utf-8") as handle:
        glosses = json.load(handle)
    return glosses.get(field, {}).get(str(value), str(value))


def inventory(graph, field):
    values = []
    for node in graph["nodes"].values():
        value = node["vector"][field]
        if value not in values:
            values.append(value)
    return sorted(values)


def question(task, parameters, options, videos, fields=None, definitions=None):
    roles = []
    for index, spec in enumerate(videos, 1):
        roles.append(f"Video {index}: {spec['role']}")
    context = "Video order:\n" + "\n".join(roles) if roles else ""
    if definitions is None:
        definitions = definition_text(fields or [])
    return render_diagnostic(task, parameters, options, context, definitions)


def instance(experiment, condition, task, prompt, videos, options, gold, source, cluster, metadata):
    row = {
        "instance_id": identity(experiment, condition, task, source),
        "source_id": source, "cluster_id": cluster, "experiment": experiment,
        "condition": condition, "task": task, "prompt": prompt,
        "videos": copy.deepcopy(videos), "choices": copy.deepcopy(options),
        "gold_option": gold_option(options, gold), "metadata": copy.deepcopy(metadata),
    }
    return row


def clone(row, experiment, condition):
    result = copy.deepcopy(row)
    result["experiment"] = experiment
    result["condition"] = condition
    result["instance_id"] = identity(experiment, condition, row["instance_id"])
    result["metadata"]["base_instance_id"] = row["instance_id"]
    return result


def choose_pair(graph, edge, rng):
    left = list(graph["nodes"][edge["left"]]["video_ids"])
    right = list(graph["nodes"][edge["right"]]["video_ids"])
    rng.shuffle(left)
    rng.shuffle(right)
    for a in left:
        for b in right:
            if a != b:
                return a, b
    return None


def selected_edges(graph, count, rng):
    """Round-robin across nonempty edge-label groups, without repeating edges."""
    groups = {}
    for edge in graph["edges"]:
        groups.setdefault(edge["field"], []).append(edge)
    fields = sorted(groups)
    rng.shuffle(fields)
    for group in groups.values():
        rng.shuffle(group)
    output = []
    while len(output) < count:
        found = False
        for field in fields:
            if groups[field] and len(output) < count:
                edge = groups[field].pop()
                if choose_pair(graph, edge, rng) is not None:
                    output.append(edge)
                found = True
        if not found:
            break
    return output


def edge_id(edge):
    return "edge-" + identity(edge["left"], edge["right"], edge["field"])


def pair_metadata(edge, left, right):
    return {
        "edge_id": edge_id(edge), "gold_field": edge["field"], "queried_field": edge["field"],
        "left_label": edge["left"], "right_label": edge["right"],
        "left_video_id": left, "right_video_id": right,
        "matched_pair_id": "pair-" + identity(edge_id(edge), left, right),
        "hard_distractor_field": HARD_DISTRACTORS[edge["field"]][0],
    }


def videos_for_labels(graph, labels, rng):
    """Choose distinct tokens for the supplied ordered lexical labels."""
    videos = []
    used = []
    for index, label in enumerate(labels, 1):
        candidates = list(graph["nodes"][label]["video_ids"])
        rng.shuffle(candidates)
        selected = None
        for candidate in candidates:
            if candidate not in used:
                selected = candidate
                break
        if selected is None:
            return None
        used.append(selected)
        videos.append(video(graph, label, selected, f"Sign {index}"))
    return videos
