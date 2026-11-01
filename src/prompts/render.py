"""Render questions, video roles and shuffled answer choices."""

import json
from pathlib import Path


def load_templates(filename):
    with Path(__file__).with_name(filename).open(encoding="utf-8") as handle:
        return json.load(handle)


def render_prompt(instruction, choices, context="", definitions=None):
    sections = []
    if context:
        sections.append(context)
    if definitions:
        lines = ["Definitions:"]
        for definition in definitions:
            lines.append("- " + definition)
        sections.append("\n".join(lines))
    sections.append("Task:\n" + instruction)
    lines = ["Options:"]
    for choice in choices:
        lines.append(f"{choice['option_id']}. {choice['text']}")
    sections.append("\n".join(lines))
    sections.append('Return only valid JSON in exactly this form: {"answer": "<option-id>"}')
    return "\n\n".join(sections)


def render_program(record):
    program = record["program"]
    instruction = load_templates("tasks.json")[program["op"]].format(**program)
    order = program["item_order"]
    roles = ["Video order:"]
    for index, key in enumerate(order, 1):
        roles.append(f"Video {index}: {record['items'][key]['role']}")
    if program["op"] == "feature_transformation":
        instruction += " Videos 1 and 2 illustrate the supplied change; apply it to Video 3 (anchor C)."
    return render_prompt(instruction, record["choices"], context="\n".join(roles))


def render_diagnostic(task, parameters, choices, context="", definitions=None):
    instruction = load_templates("diagnostics.json")[task].format(**parameters)
    return render_prompt(instruction, choices, context, definitions)


def program_instance(record):
    """Convert a validated formal program to the model input schema."""
    videos = []
    for key in record["program"]["item_order"]:
        item = record["items"][key]
        videos.append({
            "path": f"{item['video_id']}/videos/rgb_clean_480p.mp4",
            "role": item["role"],
            "num_frames": 32,
            "sampling": "uniform",
            "crop": "none",
            "seed": 17,
            "black": False,
            "repeat_middle": False,
            "min_pixels": 16384,
            "max_pixels": 307200,
            "metadata": {"video_id": item["video_id"], "split": "test"},
        })
    gold = []
    for choice in record["choices"]:
        if type(choice["semantic"]) is type(record["gold_semantic"]) and choice["semantic"] == record["gold_semantic"]:
            gold.append(choice["option_id"])
    if len(gold) != 1:
        raise ValueError("A program must have exactly one correct displayed choice")
    return {
        "instance_id": record["program_id"],
        "source_id": record["program_id"],
        "cluster_id": record["cluster_id"],
        "experiment": "benchmark",
        "condition": "full_video",
        "task": record["task"],
        "prompt": render_program(record),
        "videos": videos,
        "choices": record["choices"],
        "gold_option": gold[0],
        "metadata": {"program": record["program"]},
    }
