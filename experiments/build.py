"""Build experiment inputs from the metadata and the videos available locally."""

import argparse
import json
from pathlib import Path

from src.files import file_hash, write_json, write_jsonl
from src.graph.data import load_metadata
from src.graph.fields import TASKS
from src.graph.graph import build_graph, graph_audit
from src.graph.programs import generate_programs
from src.prompts.render import program_instance
from .diagnostics import NAMES, build


def build_tasks(metadata="resources/semlex_metadata.csv", video_root=None,
                experiment="benchmark", count=1000, seed=17,
                output="resources/tasks/generated", metadata_only=False,
                overwrite=False):
    """Write selected experiments; count bounds tasks or diagnostic base cases.

    Matched conditions can create several model inputs per diagnostic base case.
    Graph availability determines how many valid, unique examples can be built.
    """
    experiments = ["benchmark"] + NAMES
    if experiment not in experiments + ["all"]:
        raise ValueError("Unknown experiment: " + str(experiment))
    if count < 1:
        raise ValueError("count must be positive")
    if not metadata_only and video_root is None:
        raise ValueError("Supply video_root, or explicitly request metadata_only=True")
    selected = experiments if experiment == "all" else [experiment]
    output = Path(output)
    if output.exists() and not output.is_dir():
        raise ValueError("Output must be a directory")
    generated_names = ["graph.json", "nodes.json", "edges.jsonl", "programs.jsonl", "summary.json"]
    for name in experiments:
        generated_names.append(name + ".jsonl")
    if output.exists():
        for path in output.iterdir():
            if path.name == ".gitkeep":
                continue
            if not overwrite:
                raise ValueError("Output is not empty; use --overwrite to replace generated files")
            if path.name in generated_names and not path.is_file():
                raise ValueError("A generated filename is occupied by a directory: " + str(path))

    nodes, data_summary = load_metadata(metadata, video_root, metadata_only)
    graph = build_graph(nodes)
    programs = []
    if "benchmark" in selected or "gold_features" in selected:
        programs = generate_programs(graph, count, seed)
    program_counts = {}
    for task in TASKS:
        program_counts[task] = 0
    for record in programs:
        program_counts[record["task"]] += 1

    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in generated_names:
            path = output / name
            if path.is_file():
                path.unlink()
    write_json(output / "graph.json", graph)
    if "benchmark" in selected or "gold_features" in selected:
        write_jsonl(output / "programs.jsonl", programs)
    experiment_counts = {}
    for name in selected:
        if name == "benchmark":
            instances = []
            for record in programs:
                instances.append(program_instance(record))
        else:
            instances = build(name, graph, count, seed, programs)
        write_jsonl(output / (name + ".jsonl"), instances)
        task_counts = {}
        condition_counts = {}
        for row in instances:
            task = row["task"]
            condition = row["condition"]
            task_counts[task] = task_counts.get(task, 0) + 1
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        experiment_counts[name] = {
            "instances": len(instances), "tasks": task_counts, "conditions": condition_counts,
        }

    summary = {
        "arguments": {
            "metadata": str(metadata),
            "video_root": str(video_root) if video_root is not None else None,
            "experiment": experiment, "count": count, "seed": seed,
            "output": str(output), "metadata_only": metadata_only, "overwrite": overwrite,
        },
        "metadata_sha256": file_hash(metadata),
        "data": data_summary, "graph": graph_audit(graph),
        "program_counts": program_counts, "experiments": experiment_counts,
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="resources/semlex_metadata.csv")
    parser.add_argument("--video-root", help="Sem-Lex test directory, or its parent")
    parser.add_argument("--experiment", choices=["benchmark"] + NAMES + ["all"], default="benchmark")
    parser.add_argument("--count", type=int, default=1000,
                        help="Upper bound per formal task or diagnostic base cases")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="resources/tasks/generated")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Build symbolic inputs without checking local video files")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace generated files in the output directory")
    args = parser.parse_args()
    try:
        summary = build_tasks(args.metadata, args.video_root, args.experiment,
                              args.count, args.seed, args.output,
                              args.metadata_only, args.overwrite)
    except ValueError as error:
        parser.error(str(error))
    counts = {}
    for name, result in summary["experiments"].items():
        counts[name] = result["instances"]
    print(json.dumps({"output": args.output, "nodes": summary["graph"]["nodes"],
                      "edges": summary["graph"]["edges"], "instances": counts}, indent=2))


if __name__ == "__main__":
    main()
