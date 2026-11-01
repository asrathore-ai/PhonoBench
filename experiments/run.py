"""Run a generated task file with Qwen or InternVL, one instance at a time."""

import argparse
import json
from pathlib import Path

from src.files import file_hash, read_json, read_jsonl, write_json, write_jsonl
from src.parsing import parse_answer
from src.qwen.features import field_name
from src.video import resolve_video


def select_instances(rows, task=None, condition=None, field=None, limit=None):
    """Validate the input file, then select the requested evaluation subset."""
    selected = []
    identities = set()
    if field is not None:
        field = field_name(field)
    for row in rows:
        identity = row["instance_id"]
        if identity in identities:
            raise ValueError("Repeated instance_id: " + identity)
        identities.add(identity)
        options = []
        for choice in row["choices"]:
            if choice["option_id"] in options:
                raise ValueError("Repeated answer option: " + identity)
            options.append(choice["option_id"])
        if not options or row["gold_option"] not in options:
            raise ValueError("Missing or invalid gold answer: " + identity)
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError("Empty prompt: " + identity)
        if task is not None and row["task"] != task:
            continue
        if condition is not None and row["condition"] != condition:
            continue
        if field is not None:
            metadata = row.get("metadata", {})
            program = metadata.get("program", {})
            fields = [metadata.get("queried_field"), metadata.get("field")]
            for key in ["field", "field_a", "field_b"]:
                fields.append(program.get(key))
            if field not in fields:
                continue
        selected.append(row)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No instances match the requested selection")
    return selected


def adapter_settings(args):
    """Use the saved base model and identify the adapter bytes used in this run."""
    hashes = {}
    if args.adapter is None:
        return hashes
    if args.model != "qwen":
        raise ValueError("--adapter is supported by the Qwen runner")
    directory = Path(args.adapter)
    saved = read_json(directory / "feature_config.json")
    if args.checkpoint is not None and args.checkpoint != saved["checkpoint"]:
        raise ValueError("Use the base checkpoint recorded in the adapter's feature_config.json")
    args.checkpoint = saved["checkpoint"]
    revision = saved.get("environment", {}).get("resolved_revision") or saved.get("revision")
    if revision is not None:
        if args.revision is not None and args.revision != revision:
            raise ValueError("The adapter requires base model revision " + revision)
        args.revision = revision
    for filename in ["adapter_config.json", "adapter_model.safetensors", "feature_config.json"]:
        hashes[filename] = file_hash(directory / filename)
    args.adapter = str(directory.resolve())
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL file produced by experiments.build")
    parser.add_argument("--model", choices=["qwen", "internvl"], default="qwen")
    parser.add_argument("--video-root", help="Directory containing the Sem-Lex videos")
    parser.add_argument("--output", help="Output JSONL file, required unless --dry-run")
    parser.add_argument("--checkpoint")
    parser.add_argument("--revision", help="Model revision or commit on Hugging Face")
    parser.add_argument("--adapter", help="Feature adapter directory produced by src.qwen.train")
    parser.add_argument("--load-4bit", action="store_true", help="Use NF4 quantization with Qwen")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attention", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--task")
    parser.add_argument("--condition")
    parser.add_argument("--field", help="Select questions that explicitly query this annotation field")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and video paths without loading a model")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="Retry failed rows when resuming")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--max-new-tokens and --limit must be positive")
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    if args.load_4bit and args.model != "qwen":
        parser.error("--load-4bit is supported by the Qwen runner")
    if args.video_root is not None:
        args.video_root = str(Path(args.video_root).expanduser().resolve())
        if not Path(args.video_root).is_dir():
            parser.error("--video-root must be an existing directory")
    if args.field is not None:
        args.field = field_name(args.field)
    adapter_hashes = adapter_settings(args)
    if args.checkpoint is None:
        args.checkpoint = "Qwen/Qwen3-VL-8B-Instruct" if args.model == "qwen" else "OpenGVLab/InternVL3-8B-hf"

    rows = select_instances(read_jsonl(args.manifest), args.task, args.condition, args.field, args.limit)
    if args.model == "internvl":
        for row in rows:
            if row["experiment"] == "evidence":
                raise ValueError("Use Qwen for evidence experiments: InternVL uses a fixed 28-frame composite")
    checked = set()
    counts = {}
    for row in rows:
        key = row["experiment"] + "/" + row["task"] + "/" + row["condition"]
        counts[key] = counts.get(key, 0) + 1
        for spec in row.get("videos", []):
            metadata = spec.get("metadata", {})
            identity = (spec.get("path"), metadata.get("split"), metadata.get("video_id"))
            if identity not in checked:
                resolve_video(spec, args.video_root)
                checked.add(identity)
    print(json.dumps({"instances": len(rows), "videos": len(checked), "groups": counts}, indent=2))
    if args.dry_run:
        return
    if args.output is None or Path(args.output).suffix != ".jsonl":
        parser.error("--output must be a .jsonl path")

    output = Path(args.output)
    inputs_path = output.with_suffix(".inputs.jsonl")
    run_path = output.with_suffix(".run.json")
    configuration = {
        "manifest_sha256": file_hash(args.manifest), "model": args.model,
        "checkpoint": args.checkpoint, "revision": args.revision,
        "adapter": args.adapter, "adapter_sha256": adapter_hashes,
        "load_in_4bit": args.load_4bit, "video_root": args.video_root,
        "device": args.device, "dtype": args.dtype, "attention": args.attention,
        "max_new_tokens": args.max_new_tokens, "task": args.task,
        "condition": args.condition, "field": args.field, "limit": args.limit,
    }
    previous_environment = None
    if args.resume:
        previous = read_json(run_path)
        if previous["configuration"] != configuration or read_jsonl(inputs_path) != rows:
            raise ValueError("Run settings or inputs changed; choose a new --output path")
        previous_environment = previous.get("environment")
    else:
        for path in [output, inputs_path, run_path]:
            if path.exists():
                raise FileExistsError("Output already exists; use --resume or choose a new --output path")
        write_jsonl(inputs_path, rows)
        write_json(run_path, {"configuration": configuration, "environment": None})

    completed = {}
    identities = set()
    for row in rows:
        identities.add(row["instance_id"])
    if output.exists():
        for result in read_jsonl(output):
            if result["instance_id"] not in identities:
                raise ValueError("An output instance is absent from this run's selected inputs")
            completed[result["instance_id"]] = result
    pending = []
    for row in rows:
        previous = completed.get(row["instance_id"])
        if previous is None or (args.retry_errors and previous.get("status") == "error"):
            pending.append(row)
    if not pending:
        print("All selected instances already have results.")
        return

    if args.model == "qwen":
        from src.qwen.run import generate, load_model
        bundle = load_model(args.checkpoint, args.revision, args.dtype, args.device,
                            args.attention, args.max_new_tokens, args.adapter, args.load_4bit)
    else:
        from src.internvl.run import generate, load_model
        bundle = load_model(args.checkpoint, args.revision, args.dtype, args.device,
                            args.attention, args.max_new_tokens)
    if previous_environment is not None and previous_environment != bundle["environment"]:
        raise ValueError("Model revision or software environment changed; choose a new --output path")
    write_json(run_path, {"configuration": configuration, "environment": bundle["environment"]})
    consecutive_errors = 0
    with output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(pending, 1):
            record = {"instance_id": row["instance_id"]}
            try:
                result = generate(bundle, row, args.video_root)
                result["parsed"] = parse_answer(result["raw_output"], row["choices"])
                record.update({"status": "ok", "result": result})
                consecutive_errors = 0
            except Exception as error:
                record.update({"status": "error", "error": type(error).__name__ + ": " + str(error)})
                consecutive_errors += 1
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(f"{index}/{len(pending)} {row['instance_id']} {record['status']}", flush=True)
            if consecutive_errors >= 3:
                raise SystemExit("Stopped after three consecutive errors. Fix the cause and use --resume --retry-errors.")


if __name__ == "__main__":
    main()
