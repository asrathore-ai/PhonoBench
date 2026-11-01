"""Evaluate one annotated feature with Qwen, optionally loading a trained adapter."""

import argparse
import json
from pathlib import Path

from src.files import file_hash, read_json, read_jsonl, write_json
from src.qwen.features import field_name, feature_instance, feature_metrics
from src.qwen.features import feature_prompt, load_feature_data, parse_label
from src.qwen.run import generate, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True)
    parser.add_argument("--metadata", default="resources/semlex_metadata.csv")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", help="Defaults to the adapter's base model or Qwen3-VL-8B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--adapter", help="Adapter directory produced by src.qwen.train")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attention", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit", type=int, help="Evaluate the first N available videos")
    parser.add_argument("--resume", action="store_true", help="Reuse successful predictions from this output directory")
    args = parser.parse_args()
    if args.frames < 1 or args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--frames, --max-new-tokens, and --limit must be positive")

    field = field_name(args.field)
    splits, labels, selection = load_feature_data(args.metadata, args.video_root, field)
    if args.adapter:
        args.adapter = str(Path(args.adapter).expanduser().resolve())
        adapter = read_json(Path(args.adapter) / "feature_config.json")
        if adapter["field"] != field:
            raise ValueError("The adapter was trained for " + adapter["field"])
        if args.checkpoint is not None and args.checkpoint != adapter["checkpoint"]:
            raise ValueError("Use the base checkpoint recorded in the adapter's feature_config.json")
        args.checkpoint = adapter["checkpoint"]
        revision = adapter.get("environment", {}).get("resolved_revision") or adapter.get("revision")
        if args.revision is not None and revision is not None and args.revision != revision:
            raise ValueError("Use the saved base model revision for this adapter: " + revision)
        args.revision = revision or args.revision
        labels = adapter["allowed_labels"]
    if args.checkpoint is None:
        args.checkpoint = "Qwen/Qwen3-VL-8B-Instruct"
    examples = splits[args.split]
    if args.limit is not None:
        examples = examples[:args.limit]
    if not examples:
        raise ValueError("No labelled videos are available in the selected split")
    for example in examples:
        if example["value"] not in labels:
            raise ValueError("A label is outside this adapter's vocabulary: " + example["value"])

    prompt = feature_prompt(field, labels)
    if args.adapter:
        prompt = adapter.get("prompt", prompt)
    configuration = vars(args).copy()
    configuration.pop("resume")
    configuration.update({"field": field, "allowed_labels": labels,
                          "metadata_sha256": file_hash(args.metadata), "prompt": prompt,
                          "selection": selection, "examples": examples})
    if args.adapter:
        configuration["adapter_sha256"] = {}
        for filename in ["adapter_config.json", "adapter_model.safetensors", "feature_config.json"]:
            configuration["adapter_sha256"][filename] = file_hash(Path(args.adapter) / filename)
    output = Path(args.output)
    config_path = output / "config.json"
    if args.resume:
        if read_json(config_path) != configuration:
            raise ValueError("Run settings or available examples changed; use a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory or pass --resume")
    write_json(config_path, configuration)

    predictions = output / "predictions.jsonl"
    results = {}
    if args.resume and predictions.exists():
        for row in read_jsonl(predictions):
            results[row["video_id"]] = row
    pending = []
    for example in examples:
        if results.get(example["video_id"], {}).get("status") != "ok":
            pending.append(example)
    errors = 0
    if pending:
        bundle = load_model(args.checkpoint, revision=args.revision, dtype=args.dtype,
                            device=args.device, attention=args.attention,
                            max_new_tokens=args.max_new_tokens, adapter=args.adapter,
                            load_4bit=args.load_4bit)
        if args.adapter:
            base_revision = adapter.get("environment", {}).get("resolved_revision")
            if base_revision is not None and base_revision != bundle["environment"]["resolved_revision"]:
                raise ValueError("The base model revision differs from the adapter's training checkpoint")
        environment_path = output / "environment.json"
        if args.resume and environment_path.exists():
            if read_json(environment_path) != bundle["environment"]:
                raise ValueError("Model revision or software environment changed; use a new output directory")
        write_json(environment_path, bundle["environment"])
        with predictions.open("a", encoding="utf-8") as handle:
            for index, example in enumerate(pending, 1):
                row = {"video_id": example["video_id"], "gold": example["value"]}
                try:
                    row.update(generate(bundle, feature_instance(example, prompt, args.frames)))
                    row["prediction"] = parse_label(row["raw_output"], labels)
                    row["status"] = "ok"
                    errors = 0
                except Exception as error:
                    row.update({"status": "error", "prediction": None,
                                "error": type(error).__name__ + ": " + str(error)})
                    errors += 1
                row["valid"] = row["prediction"] in labels
                row["correct"] = row["valid"] and row["prediction"] == row["gold"]
                results[example["video_id"]] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"{index}/{len(pending)} {example['video_id']} {row['status']}", flush=True)
                if errors >= 3:
                    break
    metrics = feature_metrics(examples, results, labels)
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    if errors >= 3:
        raise SystemExit("Stopped after three consecutive errors; inspect predictions.jsonl and use --resume after fixing the cause")


if __name__ == "__main__":
    main()
