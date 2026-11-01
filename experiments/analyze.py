"""Score a run against every selected input, including missing or invalid answers."""

import argparse
from pathlib import Path

from src.files import read_jsonl, write_json
from src.metrics import analyze, compare_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="JSONL file written by experiments.run")
    parser.add_argument("--manifest", type=Path, help="Selected inputs; default: RESULTS.inputs.jsonl")
    parser.add_argument("--output", type=Path, help="Metrics JSON; default: RESULTS.metrics.json")
    parser.add_argument("--reparse", action="store_true", help="Parse raw model text again with the current parser")
    parser.add_argument("--bootstrap", type=int, default=0, help="Cluster bootstrap repetitions; 0 disables intervals")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--compare", type=Path, help="Baseline results for paired accuracy changes on these inputs")
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    manifest_path = args.manifest or args.results.with_suffix(".inputs.jsonl")
    output_path = args.output or args.results.with_suffix(".metrics.json")
    protected = [args.results, manifest_path]
    baseline_manifest_path = None
    if args.compare is not None:
        baseline_manifest_path = args.compare.with_suffix(".inputs.jsonl")
        protected.extend([args.compare, baseline_manifest_path])
    for path in protected:
        if output_path.resolve() == path.resolve():
            parser.error("--output must not overwrite an input or result file")

    manifest = read_jsonl(manifest_path)
    results = read_jsonl(args.results)
    report = analyze(manifest, results, args.reparse, args.bootstrap, args.seed)
    if args.compare is not None:
        baseline_manifest = None
        if baseline_manifest_path.is_file():
            baseline_manifest = read_jsonl(baseline_manifest_path)
        report["comparison"] = compare_results(manifest, results, read_jsonl(args.compare),
                                                baseline_manifest, args.reparse)
    write_json(output_path, report)
    print(f"Scored {len(manifest)} selected inputs. Metrics: {output_path}")


if __name__ == "__main__":
    main()
