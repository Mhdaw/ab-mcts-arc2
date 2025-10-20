#!/usr/bin/env python3
"""
Generate a CSV of prompts for all ARC problems.

Usage:
    python experiments/arc2/generate_prompts.py \
        --repo-root . \
        --evaluation-dir ARC-AGI-2/data/evaluation \
        --out-file all_prompts.csv

If you run this script from the repository root and the default paths match your layout,
you can omit --repo-root and --evaluation-dir.
"""
from pathlib import Path
import argparse
import csv
import sys
import traceback

def main():
    parser = argparse.ArgumentParser(description="Generate CSV of task_id -> prompt")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Path to the repository root (so imports like `from prompt import BaselinePrompt` work).",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=str,
        default="ARC-AGI-2/data/evaluation",
        help="Directory containing the ARC problem json files.",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="prompts.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress and warnings.",
    )

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    eval_dir = Path(args.evaluation_dir).resolve()
    out_file = Path(args.out_file).resolve()

    # Ensure repo root is on sys.path so local imports work
    sys.path.insert(0, str(repo_root / 'src'))
    # Add the repo root for other imports
    sys.path.insert(0, str(repo_root))
    # Add the script's directory to sys.path for imports like 'from prompt import BaselinePrompt'
    script_dir = Path(__file__).parent
    sys.path.append(str(script_dir))

    if args.verbose:
        print(f"Using repo root: {repo_root}")
        print(f"Reading problems from: {eval_dir}")
        print(f"Writing CSV to: {out_file}")

    if not eval_dir.exists() or not eval_dir.is_dir():
        print(f"Evaluation directory not found: {eval_dir}")
        return

    # Import inside runtime so sys.path modification takes effect
    try:
        from ab_mcts_arc2.tasks.arc.task import ARCProblem
        from ab_mcts_arc2.prompts.prompt_configs import PromptConfig
        # BaselinePrompt is expected to be in prompt.py at repo root (same as your code).
        from prompt import BaselinePrompt
    except Exception:
        print("Failed to import required modules. Traceback:")
        traceback.print_exc()
        print(
            "\nMake sure you ran this script with --repo-root pointing to your project root "
            "so Python can import local modules (e.g. `from prompt import BaselinePrompt`)."
        )
        return

    json_files = sorted(eval_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {eval_dir}")
        return

    # Write CSV with quoting to preserve newlines in prompts
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "prompt"], quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for path in json_files:
            task_id = path.stem
            try:
                problem = ARCProblem.load_file(path)
            except Exception:
                print(f"Warning: failed to load {path}, skipping. Traceback:")
                traceback.print_exc()
                continue

            try:
                prompt_template = BaselinePrompt(prompt_config=PromptConfig(), problem=problem)
                prompt_text = prompt_template.initial_prompt()
            except Exception:
                print(f"Warning: failed to build prompt for {task_id}. Traceback:")
                traceback.print_exc()
                # You can still write an empty prompt or skip; we'll write empty
                prompt_text = ""

            writer.writerow({"task_id": task_id, "prompt": prompt_text})
            if args.verbose:
                print(f"Wrote prompt for {task_id}")

    if args.verbose:
        print("Done.")

if __name__ == "__main__":
    main()
