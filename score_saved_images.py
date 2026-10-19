"""Compute ImageReward/HPSv2 from saved images without contaminating generation timing."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils import do_eval


STATISTICS = ("mean", "max", "min", "std")


def stats(values, ddof=0):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "max": float(array.max()),
        "min": float(array.min()),
        "std": float(array.std(ddof=ddof)) if len(array) > ddof else 0.0,
    }


def selected_position(result):
    if result.get("particle_order") == "generation_order":
        return int(result["selected_order"][0])
    # Runs created before generation-order bookkeeping sorted samples by reward.
    return 0


def aggregate(rows, metrics):
    legacy = {}
    global_all = {}
    selected_top1 = {}
    for metric in metrics:
        per_prompt = []
        all_values = []
        top1_values = []
        for row in rows:
            values = row[metric]
            per_prompt.append(stats(values, ddof=1 if len(values) > 1 else 0))
            all_values.extend(values)
            top1_values.append(values[row["selected_position"]])
        legacy[metric] = {
            statistic: float(np.mean([item[statistic] for item in per_prompt]))
            for statistic in STATISTICS
        }
        global_all[metric] = stats(all_values)
        selected_top1[metric] = stats(top1_values)
    return {
        "aggregation_notes": {
            "legacy_prompt_aggregated": (
                "For each prompt, compute mean/max/min/sample-std over particles; "
                "then average each statistic over prompts. N=1 std is defined as zero."
            ),
            "global_all_particles": "Statistics over every saved particle image.",
            "selected_top1": "Statistics over the ImageReward-selected final image per prompt.",
        },
        "num_prompts": len(rows),
        "num_all_particle_images": sum(len(row[metrics[0]]) for row in rows),
        "selection_policy": "highest ImageReward particle",
        "legacy_prompt_aggregated": legacy,
        "global_all_particles": global_all,
        "selected_top1": selected_top1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--metrics", default="ImageReward#HumanPreference")
    parser.add_argument("--max_prompts", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metrics = args.metrics.split("#")
    score_path = run_dir / "offline_reward_scores.jsonl"
    rows_by_index = {}
    if args.resume and score_path.exists():
        for line in score_path.read_text().splitlines():
            row = json.loads(line)
            rows_by_index[int(row["prompt_index"])] = row

    prompt_dirs = sorted(
        path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )
    if args.max_prompts is not None:
        prompt_dirs = prompt_dirs[: args.max_prompts]

    for position, prompt_dir in enumerate(prompt_dirs):
        prompt_index = int(prompt_dir.name)
        if prompt_index in rows_by_index:
            continue
        metadata = json.loads((prompt_dir / "metadata.jsonl").read_text())
        result = json.loads((prompt_dir / "results.json").read_text())
        image_paths = sorted((prompt_dir / "samples").glob("*.png"))
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB").copy())
        prompt_batch = [metadata["prompt"]] * len(images)
        scored = do_eval(
            prompt=prompt_batch, images=images, metrics_to_compute=metrics
        )
        row = {
            "prompt_index": prompt_index,
            "prompt": metadata["prompt"],
            "image_files": [path.name for path in image_paths],
            "selected_position": selected_position(result),
        }
        for metric in metrics:
            row[metric] = [float(value) for value in scored[metric]["result"]]
        rows_by_index[prompt_index] = row
        with open(score_path, "a") as output:
            output.write(json.dumps(row) + "\n")
        print(
            f"REWARD_PROGRESS run={run_dir.name} prompt={position + 1}/{len(prompt_dirs)}",
            flush=True,
        )

    rows = [rows_by_index[int(path.name)] for path in prompt_dirs]
    if len(rows) != len(prompt_dirs):
        raise RuntimeError("Offline reward scoring is incomplete")
    summary = aggregate(rows, metrics)
    (run_dir / "offline_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "REWARDS_DONE").write_text("complete\n")
    print("REWARD_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
