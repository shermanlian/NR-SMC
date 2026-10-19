"""Collect official BLIP-VQA outputs and report all-particle/IR-selected Top-1."""

import argparse
import json
import statistics
from pathlib import Path


def stats(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    root = run_dir / "evaluation" / "t2icomp_bvqa"
    grouped = {}
    for category in ("color", "shape", "texture"):
        view = root / category
        mapping = [json.loads(line) for line in (view / "mapping.jsonl").read_text().splitlines()]
        result_path = view / "annotation_blip" / "vqa_result.json"
        results = json.loads(result_path.read_text())
        if len(mapping) != len(results):
            raise RuntimeError(f"{category}: mapping/results length mismatch")
        by_id = {int(item["question_id"]): float(item["answer"]) for item in results}
        for item in mapping:
            serial = int(item["serial"])
            if serial not in by_id:
                raise RuntimeError(f"{category}: missing question_id {serial}")
            prompt_index = int(item["prompt_index"])
            row = grouped.setdefault(
                prompt_index,
                {
                    "prompt_index": prompt_index,
                    "prompt": item["prompt"],
                    "category": category,
                    "bvqa_by_particle": {},
                },
            )
            row["bvqa_by_particle"][int(item["particle_index"])] = by_id[serial]

    rows = []
    category_values = {
        category: {"all": [], "selected": []}
        for category in ("color", "shape", "texture")
    }
    for prompt_index in sorted(grouped):
        row = grouped[prompt_index]
        particle_map = row.pop("bvqa_by_particle")
        indices = sorted(particle_map)
        if indices != list(range(len(indices))):
            raise RuntimeError(f"Non-contiguous particles for prompt {prompt_index}")
        row["bvqa"] = [particle_map[index] for index in indices]
        result = json.loads((run_dir / f"{prompt_index:05d}" / "results.json").read_text())
        selected_position = int(result["selected_order"][0])
        row["selected_position"] = selected_position
        row["selection_policy"] = "highest ImageReward particle"
        rows.append(row)
        category_values[row["category"]]["all"].extend(row["bvqa"])
        category_values[row["category"]]["selected"].append(row["bvqa"][selected_position])

    evaluation_dir = run_dir / "evaluation"
    with (evaluation_dir / "t2icomp_bvqa_scores.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    all_values = [value for row in rows for value in row["bvqa"]]
    selected_values = [row["bvqa"][row["selected_position"]] for row in rows]
    summary = {
        "num_prompts": len(rows),
        "num_all_particle_images": len(all_values),
        "selection_policy": "highest ImageReward particle",
        "global_all_particles": {
            "overall": stats(all_values),
            "categories": {
                category: stats(values["all"])
                for category, values in category_values.items()
            },
        },
        "selected_top1": {
            "overall": stats(selected_values),
            "categories": {
                category: stats(values["selected"])
                for category, values in category_values.items()
            },
        },
    }
    (evaluation_dir / "t2icomp_bvqa_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (run_dir / "T2ICOMP_DONE").write_text("complete\n")
    print("T2ICOMP_COMPLETE", json.dumps(summary))


if __name__ == "__main__":
    main()
