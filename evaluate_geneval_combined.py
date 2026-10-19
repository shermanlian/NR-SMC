"""Run GenEval once per particle and emit all-particle plus selected-Top-1 summaries."""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def selected_position(result):
    if result.get("particle_order") == "generation_order":
        return int(result["selected_order"][0])
    return 0


def summarize(rows):
    metadata_groups = OrderedDict()
    task_groups = OrderedDict()
    for row in rows:
        metadata_groups.setdefault(row["metadata"], []).append(bool(row["correct"]))
        task_groups.setdefault(row["tag"], []).append(bool(row["correct"]))
    correct_images = float(np.mean([row["correct"] for row in rows]))
    correct_prompts = float(np.mean([any(values) for values in metadata_groups.values()]))
    tasks = {
        tag: {
            "score": float(np.mean(values)),
            "correct": int(sum(values)),
            "total": len(values),
        }
        for tag, values in task_groups.items()
    }
    return {
        "total_images": len(rows),
        "total_prompts": len(metadata_groups),
        "correct_images": correct_images,
        "correct_prompts": correct_prompts,
        "tasks": tasks,
        "overall_score": float(np.mean([value["score"] for value in tasks.values()])),
    }


def format_summary(summary):
    lines = [
        "Summary",
        "=======",
        f"Total images: {summary['total_images']}",
        f"Total prompts: {summary['total_prompts']}",
        f"% correct images: {summary['correct_images']:.2%}",
        f"% correct prompts: {summary['correct_prompts']:.2%}",
        "",
        "Task breakdown",
        "==============",
    ]
    for tag, values in summary["tasks"].items():
        lines.append(
            f"{tag:<16} = {values['score']:.2%} "
            f"({values['correct']} / {values['total']})"
        )
    lines.extend(
        ["", f"Overall score (avg. over tasks): {summary['overall_score']:.5f}"]
    )
    return "\n".join(lines) + "\n"


def write_jsonl(path, rows):
    with open(path, "w") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--geneval_root", default="../geneval")
    parser.add_argument("--max_prompts", type=int)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    geneval_root = Path(args.geneval_root).resolve()
    sys.path.insert(0, str(geneval_root))
    sys.path.insert(0, str(geneval_root / "evaluation"))
    import mmdet
    import evaluate_images as evaluator

    model_config = os.path.join(
        os.path.dirname(mmdet.__file__),
        "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py",
    )
    model_args = SimpleNamespace(
        model_config=model_config,
        model_path=str(geneval_root / "models"),
        options={},
    )
    evaluator.args = model_args
    evaluator.object_detector, clip_bundle, evaluator.classnames = evaluator.load_models(
        model_args
    )
    evaluator.clip_model, evaluator.transform, evaluator.tokenizer = clip_bundle
    evaluator.THRESHOLD = 0.3
    evaluator.COUNTING_THRESHOLD = 0.9
    evaluator.MAX_OBJECTS = 16
    evaluator.NMS_THRESHOLD = 1.0
    evaluator.POSITION_THRESHOLD = 0.1

    prompt_dirs = sorted(
        path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )
    if args.max_prompts is not None:
        prompt_dirs = prompt_dirs[: args.max_prompts]

    all_rows = []
    top1_rows = []
    for prompt_number, prompt_dir in enumerate(prompt_dirs):
        metadata = json.loads((prompt_dir / "metadata.jsonl").read_text())
        result = json.loads((prompt_dir / "results.json").read_text())
        image_paths = sorted((prompt_dir / "samples").glob("*.png"))
        prompt_rows = []
        for particle_index, image_path in enumerate(image_paths):
            row = evaluator.evaluate_image(str(image_path), metadata)
            row["prompt_index"] = int(prompt_dir.name)
            row["particle_index"] = particle_index
            prompt_rows.append(row)
        all_rows.extend(prompt_rows)
        selected = selected_position(result)
        selected_row = dict(prompt_rows[selected])
        selected_row["selection_policy"] = "highest ImageReward particle"
        top1_rows.append(selected_row)
        print(
            f"GENEVAL_PROGRESS run={run_dir.name} prompt={prompt_number + 1}/{len(prompt_dirs)}",
            flush=True,
        )

    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    for label, rows in (("all_particles", all_rows), ("selected_top1", top1_rows)):
        summary = summarize(rows)
        write_jsonl(evaluation_dir / f"geneval_{label}.jsonl", rows)
        (evaluation_dir / f"geneval_{label}_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        text = format_summary(summary)
        (evaluation_dir / f"geneval_{label}_summary.txt").write_text(text)
        print(text, flush=True)
    (run_dir / "GENEVAL_DONE").write_text("complete\n")


if __name__ == "__main__":
    main()
