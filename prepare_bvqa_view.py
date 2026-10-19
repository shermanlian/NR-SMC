"""Create an official T2I-CompBench flat samples view for one run/category."""

import argparse
import json
import os
import re
from pathlib import Path


def safe_prompt(prompt):
    prompt = prompt.replace("_", " ")
    prompt = re.sub(r"[\\/:*?\"<>|]", " ", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt[:180]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("category", choices=("color", "shape", "texture"))
    parser.add_argument("view_dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    view_dir = Path(args.view_dir).resolve()
    sample_dir = view_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    serial = 0
    prompt_dirs = sorted(
        path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )
    for prompt_dir in prompt_dirs:
        metadata = json.loads((prompt_dir / "metadata.jsonl").read_text())
        if metadata.get("category") != args.category:
            continue
        prompt = metadata["prompt"]
        for particle_index, source in enumerate(sorted((prompt_dir / "samples").glob("*.png"))):
            filename = f"{safe_prompt(prompt)}_{serial:08d}.png"
            destination = sample_dir / filename
            if not destination.exists() and not destination.is_symlink():
                destination.symlink_to(os.path.relpath(source, destination.parent))
            rows.append(
                {
                    "serial": serial,
                    "filename": filename,
                    "prompt_index": int(prompt_dir.name),
                    "particle_index": particle_index,
                    "prompt": prompt,
                    "category": args.category,
                }
            )
            serial += 1
    if not rows:
        raise RuntimeError(f"No {args.category} images found in {run_dir}")
    expected = {row["filename"] for row in rows}
    actual = {path.name for path in sample_dir.glob("*.png")}
    if actual != expected:
        raise RuntimeError(f"Stale or missing sample links in {sample_dir}")
    with (view_dir / "mapping.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"BVQA_VIEW category={args.category} images={len(rows)} dir={view_dir}")


if __name__ == "__main__":
    main()
