"""Build one ordered JSONL prompt file from the three T2I-CompBench attribute sets."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dir", default="prompt_files/t2i")
    parser.add_argument(
        "--output", default="prompt_files/t2i/t2icompbench_attr_metadata.jsonl"
    )
    parser.add_argument(
        "--smoke-output", default="prompt_files/t2i/t2icompbench_attr_smoke.jsonl"
    )
    args = parser.parse_args()

    prompt_dir = Path(args.prompt_dir)
    output = Path(args.output)
    rows = []
    for category in ("color", "shape", "texture"):
        prompts = (prompt_dir / f"{category}_val.txt").read_text().splitlines()
        if len(prompts) != 300:
            raise RuntimeError(f"Expected 300 {category} prompts, found {len(prompts)}")
        for category_index, prompt in enumerate(prompts):
            rows.append(
                {
                    "prompt": prompt,
                    "category": category,
                    "category_index": category_index,
                    "benchmark": "t2icomp",
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for prompt_index, row in enumerate(rows):
            row["prompt_index"] = prompt_index
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} prompts to {output}")
    smoke = [row for row in rows if row["category_index"] == 0]
    smoke_output = Path(args.smoke_output)
    smoke_output.parent.mkdir(parents=True, exist_ok=True)
    with smoke_output.open("w") as handle:
        for prompt_index, row in enumerate(smoke):
            item = dict(row)
            item["prompt_index"] = prompt_index
            handle.write(json.dumps(item) + "\n")
    print(f"Wrote {len(smoke)} smoke prompts to {smoke_output}")


if __name__ == "__main__":
    main()
