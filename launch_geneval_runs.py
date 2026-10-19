"""Generate GenEval samples with reproducible experiment bookkeeping."""

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append("smc_diffusers")

from pipeline_flux import SMCFluxPipeline
from pipeline_sd3 import SMCStableDiffusion3Pipeline
from pipeline_zimage import SMCZImagePipeline
from utils import do_eval


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def save_rng_checkpoint(path):
    state = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_rng_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def load_geneval_metadata(prompt_path, max_prompts=None):
    if prompt_path.endswith(".json"):
        with open(prompt_path, "r") as f:
            data = json.load(f)
    else:
        assert prompt_path.endswith(".jsonl")
        with open(prompt_path, "r") as f:
            data = [json.loads(line) for line in f]

    assert isinstance(data, list)
    if "prompt" not in data[0]:
        assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"
        for item in data:
            item["prompt"] = item["text"]

    return data if max_prompts is None else data[:max_prompts]


def prepare_output_dir(args):
    if args.experiment_name:
        if Path(args.experiment_name).name != args.experiment_name:
            raise ValueError("experiment_name must be a single directory name")
        output_dir = Path(args.run_root) / args.experiment_name
        if output_dir.exists() and not args.resume:
            raise FileExistsError(
                f"Run already exists: {output_dir}. Use --resume or a new experiment name."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    legacy_root = Path(args.prompt_path.replace(".json", "_outputs"))
    output_dir = legacy_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def build_pipeline(args):
    if args.model_name == "sd3":
        print("Using SD3:", args.model_name, flush=True)
        return SMCStableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
        )
    if args.model_name == "flux":
        print("Using Flux1.dev", flush=True)
        return SMCFluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
        )
    if args.model_name == "zimage":
        print("Using Z-Image", flush=True)
        return SMCZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
    raise ValueError(f"Unknown model: {args.model_name}")


def metric_accumulator(metrics):
    return {metric: dict(mean=0.0, max=0.0, min=0.0, std=0.0) for metric in metrics}


def add_result(metrics_arr, compute_records, results, metrics):
    for metric in metrics:
        for statistic in ("mean", "max", "min", "std"):
            metrics_arr[metric][statistic] += float(results[metric][statistic])
    compute_records.append(results["compute"])


def summarize_values(records, key):
    values = [float(record[key]) for record in records]
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def save_run_summary(output_dir, args, metrics_arr, compute_records):
    n_samples = len(compute_records)
    final_metrics = {
        metric: {key: value / n_samples for key, value in aggregates.items()}
        for metric, aggregates in metrics_arr.items()
    }
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    compute_summary = {
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "benchmark": args.benchmark,
        "seed": args.seed,
        "num_prompts": n_samples,
        "num_particles": args.num_particles,
        "num_inference_steps": args.num_inference_steps,
        "nfe_per_prompt": summarize_values(compute_records, "nfe"),
        "guidance_reward_evals_per_prompt": summarize_values(
            compute_records, "guidance_reward_evals"
        ),
        "selection_reward_evals_per_prompt": summarize_values(
            compute_records, "selection_reward_evals"
        ),
        "vae_decode_evals_per_prompt": summarize_values(
            compute_records, "total_vae_decode_evals"
        ),
        "generation_seconds_per_prompt": summarize_values(
            compute_records, "generation_seconds"
        ),
        "selection_seconds_per_prompt": summarize_values(
            compute_records, "selection_seconds"
        ),
        "method_wall_seconds_per_prompt": summarize_values(
            compute_records, "method_wall_seconds"
        ),
        "peak_allocated_gib": summarize_values(
            compute_records, "peak_allocated_gib"
        ),
        "peak_reserved_gib": summarize_values(
            compute_records, "peak_reserved_gib"
        ),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with open(output_dir / "compute.json", "w") as f:
        json.dump(compute_summary, f, indent=2)

    with open(output_dir / "prompt_results.jsonl", "w") as out:
        for prompt_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
            result_path = prompt_dir / "results.json"
            if result_path.exists():
                with open(result_path, "r") as f:
                    out.write(json.dumps(json.load(f)) + "\n")

    return final_metrics, compute_summary


def main(args):
    seed_everything(args.seed)

    if args.resample_t_end is None:
        args.resample_t_end = args.num_inference_steps
    if args.use_smc:
        assert args.resample_frequency > 0
        assert args.num_particles > 0

    output_dir = prepare_output_dir(args)
    args.output_dir = str(output_dir)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    prompt_data = load_geneval_metadata(args.prompt_path, args.max_prompts)
    print(f"EXPERIMENT_DIR: {output_dir}", flush=True)
    print(f"PROMPTS: {len(prompt_data)}", flush=True)

    pipe = build_pipeline(args)
    if args.stochastic_sampling:
        pipe.scheduler.config.stochastic_sampling = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    manifest = {
        "experiment_name": args.experiment_name,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "args": vars(args),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    metrics_to_compute = args.metrics_to_compute.split("#")
    if args.guidance_reward_fn not in metrics_to_compute:
        metrics_to_compute.insert(0, args.guidance_reward_fn)

    # Load/cache only the metrics used in this run before starting memory/timing stats.
    do_eval(
        prompt=["test"],
        images=[Image.new("RGB", (224, 224))],
        metrics_to_compute=metrics_to_compute,
    )
    if torch.cuda.is_available():
        cuda_sync()
        torch.cuda.empty_cache()

    if args.resume and args.checkpoint_global_rng and not args.seed_per_prompt:
        latest_rng_checkpoint = None
        for prompt_idx in range(len(prompt_data)):
            prompt_dir = output_dir / f"{prompt_idx:05d}"
            result_exists = (prompt_dir / "results.json").exists()
            rng_path = prompt_dir / "rng_state_after.pt"
            if result_exists and rng_path.exists():
                latest_rng_checkpoint = rng_path
                continue
            break
        if latest_rng_checkpoint is not None:
            load_rng_checkpoint(latest_rng_checkpoint)
            print(f"RNG_RESUME checkpoint={latest_rng_checkpoint}", flush=True)

    metrics_arr = metric_accumulator(metrics_to_compute)
    compute_records = []

    for prompt_idx, item in enumerate(tqdm(prompt_data)):
        prompt_path = output_dir / f"{prompt_idx:05d}"
        result_path = prompt_path / "results.json"
        rng_checkpoint_path = prompt_path / "rng_state_after.pt"
        checkpoint_complete = result_path.exists()
        if args.checkpoint_global_rng and not args.seed_per_prompt:
            checkpoint_complete = checkpoint_complete and rng_checkpoint_path.exists()
        if args.resume and checkpoint_complete:
            with open(result_path, "r") as f:
                previous_result = json.load(f)
            add_result(
                metrics_arr, compute_records, previous_result, metrics_to_compute
            )
            print(f"RESUME_SKIP prompt={prompt_idx}", flush=True)
            continue

        prompt_path.mkdir(parents=True, exist_ok=True)
        with open(prompt_path / "metadata.jsonl", "w") as f:
            json.dump(item, f)

        prompt_seed = args.seed
        if args.seed_per_prompt:
            prompt_seed = args.seed * 1_000_000 + prompt_idx * 100
            seed_everything(prompt_seed)
        prompt = [item["prompt"]] * args.num_particles

        smc_args = dict(
            lmbda=args.lmbda,
            num_particles=args.num_particles,
            use_smc=args.use_smc,
            rejuvenation=args.rejuvenation,
            naive_noise=args.naive_noise,
            look_ahead=args.look_ahead,
            adaptive_resampling=args.adaptive_resampling,
            resample_frequency=args.resample_frequency,
            time_steps=args.num_inference_steps,
            resampling_t_start=args.resample_t_start,
            resampling_t_end=args.resample_t_end,
            guidance_reward_fn=args.guidance_reward_fn,
        )

        if torch.cuda.is_available():
            cuda_sync()
            torch.cuda.reset_peak_memory_stats()

        generation_start = time.perf_counter()
        images, nfe = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            smc_args=smc_args,
            return_dict=False,
        )
        raw_pipeline_nfe = int(nfe)
        # SD3 counts vanilla denoiser evaluations per particle, while the Flux
        # and Z-Image pipelines return only the number of batched solver steps.
        # Store one cross-model particle-level convention in the experiment
        # table, while retaining the pipeline-native value for auditability.
        if not args.use_smc:
            nfe = args.num_particles * args.num_inference_steps
        nfe = int(nfe)
        cuda_sync()
        generation_seconds = time.perf_counter() - generation_start

        selection_start = time.perf_counter()
        results = do_eval(
            prompt=prompt,
            images=images,
            metrics_to_compute=[args.guidance_reward_fn],
            topk=args.topk,
        )
        cuda_sync()
        selection_seconds = time.perf_counter() - selection_start

        reporting_metrics = [
            metric for metric in metrics_to_compute if metric != args.guidance_reward_fn
        ]
        if reporting_metrics:
            results.update(
                do_eval(
                    prompt=prompt,
                    images=images,
                    metrics_to_compute=reporting_metrics,
                    topk=args.topk,
                )
            )
            cuda_sync()

        pipeline_stats = getattr(pipe, "last_compute_stats", {})
        compute = {
            "generation_seconds": generation_seconds,
            "selection_seconds": selection_seconds,
            "method_wall_seconds": generation_seconds + selection_seconds,
            "nfe": int(nfe),
            "raw_pipeline_nfe": raw_pipeline_nfe,
            "resampling_events": int(pipeline_stats.get("resampling_events", 0)),
            "guidance_reward_evals": int(
                pipeline_stats.get("guidance_reward_evals", 0)
            ),
            "selection_reward_evals": len(images),
            "intermediate_vae_decode_evals": int(
                pipeline_stats.get("intermediate_vae_decode_evals", 0)
            ),
            "final_vae_decode_evals": int(
                pipeline_stats.get("final_vae_decode_evals", len(images))
            ),
            "total_vae_decode_evals": int(
                pipeline_stats.get("total_vae_decode_evals", len(images))
            ),
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            ),
        }
        results["compute"] = compute
        results["nfe"] = int(nfe)
        results["time_taken"] = compute["method_wall_seconds"]
        results["prompt"] = prompt
        results["prompt_index"] = prompt_idx
        results["prompt_seed"] = prompt_seed

        guidance_reward = np.asarray(
            results[args.guidance_reward_fn]["result"], dtype=float
        )
        sorted_idx = np.argsort(guidance_reward)[::-1]
        if args.topk > 0:
            sorted_idx = sorted_idx[: args.topk]
        results["selected_order"] = [int(index) for index in sorted_idx]
        results["particle_order"] = "generation_order"
        results["particle_indices"] = list(range(len(images)))
        best_index = int(sorted_idx[0])

        if args.save_individual_images:
            sample_path = prompt_path / "samples"
            sample_path.mkdir(exist_ok=True)
            for image_idx, image in enumerate(images):
                image.save(sample_path / f"{image_idx:05d}.png")

            best_path = prompt_path / "best_of_n_samples"
            best_path.mkdir(exist_ok=True)
            images[best_index].save(best_path / "00000.png")

        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        if args.save_grid:
            if args.num_particles > 1:
                _, axes = plt.subplots(
                    1, args.num_particles, figsize=(args.num_particles * 5, 5)
                )
                for image_idx, image in enumerate(images):
                    axes[image_idx].imshow(image)
                    axes[image_idx].axis("off")
            else:
                plt.imshow(images[0])
                plt.axis("off")
            plt.suptitle(prompt[0])
            plt.savefig(prompt_path / "grid.png")
            plt.close()

        if args.checkpoint_global_rng and not args.seed_per_prompt:
            save_rng_checkpoint(rng_checkpoint_path)

        add_result(metrics_arr, compute_records, results, metrics_to_compute)
        print(
            "RUN_PROGRESS "
            f"experiment={args.experiment_name} "
            f"prompt={prompt_idx + 1}/{len(prompt_data)} "
            f"wall={compute['method_wall_seconds']:.3f}s nfe={nfe} "
            f"peak_allocated={compute['peak_allocated_gib']:.3f}GiB",
            flush=True,
        )

    if not compute_records:
        raise RuntimeError("No completed prompts were found")

    _, compute_summary = save_run_summary(
        output_dir, args, metrics_arr, compute_records
    )
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    manifest["num_completed_prompts"] = len(compute_records)
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    (output_dir / "DONE").write_text(datetime.now().astimezone().isoformat() + "\n")
    failed_marker = output_dir / "FAILED"
    if failed_marker.exists():
        failed_marker.unlink()
    print("RUN_COMPLETE", json.dumps(compute_summary), flush=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="geneval_outputs")
    parser.add_argument(
        "--save_individual_images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_grid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save a large per-prompt contact sheet in addition to individual images.",
    )
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--stochastic_sampling", action="store_true", default=False)
    parser.add_argument("--use_smc", action="store_true", default=False)
    parser.add_argument("--rejuvenation", action="store_true", default=False)
    parser.add_argument("--naive_noise", action="store_true", default=False)
    parser.add_argument("--look_ahead", action="store_true", default=False)
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument(
        "--metrics_to_compute",
        type=str,
        default="ImageReward#HumanPreference",
        help="# separated list of metrics",
    )
    parser.add_argument(
        "--prompt_path", type=str, default="prompt_files/geneval_metadata.jsonl"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="geneval",
        help="Benchmark label stored in the run metadata (generation is benchmark-agnostic).",
    )
    parser.add_argument("--model_name", type=str, default="zimage")
    parser.add_argument("--lmbda", type=float, default=10.0)
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_per_prompt", action="store_true", default=False)
    parser.add_argument(
        "--checkpoint_global_rng",
        action="store_true",
        default=False,
        help="Persist per-prompt torch RNG state so continuous-global-RNG runs can resume exactly.",
    )
    parser.add_argument("--adaptive_resampling", action="store_true", default=False)
    parser.add_argument("--resample_frequency", type=int, default=5)
    parser.add_argument("--resample_t_start", type=int, default=5)
    parser.add_argument("--resample_t_end", type=int, default=35)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--run_root", type=str, default="runs")
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    try:
        main(args)
    except Exception:
        if args.experiment_name:
            output_dir = Path(args.run_root) / args.experiment_name
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "FAILED").write_text(traceback.format_exc())
        raise
