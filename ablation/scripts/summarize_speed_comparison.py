#!/usr/bin/env python3
"""Validate and summarize Diffusion versus E7 speed benchmark JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


PROTOCOL_FIELDS = (
    "schema_version",
    "num_samples",
    "batch_size",
    "repeats",
    "tasks",
    "include_postprocess",
    "sample_key_sha256",
    "gpu",
    "torch_version",
    "cuda_version",
    "inference_dtype",
    "float32_matmul_precision",
    "cudnn_benchmark",
)


def _read_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _validate_protocol(diffusion: dict, e7: dict):
    mismatches = []
    for field in PROTOCOL_FIELDS:
        if diffusion.get(field) != e7.get(field):
            mismatches.append(
                f"{field}: Diffusion={diffusion.get(field)!r}, E7={e7.get(field)!r}"
            )
    if mismatches:
        raise ValueError("Benchmark protocols do not match:\n" + "\n".join(mismatches))
    if diffusion["generative_type"] != "diffusion":
        raise ValueError("The reference JSON is not a Diffusion benchmark")
    if e7["generative_type"] not in {"flow", "flow_matching", "flow-matching"}:
        raise ValueError("The E7 JSON is not a Flow Matching benchmark")


def _method_row(method: dict, task: str):
    result = method["results"][task]
    return {
        "task": task,
        "method": method["label"],
        "type": method["generative_type"],
        "solver": method["solver"],
        "nfe": method["nfe"],
        "gpu_seconds_256": result["gpu_seconds_mean_per_run"],
        "gpu_seconds_std": result["gpu_seconds_std_per_run"],
        "gpu_ms_per_sample": result["gpu_ms_per_sample"],
        "samples_per_second": result["throughput_samples_per_second"],
        "realtime_multiplier": result["realtime_multiplier"],
        "batch_p50_seconds": result["batch_gpu_p50_seconds"],
        "batch_p90_seconds": result["batch_gpu_p90_seconds"],
        "batch_p95_seconds": result["batch_gpu_p95_seconds"],
        "peak_allocated_mib": result["peak_allocated_mib"],
        "prepared_pipeline_seconds": result["prepared_pipeline_seconds_one_run"],
    }


def _comparison_row(diffusion: dict, e7: dict, task: str):
    diff_result = diffusion["results"][task]
    e7_result = e7["results"][task]
    paired_speedups = [
        diff_repeat["gpu_seconds"] / e7_repeat["gpu_seconds"]
        for diff_repeat, e7_repeat in zip(
            diff_result["repeat_summaries"], e7_result["repeat_summaries"]
        )
    ]
    pipeline_speedup = (
        diff_result["prepared_pipeline_seconds_one_run"]
        / e7_result["prepared_pipeline_seconds_one_run"]
    )
    return {
        "task": task,
        "nfe_reduction": diffusion["nfe"] / e7["nfe"],
        "sampling_speedup": (
            diff_result["gpu_seconds_mean_per_run"]
            / e7_result["gpu_seconds_mean_per_run"]
        ),
        "paired_speedup_mean": statistics.fmean(paired_speedups),
        "paired_speedup_std": (
            statistics.stdev(paired_speedups) if len(paired_speedups) > 1 else 0.0
        ),
        "latency_reduction_percent": 100.0
        * (
            1.0
            - e7_result["gpu_seconds_mean_per_run"]
            / diff_result["gpu_seconds_mean_per_run"]
        ),
        "pipeline_speedup": pipeline_speedup,
    }


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(diffusion: dict, e7: dict, method_rows: list[dict], comparisons: list[dict]):
    lines = [
        "# UniEgoMotion Diffusion vs E7 Speed Benchmark",
        "",
        "## Protocol",
        "",
        f"- GPU: `{diffusion['gpu']}`",
        f"- Samples per task: `{diffusion['num_samples']}`",
        f"- Batch size: `{diffusion['batch_size']}`",
        f"- Repeats: `{diffusion['repeats']}`",
        f"- Sample key SHA-256: `{diffusion['sample_key_sha256']}`",
        f"- Diffusion NFE: `{diffusion['nfe']}`",
        f"- E7 NFE: `{e7['nfe']}`",
        f"- SMPL-X post-processing: `{diffusion['include_postprocess']}`",
        "",
        "## Per-method results",
        "",
        "| Task | Method | NFE | GPU s / 256 | ms / sample | samples/s | Real-time × | P95 batch s | Peak MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['task']} | {row['method']} | {row['nfe']} | "
            f"{row['gpu_seconds_256']:.4f} ± {row['gpu_seconds_std']:.4f} | "
            f"{row['gpu_ms_per_sample']:.3f} | {row['samples_per_second']:.3f} | "
            f"{row['realtime_multiplier']:.2f} | {row['batch_p95_seconds']:.4f} | "
            f"{row['peak_allocated_mib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## E7 speedup over Diffusion",
            "",
            "| Task | NFE reduction | Sampling speedup | Paired speedup | Latency reduction | Prepared-pipeline speedup |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['task']} | {row['nfe_reduction']:.1f}× | "
            f"{row['sampling_speedup']:.2f}× | "
            f"{row['paired_speedup_mean']:.2f} ± {row['paired_speedup_std']:.2f}× | "
            f"{row['latency_reduction_percent']:.2f}% | {row['pipeline_speedup']:.2f}× |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion", required=True)
    parser.add_argument("--e7", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    diffusion = _read_json(args.diffusion)
    e7 = _read_json(args.e7)
    _validate_protocol(diffusion, e7)
    tasks = diffusion["tasks"]

    method_rows = []
    comparisons = []
    for task in tasks:
        method_rows.extend((_method_row(diffusion, task), _method_row(e7, task)))
        comparisons.append(_comparison_row(diffusion, e7, task))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "speed_methods.csv", method_rows)
    _write_csv(output_dir / "speed_comparison.csv", comparisons)
    summary = {
        "protocol": {field: diffusion[field] for field in PROTOCOL_FIELDS},
        "methods": method_rows,
        "comparisons": comparisons,
    }
    (output_dir / "speed_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown = _markdown(diffusion, e7, method_rows, comparisons)
    (output_dir / "speed_comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
