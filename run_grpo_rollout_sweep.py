#!/usr/bin/env python3
"""Run quickstart-style GRPO jobs with rollout/context sweeps and GPU telemetry."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import random
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List


def parse_int_list(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_dataset_file(path: str):
    import datasets

    if path.endswith(".parquet"):
        return datasets.load_dataset("parquet", data_files=path)["train"]
    if path.endswith(".json"):
        return datasets.load_dataset("json", data_files=path)["train"]
    raise ValueError(f"Unsupported train-file format: {path}")


def _build_stub_samples(count: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    instruction = 'Let\'s think step by step and output the final answer after "####".'
    rows: List[Dict[str, Any]] = []
    for i in range(count):
        a = rng.randint(1, 999)
        b = rng.randint(1, 999)
        answer = str(a + b)
        question_raw = f"What is {a} + {b}?"
        question = f"{question_raw} {instruction}"
        rows.append(
            {
                # Keep GSM8K scorer compatibility in verl reward dispatch.
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": "train_stub",
                    "index": i,
                    "answer": f"#### {answer}",
                    "question": question_raw,
                    "is_stub": True,
                },
            }
        )
    return rows


def prepare_train_file(args: argparse.Namespace, out_dir: pathlib.Path) -> str:
    repeat_factor = args.train_repeat_factor
    stub_count = args.append_stub_samples
    if repeat_factor <= 1 and stub_count <= 0:
        return args.train_file

    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            "datasets package is required for train-data augmentation. "
            "Install it or run with --train-repeat-factor 1 --append-stub-samples 0."
        ) from exc

    base_ds = _load_dataset_file(args.train_file)
    datasets_to_concat = [base_ds]
    if repeat_factor > 1:
        datasets_to_concat.extend(base_ds for _ in range(repeat_factor - 1))

    if stub_count > 0:
        stub_rows = _build_stub_samples(stub_count, seed=args.stub_seed)
        stub_ds = datasets.Dataset.from_list(stub_rows)
        datasets_to_concat.append(stub_ds)

    merged_ds = datasets.concatenate_datasets(datasets_to_concat)
    aug_path = out_dir / (
        f"train_augmented_repeat{repeat_factor}_stub{stub_count}_{now_stamp()}.parquet"
    )
    merged_ds.to_parquet(str(aug_path))
    print(
        "Prepared augmented train data:",
        f"base={len(base_ds)}",
        f"repeat_factor={repeat_factor}",
        f"stub_samples={stub_count}",
        f"total={len(merged_ds)}",
        f"path={aug_path}",
    )
    return str(aug_path)


def query_gpu_inventory(gpu_ids: List[int]) -> List[Dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        f"--id={','.join(str(x) for x in gpu_ids)}",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    rows: List[Dict[str, Any]] = []
    if not out:
        return rows

    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpu_idx = int(parts[0])
            mem_total = float(parts[2])
        except ValueError:
            continue
        rows.append(
            {
                "gpu_index": gpu_idx,
                "name": parts[1],
                "memory_total_mib": mem_total,
            }
        )
    return rows


def query_gpu_stats(gpu_ids: List[int]) -> List[Dict[str, float]]:
    cmd = [
        "nvidia-smi",
        f"--id={','.join(str(x) for x in gpu_ids)}",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    rows: List[Dict[str, float]] = []
    if not out:
        return rows

    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpu_idx = int(parts[0])
            util = float(parts[1])
            mem_used = float(parts[2])
            mem_total = float(parts[3])
        except ValueError:
            continue
        rows.append(
            {
                "gpu_index": gpu_idx,
                "utilization_gpu": util,
                "memory_used_mib": mem_used,
                "memory_total_mib": mem_total,
            }
        )
    return rows


class GPUMonitor:
    def __init__(self, gpu_ids: List[int], interval_s: float) -> None:
        self.gpu_ids = gpu_ids
        self.interval_s = interval_s
        self.samples: List[Dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            ts = time.time()
            try:
                stats = query_gpu_stats(self.gpu_ids)
                for row in stats:
                    self.samples.append({"sample_ts": ts, **row})
            except Exception:
                # Training should not fail because metrics sampling failed.
                pass
            self._stop.wait(self.interval_s)


def build_base_overrides(args: argparse.Namespace) -> List[str]:
    return [
        "algorithm.adv_estimator=grpo",
        "trainer.val_before_train=False",
        f"data.train_files={args.train_file_prepared}",
        f"data.val_files={args.val_file}",
        f"data.train_batch_size={args.train_batch_size}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={args.model_path}",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.rollout_gpu_mem_util}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "critic.optim.lr=1e-5",
        f"critic.model.path={args.model_path}",
        "critic.ppo_micro_batch_size_per_gpu=4",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=console",
        "trainer.nnodes=1",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.test_freq={args.test_freq}",
        f"trainer.total_epochs={args.total_epochs}",
    ]


def resolve_context_settings(args: argparse.Namespace) -> List[Dict[str, int]]:
    if args.context_sizes:
        context_sizes = parse_int_list(args.context_sizes)
        if not context_sizes:
            raise ValueError("No valid values in --context-sizes.")
        settings = []
        for ctx in context_sizes:
            if ctx < 2:
                raise ValueError("Each context size must be >= 2.")
            prompt_len = int(round(ctx * args.prompt_fraction))
            prompt_len = max(1, min(prompt_len, ctx - 1))
            response_len = ctx - prompt_len
            settings.append(
                {
                    "context_size": ctx,
                    "max_prompt_length": prompt_len,
                    "max_response_length": response_len,
                }
            )
        return settings

    if args.max_prompt_length < 1 or args.max_response_length < 1:
        raise ValueError("--max-prompt-length and --max-response-length must be >= 1.")
    return [
        {
            "context_size": args.max_prompt_length + args.max_response_length,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
        }
    ]


def summarize_samples(samples: List[Dict[str, float]], gpu_ids: List[int]) -> Dict[str, object]:
    per_gpu = {}
    for gpu in gpu_ids:
        gpu_rows = [x for x in samples if int(x["gpu_index"]) == gpu]
        if not gpu_rows:
            per_gpu[gpu] = {
                "samples": 0,
                "peak_memory_mib": None,
                "avg_utilization_pct": None,
                "peak_utilization_pct": None,
                "memory_total_mib": None,
            }
            continue
        utils = [float(x["utilization_gpu"]) for x in gpu_rows]
        mems = [float(x["memory_used_mib"]) for x in gpu_rows]
        per_gpu[gpu] = {
            "samples": len(gpu_rows),
            "peak_memory_mib": round(max(mems), 2),
            "avg_utilization_pct": round(sum(utils) / len(utils), 2),
            "peak_utilization_pct": round(max(utils), 2),
            "memory_total_mib": round(float(gpu_rows[-1]["memory_total_mib"]), 2),
        }

    all_mems = [float(x["memory_used_mib"]) for x in samples]
    all_utils = [float(x["utilization_gpu"]) for x in samples]
    return {
        "overall_peak_memory_mib": round(max(all_mems), 2) if all_mems else None,
        "overall_avg_utilization_pct": round(sum(all_utils) / len(all_utils), 2) if all_utils else None,
        "overall_peak_utilization_pct": round(max(all_utils), 2) if all_utils else None,
        "per_gpu": per_gpu,
    }


def write_samples_csv(path: pathlib.Path, samples: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_ts", "gpu_index", "utilization_gpu", "memory_used_mib", "memory_total_mib"],
        )
        writer.writeheader()
        writer.writerows(samples)


def detect_oom_in_log(log_path: pathlib.Path) -> bool:
    patterns = [
        "cuda out of memory",
        "out of memory",
        "cublas_status_alloc_failed",
        "hip out of memory",
        "oom",
    ]
    if not log_path.exists():
        return False
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            lower = line.lower()
            if any(p in lower for p in patterns):
                return True
    return False


def is_h200_pair(gpu_inventory: List[Dict[str, Any]]) -> bool:
    return len(gpu_inventory) == 2 and all("h200" in str(x.get("name", "")).lower() for x in gpu_inventory)


def result_row(result: Dict[str, Any], h200_pair: bool) -> Dict[str, Any]:
    feasible = result.get("feasible_on_selected_2_gpus")
    if not h200_pair:
        possible_2_h200 = "unknown"
    elif feasible is True:
        possible_2_h200 = "yes"
    elif feasible is False:
        possible_2_h200 = "no"
    else:
        possible_2_h200 = "unknown"

    gpu_metrics = result.get("gpu_metrics", {}) or {}
    return {
        "rollout_n": result.get("rollout_n"),
        "context_size": result.get("context_size"),
        "max_prompt_length": result.get("max_prompt_length"),
        "max_response_length": result.get("max_response_length"),
        "train_repeat_factor": result.get("train_repeat_factor"),
        "append_stub_samples": result.get("append_stub_samples"),
        "gpu_mem_util_setting": result.get("rollout_gpu_mem_util"),
        "exit_code": result.get("exit_code"),
        "oom_detected": result.get("oom_detected"),
        "feasible_on_selected_2_gpus": feasible,
        "h200_pair_detected": h200_pair,
        "possible_on_2_h200": possible_2_h200,
        "overall_peak_memory_mib": gpu_metrics.get("overall_peak_memory_mib"),
        "overall_avg_utilization_pct": gpu_metrics.get("overall_avg_utilization_pct"),
        "overall_peak_utilization_pct": gpu_metrics.get("overall_peak_utilization_pct"),
        "duration_sec": result.get("duration_sec"),
        "experiment_name": result.get("experiment_name"),
    }


def write_table_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rollout_n",
        "context_size",
        "max_prompt_length",
        "max_response_length",
        "train_repeat_factor",
        "append_stub_samples",
        "gpu_mem_util_setting",
        "exit_code",
        "oom_detected",
        "feasible_on_selected_2_gpus",
        "h200_pair_detected",
        "possible_on_2_h200",
        "overall_peak_memory_mib",
        "overall_avg_utilization_pct",
        "overall_peak_utilization_pct",
        "duration_sec",
        "experiment_name",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def table_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "rollout_n",
        "context_size",
        "prompt",
        "response",
        "repeat",
        "stub",
        "gpu_mem_util",
        "exit",
        "feasible_2gpu",
        "possible_2h200",
        "peak_mem_mib",
        "avg_gpu_util_%",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = [
            row.get("rollout_n"),
            row.get("context_size"),
            row.get("max_prompt_length"),
            row.get("max_response_length"),
            row.get("train_repeat_factor"),
            row.get("append_stub_samples"),
            row.get("gpu_mem_util_setting"),
            row.get("exit_code"),
            row.get("feasible_on_selected_2_gpus"),
            row.get("possible_on_2_h200"),
            row.get("overall_peak_memory_mib"),
            row.get("overall_avg_utilization_pct"),
        ]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    return "\n".join(lines)


def write_table_md(path: pathlib.Path, markdown_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_text + "\n", encoding="utf-8")


def run_one(
    args: argparse.Namespace,
    rollout_n: int,
    context_setting: Dict[str, int],
    gpu_ids: List[int],
    base_overrides: List[str],
    out_dir: pathlib.Path,
) -> Dict[str, object]:
    context_size = context_setting["context_size"]
    prompt_len = context_setting["max_prompt_length"]
    response_len = context_setting["max_response_length"]
    exp_name = f"{args.experiment_prefix}_ctx{context_size}_n{rollout_n}_{now_stamp()}"
    train_log_path = out_dir / f"{exp_name}.log"
    metrics_csv_path = out_dir / f"{exp_name}_gpu_metrics.csv"

    dynamic_context_overrides = [
        f"data.max_prompt_length={prompt_len}",
        f"data.max_response_length={response_len}",
        f"actor_rollout_ref.rollout.max_model_len={context_size}",
    ]

    cmd = [
        args.python,
        "-m",
        "verl.trainer.main_ppo",
        *base_overrides,
        *dynamic_context_overrides,
        f"actor_rollout_ref.rollout.n={rollout_n}",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={exp_name}",
        f"trainer.n_gpus_per_node={len(gpu_ids)}",
        *args.extra_override,
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)

    print(
        f"\n=== rollout.n={rollout_n} | context={context_size} "
        f"(prompt={prompt_len}, response={response_len}) | experiment={exp_name} ==="
    )
    print("CMD:", " ".join(cmd))
    print("CUDA_VISIBLE_DEVICES:", env["CUDA_VISIBLE_DEVICES"])
    print("Log:", train_log_path)

    if args.dry_run:
        return {
            "rollout_n": rollout_n,
            "context_size": context_size,
            "max_prompt_length": prompt_len,
            "max_response_length": response_len,
            "train_repeat_factor": args.train_repeat_factor,
            "append_stub_samples": args.append_stub_samples,
            "rollout_gpu_mem_util": args.rollout_gpu_mem_util,
            "experiment_name": exp_name,
            "exit_code": None,
            "duration_sec": 0.0,
            "gpu_metrics": {},
            "oom_detected": False,
            "feasible_on_selected_2_gpus": None,
            "dry_run": True,
        }

    monitor = GPUMonitor(gpu_ids=gpu_ids, interval_s=args.sample_interval)
    start = time.time()
    with train_log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        monitor.start()
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        proc.wait()
    monitor.stop()
    # One last sample after process end.
    ts = time.time()
    try:
        for row in query_gpu_stats(gpu_ids):
            monitor.samples.append({"sample_ts": ts, **row})
    except Exception:
        pass
    duration = time.time() - start

    write_samples_csv(metrics_csv_path, monitor.samples)
    gpu_summary = summarize_samples(monitor.samples, gpu_ids)
    oom_detected = detect_oom_in_log(train_log_path)
    feasible = proc.returncode == 0 and not oom_detected
    result = {
        "rollout_n": rollout_n,
        "context_size": context_size,
        "max_prompt_length": prompt_len,
        "max_response_length": response_len,
        "train_repeat_factor": args.train_repeat_factor,
        "append_stub_samples": args.append_stub_samples,
        "rollout_gpu_mem_util": args.rollout_gpu_mem_util,
        "experiment_name": exp_name,
        "exit_code": proc.returncode,
        "duration_sec": round(duration, 2),
        "gpu_metrics": gpu_summary,
        "oom_detected": oom_detected,
        "feasible_on_selected_2_gpus": feasible,
        "log_path": str(train_log_path),
        "gpu_metrics_csv": str(metrics_csv_path),
    }
    return result


def write_summary(summary_path: pathlib.Path, results: List[Dict[str, object]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": dt.datetime.now().isoformat(),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep GRPO rollout.n and context size for verl quickstart-like training."
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to launch training.")
    parser.add_argument("--train-file", default="~/data/gsm8k/train.parquet")
    parser.add_argument("--val-file", default="~/data/gsm8k/test.parquet")
    parser.add_argument("--train-repeat-factor", type=int, default=1, help="Repeat train dataset N times.")
    parser.add_argument(
        "--append-stub-samples",
        type=int,
        default=0,
        help="Append N synthetic arithmetic stub samples to train data.",
    )
    parser.add_argument("--stub-seed", type=int, default=42, help="Seed for synthetic stub generation.")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--rollout-sizes", default="1,2,4,8", help="Comma-separated rollout.n values.")
    parser.add_argument(
        "--context-sizes",
        default="",
        help=(
            "Comma-separated TOTAL context sizes to sweep. For each size S, "
            "prompt_len=round(S*prompt_fraction), response_len=S-prompt_len."
        ),
    )
    parser.add_argument(
        "--prompt-fraction",
        type=float,
        default=0.5,
        help="Prompt fraction when --context-sizes is used.",
    )
    parser.add_argument("--gpus", default="0,1", help="Exactly two GPU ids, e.g. '0,1'.")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="nvidia-smi poll interval in seconds.")
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-response-length", type=int, default=512)
    parser.add_argument("--rollout-gpu-mem-util", type=float, default=0.4)
    parser.add_argument("--project-name", default="verl_grpo_rollout_sweep")
    parser.add_argument("--experiment-prefix", default="quickstart_grpo")
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--test-freq", type=int, default=10)
    parser.add_argument("--output-dir", default="runs/grpo_rollout_sweep")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop sweep at first failing run. By default continue to collect a full table.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Extra Hydra override; can be passed multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")

    args = parser.parse_args()
    args.train_file = os.path.expanduser(args.train_file)
    args.val_file = os.path.expanduser(args.val_file)

    try:
        rollout_sizes = parse_int_list(args.rollout_sizes)
        gpu_ids = parse_int_list(args.gpus)
    except ValueError as exc:
        print(f"Invalid numeric list: {exc}", file=sys.stderr)
        return 2

    if not rollout_sizes:
        print("No rollout sizes provided.", file=sys.stderr)
        return 2

    if len(gpu_ids) != 2:
        print("--gpus must contain exactly two GPU ids, for example: --gpus 0,1", file=sys.stderr)
        return 2

    if not (0.0 < args.prompt_fraction < 1.0):
        print("--prompt-fraction must be between 0 and 1 (exclusive).", file=sys.stderr)
        return 2

    try:
        context_settings = resolve_context_settings(args)
    except ValueError as exc:
        print(f"Invalid context settings: {exc}", file=sys.stderr)
        return 2

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.train_repeat_factor < 1:
        print("--train-repeat-factor must be >= 1.", file=sys.stderr)
        return 2
    if args.append_stub_samples < 0:
        print("--append-stub-samples must be >= 0.", file=sys.stderr)
        return 2

    try:
        args.train_file_prepared = prepare_train_file(args, out_dir)
    except Exception as exc:
        print(f"Failed to prepare augmented train data: {exc}", file=sys.stderr)
        return 2

    print(f"Train data file used: {args.train_file_prepared}")
    print(f"Train repeat factor: {args.train_repeat_factor}")
    print(f"Appended stub samples: {args.append_stub_samples}")
    base_overrides = build_base_overrides(args)

    gpu_inventory: List[Dict[str, Any]] = []
    try:
        gpu_inventory = query_gpu_inventory(gpu_ids)
    except Exception as exc:
        print(f"Warning: unable to query GPU inventory: {exc}", file=sys.stderr)
    if gpu_inventory:
        print("Selected GPUs:")
        for gpu in gpu_inventory:
            print(
                f"  GPU {gpu['gpu_index']}: {gpu['name']} | total_mem={gpu['memory_total_mib']} MiB"
            )
    h200_pair = is_h200_pair(gpu_inventory)
    print(f"H200 pair detected: {h200_pair}")
    print(f"Configured rollout.gpu_memory_utilization: {args.rollout_gpu_mem_util}")

    results = []
    for context_setting in context_settings:
        for rollout_n in rollout_sizes:
            result = run_one(args, rollout_n, context_setting, gpu_ids, base_overrides, out_dir)
            results.append(result)
            if args.stop_on_failure and result.get("exit_code") not in (0, None):
                print(
                    "Run failed for "
                    f"context={context_setting['context_size']}, rollout.n={rollout_n} "
                    f"(exit_code={result['exit_code']}). Stopping."
                )
                break
        else:
            continue
        break

    summary_path = out_dir / f"summary_{now_stamp()}.json"
    write_summary(summary_path, results)
    print(f"\nSummary written to: {summary_path}")

    rows = [result_row(x, h200_pair=h200_pair) for x in results]
    table_csv_path = out_dir / f"table_{now_stamp()}.csv"
    write_table_csv(table_csv_path, rows)
    table_md = table_markdown(rows)
    table_md_path = out_dir / f"table_{now_stamp()}.md"
    write_table_md(table_md_path, table_md)
    print(f"Table CSV written to: {table_csv_path}")
    print(f"Table Markdown written to: {table_md_path}")
    print("\n" + table_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
