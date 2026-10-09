#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract, verify, and plot RL reward & validation accuracy curves from verl logs.

Supports both:
1. Structured JSONL files written by `FileLogger` (`trainer.logger="['console','tensorboard','file']"`
   with `VERL_FILE_LOGGER_ROOT` or `VERL_FILE_LOGGER_PATH`).
2. Raw console / `ray job logs` output containing `step:<N> - key:val - ...` lines.

Usage examples:
  # Single TPU run (JSONL or Ray console log):
  python3 examples/tpu/grpo/plot_reward_curve.py \
      --inputs /data/jialei/logs/verl_tpu_h2h/qwen3_4b_base_gsm8k_tpu.jsonl \
      --labels "TPU v6e-8 (Qwen3-4B-Base)" \
      --output reward_curve_4b.png \
      --csv-out reward_curve_4b.csv \
      --check-gates

  # H2H comparison (TPU v6e vs. GPU H100 baseline):
  python3 examples/tpu/grpo/plot_reward_curve.py \
      --inputs tpu_run.jsonl gpu_run.jsonl \
      --labels "TPU v6e" "GPU H100" \
      --output h2h_reward_curve.png \
      --check-gates
"""

import argparse
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import statistics
from typing import Sequence

STEP_CONSOLE_PATTERN = re.compile(r"step:(\d+)\s+-\s+(.*)")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")

DEFAULT_REWARD_KEY = "critic/rewards/mean"
DEFAULT_SCORE_KEY = "critic/score/mean"
DEFAULT_GRAD_NORM_KEY = "actor/grad_norm"
DEFAULT_PG_LOSS_KEY = "actor/pg_loss"
DEFAULT_PEARSON_KEY = "training/rollout_actor_probs_pearson_corr"
DEFAULT_STEP_TIME_KEY = "timing_s/step"
DEFAULT_RAIDEN_SYNC_KEY = "timing_s/tpu-sync/total_sync"
DEFAULT_RAIDEN_P2P_KEY = "timing_s/tpu-sync/p2p_transfer"


@dataclass
class RunMetrics:
    """Aggregated step-indexed metrics for a single RL training run."""

    label: str
    source_path: Path
    steps: dict[int, dict[str, float]] = field(default_factory=dict)

    @property
    def sorted_steps(self) -> list[int]:
        return sorted(self.steps.keys())

    def get_series(self, key: str) -> tuple[list[int], list[float]]:
        """Return `(step_list, value_list)` for all steps where `key` is present (including NaN/Inf)."""
        xs: list[int] = []
        ys: list[float] = []
        for s in self.sorted_steps:
            if key in self.steps[s]:
                xs.append(s)
                ys.append(self.steps[s][key])
        return xs, ys

    def detect_val_acc_keys(self) -> list[str]:
        """Auto-detect validation accuracy metric keys (`val-core/.../acc/mean...`)."""
        found: set[str] = set()
        for step_dict in self.steps.values():
            for k in step_dict:
                if k.startswith("val-core/") and "/acc/mean" in k:
                    found.add(k)
        return sorted(found)


@dataclass
class GateResult:
    """Pass/fail evaluation result for one H2H parity gate."""

    gate_name: str
    passed: bool
    summary: str


def _try_parse_float(raw: str) -> float | None:
    cleaned = raw.strip()
    if cleaned.startswith("np.float"):
        match = re.search(r"\(([-+0-9.eEnaNiIfF]+)\)", cleaned)
        if match:
            cleaned = match.group(1)
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def parse_log_file(path: str | Path, label: str | None = None) -> RunMetrics:
    """Parse either a `FileLogger` JSONL file or a Ray console log into `RunMetrics`.

    Multiple lines for the same step (e.g. separate validation and training log calls)
    are merged into a single step dictionary so no metrics are lost.
    """
    file_path = Path(path)
    run_label = label or file_path.stem
    metrics = RunMetrics(label=run_label, source_path=file_path)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = ANSI_ESCAPE_PATTERN.sub("", raw_line).strip()
            if not line:
                continue

            # 1. Try FileLogger JSONL format: {"step": <int>, "data": {...}}
            if line.startswith("{") and line.endswith("}"):
                try:
                    payload = json.loads(line)
                    if isinstance(payload, dict) and "step" in payload and isinstance(payload.get("data"), dict):
                        step_num = int(payload["step"])
                        step_bucket = metrics.steps.setdefault(step_num, {})
                        for k, v in payload["data"].items():
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                step_bucket[str(k)] = float(v)
                            elif isinstance(v, str):
                                parsed = _try_parse_float(v)
                                if parsed is not None:
                                    step_bucket[str(k)] = parsed
                        continue
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

            # 2. Try Console / Ray log format: "step:10 - k1:v1 - k2:v2"
            match = STEP_CONSOLE_PATTERN.search(line)
            if match:
                step_num = int(match.group(1))
                rest = match.group(2)
                step_bucket = metrics.steps.setdefault(step_num, {})
                for item in rest.split(" - "):
                    if ":" not in item:
                        continue
                    k, v_str = item.split(":", 1)
                    parsed = _try_parse_float(v_str)
                    if parsed is not None:
                        step_bucket[k.strip()] = parsed

    return metrics


def compute_ema(values: Sequence[float], alpha: float = 0.9) -> list[float]:
    """Compute Exponential Moving Average (EMA) with smoothing factor `alpha` in `[0, 1)`."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0), got {alpha}")
    smoothed: list[float] = []
    ema_val: float | None = None
    for v in values:
        if math.isnan(v) or math.isinf(v):
            smoothed.append(v)
            continue
        if ema_val is None:
            ema_val = v
        else:
            ema_val = alpha * ema_val + (1.0 - alpha) * v
        smoothed.append(ema_val)
    return smoothed


def evaluate_h2h_gates(
    run: RunMetrics,
    min_steps: int = 1,
    reward_key: str = DEFAULT_REWARD_KEY,
    val_key: str | None = None,
    pearson_threshold: float = 0.99,
) -> list[GateResult]:
    """Evaluate the 6 H2H Parity Gates defined in the Trellis RL Recipe Selection doc."""
    results: list[GateResult] = []

    # Gate 1: Functional completion (steps recorded)
    reward_steps, reward_vals = run.get_series(reward_key)
    if not reward_steps:
        reward_steps, reward_vals = run.get_series(DEFAULT_SCORE_KEY)
    num_train_steps = len(reward_steps)
    g1_pass = num_train_steps >= min_steps
    results.append(
        GateResult(
            gate_name="Gate 1: Functional Completion",
            passed=g1_pass,
            summary=f"Recorded {num_train_steps} training reward steps (required >= {min_steps})",
        )
    )

    # Gate 2: Reward curve monotonicity / positive lift
    finite_rewards = [v for v in reward_vals if math.isfinite(v)]
    if len(finite_rewards) >= 2:
        window = max(1, min(10, len(finite_rewards) // 3))
        init_mean = statistics.mean(finite_rewards[:window])
        final_mean = statistics.mean(finite_rewards[-window:])
        delta = final_mean - init_mean
        g2_pass = delta > 0.0
        results.append(
            GateResult(
                gate_name="Gate 2: Reward Lift",
                passed=g2_pass,
                summary=f"Initial({window})={init_mean:.4f} -> Final({window})={final_mean:.4f} (delta={delta:+.4f})",
            )
        )
    else:
        results.append(
            GateResult(
                gate_name="Gate 2: Reward Lift",
                passed=False,
                summary=f"Insufficient finite reward points ({len(finite_rewards)}) to compute lift",
            )
        )

    # Gate 3: Validation accuracy lift
    detected_val_keys = [val_key] if val_key else run.detect_val_acc_keys()
    if detected_val_keys:
        vkey = detected_val_keys[0]
        val_steps, val_vals = run.get_series(vkey)
        finite_vals = [v for v in val_vals if math.isfinite(v)]
        if len(finite_vals) >= 2:
            v_init, v_final, v_peak = finite_vals[0], finite_vals[-1], max(finite_vals)
            v_delta = v_final - v_init
            results.append(
                GateResult(
                    gate_name=f"Gate 3: Validation Accuracy ({vkey})",
                    passed=v_final >= v_init,
                    summary=f"Step {val_steps[0]}={v_init:.4f} -> Step {val_steps[-1]}={v_final:.4f} "
                    f"(peak={v_peak:.4f}, delta={v_delta:+.4f})",
                )
            )
        elif len(finite_vals) == 1:
            results.append(
                GateResult(
                    gate_name=f"Gate 3: Validation Accuracy ({vkey})",
                    passed=True,
                    summary=f"Single validation point recorded at step {val_steps[0]}: {finite_vals[0]:.4f}",
                )
            )
    else:
        results.append(
            GateResult(
                gate_name="Gate 3: Validation Accuracy",
                passed=True,
                summary="No val-core/*/acc/mean* keys present in log (validation skipped)",
            )
        )

    # Gate 4: Loss & Grad Norm stability (no NaN/Inf)
    _, grad_vals = run.get_series(DEFAULT_GRAD_NORM_KEY)
    _, loss_vals = run.get_series(DEFAULT_PG_LOSS_KEY)
    all_checked = grad_vals + loss_vals + reward_vals
    non_finite_count = sum(1 for v in all_checked if not math.isfinite(v))
    g4_pass = len(all_checked) > 0 and non_finite_count == 0
    results.append(
        GateResult(
            gate_name="Gate 4: Loss & Grad Norm Finite",
            passed=g4_pass,
            summary=f"Checked {len(all_checked)} values across reward/pg_loss/grad_norm; "
            f"non-finite (NaN/Inf) count = {non_finite_count}",
        )
    )

    # Gate 5: Steady-state step timing & Raiden weight sync latency (excluding step 1 XLA compile warmup)
    time_steps, time_vals = run.get_series(DEFAULT_STEP_TIME_KEY)
    steady_times = [v for s, v in zip(time_steps, time_vals) if s > min(time_steps, default=0) and math.isfinite(v)]
    if not steady_times:
        steady_times = [v for v in time_vals if math.isfinite(v)]
    sync_steps, sync_vals = run.get_series(DEFAULT_RAIDEN_SYNC_KEY)
    steady_syncs = [v for s, v in zip(sync_steps, sync_vals) if s > min(sync_steps, default=0) and math.isfinite(v)]
    if not steady_syncs:
        steady_syncs = [v for v in sync_vals if math.isfinite(v)]
    if steady_times:
        med_time = statistics.median(steady_times)
        sync_note = (
            f", median {DEFAULT_RAIDEN_SYNC_KEY} = {statistics.median(steady_syncs):.2f}s" if steady_syncs else ""
        )
        results.append(
            GateResult(
                gate_name="Gate 5: Step Latency Recorded",
                passed=med_time > 0.0,
                summary=f"Steady-state median {DEFAULT_STEP_TIME_KEY} = {med_time:.2f}s "
                f"(over {len(steady_times)} steps{sync_note})",
            )
        )
    else:
        results.append(
            GateResult(
                gate_name="Gate 5: Step Latency Recorded",
                passed=False,
                summary=f"No {DEFAULT_STEP_TIME_KEY} entries found",
            )
        )

    # Gate 6: Trainer-Rollout Log-Prob Parity (Pearson correlation)
    _, pearson_vals = run.get_series(DEFAULT_PEARSON_KEY)
    finite_pearson = [v for v in pearson_vals if math.isfinite(v)]
    if finite_pearson:
        mean_pearson = statistics.mean(finite_pearson)
        min_pearson = min(finite_pearson)
        results.append(
            GateResult(
                gate_name="Gate 6: Rollout-Actor Log-Prob Parity",
                passed=mean_pearson >= pearson_threshold,
                summary=f"Mean Pearson r = {mean_pearson:.5f} (min={min_pearson:.5f}, threshold={pearson_threshold})",
            )
        )
    else:
        results.append(
            GateResult(
                gate_name="Gate 6: Rollout-Actor Log-Prob Parity",
                passed=True,
                summary=f"Metric {DEFAULT_PEARSON_KEY} not present in this run mode",
            )
        )

    return results


def export_csv(runs: Sequence[RunMetrics], csv_path: str | Path, reward_key: str = DEFAULT_REWARD_KEY) -> Path:
    """Export key training & validation metrics across all runs into a flat CSV file."""
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "label",
        "step",
        reward_key,
        DEFAULT_SCORE_KEY,
        "val_acc",
        DEFAULT_GRAD_NORM_KEY,
        DEFAULT_PEARSON_KEY,
        DEFAULT_STEP_TIME_KEY,
        DEFAULT_RAIDEN_SYNC_KEY,
        DEFAULT_RAIDEN_P2P_KEY,
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            val_keys = run.detect_val_acc_keys()
            primary_val_key = val_keys[0] if val_keys else None
            for step in run.sorted_steps:
                d = run.steps[step]
                writer.writerow(
                    {
                        "label": run.label,
                        "step": step,
                        reward_key: d.get(reward_key, ""),
                        DEFAULT_SCORE_KEY: d.get(DEFAULT_SCORE_KEY, ""),
                        "val_acc": d.get(primary_val_key, "") if primary_val_key else "",
                        DEFAULT_GRAD_NORM_KEY: d.get(DEFAULT_GRAD_NORM_KEY, ""),
                        DEFAULT_PEARSON_KEY: d.get(DEFAULT_PEARSON_KEY, ""),
                        DEFAULT_STEP_TIME_KEY: d.get(DEFAULT_STEP_TIME_KEY, ""),
                        DEFAULT_RAIDEN_SYNC_KEY: d.get(DEFAULT_RAIDEN_SYNC_KEY, ""),
                        DEFAULT_RAIDEN_P2P_KEY: d.get(DEFAULT_RAIDEN_P2P_KEY, ""),
                    }
                )
    return out_path


def plot_runs(
    runs: Sequence[RunMetrics],
    output_path: str | Path,
    reward_key: str = DEFAULT_REWARD_KEY,
    val_key: str | None = None,
    ema_alpha: float = 0.9,
) -> Path | None:
    """Render a 2x2 diagnostic plot (Training Reward, Validation Accuracy, Grad Norm, Step Time)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_reward_curve] matplotlib is not installed; skipping PNG plot generation.")
        return None

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_rew, ax_val = axes[0, 0], axes[0, 1]
    ax_grad, ax_time = axes[1, 0], axes[1, 1]

    for run in runs:
        # 1. Training Reward (raw + EMA)
        r_steps, r_vals = run.get_series(reward_key)
        if not r_steps:
            r_steps, r_vals = run.get_series(DEFAULT_SCORE_KEY)
        if r_steps:
            line = ax_rew.plot(r_steps, r_vals, alpha=0.3, linewidth=1.0)[0]
            ema_vals = compute_ema(r_vals, alpha=ema_alpha)
            ax_rew.plot(
                r_steps,
                ema_vals,
                color=line.get_color(),
                linewidth=2.2,
                label=f"{run.label} (EMA {ema_alpha})",
            )

        # 2. Validation Accuracy
        v_keys = [val_key] if val_key else run.detect_val_acc_keys()
        if v_keys:
            v_steps, v_vals = run.get_series(v_keys[0])
            if v_steps:
                ax_val.plot(v_steps, v_vals, marker="o", linewidth=2.0, label=f"{run.label} ({v_keys[0]})")

        # 3. Actor Grad Norm
        g_steps, g_vals = run.get_series(DEFAULT_GRAD_NORM_KEY)
        if g_steps:
            ax_grad.plot(g_steps, g_vals, linewidth=1.6, label=run.label)

        # 4. Step Latency
        t_steps, t_vals = run.get_series(DEFAULT_STEP_TIME_KEY)
        if t_steps:
            ax_time.plot(t_steps, t_vals, linewidth=1.6, label=run.label)

    ax_rew.set_title(f"Training Reward Curve ({reward_key})")
    ax_rew.set_xlabel("Training Step")
    ax_rew.set_ylabel("Mean Reward")
    ax_rew.grid(True, alpha=0.3)
    if ax_rew.get_legend_handles_labels()[0]:
        ax_rew.legend()

    ax_val.set_title("Validation Accuracy Curve")
    ax_val.set_xlabel("Training Step")
    ax_val.set_ylabel("Accuracy (pass@1)")
    ax_val.grid(True, alpha=0.3)
    if ax_val.get_legend_handles_labels()[0]:
        ax_val.legend()

    ax_grad.set_title(f"Actor Gradient Norm ({DEFAULT_GRAD_NORM_KEY})")
    ax_grad.set_xlabel("Training Step")
    ax_grad.set_ylabel("Grad Norm")
    ax_grad.grid(True, alpha=0.3)
    if ax_grad.get_legend_handles_labels()[0]:
        ax_grad.legend()

    ax_time.set_title(f"Wall-Clock Step Time ({DEFAULT_STEP_TIME_KEY})")
    ax_time.set_xlabel("Training Step")
    ax_time.set_ylabel("Seconds / Step")
    ax_time.grid(True, alpha=0.3)
    if ax_time.get_legend_handles_labels()[0]:
        ax_time.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse and plot verl RL reward & validation accuracy curves.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to one or more FileLogger .jsonl files or Ray console log files.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display labels corresponding to --inputs.",
    )
    parser.add_argument(
        "--reward-key",
        default=DEFAULT_REWARD_KEY,
        help=f"Metric key for training reward (default: {DEFAULT_REWARD_KEY}).",
    )
    parser.add_argument(
        "--val-key",
        default=None,
        help="Explicit validation accuracy key (auto-detected from val-core/*/acc/mean* if omitted).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.9,
        help="Exponential moving average smoothing factor in [0, 1) (default: 0.9).",
    )
    parser.add_argument(
        "--output",
        default="reward_curve.png",
        help="Output PNG/PDF path for the 2x2 diagnostic plot.",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV path to export aligned step-by-step metrics.",
    )
    parser.add_argument(
        "--check-gates",
        action="store_true",
        help="Evaluate the 6 H2H parity gates and exit with non-zero status if any gate fails.",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="Minimum number of training steps required for Gate 1 (default: 1).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError(f"--labels count ({len(args.labels)}) must match --inputs count ({len(args.inputs)})")

    runs: list[RunMetrics] = []
    for idx, inp in enumerate(args.inputs):
        lbl = args.labels[idx] if args.labels else None
        runs.append(parse_log_file(inp, label=lbl))

    if args.csv_out:
        csv_file = export_csv(runs, args.csv_out, reward_key=args.reward_key)
        print(f"[plot_reward_curve] Exported CSV metrics to: {csv_file}")

    if args.output:
        plot_file = plot_runs(
            runs,
            args.output,
            reward_key=args.reward_key,
            val_key=args.val_key,
            ema_alpha=args.ema_alpha,
        )
        if plot_file is not None:
            print(f"[plot_reward_curve] Saved diagnostic plot to: {plot_file}")

    all_gates_passed = True
    for run in runs:
        print(f"\n=== H2H Parity Gate Summary: {run.label} ({run.source_path}) ===")
        gate_results = evaluate_h2h_gates(
            run,
            min_steps=args.min_steps,
            reward_key=args.reward_key,
            val_key=args.val_key,
        )
        for gr in gate_results:
            status_str = "PASS" if gr.passed else "FAIL"
            print(f"  [{status_str}] {gr.gate_name}: {gr.summary}")
            if not gr.passed:
                all_gates_passed = False

    if args.check_gates and not all_gates_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
