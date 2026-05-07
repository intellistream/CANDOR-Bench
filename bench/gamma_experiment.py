from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from bench.algorithms.registry import get_algorithm
from bench.io_utils import (
    create_timestamped_output_dir,
    write_manifest_json,
    write_rows_csv,
)
from bench.runner import BenchmarkRunner
from datasets.base import Dataset


DEFAULT_GAMMA_DIM = 32
DEFAULT_GAMMA_THREADS = 1


@dataclass(frozen=True)
class GammaOperation:
    op_type: str
    target_id: int
    phase: str


@dataclass(frozen=True)
class GammaSweepConfig:
    dataset_size: int
    operations: int
    dim: int
    topk: int
    gamma_values: list[float]
    indices: list[str]
    random_seed: int
    prefill_ratio: float
    zipf_alpha: float
    delete_ratio: float
    threads: int
    compute_recall: bool = True
    algorithm_dataset_key: str = "random-xs"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GammaSweepConfig":
        indices_value = data.get("indices", data.get("algorithms"))
        required = [
            "dataset_size",
            "operations",
            "topk",
            "gamma_values",
            "random_seed",
            "prefill_ratio",
            "zipf_alpha",
            "delete_ratio",
        ]
        missing = [key for key in required if key not in data]
        if indices_value is None:
            missing.append("indices or algorithms")
        if missing:
            raise ValueError(f"gamma_sweep missing keys: {', '.join(missing)}")

        cfg = cls(
            dataset_size=int(data["dataset_size"]),
            operations=int(data["operations"]),
            dim=int(data.get("dim", DEFAULT_GAMMA_DIM)),
            topk=int(data["topk"]),
            gamma_values=[float(v) for v in data["gamma_values"]],
            indices=[str(v) for v in indices_value],
            random_seed=int(data["random_seed"]),
            prefill_ratio=float(data["prefill_ratio"]),
            zipf_alpha=float(data["zipf_alpha"]),
            delete_ratio=float(data["delete_ratio"]),
            threads=int(data.get("threads", DEFAULT_GAMMA_THREADS)),
            compute_recall=bool(data.get("compute_recall", True)),
            algorithm_dataset_key=str(data.get("algorithm_dataset_key", "random-xs")),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.dataset_size <= 0:
            raise ValueError("gamma_sweep.dataset_size must be > 0")
        if self.operations <= 0:
            raise ValueError("gamma_sweep.operations must be > 0")
        if self.dim <= 0:
            raise ValueError("gamma_sweep.dim must be > 0")
        if self.topk <= 0:
            raise ValueError("gamma_sweep.topk must be > 0")
        if not self.gamma_values:
            raise ValueError("gamma_sweep.gamma_values must not be empty")
        if any(gamma < 0.0 for gamma in self.gamma_values):
            raise ValueError("gamma_sweep.gamma_values must be >= 0")
        if not self.indices:
            raise ValueError("gamma_sweep.indices must not be empty")
        if not 0.0 <= self.prefill_ratio <= 1.0:
            raise ValueError("gamma_sweep.prefill_ratio must be in [0, 1]")
        if self.zipf_alpha < 0.0:
            raise ValueError("gamma_sweep.zipf_alpha must be >= 0")
        if not 0.0 <= self.delete_ratio <= 1.0:
            raise ValueError("gamma_sweep.delete_ratio must be in [0, 1]")
        if self.threads <= 0:
            raise ValueError("gamma_sweep.threads must be > 0")


class _GammaVectorDataset(Dataset):
    """Dataset adapter for gamma-generated or sliced vectors."""

    def __init__(self, vectors: np.ndarray, name: str, distance: str = "euclidean"):
        super().__init__()
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._name = name
        self._distance = distance
        self.nb = int(self._vectors.shape[0])
        self.nq = min(10, self.nb)
        self.d = int(self._vectors.shape[1]) if self._vectors.ndim == 2 else 0
        self.dtype = "float32"

    def prepare(self, skip_data: bool = False) -> None:
        return None

    def get_dataset_fn(self) -> str:
        return self._name

    def get_dataset(self) -> np.ndarray:
        return self._vectors

    def get_dataset_iterator(self, bs: int = 512, split: tuple[int, int] = (1, 0)):
        for start in range(0, self.nb, bs):
            yield self._vectors[start : start + bs]

    def get_queries(self) -> np.ndarray:
        return self._vectors[: self.nq]

    def get_groundtruth(self, k: int | None = None) -> None:
        return None

    def distance(self) -> str:
        return self._distance

    def short_name(self) -> str:
        return self._name


def load_gamma_sweep_config(path: str | Path) -> GammaSweepConfig:
    with open(path, "r", encoding="utf-8") as handle:
        root = yaml.safe_load(handle) or {}
    section = root.get("gamma_sweep", root)
    if not isinstance(section, dict):
        raise ValueError("gamma_sweep config must be a mapping")
    return GammaSweepConfig.from_dict(section)


def create_gamma_output_dir(base_dir: str | Path, experiment_name: str = "gamma_sweep") -> Path:
    return create_timestamped_output_dir(
        base_dir,
        experiment_name,
        subdirs=("data", Path("figures") / "main"),
    )


def generate_gamma_operation_sequence(
    cfg: GammaSweepConfig,
    gamma: float,
) -> list[GammaOperation]:
    rng = np.random.default_rng(cfg.random_seed)
    all_ids = np.arange(cfg.dataset_size, dtype=np.uint64)
    rng.shuffle(all_ids)

    prefill_count = min(cfg.dataset_size, int(cfg.dataset_size * cfg.prefill_ratio))
    active_ids = [int(v) for v in all_ids[:prefill_count]]
    available_ids = [int(v) for v in all_ids[prefill_count:]]
    operations: list[GammaOperation] = [
        GammaOperation("insert", target_id, "prefill") for target_id in active_ids
    ]

    write_prob = 1.0 if gamma == 0.0 else 1.0 / (1.0 + gamma)
    query_weights = _zipf_weights(cfg.dataset_size, cfg.zipf_alpha)

    for _ in range(cfg.operations):
        if rng.random() >= write_prob:
            operations.append(
                GammaOperation(
                    "query",
                    _sample_query_id(rng, cfg.dataset_size, query_weights),
                    "measurement",
                )
            )
            continue

        do_delete = rng.random() < cfg.delete_ratio
        if do_delete and not active_ids:
            do_delete = False
        if not do_delete and not available_ids:
            do_delete = True

        if do_delete and active_ids:
            pos = int(rng.integers(0, len(active_ids)))
            target_id = active_ids[pos]
            active_ids[pos] = active_ids[-1]
            active_ids.pop()
            available_ids.append(target_id)
            operations.append(GammaOperation("delete", target_id, "measurement"))
        elif available_ids:
            pos = int(rng.integers(0, len(available_ids)))
            target_id = available_ids[pos]
            available_ids[pos] = available_ids[-1]
            available_ids.pop()
            active_ids.append(target_id)
            operations.append(GammaOperation("insert", target_id, "measurement"))
        else:
            operations.append(
                GammaOperation(
                    "query",
                    _sample_query_id(rng, cfg.dataset_size, query_weights),
                    "measurement",
                )
            )

    return operations


def run_gamma_sweep(
    cfg: GammaSweepConfig,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    dataset: Dataset | None = None,
    dataset_name: str | None = None,
    make_plots: bool = False,
) -> Path:
    run_dir = create_gamma_output_dir(output_dir)
    data_dir = run_dir / "data"
    cfg, vectors, runner_dataset, dataset_label = _resolve_vectors(cfg, dataset, dataset_name)

    sequence_by_gamma = {
        gamma: generate_gamma_operation_sequence(cfg, gamma) for gamma in cfg.gamma_values
    }
    for gamma, sequence in sequence_by_gamma.items():
        _write_operation_sequence(
            data_dir / f"operation_sequence_gamma_{_gamma_label(gamma)}.csv",
            sequence,
        )

    runs: list[dict[str, Any]] = []
    latencies: list[dict[str, Any]] = []
    timeseries: list[dict[str, Any]] = []
    custom_metrics: list[dict[str, Any]] = []

    if cfg.threads != 1:
        print("Warning: gamma_sweep.threads is recorded but this runner executes serially.")

    for gamma in cfg.gamma_values:
        sequence = sequence_by_gamma[gamma]
        for index_name in cfg.indices:
            run_rows = _run_one_index_gamma(
                cfg=cfg,
                gamma=gamma,
                index_name=index_name,
                vectors=vectors,
                runner_dataset=runner_dataset,
                dataset_label=dataset_label,
                sequence=sequence,
            )
            runs.append(run_rows["run"])
            latencies.extend(run_rows["latencies"])
            timeseries.extend(run_rows["timeseries"])
            custom_metrics.extend(run_rows["custom_metrics"])

    write_rows_csv(data_dir / "benchmark_runs.csv", runs, BENCHMARK_RUN_FIELDS)
    write_rows_csv(data_dir / "benchmark_latencies.csv", latencies, BENCHMARK_LATENCY_FIELDS)
    write_rows_csv(data_dir / "timeseries.csv", timeseries, TIMESERIES_FIELDS)
    write_rows_csv(data_dir / "custom_metrics.csv", custom_metrics, CUSTOM_METRIC_FIELDS)
    _write_manifest(run_dir, cfg, config_path=config_path, dataset_label=dataset_label)

    if make_plots:
        plot_gamma_sweep(run_dir)
    return run_dir


def plot_gamma_sweep(run_dir: str | Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/candor_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    data_dir = run_dir / "data"
    figures_dir = run_dir / "figures" / "main"
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(data_dir / "benchmark_runs.csv")
    latencies = pd.read_csv(data_dir / "benchmark_latencies.csv")
    outputs: list[Path] = []

    outputs.append(
        _plot_gamma_runs(
            runs,
            figures_dir / "gamma_vs_throughput.png",
            y_column="system_ops_per_sec",
            y_label="System ops/sec",
            title="Gamma Sweep Throughput",
        )
    )
    outputs.append(_plot_recall_adjusted_throughput(runs, figures_dir / "gamma_vs_recall_adjusted_throughput.png"))
    outputs.append(
        _plot_gamma_latency(
            latencies,
            figures_dir / "gamma_vs_query_latency.png",
            op_type="query_latency",
            y_label="Average query latency (ms)",
            title="Gamma Sweep Query Latency",
            empty_message="No query latency data available",
        )
    )
    outputs.append(
        _plot_gamma_latency(
            latencies,
            figures_dir / "gamma_vs_insert_latency.png",
            op_type="insert_latency",
            y_label="Average insert latency (ms)",
            title="Gamma Sweep Insert Latency",
            empty_message="No insert latency data available",
        )
    )
    outputs.append(
        _plot_gamma_latency(
            latencies,
            figures_dir / "gamma_vs_delete_latency.png",
            op_type="delete_latency",
            y_label="Average delete latency (ms)",
            title="Gamma Sweep Delete Latency",
            empty_message="No delete latency data available",
        )
    )

    return outputs


def _plot_gamma_runs(
    runs: pd.DataFrame,
    out: Path,
    *,
    y_column: str,
    y_label: str,
    title: str,
    empty_message: str = "No run data available",
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    valid = runs.dropna(subset=["gamma", y_column]).copy() if y_column in runs else pd.DataFrame()
    if valid.empty:
        _annotate_empty_axis(ax, empty_message)
    else:
        for index_name, frame in valid.groupby("index"):
            frame = frame.sort_values("gamma")
            ax.plot(frame["gamma"], frame[y_column], marker="o", label=index_name)
        ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("Gamma (Q / writes)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_recall_adjusted_throughput(runs: pd.DataFrame, out: Path) -> Path:
    frame = runs.copy()
    if "recall" in frame:
        frame["recall"] = pd.to_numeric(frame["recall"], errors="coerce")
    else:
        frame["recall"] = math.nan
    frame["recall_adjusted_ops_per_sec"] = frame["system_ops_per_sec"] * frame["recall"]
    return _plot_gamma_runs(
        frame,
        out,
        y_column="recall_adjusted_ops_per_sec",
        y_label="System ops/sec x recall",
        title="Gamma Sweep Recall-Adjusted Throughput",
        empty_message="Recall data unavailable",
    )


def _plot_gamma_latency(
    latencies: pd.DataFrame,
    out: Path,
    *,
    op_type: str,
    y_label: str,
    title: str,
    empty_message: str,
) -> Path:
    import matplotlib.pyplot as plt

    frame = latencies[latencies["op_type"] == op_type].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    if frame.empty:
        _annotate_empty_axis(ax, empty_message)
    else:
        for index_name, group in frame.groupby("index"):
            group = group.sort_values("gamma")
            ax.plot(group["gamma"], group["avg_latency_ms"], marker="o", label=index_name)
        ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("Gamma (Q / writes)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _annotate_empty_axis(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def _run_one_index_gamma(
    *,
    cfg: GammaSweepConfig,
    gamma: float,
    index_name: str,
    vectors: np.ndarray,
    runner_dataset: Dataset,
    dataset_label: str,
    sequence: list[GammaOperation],
) -> dict[str, list[dict[str, Any]] | dict[str, Any]]:
    algorithm = get_algorithm(index_name, dataset=cfg.algorithm_dataset_key)
    run_id = f"{index_name}__gamma_{_gamma_label(gamma)}__seed_{cfg.random_seed}"

    runner = BenchmarkRunner(
        algorithm=algorithm,
        dataset=runner_dataset,
        k=cfg.topk,
        num_workers=cfg.threads,
        save_timestamps=False,
        output_dir="",
        use_worker=False,
    )
    sequence_result = runner.run_operation_sequence(
        sequence,
        vectors,
        run_id=run_id,
        index_name=index_name,
        compute_recall=cfg.compute_recall,
    )

    duration = float(sequence_result["duration_sec"])
    op_counts = sequence_result["op_counts"]
    op_latencies = sequence_result["op_latencies"]
    total_measurement_ops = sum(op_counts.values())
    system_ops_per_sec = total_measurement_ops / duration if duration > 0 else 0.0

    run_row = {
        "run_id": run_id,
        "timestamp": int(time.time() * 1000),
        "index": index_name,
        "gamma": gamma,
        "dataset": dataset_label,
        "dataset_size": cfg.dataset_size,
        "dim": cfg.dim,
        "topk": cfg.topk,
        "operations": cfg.operations,
        "prefill_ratio": cfg.prefill_ratio,
        "zipf": cfg.zipf_alpha,
        "delete_ratio": cfg.delete_ratio,
        "threads": cfg.threads,
        "compute_recall": cfg.compute_recall,
        "random_seed": cfg.random_seed,
        "total_duration_sec": duration,
        "total_operations": total_measurement_ops,
        "insert_count": op_counts["insert"],
        "delete_count": op_counts["delete"],
        "query_count": op_counts["query"],
        "system_ops_per_sec": system_ops_per_sec,
        "recall": sequence_result["recall"],
    }

    latency_rows = [
        _latency_summary(run_id, index_name, gamma, op_type, values)
        for op_type, values in op_latencies.items()
        if values
    ]
    custom_rows = [
        {"run_id": run_id, "key": "prefill_count", "value": int(cfg.dataset_size * cfg.prefill_ratio)},
        {"run_id": run_id, "key": "measurement_insert_count", "value": op_counts["insert"]},
        {"run_id": run_id, "key": "measurement_delete_count", "value": op_counts["delete"]},
        {"run_id": run_id, "key": "measurement_query_count", "value": op_counts["query"]},
        {"run_id": run_id, "key": "final_live_count", "value": sequence_result["final_live_count"]},
        {"run_id": run_id, "key": "mean_recall", "value": sequence_result["recall"]},
    ]

    return {
        "run": run_row,
        "latencies": latency_rows,
        "timeseries": sequence_result["timeseries"],
        "custom_metrics": custom_rows,
    }


def _resolve_vectors(
    cfg: GammaSweepConfig,
    dataset: Dataset | None,
    dataset_name: str | None,
) -> tuple[GammaSweepConfig, np.ndarray, Dataset, str]:
    if dataset is None:
        rng = np.random.default_rng(cfg.random_seed)
        vectors = rng.standard_normal((cfg.dataset_size, cfg.dim), dtype=np.float32)
        resolved_name = dataset_name or "synthetic"
        algorithm_dataset_key = dataset_name or cfg.algorithm_dataset_key
        runner_dataset = _GammaVectorDataset(vectors, resolved_name)
        return replace(cfg, algorithm_dataset_key=algorithm_dataset_key), vectors, runner_dataset, resolved_name

    dataset.prepare()
    data = np.asarray(dataset.get_dataset(), dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Dataset {dataset.short_name()} must return a 2D dense array")
    if data.shape[0] == 0:
        raise ValueError(f"Dataset {dataset.short_name()} returned no vectors")

    dataset_size = min(cfg.dataset_size, int(data.shape[0]))
    if dataset_size <= 0:
        raise ValueError("gamma_sweep.dataset_size must leave at least one vector")

    vectors = np.ascontiguousarray(data[:dataset_size], dtype=np.float32)
    resolved_name = dataset_name or dataset.short_name()
    resolved_cfg = replace(
        cfg,
        dataset_size=dataset_size,
        dim=int(vectors.shape[1]),
        algorithm_dataset_key=resolved_name,
    )
    runner_dataset = _GammaVectorDataset(vectors, resolved_name, distance=dataset.distance())
    return resolved_cfg, vectors, runner_dataset, resolved_name


def _latency_summary(
    run_id: str,
    index_name: str,
    gamma: float,
    op_type: str,
    values: list[float],
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    total_sec = float(arr.sum() / 1000.0)
    return {
        "run_id": run_id,
        "index": index_name,
        "gamma": gamma,
        "op_type": op_type,
        "count": int(arr.size),
        "avg_latency_ms": float(arr.mean()),
        "p50_latency_ms": float(np.percentile(arr, 50)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "p99_latency_ms": float(np.percentile(arr, 99)),
        "min_latency_ms": float(arr.min()),
        "max_latency_ms": float(arr.max()),
        "qps": float(arr.size / total_sec) if total_sec > 0 else 0.0,
    }


def _zipf_weights(size: int, alpha: float) -> np.ndarray | None:
    if alpha <= 0.0:
        return None
    ranks = np.arange(1, size + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, alpha)
    weights /= weights.sum()
    return weights


def _sample_query_id(
    rng: np.random.Generator,
    dataset_size: int,
    weights: np.ndarray | None,
) -> int:
    if weights is None:
        return int(rng.integers(0, dataset_size))
    return int(rng.choice(dataset_size, p=weights))


def _gamma_label(gamma: float) -> str:
    return f"{float(gamma):.6g}".replace("-", "m").replace(".", "p")


def _write_operation_sequence(path: Path, sequence: Iterable[GammaOperation]) -> None:
    rows = [
        {
            "sequence_index": idx,
            "phase": operation.phase,
            "type": operation.op_type,
            "target_id": operation.target_id,
        }
        for idx, operation in enumerate(sequence, start=1)
    ]
    write_rows_csv(path, rows, ["sequence_index", "phase", "type", "target_id"])


def _write_manifest(
    run_dir: Path,
    cfg: GammaSweepConfig,
    *,
    config_path: str | Path | None,
    dataset_label: str,
) -> None:
    manifest = {
        "experiment": "gamma_sweep",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path) if config_path else None,
        "dataset": dataset_label,
        "outputs": {
            "benchmark_runs": "data/benchmark_runs.csv",
            "benchmark_latencies": "data/benchmark_latencies.csv",
            "timeseries": "data/timeseries.csv",
            "custom_metrics": "data/custom_metrics.csv",
        },
        "config": {
            "dataset_size": cfg.dataset_size,
            "operations": cfg.operations,
            "dim": cfg.dim,
            "topk": cfg.topk,
            "gamma_values": cfg.gamma_values,
            "indices": cfg.indices,
            "random_seed": cfg.random_seed,
            "prefill_ratio": cfg.prefill_ratio,
            "zipf_alpha": cfg.zipf_alpha,
            "delete_ratio": cfg.delete_ratio,
            "threads": cfg.threads,
            "compute_recall": cfg.compute_recall,
            "algorithm_dataset_key": cfg.algorithm_dataset_key,
        },
        "notes": [
            "This CANDOR-Bench runner uses the algorithm registry and BaseStreamingANN API.",
            "GammaFresh internals are not modified by this experiment path.",
            "threads is recorded in this first version; execution is serial.",
            (
                "recall is exact dynamic Recall@k against the live vector set at each measurement query."
                if cfg.compute_recall
                else "recall computation is disabled; recall fields are NaN and query latency excludes exact recall."
            ),
        ],
    }
    write_manifest_json(run_dir, manifest)


BENCHMARK_RUN_FIELDS = [
    "run_id",
    "timestamp",
    "index",
    "gamma",
    "dataset",
    "dataset_size",
    "dim",
    "topk",
    "operations",
    "prefill_ratio",
    "zipf",
    "delete_ratio",
    "threads",
    "compute_recall",
    "random_seed",
    "total_duration_sec",
    "total_operations",
    "insert_count",
    "delete_count",
    "query_count",
    "system_ops_per_sec",
    "recall",
]

BENCHMARK_LATENCY_FIELDS = [
    "run_id",
    "index",
    "gamma",
    "op_type",
    "count",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "qps",
]

TIMESERIES_FIELDS = [
    "run_id",
    "timestamp_sec",
    "index_type",
    "op_type",
    "latency_ms",
    "current_size",
    "workload_op_seq",
    "target_id",
    "phase",
]

CUSTOM_METRIC_FIELDS = ["run_id", "key", "value"]
