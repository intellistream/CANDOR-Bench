from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bench.algorithms.base import DummyStreamingANN
from bench.gamma_experiment import (
    GammaSweepConfig,
    _GammaVectorDataset,
    generate_gamma_operation_sequence,
    plot_gamma_sweep,
    run_gamma_sweep,
)
from bench.runner import BenchmarkRunner


def _small_config() -> GammaSweepConfig:
    return GammaSweepConfig(
        dataset_size=64,
        operations=40,
        dim=8,
        topk=3,
        gamma_values=[0.1, 2.0],
        indices=["dummy"],
        random_seed=7,
        prefill_ratio=0.5,
        zipf_alpha=0.0,
        delete_ratio=0.5,
        threads=1,
    )


def test_gamma_config_validation() -> None:
    cfg = GammaSweepConfig.from_dict(
        {
            "dataset_size": 100,
            "operations": 10,
            "dim": 4,
            "topk": 2,
            "gamma_values": [0.01, 1.0],
            "indices": ["dummy"],
            "random_seed": 1,
            "prefill_ratio": 0.5,
            "zipf_alpha": 0.0,
            "delete_ratio": 0.5,
            "threads": 1,
        }
    )
    assert cfg.dataset_size == 100
    assert cfg.gamma_values == [0.01, 1.0]


def test_gamma_config_accepts_algorithms_alias() -> None:
    cfg = GammaSweepConfig.from_dict(
        {
            "dataset_size": 100,
            "operations": 10,
            "topk": 2,
            "gamma_values": [1.0],
            "algorithms": ["dummy"],
            "random_seed": 1,
            "prefill_ratio": 0.5,
            "zipf_alpha": 0.0,
            "delete_ratio": 0.5,
        }
    )
    assert cfg.indices == ["dummy"]
    assert cfg.dim == 32
    assert cfg.threads == 1


def test_gamma_operation_sequence_is_deterministic() -> None:
    cfg = _small_config()
    first = generate_gamma_operation_sequence(cfg, gamma=1.0)
    second = generate_gamma_operation_sequence(cfg, gamma=1.0)

    assert first == second
    assert len(first) == int(cfg.dataset_size * cfg.prefill_ratio) + cfg.operations
    assert any(op.op_type == "query" for op in first if op.phase == "measurement")
    assert any(op.op_type in {"insert", "delete"} for op in first if op.phase == "measurement")


def test_benchmark_runner_executes_generated_operation_sequence() -> None:
    vectors = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype="float32",
    )
    dataset = _GammaVectorDataset(vectors, "unit-gamma")
    runner = BenchmarkRunner(
        algorithm=DummyStreamingANN(),
        dataset=dataset,
        k=2,
        save_timestamps=False,
        use_worker=False,
    )

    result = runner.run_operation_sequence(
        [
            {"type": "insert", "target_id": 0, "phase": "prefill"},
            {"type": "insert", "target_id": 1, "phase": "measurement"},
            {"type": "query", "target_id": 0, "phase": "measurement"},
            {"type": "delete", "target_id": 1, "phase": "measurement"},
        ],
        vectors,
        run_id="unit-run",
        index_name="dummy",
        compute_recall=True,
    )

    assert result["op_counts"] == {"insert": 1, "delete": 1, "query": 1}
    assert result["final_live_count"] == 1
    assert result["recall"] == 1.0
    assert len(result["timeseries"]) == 4


def test_benchmark_runner_batches_prefill_initial_load() -> None:
    class InitialLoadTrackingANN(DummyStreamingANN):
        def __init__(self):
            super().__init__()
            self.initial_load_sizes = []

        def initial_load(self, X, ids):
            self.initial_load_sizes.append(len(ids))
            return super().initial_load(X, ids)

    vectors = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype="float32",
    )
    dataset = _GammaVectorDataset(vectors, "unit-gamma-prefill")
    algorithm = InitialLoadTrackingANN()
    runner = BenchmarkRunner(
        algorithm=algorithm,
        dataset=dataset,
        k=2,
        save_timestamps=False,
        use_worker=False,
    )

    result = runner.run_operation_sequence(
        [
            {"type": "insert", "target_id": 0, "phase": "prefill"},
            {"type": "insert", "target_id": 1, "phase": "prefill"},
            {"type": "query", "target_id": 0, "phase": "measurement"},
        ],
        vectors,
        run_id="unit-prefill-run",
        index_name="dummy",
    )

    assert algorithm.initial_load_sizes == [2]
    assert result["op_counts"] == {"insert": 0, "delete": 0, "query": 1}
    assert result["final_live_count"] == 2
    assert len(result["timeseries"]) == 3


def test_gamma_sweep_writes_standard_outputs_and_plots(tmp_path: Path) -> None:
    cfg = _small_config()
    run_dir = run_gamma_sweep(cfg, tmp_path, make_plots=True)

    data_dir = run_dir / "data"
    expected = [
        data_dir / "benchmark_runs.csv",
        data_dir / "benchmark_latencies.csv",
        data_dir / "timeseries.csv",
        data_dir / "custom_metrics.csv",
        run_dir / "manifest.json",
    ]
    for path in expected:
        assert path.exists(), path

    runs = pd.read_csv(data_dir / "benchmark_runs.csv")
    latencies = pd.read_csv(data_dir / "benchmark_latencies.csv")
    timeseries = pd.read_csv(data_dir / "timeseries.csv")

    assert len(runs) == len(cfg.gamma_values) * len(cfg.indices)
    assert runs["recall"].notna().all()
    assert (runs["recall"] == 1.0).all()
    assert {"query_latency", "insert_latency", "delete_latency"} & set(latencies["op_type"])
    assert not timeseries.empty

    plot_outputs = plot_gamma_sweep(run_dir)
    assert {path.name for path in plot_outputs} == {
        "gamma_vs_throughput.png",
        "gamma_vs_recall_adjusted_throughput.png",
        "gamma_vs_query_latency.png",
        "gamma_vs_insert_latency.png",
        "gamma_vs_delete_latency.png",
    }
    for path in plot_outputs:
        assert path.exists(), path
