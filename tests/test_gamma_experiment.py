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
    assert cfg.compute_recall is True


def test_gamma_config_can_disable_recall() -> None:
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
            "compute_recall": False,
        }
    )
    assert cfg.compute_recall is False


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
    # 2 inserts (prefill + measurement) + 1 query + 1 recall_eval + 1 delete
    # = 5 timeseries entries (recall_eval split was added in commit 0c0dc625)
    assert len(result["timeseries"]) == 5


def test_benchmark_runner_records_recall_eval_latency_separately(monkeypatch) -> None:
    import time

    def slow_recall(**kwargs):
        time.sleep(0.03)
        return 1.0

    monkeypatch.setattr("bench.runner._exact_dynamic_recall", slow_recall)

    vectors = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype="float32",
    )
    dataset = _GammaVectorDataset(vectors, "unit-gamma-recall")
    runner = BenchmarkRunner(
        algorithm=DummyStreamingANN(),
        dataset=dataset,
        k=1,
        save_timestamps=False,
        use_worker=False,
    )

    result = runner.run_operation_sequence(
        [
            {"type": "insert", "target_id": 0, "phase": "prefill"},
            {"type": "query", "target_id": 0, "phase": "measurement"},
        ],
        vectors,
        run_id="unit-recall-timing",
        index_name="dummy",
        compute_recall=True,
    )

    query_latency = result["op_latencies"]["query_latency"][0]
    recall_latency = result["op_latencies"]["recall_eval_latency"][0]

    assert result["recall"] == 1.0
    assert recall_latency >= 25.0
    assert query_latency < recall_latency
    assert result["duration_sec"] < result["wall_duration_sec"]
    assert any(row["op_type"] == "recall_eval_latency" for row in result["timeseries"])


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
    assert runs["compute_recall"].eq(True).all()
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


def test_gamma_sweep_can_skip_exact_recall(tmp_path: Path) -> None:
    cfg = GammaSweepConfig(
        dataset_size=32,
        operations=12,
        dim=4,
        topk=2,
        gamma_values=[1.0],
        indices=["dummy"],
        random_seed=3,
        prefill_ratio=0.5,
        zipf_alpha=0.0,
        delete_ratio=0.5,
        threads=1,
        compute_recall=False,
    )
    run_dir = run_gamma_sweep(cfg, tmp_path, make_plots=False)
    runs = pd.read_csv(run_dir / "data" / "benchmark_runs.csv")

    assert runs["compute_recall"].eq(False).all()
    assert runs["recall"].isna().all()


def test_gamma_sweep_applies_algorithm_query_arguments(tmp_path: Path, monkeypatch) -> None:
    class QueryArgTrackingANN(DummyStreamingANN):
        def __init__(self):
            super().__init__()
            self.query_args = None

        def set_query_arguments(self, query_args):
            self.query_args = dict(query_args)

    created = []

    def fake_get_algorithm(name, dataset="random-xs", **kwargs):
        algorithm = QueryArgTrackingANN()
        created.append(
            {
                "name": name,
                "dataset": dataset,
                "kwargs": kwargs,
                "algorithm": algorithm,
            }
        )
        return algorithm

    def fake_get_algorithm_params_from_config(name, dataset="random-xs"):
        return {
            "build_params": {"indexkey": "unit-index"},
            "query_params": {"ef": 123},
        }

    monkeypatch.setattr("bench.gamma_experiment.get_algorithm", fake_get_algorithm)
    monkeypatch.setattr(
        "bench.gamma_experiment.get_algorithm_params_from_config",
        fake_get_algorithm_params_from_config,
    )

    cfg = GammaSweepConfig(
        dataset_size=16,
        operations=8,
        dim=4,
        topk=2,
        gamma_values=[1.0],
        indices=["unit_algo"],
        random_seed=11,
        prefill_ratio=0.5,
        zipf_alpha=0.0,
        delete_ratio=0.5,
        threads=1,
        compute_recall=False,
        algorithm_dataset_key="unit-dataset",
    )

    run_gamma_sweep(cfg, tmp_path, make_plots=False)

    assert created == [
        {
            "name": "unit_algo",
            "dataset": "unit-dataset",
            "kwargs": {"index_params": {"indexkey": "unit-index"}},
            "algorithm": created[0]["algorithm"],
        }
    ]
    assert created[0]["algorithm"].query_args == {"ef": 123}
