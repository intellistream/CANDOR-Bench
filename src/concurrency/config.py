"""YAML config loading and variant expansion.

The schema is unchanged from earlier releases, including the sweep
mechanics: any scalar field given as a list becomes a sweep axis (cross
product across all such fields), and ``workload.rate_groups(r/w)`` pairs
``[search_rate, insert_rate]`` expand into one variant each.
"""

from __future__ import annotations

import copy
import itertools
import os
from typing import Any, Dict, List, Tuple

import yaml

QUERY_MODES = {"round_robin", "chasing", "peeking", "zipfian"}


def _ensure(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    if not isinstance(d.get(key), dict):
        d[key] = {}
    return d[key]


def _extract_rate_groups(raw: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Returns [(insert_rate, search_rate), ...]; YAML pairs are [r, w]."""
    workload = raw.get("workload")
    if not isinstance(workload, dict):
        return []
    groups_raw = workload.pop("rate_groups(r/w)", None)
    workload.pop("rate_groups", None)
    if not groups_raw:
        return []
    workload.pop("insert_event_rate", None)
    workload.pop("search_event_rate", None)
    groups: List[Tuple[float, float]] = []
    for entry in groups_raw:
        if isinstance(entry, dict):
            insert = entry.get(
                "insert", entry.get("write", entry.get("w", 0.0))
            )
            search = entry.get("search", entry.get("read", entry.get("r", 0.0)))
            groups.append((float(insert), float(search)))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            groups.append((float(entry[1]), float(entry[0])))
    return groups


def _collect_sweep_params(node: Any, path: Tuple[str, ...] = ()) -> List[Tuple[Tuple[str, ...], List[Any]]]:
    # rate_groups keys never reach this walk: _extract_rate_groups pops
    # them from the raw config first.
    params = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict):
                params.extend(_collect_sweep_params(value, path + (key,)))
            elif isinstance(value, list) and value and all(
                not isinstance(v, (dict, list)) for v in value
            ):
                params.append((path + (key,), value))
    return params


def _set_path(d: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


def load_config_variants(path: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Expand a config file into [(variant_name, config_dict), ...]."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    base_name = os.path.splitext(os.path.basename(path))[0]

    rate_groups = _extract_rate_groups(raw)
    sweep_params = _collect_sweep_params(raw)

    if not sweep_params and not rate_groups:
        return [(base_name, _normalize(raw))]

    combos = (
        list(itertools.product(*(range(len(v)) for _, v in sweep_params)))
        if sweep_params
        else [()]
    )

    variants: List[Tuple[str, Dict[str, Any]]] = []
    for combo in combos:
        materialized = copy.deepcopy(raw)
        labels = []
        for (param_path, values), idx in zip(sweep_params, combo):
            _set_path(materialized, param_path, values[idx])
            labels.append(f"{param_path[-1]}{values[idx]}")
        sweep_label = "__".join(labels)

        if rate_groups:
            for insert_rate, search_rate in rate_groups:
                variant = copy.deepcopy(materialized)
                workload = _ensure(variant, "workload")
                workload["insert_event_rate"] = insert_rate
                workload["search_event_rate"] = search_rate
                label = f"w{insert_rate:g}_r{search_rate:g}"
                if sweep_label:
                    label += "__" + sweep_label
                variants.append((label, _normalize(variant)))
        else:
            variants.append((sweep_label or base_name, _normalize(materialized)))
    return variants


def _normalize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the query-path fallbacks and config validation."""
    data = _ensure(cfg, "data")
    workload = _ensure(cfg, "workload")
    _ensure(cfg, "index")
    _ensure(cfg, "search")
    _ensure(cfg, "result")
    _ensure(cfg, "profile")

    if not data.get("incr_query_path"):
        data["incr_query_path"] = data.get("query_path") or data.get(
            "overall_query_path"
        )
    if not data.get("overall_query_path"):
        data["overall_query_path"] = data.get("query_path") or data.get(
            "incr_query_path"
        )
    if not data.get("incr_query_path"):
        raise ValueError(
            "incr_query_path (or query_path/overall_query_path) must be provided"
        )

    batch_size = int(workload.get("batch_size", 0))
    if batch_size <= 0:
        raise ValueError("workload.batch_size must be set and greater than 0")
    if batch_size % 10 != 0:
        raise ValueError(
            f"workload.batch_size ({batch_size}) must be a multiple of 10"
        )
    if float(workload.get("insert_event_rate", 0) or 0) < 0:
        raise ValueError("insert_event_rate cannot be negative")
    if float(workload.get("search_event_rate", 0) or 0) < 0:
        raise ValueError("search_event_rate cannot be negative")

    workload.setdefault("query_mode", "round_robin")
    if workload["query_mode"] not in QUERY_MODES:
        raise ValueError(f"unsupported query_mode {workload['query_mode']}")
    return cfg
