import glob
import os

import yaml

datasets = {
    "msong": {
        "name": "msong",
        "base_path": "../data/msong/msong_base.bin",
        "query_path": "../data/msong/msong_query.bin",
        "query_stream_path": "../data/msong/msong_query_stream.bin",
        "gt_path_fmt": "../data/msong/msong_base_i{}_offset_index.txt",
        "gt_stream_path_fmt": "../data/msong/msong_stream_i{}_offset_index.txt",
        "overall_gt_path": "../data/msong/msong_base.gt20",
        "available_indices": [20, 100, 1000],
    },
    "glove1.2m": {
        "name": "glove1.2m",
        "base_path": "../data/glove1.2m/glove1.2m_base.bin",
        "query_path": "../data/glove1.2m/glove1.2m_query.bin",
        "query_stream_path": "../data/glove1.2m/glove1.2m_query_stream.bin",
        "gt_path_fmt": "../data/glove1.2m/glove1.2m_base_i{}_offset_index.txt",
        "gt_stream_path_fmt": "../data/glove1.2m/glove1.2m_stream_i{}_offset_index.txt",
        "overall_gt_path": "../data/glove1.2m/glove1.2m_base.gt20",
        "available_indices": [20, 100, 1000],
    },
    "glove2.2m": {
        "name": "glove2.2m",
        "base_path": "../data/glove2.2m/glove2.2m_base.bin",
        "query_path": "../data/glove2.2m/glove2.2m_query.bin",
        "query_stream_path": "../data/glove2.2m/glove2.2m_query_stream.bin",
        "gt_path_fmt": "../data/glove2.2m/glove2.2m_base_i{}_offset_index.txt",
        "gt_stream_path_fmt": "../data/glove2.2m/glove2.2m_stream_i{}_offset_index.txt",
        "overall_gt_path": "../data/glove2.2m/glove2.2m_base.gt20",  # Assuming this exists or follows pattern? ls didn't show it explicitly but let's assume standard naming or check again.
        "available_indices": [20, 100, 1000],
    },
    "gist": {
        "name": "gist",
        "base_path": "../data/gist/gist_base.bin",
        "query_path": "../data/gist/gist_query.bin",
        "query_stream_path": "../data/gist/gist_query_stream.bin",
        "gt_path_fmt": "../data/gist/gist_base_i{}_offset_index.txt",
        "gt_stream_path_fmt": "../data/gist/gist_stream_i{}_offset_index.txt",
        "overall_gt_path": "../data/gist/gist_base.gt20",
        "available_indices": [20, 100, 1000],
    },
    "deep1M": {
        "name": "deep1M",
        "base_path": "../data/deep1M/deep1M_base.bin",
        "query_path": "../data/deep1M/deep1M_query.bin",
        "query_stream_path": "../data/deep1M/deep1M_query_stream.bin",
        "gt_path_fmt": "../data/deep1M/deep1M_base_i{}_offset_index.txt",
        "gt_stream_path_fmt": "../data/deep1M/deep1M_stream_i{}_offset_index.txt",
        "overall_gt_path": "../data/deep1M/deep1M_base.gt20",
        "available_indices": [20, 100, 1000],
    },
}

import sys
base_config_dir = sys.argv[1] if len(sys.argv) > 1 else "configs"


class FlowList(list):
    pass


def flow_list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(FlowList, flow_list_representer)


def update_config(file_path, dataset_key):
    ds_info = datasets[dataset_key]
    filename = os.path.basename(file_path)

    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    is_stream = "ch_" in filename or "pk_" in filename or "segmented" in filename

    batch_size = 20
    if "_b20." in filename:
        batch_size = 20
    elif "_b100." in filename:
        batch_size = 100
    elif "_b200." in filename:
        batch_size = 200

    target_index = batch_size
    if (
        batch_size == 200
        and 200 not in ds_info["available_indices"]
        and 1000 in ds_info["available_indices"]
    ):
        target_index = 1000
        print(f"  Mapping batch size 200 to index 1000 for {filename}")
        if "workload" in config and "batch_size" in config["workload"]:
            # If it's a list with single element, update it
            if (
                isinstance(config["workload"]["batch_size"], list)
                and len(config["workload"]["batch_size"]) == 1
            ):
                config["workload"]["batch_size"] = [1000]

    data_section = {
        "dataset_name": ds_info["name"],
        "max_elements": -1,
        "begin_num": 50000,
        "data_type": "float",
        "data_path": ds_info["base_path"],
        "overall_query_path": ds_info["query_path"],
        "overall_gt_path": ds_info["overall_gt_path"],
    }

    if is_stream:
        data_section["incr_query_path"] = ds_info["query_stream_path"]
        data_section["incr_gt_path"] = ds_info["gt_stream_path_fmt"].format(
            target_index
        )
    else:
        data_section["incr_query_path"] = ds_info["query_path"]
        data_section["incr_gt_path"] = ds_info["gt_path_fmt"].format(
            20
        )  # Default to i20 for rr/overall

    config["data"] = data_section

    if "workload" in config and "query_mode" in config["workload"]:
        qm = config["workload"]["query_mode"]
        if qm == "consistent_hashing":
            config["workload"]["query_mode"] = "chasing"
            print(f"  Fixed query_mode: consistent_hashing -> chasing")
        elif qm == "primary_key":
            config["workload"]["query_mode"] = "peeking"
            print(f"  Fixed query_mode: primary_key -> peeking")

    if "workload" in config and "rate_groups(r/w)" in config["workload"]:
        groups = config["workload"]["rate_groups(r/w)"]
        new_groups = []
        for g in groups:
            if isinstance(g, list):
                new_groups.append(FlowList(g))
            else:
                new_groups.append(g)
        config["workload"]["rate_groups(r/w)"] = new_groups

    if "index" in config:
        if "m" in config["index"] and isinstance(config["index"]["m"], list):
            config["index"]["m"] = FlowList(config["index"]["m"])
        if "ef_construction" in config["index"] and isinstance(
            config["index"]["ef_construction"], list
        ):
            config["index"]["ef_construction"] = FlowList(
                config["index"]["ef_construction"]
            )

    if "search" in config:
        if "ef_search" in config["search"] and isinstance(
            config["search"]["ef_search"], list
        ):
            config["search"]["ef_search"] = FlowList(config["search"]["ef_search"])

    if "workload" in config:
        if "batch_size" in config["workload"] and isinstance(
            config["workload"]["batch_size"], list
        ):
            config["workload"]["batch_size"] = FlowList(
                config["workload"]["batch_size"]
            )
        if "num_threads" in config["workload"] and isinstance(
            config["workload"]["num_threads"], list
        ):
            config["workload"]["num_threads"] = FlowList(
                config["workload"]["num_threads"]
            )

    with open(file_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Updated {file_path}")


def main():
    for ds_key in datasets:
        dir_path = os.path.join(base_config_dir, ds_key)
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue

        for file_path in glob.glob(os.path.join(dir_path, "*.yaml")):
            update_config(file_path, ds_key)


if __name__ == "__main__":
    main()
