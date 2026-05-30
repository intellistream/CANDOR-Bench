#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "knn_processor.h"
#include "utils.hpp"

using PointPair = std::pair<int, float>;

struct BatchOutputPaths {
    std::filesystem::path parent_dir;
    std::filesystem::path slices_dir;
    std::string stem;
};

BatchOutputPaths prepare_batch_output_paths(const std::string &base_path) {
    std::filesystem::path base(base_path);
    std::filesystem::path parent = base.parent_path();

    std::string stem = base.stem().string();
    if (stem.empty()) {
        auto filename = base.filename();
        if (!filename.empty()) {
            stem = filename.string();
        } else if (!parent.empty()) {
            auto parent_name = parent.filename();
            if (!parent_name.empty()) {
                stem = parent_name.string();
            }
        }
    }
    if (stem.empty()) {
        stem = "groundtruth";
    }

    std::filesystem::path slices_dir =
        parent.empty() ? std::filesystem::path(stem + "_slices")
                       : parent / (stem + "_slices");

    std::error_code ec;
    std::filesystem::create_directories(slices_dir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create slices directory '" +
                                 slices_dir.string() +
                                 "': " + ec.message());
    }

    return {parent, slices_dir, stem};
}

std::string generate_batch_filename(const BatchOutputPaths &paths,
                                    int start_offset, int end_offset, int k) {
    std::ostringstream oss;
    oss << paths.stem << "_offset_" << start_offset << "_" << end_offset
        << ".gt" << k;
    return (paths.slices_dir / oss.str()).string();
}

std::string generate_index_filename(const BatchOutputPaths &paths) {
    std::ostringstream oss;
    oss << paths.stem << "_offset_index.txt";
    if (paths.parent_dir.empty()) {
        return oss.str();
    }
    return (paths.parent_dir / oss.str()).string();
}

std::vector<PointPair> exact_knn(const std::vector<float> &query,
                                 const std::vector<std::vector<float>> &base,
                                 size_t b_size, int k, int query_idx = -1) {
    using HeapPair = std::pair<float, int>;  // (distance, point_id)
    std::priority_queue<HeapPair, std::vector<HeapPair>, std::less<HeapPair>>
        max_heap;

    for (size_t j = 0; j < b_size && j < base.size(); ++j) {
        if (static_cast<int>(j) == query_idx) {
            continue;
        }

        float dist = euclidean_distance_simd(query, base[j]);
        if (max_heap.size() < static_cast<size_t>(k)) {
            max_heap.emplace(dist, static_cast<int>(j));
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, static_cast<int>(j));
        }
    }

    std::vector<PointPair> topk;
    topk.reserve(k);
    while (!max_heap.empty()) {
        topk.emplace_back(max_heap.top().second, max_heap.top().first);
        max_heap.pop();
    }
    std::reverse(topk.begin(), topk.end());

    while (topk.size() < static_cast<size_t>(k)) {
        topk.emplace_back(-1, std::numeric_limits<float>::infinity());
    }

    return topk;
}

struct Args {
    std::string base_path;
    std::string query_path;
    std::string batch_gt_path;
    int k = 20;
    int increment = 10;
    int num_threads = 0;
    int max_batches_per_file = 1000;
    int start_offset = 0;
    bool use_gpu = false;
    bool stream = false;
};

void print_help() {
    std::cout
        << "Usage: compute_gt [options]\n"
        << "Options:\n"
        << "  --base_path PATH         Path to base vectors file (required)\n"
        << "  --query_path PATH        Path to query vectors file (required)\n"
        << "  --batch_gt_path PATH     Path to save batch groundtruth "
           "(required)\n"
        << "  --k K                    Number of nearest neighbors (default: "
           "20)\n"
        << "  --inc INCREMENT          Increment size for batch processing "
           "(default: 10)\n"
        << "  --threads N              Number of threads to use (default: 0, "
           "use system default)\n"
        << "  --max_batches_per_file N Max batches per output file (default: "
           "1000)\n"
        << "  --gpu                    Enable CUDA acceleration if available\n"
        << "  --stream               Restrict queries to the most recent "
           "increment window\n"
        << "  --help                   Show this help message\n";
}

Args parse_args(int argc, char *argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "--h" || arg == "-h") {
            print_help();
            exit(0);
        } else if (arg == "--base_path" && i + 1 < argc)
            args.base_path = argv[++i];
        else if (arg == "--query_path" && i + 1 < argc)
            args.query_path = argv[++i];
        else if (arg == "--batch_gt_path" && i + 1 < argc)
            args.batch_gt_path = argv[++i];
        else if (arg == "--k" && i + 1 < argc)
            args.k = std::stoi(argv[++i]);
        else if (arg == "--inc" && i + 1 < argc)
            args.increment = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc)
            args.num_threads = std::stoi(argv[++i]);
        else if (arg == "--max_batches_per_file" && i + 1 < argc)
            args.max_batches_per_file = std::stoi(argv[++i]);
        else if (arg == "--start_offset" && i + 1 < argc)
            args.start_offset = std::stoi(argv[++i]);
        else if (arg == "--gpu")
            args.use_gpu = true;
        else if (arg == "--stream")
            args.stream = true;
        else {
            std::cerr << "Error: Unknown argument '" << arg << "'" << std::endl;
            exit(1);
        }
    }
    return args;
}

int main(int argc, char *argv[]) {
    Args args = parse_args(argc, argv);

    if (args.base_path.empty() || args.batch_gt_path.empty()) {
        std::cerr << "Error: --base_path and --batch_gt_path are required."
                  << std::endl;
        return 1;
    }

    if (args.query_path.empty()) {
        std::cerr << "Error: --query_path is required." << std::endl;
        return 1;
    }

    BatchOutputPaths batch_paths;
    try {
        batch_paths = prepare_batch_output_paths(args.batch_gt_path);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    size_t num_threads = args.num_threads > 0
                             ? static_cast<size_t>(args.num_threads)
                             : std::thread::hardware_concurrency();

    std::vector<std::vector<float>> base = read_bin(args.base_path);
    std::vector<std::vector<float>> queries;
    queries = read_bin(args.query_path);

    if (args.stream && queries.size() < base.size()) {
        std::cerr << "Error: stream requires query dataset "
                     "size >= base size."
                  << std::endl;
        return 1;
    }

    int n = static_cast<int>(queries.size());
    size_t total_b = base.size();
    size_t total_increments = (total_b + args.increment - 1) / args.increment;

    std::vector<std::string> output_files;
    std::vector<int> file_start_offsets;
    std::vector<int> file_end_offsets;
    std::vector<int> file_actual_batch_counts;

    size_t current_increment_idx = 0;
    int current_file_batches = 0;
    int current_file_query_count = 0;
    int current_file_start_offset = -1;
    int current_file_end_offset = -1;
    std::ofstream current_out;
    bool file_opened = false;

    size_t start_b_size = args.increment;

#if defined(UTILS_ENABLE_CUDA) && UTILS_ENABLE_CUDA
    bool gpu_execution_enabled = false;
    std::unique_ptr<GPUKNNProcessor> gpu_processor;
    if (args.use_gpu) {
        if (base.empty()) {
            std::cerr << "Cannot enable GPU execution without base vectors."
                      << std::endl;
            return 1;
        }
        int vector_dim = static_cast<int>(base[0].size());
        int max_queries = static_cast<int>(queries.size());
        int max_base = static_cast<int>(base.size());
        try {
            gpu_processor = std::make_unique<GPUKNNProcessor>(
                args.k, max_queries, max_base, vector_dim);
            std::cout << "GPU acceleration enabled." << std::endl;
            gpu_execution_enabled = true;
        } catch (const std::exception &e) {
            std::cerr << "GPU initialization failed: " << e.what()
                      << ". Falling back to CPU execution." << std::endl;
            args.use_gpu = false;
        }
        args.use_gpu = gpu_execution_enabled;
    }
#else
    if (args.use_gpu) {
        std::cerr << "This binary was built without CUDA support. "
                     "Falling back to CPU execution."
                  << std::endl;
        args.use_gpu = false;
    }
#endif

    if (!args.use_gpu) {
        std::cout << "Using " << num_threads << " CPU threads" << std::endl;
    }

    auto finalize_current_file = [&]() {
        if (!file_opened) return;
        current_out.seekp(8);
        current_out.write(reinterpret_cast<const char *>(&current_file_batches),
                          sizeof(int));
        current_out.close();
        file_actual_batch_counts.push_back(current_file_batches);
        file_end_offsets.back() = current_file_end_offset;
        std::string old_name = output_files.back();
        std::string new_name = generate_batch_filename(
            batch_paths, file_start_offsets.back(), current_file_end_offset,
            args.k);
        if (new_name != old_name) {
            std::filesystem::path new_path(new_name);
            if (std::filesystem::exists(new_path)) {
                std::filesystem::remove(new_path);
            }
            std::filesystem::rename(old_name, new_name);
            output_files.back() = new_name;
        }
        file_opened = false;
        current_file_batches = 0;
        current_file_query_count = 0;
        current_file_start_offset = -1;
        current_file_end_offset = -1;
    };

    const int max_batches_per_file = args.max_batches_per_file > 0
                                         ? args.max_batches_per_file
                                         : std::numeric_limits<int>::max();

    for (size_t current_base_size = start_b_size; current_base_size <= total_b;
         current_base_size += args.increment) {
        current_increment_idx++;

        std::cout << "Batch " << current_increment_idx << "/"
                  << total_increments << " (offset: " << current_base_size
                  << ") (" << (current_increment_idx * 100 / total_increments)
                  << "%) - " << (args.use_gpu ? "GPU" : "CPU") << std::endl;

        const auto &current_queries = queries;
        size_t query_start_idx = 0;
        size_t query_end_idx = current_queries.size();
        if (args.stream) {
            query_end_idx = std::min(current_base_size, current_queries.size());
            size_t available = query_end_idx - query_start_idx;
            size_t slice_size =
                std::min(static_cast<size_t>(args.increment), available);
            if (available > slice_size) {
                query_start_idx = query_end_idx - slice_size;
            }
        }
        size_t queries_in_batch = query_end_idx - query_start_idx;
        if (queries_in_batch == 0) {
            std::cerr << "Warning: No queries selected for batch "
                      << current_increment_idx << ", skipping." << std::endl;
            continue;
        }

        if (file_opened && args.stream &&
            static_cast<int>(queries_in_batch) != current_file_query_count) {
            finalize_current_file();
        }

        if (file_opened && current_file_batches >= max_batches_per_file) {
            finalize_current_file();
        }

        if (!file_opened) {
            current_file_start_offset = static_cast<int>(current_base_size);
            current_file_end_offset = current_file_start_offset;
            std::string filename = generate_batch_filename(
                batch_paths, current_file_start_offset, current_file_end_offset,
                args.k);
            current_out.open(filename, std::ios::binary);
            if (!current_out.is_open())
                throw std::runtime_error("Error: Failed to open file: " +
                                         filename);

            current_file_query_count = args.stream
                                           ? static_cast<int>(queries_in_batch)
                                           : n;
            int header_n = current_file_query_count;
            int placeholder_batch_count = 0;
            current_out.write(reinterpret_cast<const char *>(&header_n),
                              sizeof(int));
            current_out.write(reinterpret_cast<const char *>(&args.k),
                              sizeof(int));
            current_out.write(
                reinterpret_cast<const char *>(&placeholder_batch_count),
                sizeof(int));

            output_files.push_back(filename);
            file_start_offsets.push_back(current_file_start_offset);
            file_end_offsets.push_back(current_file_end_offset);
            current_file_batches = 0;
            file_opened = true;
        }

        std::vector<std::vector<PointPair>> current_batch_results;
        std::vector<std::vector<float>> slice_queries_storage;
        std::vector<uint32_t> query_ids;
        if (args.stream) {
            query_ids.reserve(queries_in_batch);
            for (size_t qi = 0; qi < queries_in_batch; ++qi) {
                query_ids.push_back(
                    static_cast<uint32_t>(query_start_idx + qi));
            }
        }

#if defined(UTILS_ENABLE_CUDA) && UTILS_ENABLE_CUDA
        if (gpu_execution_enabled) {
            const std::vector<std::vector<float>> *gpu_queries =
                &current_queries;
            if (args.stream) {
                slice_queries_storage.assign(
                    current_queries.begin() + query_start_idx,
                    current_queries.begin() + query_end_idx);
                gpu_queries = &slice_queries_storage;
            }
            current_batch_results = gpu_processor->compute_single_increment_knn(
                base, *gpu_queries, static_cast<int>(current_base_size));
        } else
#endif
        {
            current_batch_results.resize(queries_in_batch);

            size_t num_queries_in_batch = queries_in_batch;
            std::vector<std::thread> threads;
            size_t chunk_size =
                (num_queries_in_batch + num_threads - 1) / num_threads;

            auto worker = [&](size_t thread_idx) {
                size_t start = thread_idx * chunk_size;
                size_t end = std::min(start + chunk_size, num_queries_in_batch);
                for (size_t i = start; i < end; ++i) {
                    size_t query_idx = query_start_idx + i;
                    current_batch_results[i] =
                        exact_knn(current_queries[query_idx], base,
                                  current_base_size, args.k, -1);
                }
            };

            for (size_t t = 0; t < num_threads; ++t) {
                threads.emplace_back(worker, t);
            }
            for (auto &th : threads) {
                th.join();
            }
        }

        uint64_t current_offset = static_cast<uint64_t>(current_base_size);
        current_out.write(reinterpret_cast<const char *>(&current_offset),
                          sizeof(uint64_t));
        current_file_end_offset = static_cast<int>(current_base_size);

        if (args.stream) {
            current_out.write(reinterpret_cast<const char *>(query_ids.data()),
                              sizeof(uint32_t) * query_ids.size());
        }

        // Write all distances first, then all IDs
        for (const auto &result : current_batch_results) {
            for (const auto &pair : result) {
                current_out.write(reinterpret_cast<const char *>(&pair.second),
                                  sizeof(float));
            }
        }
        for (const auto &result : current_batch_results) {
            for (const auto &pair : result) {
                current_out.write(reinterpret_cast<const char *>(&pair.first),
                                  sizeof(int));
            }
        }
        current_file_batches++;
    }

    finalize_current_file();

    std::filesystem::path index_dir = batch_paths.parent_dir.empty()
                                           ? std::filesystem::current_path()
                                           : std::filesystem::absolute(
                                                 batch_paths.parent_dir);
    std::string index_filename = generate_index_filename(batch_paths);
    std::ofstream index_out(index_filename);
    if (index_out.is_open()) {
        index_out << "# Batch Groundtruth Index File\n";
        for (size_t i = 0; i < output_files.size(); ++i) {
            int actual_batches_in_file = file_actual_batch_counts[i];
            int end_offset = file_end_offsets[i];
            std::filesystem::path output_path =
                std::filesystem::absolute(output_files[i]);
            std::error_code ec;
            std::filesystem::path rel_path =
                std::filesystem::relative(output_path, index_dir, ec);
            if (ec || rel_path.empty()) {
                rel_path = output_path.filename();
            }
            std::string filename_only = rel_path.generic_string();
            index_out << filename_only << " " << file_start_offsets[i] << " "
                      << end_offset << " " << actual_batches_in_file << "\n";
        }
        index_out.close();
        std::cout << "Created index file: " << index_filename << std::endl;
    }

    std::cout << "\nBatch processing completed!" << std::endl;

    return 0;
}
