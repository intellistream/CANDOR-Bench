#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils.hpp"

struct GTEntry {
    uint64_t insert_offset;
    uint64_t query_tag;
    std::vector<uint32_t> tags;
};

struct GTEntryComp {
    bool operator()(const GTEntry& a, const GTEntry& b) const {
        if (a.insert_offset != b.insert_offset) {
            return a.insert_offset < b.insert_offset;
        }
        return a.query_tag < b.query_tag;
    }
    bool operator()(const GTEntry& a, uint64_t offset) const {
        return a.insert_offset < offset;
    }
    bool operator()(uint64_t offset, const GTEntry& a) const {
        return offset < a.insert_offset;
    }
};

struct GTEntryQueryComp {
    bool operator()(const GTEntry& a, uint64_t tag) const {
        return a.query_tag < tag;
    }
    bool operator()(uint64_t tag, const GTEntry& a) const {
        return tag < a.query_tag;
    }
};

void load_gt_file(std::vector<GTEntry>& gt_entries,
                  const std::string& split_file_path) {
    std::ifstream in(split_file_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("FATAL: Failed to open split file: " +
                                 split_file_path);
    }
    gt_entries.clear();

    int32_t n_per_batch, k_file, num_batches;
    in.read(reinterpret_cast<char*>(&n_per_batch), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&k_file), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&num_batches), sizeof(int32_t));

    if (in.fail() || n_per_batch <= 0 || k_file <= 0 || num_batches <= 0) {
        throw std::runtime_error("Invalid GT file header in: " +
                                 split_file_path);
    }

    const std::streampos payload_begin = in.tellg();
    in.seekg(0, std::ios::end);
    const std::streampos file_size = in.tellg();
    in.seekg(payload_begin);

    const size_t header_size = sizeof(int32_t) * 3;
    const size_t per_batch_base =
        sizeof(uint64_t) + static_cast<size_t>(n_per_batch) *
                               static_cast<size_t>(k_file) *
                               (sizeof(float) + sizeof(uint32_t));
    const size_t expected_payload_size =
        per_batch_base * static_cast<size_t>(num_batches);
    const size_t query_tag_bytes = static_cast<size_t>(n_per_batch) *
                                   sizeof(uint32_t) *
                                   static_cast<size_t>(num_batches);
    const size_t payload_size = static_cast<size_t>(file_size) - header_size;

    bool has_query_tags = false;
    if (payload_size == expected_payload_size + query_tag_bytes) {
        has_query_tags = true;
    } else if (payload_size != expected_payload_size) {
        throw std::runtime_error(
            "GT file " + split_file_path +
            " does not follow the unified incremental GT layout. "
            "Please regenerate it with the updated compute_incr_gt.");
    }

    gt_entries.reserve(n_per_batch * num_batches);

    std::vector<uint32_t> raw_ids;
    if (has_query_tags) {
        raw_ids.resize(n_per_batch);
    }
    std::vector<float> batch_distances(n_per_batch * k_file);
    std::vector<uint32_t> batch_indices(n_per_batch * k_file);

    for (int b = 0; b < num_batches; ++b) {
        uint64_t current_batch_offset;
        in.read(reinterpret_cast<char*>(&current_batch_offset),
                sizeof(uint64_t));

        std::vector<int> query_ids_for_batch(n_per_batch);
        if (has_query_tags) {
            in.read(reinterpret_cast<char*>(raw_ids.data()),
                    n_per_batch * sizeof(uint32_t));
            for (int i = 0; i < n_per_batch; ++i) {
                query_ids_for_batch[i] = static_cast<int>(raw_ids[i]);
            }
        } else {
            std::iota(query_ids_for_batch.begin(), query_ids_for_batch.end(),
                      0);
        }

        in.read(reinterpret_cast<char*>(batch_distances.data()),
                n_per_batch * k_file * sizeof(float));
        in.read(reinterpret_cast<char*>(batch_indices.data()),
                n_per_batch * k_file * sizeof(uint32_t));

        if (in.fail()) {
            throw std::runtime_error("Failed to read data for batch " +
                                     std::to_string(b) + " in " +
                                     split_file_path);
        }

        for (int i = 0; i < n_per_batch; ++i) {
            uint64_t query_tag = static_cast<uint64_t>(query_ids_for_batch[i]);

            GTEntry entry;
            entry.insert_offset = current_batch_offset;
            entry.query_tag = query_tag;
            entry.tags.assign(batch_indices.begin() + i * k_file,
                              batch_indices.begin() + (i + 1) * k_file);
            // Ensure tags are sorted for faster intersection
            std::sort(entry.tags.begin(), entry.tags.end());
            gt_entries.push_back(std::move(entry));
        }
    }
    std::sort(gt_entries.begin(), gt_entries.end(), GTEntryComp());
}

float check_recall_streaming(std::vector<SearchResult<uint32_t>>& res,
                             const std::string& index_path,
                             const std::string& recall_path, size_t recall_at,
                             int increment) {
    std::vector<SplitFileInfo> split_files;
    parse_index_file_with_ranges(index_path, split_files);
    if (split_files.empty()) {
        throw std::runtime_error(
            "Index file parsing failed or file was empty: " + index_path);
    }

    std::map<uint64_t, std::vector<SearchResult<uint32_t>*>> res_by_offset;
    for (auto& res_entry : res) {
        res_by_offset[res_entry.insert_offset].push_back(&res_entry);
    }

    std::string loaded_gt_filename = "";
    std::vector<GTEntry> current_gt_entries;

    double total_recall_sum = 0.0;
    size_t total_valid_entries = 0;
    std::map<uint64_t, double> batch_recall_sum;
    std::map<uint64_t, size_t> batch_entry_count;
    std::map<uint64_t, float> batch_worst_recall;

    size_t total_batches = res_by_offset.size();
    size_t processed_batches = 0;

    for (const auto& pair : res_by_offset) {
        uint64_t offset = pair.first;
        const auto& res_entries = pair.second;

        auto it = std::find_if(split_files.begin(), split_files.end(),
                               [offset](const SplitFileInfo& info) {
                                   return offset >= info.start_offset &&
                                          offset <= info.end_offset;
                               });

        if (it == split_files.end()) {
            continue;
        }

        if (loaded_gt_filename != it->filename) {
            std::string gt_dir = "";
            size_t last_slash_idx = index_path.find_last_of("/\\");
            if (last_slash_idx != std::string::npos)
                gt_dir = index_path.substr(0, last_slash_idx + 1);

            load_gt_file(current_gt_entries, gt_dir + it->filename);
            loaded_gt_filename = it->filename;
        }

        auto range = std::equal_range(current_gt_entries.begin(),
                                      current_gt_entries.end(), offset,
                                      GTEntryComp());

        if (range.first == range.second) {
            continue; 
        }

        for (auto* res_entry_ptr : res_entries) {
            const auto& res_entry = *res_entry_ptr;

            auto gt_it = std::lower_bound(range.first, range.second,
                                          res_entry.query_tag, GTEntryQueryComp());

            if (gt_it == range.second || gt_it->query_tag != res_entry.query_tag) {
                continue; 
            }

            const auto& gt_tags = gt_it->tags;
            size_t limit = std::min(recall_at, gt_tags.size());
            if (limit == 0) continue;

            std::vector<uint32_t> res_tags = res_entry.tags;
            if (res_tags.size() > recall_at) {
                res_tags.resize(recall_at);
            }
            std::sort(res_tags.begin(), res_tags.end());

            size_t matches = 0;
            size_t i = 0, j = 0;
            while (i < res_tags.size() && j < limit) {
                if (res_tags[i] < gt_tags[j]) {
                    i++;
                } else if (gt_tags[j] < res_tags[i]) {
                    j++;
                } else {
                    matches++;
                    i++;
                    j++;
                }
            }

            float recall = static_cast<float>(matches) / recall_at;

            total_recall_sum += recall;
            total_valid_entries++;

            uint64_t offset_bucket = res_entry.insert_offset;
            if (increment > 0) {
                uint64_t inc = static_cast<uint64_t>(increment);
                offset_bucket = (offset_bucket / inc) * inc;
            }

            batch_recall_sum[offset_bucket] += recall;
            batch_entry_count[offset_bucket]++;
            auto& current_worst = batch_worst_recall[offset_bucket];
            if (batch_entry_count[offset_bucket] == 1) {
                current_worst = recall;
            } else if (recall < current_worst) {
                current_worst = recall;
            }
        }
        processed_batches++;
        size_t print_interval = total_batches / 10;
        if (print_interval == 0) print_interval = 1;

        if (processed_batches % print_interval == 0 ||
            processed_batches == total_batches) {
            int percentage = (processed_batches * 100) / total_batches;
            std::cout << "Processing: " << percentage << "% ("
                      << processed_batches << "/" << total_batches
                      << ", Offset: " << offset << ")\r" << std::flush;
        }
    }
    std::cout << std::endl;

    if (total_valid_entries == 0) {
        std::cerr << "Error: No valid entries found to compute recall. Final "
                     "recall is 0."
                  << std::endl;
        return 0.0f;
    }

    float average_recall = total_recall_sum / total_valid_entries * 100.0f;

    std::stringstream ss;
    const int offset_width = 12;
    const int avg_width = 18;
    const int worst_width = 16;
    const int count_width = 12;

    ss << std::left << std::setw(offset_width) << "Offset"
       << std::setw(avg_width) << "Average_Recall_%" << std::setw(worst_width)
       << "Worst_Recall_%" << std::setw(count_width) << "Query_Count" << "\n";
    ss << std::left << std::fixed << std::setprecision(4);
    for (const auto& pair : batch_recall_sum) {
        uint64_t offset = pair.first;
        double recall_sum = pair.second;
        float batch_avg_recall =
            recall_sum / batch_entry_count.at(offset) * 100.0f;
        float worst_recall = batch_worst_recall.count(offset)
                                 ? batch_worst_recall.at(offset) * 100.0f
                                 : 0.0f;
        ss << std::setw(offset_width) << offset << std::setw(avg_width)
           << batch_avg_recall << std::setw(worst_width) << worst_recall
           << std::setw(count_width) << batch_entry_count.at(offset) << "\n";
    }

    std::ofstream out_file(recall_path);
    out_file << ss.str();
    out_file.close();

    return average_recall;
}

int main(int argc, char* argv[]) {
    std::string res_path, gt_path, recall_path;
    size_t recall_at = 10;
    int increment = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--res_path" && i + 1 < argc)
            res_path = argv[++i];
        else if (arg == "--gt_path" && i + 1 < argc)
            gt_path = argv[++i];
        else if (arg == "--recall_path" && i + 1 < argc)
            recall_path = argv[++i];
        else if (arg == "--k" && i + 1 < argc)
            recall_at = std::stoul(argv[++i]);
        else if (arg == "--inc" && i + 1 < argc)
            increment = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " --res_path <path> --gt_path <index_file_path> "
                         "--recall_path <path> [--k <K>] [--inc <offset_bucket>]\n";
            return 0;
        }
    }

    if (res_path.empty() || gt_path.empty() || recall_path.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --res_path <path> --gt_path <index_file_path> "
                     "--recall_path <path> [--k <K>] [--inc <offset_bucket>]\n";
        return 1;
    }

    try {
        std::vector<SearchResult<uint32_t>> res;
        read_results_binary(res, res_path);

        if (res.empty()) {
            std::cerr << "WARNING: ANNS result file was empty or failed to "
                         "load. No recall can be computed."
                      << std::endl;
            std::ofstream out_file(recall_path);
            out_file << "Offset      Average_Recall_%  Worst_Recall_%  Query_Count \n";
            out_file.close();
            return 0;
        }

        check_recall_streaming(res, gt_path, recall_path, recall_at, increment);

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
