#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

struct SearchResult {
    uint64_t insert_offset;
    uint64_t query_tag;
    std::vector<uint32_t> tags;
};

void read_results_binary_debug(std::vector<SearchResult> &res,
                               const std::string &res_path,
                               int max_entries = -1) {
    res.clear();

    std::ifstream in_file(res_path, std::ios::binary);
    if (!in_file.is_open()) {
        throw std::runtime_error("Failed to open results file: " + res_path);
    }

    uint64_t num_queries;
    if (!in_file.read(reinterpret_cast<char *>(&num_queries),
                      sizeof(uint64_t))) {
        throw std::runtime_error("Failed to read number of queries");
    }

    std::cout << "Number of queries: " << num_queries << std::endl;

    int entries_to_read =
        (max_entries == -1)
            ? num_queries
            : std::min(static_cast<uint64_t>(max_entries), num_queries);
    std::cout << "Reading " << entries_to_read << " entries (out of "
              << num_queries << " total)" << std::endl;

    res.reserve(entries_to_read);

    for (uint64_t i = 0;
         i < num_queries && static_cast<int>(res.size()) < entries_to_read;
         ++i) {
        SearchResult result;

        if (!in_file.read(reinterpret_cast<char *>(&result.insert_offset),
                          sizeof(uint64_t))) {
            throw std::runtime_error("Failed to read insert_offset for entry " +
                                     std::to_string(i));
        }

        if (!in_file.read(reinterpret_cast<char *>(&result.query_tag),
                          sizeof(uint64_t))) {
            throw std::runtime_error("Failed to read query_tag for entry " +
                                     std::to_string(i));
        }

        uint64_t num_tags;
        if (!in_file.read(reinterpret_cast<char *>(&num_tags),
                          sizeof(uint64_t))) {
            throw std::runtime_error("Failed to read num_tags for entry " +
                                     std::to_string(i));
        }

        result.tags.resize(num_tags);
        if (num_tags > 0) {
            if (!in_file.read(reinterpret_cast<char *>(result.tags.data()),
                              num_tags * sizeof(uint32_t))) {
                throw std::runtime_error("Failed to read tags for entry " +
                                         std::to_string(i));
            }
        }

        uint64_t num_dists;
        if (!in_file.read(reinterpret_cast<char *>(&num_dists),
                          sizeof(uint64_t))) {
            throw std::runtime_error("Failed to read num_dists for entry " +
                                     std::to_string(i));
        }

        if (num_dists > 0) {
            in_file.seekg(num_dists * sizeof(float), std::ios::cur);
        }

        res.push_back(result);

        if (i < 10) {
            std::cout << "Entry " << i
                      << ": insert_offset=" << result.insert_offset
                      << ", query_tag=" << result.query_tag
                      << ", num_tags=" << num_tags;
            if (num_tags > 0) {
                std::cout << ", first_tag=" << result.tags[0];
            }
            std::cout << std::endl;
        }
    }

    in_file.close();
}

void print_detailed_results_for_offset(const std::vector<SearchResult> &results,
                                       uint64_t target_offset) {
    std::cout << "\n[Detailed Results for Offset " << target_offset << "]"
              << std::endl;
    std::cout << "================================================"
              << std::endl;

    int count = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].insert_offset == target_offset) {
            count++;
            std::cout << "\nQuery " << std::setw(4) << count
                      << " (Global Index: " << i << "):" << std::endl;
            std::cout << "  Query Tag: " << results[i].query_tag << std::endl;
            std::cout << "  Insert Offset: " << results[i].insert_offset
                      << std::endl;
            std::cout << "  Number of Results: " << results[i].tags.size()
                      << std::endl;

            if (!results[i].tags.empty()) {
                std::cout << "  Results:" << std::endl;
                for (size_t j = 0; j < results[i].tags.size(); ++j) {
                    std::cout << "    " << std::setw(3) << j + 1
                              << ": Tag=" << results[i].tags[j] << std::endl;
                }
            } else {
                std::cout << "  No results found" << std::endl;
            }
        }
    }

    if (count == 0) {
        std::cout << "No queries found for offset " << target_offset
                  << std::endl;
    } else {
        std::cout << "\nTotal queries for offset " << target_offset << ": "
                  << count << std::endl;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <results_file> [--offset <offset_value>] [max_entries]"
                  << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr
            << "  --offset <value>  Show detailed results for specific offset"
            << std::endl;
        std::cerr << "  max_entries       Maximum number of entries to read "
                     "(default: read all)"
                  << std::endl;
        std::cerr << "                    If not specified, shows summary of "
                     "offset distribution"
                  << std::endl;
        return 1;
    }

    std::string res_path = argv[1];
    int max_entries = -1;  // -1 means read all
    uint64_t target_offset = 0;
    bool show_specific_offset = false;

    // Parse command line arguments
    if (argc >= 4 && std::string(argv[2]) == "--offset") {
        try {
            target_offset = std::stoull(argv[3]);
            show_specific_offset = true;
        } catch (const std::exception &e) {
            std::cerr << "Error: Invalid offset value '" << argv[3]
                      << "'. Must be a valid number." << std::endl;
            return 1;
        }

        if (argc == 5) {
            max_entries = std::stoi(argv[4]);
            if (max_entries <= 0) {
                std::cerr << "Error: max_entries must be a positive integer"
                          << std::endl;
                return 1;
            }
        }
    } else if (argc == 3) {
        max_entries = std::stoi(argv[2]);
        if (max_entries <= 0) {
            std::cerr << "Error: max_entries must be a positive integer"
                      << std::endl;
            return 1;
        }
    }

    std::vector<SearchResult> results;

    try {
        read_results_binary_debug(results, res_path, max_entries);
        std::cout << "Successfully read " << results.size() << " results"
                  << std::endl;

        if (show_specific_offset) {
            std::cout << "\n--- Results Inspector (Specific Offset Mode) ---"
                      << std::endl;
            std::cout << "Target Offset: " << target_offset << std::endl;
            print_detailed_results_for_offset(results, target_offset);
        } else {
            std::cout << "\n--- Results Inspector (Summary Mode) ---"
                      << std::endl;
            std::cout << "\nOffset distribution:" << std::endl;
            std::vector<uint64_t> offsets;
            for (const auto &res : results) {
                offsets.push_back(res.insert_offset);
            }
            std::sort(offsets.begin(), offsets.end());
            offsets.erase(std::unique(offsets.begin(), offsets.end()),
                          offsets.end());

            std::cout << "Unique offsets found: " << offsets.size()
                      << std::endl;
            for (size_t i = 0; i < std::min(size_t(10), offsets.size()); ++i) {
                std::cout << "  " << offsets[i] << std::endl;
            }
            if (offsets.size() > 10) {
                std::cout << "  ... and " << (offsets.size() - 10) << " more"
                          << std::endl;
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
