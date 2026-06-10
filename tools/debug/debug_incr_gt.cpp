#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

struct BatchData {
    uint64_t offset;
    std::vector<int> absolute_query_ids;
    std::vector<float> distances;
    std::vector<int> neighbor_ids;
};

void print_batch_summary(const BatchData& batch, int batch_index, int n,
                         int k) {
    std::cout << "\n[Inspecting Batch " << batch_index + 1 << "]" << std::endl;
    std::cout << "Insert Offset: " << batch.offset << std::endl;

    std::cout << "  -- First Query of Batch --" << std::endl;
    std::cout << "  Query Absolute Tag: " << batch.absolute_query_ids[0]
              << std::endl;
    for (int l = 0; l < k; ++l) {
        size_t index = 0 * k + l;
        std::cout << "    - Neighbor " << std::setw(2) << l + 1 << ": "
                  << "ID=" << std::setw(7) << batch.neighbor_ids[index] << ", "
                  << "Distance=" << batch.distances[index] << std::endl;
    }

    if (n > 1) {
        std::cout << "  -- Last Query of Batch --" << std::endl;
        std::cout << "  Query Absolute Tag: " << batch.absolute_query_ids[n - 1]
                  << std::endl;
        for (int l = 0; l < k; ++l) {
            size_t index = (size_t)(n - 1) * k + l;
            std::cout << "    - Neighbor " << std::setw(2) << l + 1 << ": "
                      << "ID=" << std::setw(7) << batch.neighbor_ids[index]
                      << ", " << "Distance=" << batch.distances[index]
                      << std::endl;
        }
    }
}

void print_batch_detailed(const BatchData& batch, int batch_index, int n,
                          int k) {
    std::cout << "\n[Detailed Ground Truth for Batch " << batch_index + 1 << "]"
              << std::endl;
    std::cout << "Insert Offset: " << batch.offset << std::endl;
    std::cout << "Total Queries: " << n << std::endl;
    std::cout << "Neighbors per Query: " << k << std::endl;
    std::cout << "================================================"
              << std::endl;

    for (int i = 0; i < n; ++i) {
        std::cout << "\nQuery " << std::setw(3) << i + 1
                  << " (Absolute Tag: " << batch.absolute_query_ids[i]
                  << "):" << std::endl;
        for (int l = 0; l < k; ++l) {
            size_t index = (size_t)i * k + l;
            std::cout << "  Neighbor " << std::setw(2) << l + 1 << ": "
                      << "ID=" << std::setw(7) << batch.neighbor_ids[index]
                      << ", " << "Distance=" << std::fixed
                      << std::setprecision(6) << batch.distances[index]
                      << std::endl;
        }
    }
}

void print_help(const char* prog_name) {
    std::cerr << "Usage: " << prog_name
              << " <groundtruth_file.gt> [--offset <offset_value>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr
        << "  --offset <value>  Show detailed ground truth for specific offset"
        << std::endl;
    std::cerr << "                    If not specified, shows summary of "
                 "first, middle, and last batches"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 4) {
        print_help(argv[0]);
        return 1;
    }

    std::string filename = argv[1];
    uint64_t target_offset = 0;
    bool show_specific_offset = false;

    if (argc == 4 && std::string(argv[2]) == "--offset") {
        try {
            target_offset = std::stoull(argv[3]);
            show_specific_offset = true;
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid offset value '" << argv[3]
                      << "'. Must be a valid number." << std::endl;
            return 1;
        }
    } else if (argc != 2) {
        print_help(argv[0]);
        return 1;
    }
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "'"
                  << std::endl;
        return 1;
    }

    int n, k, b;
    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    file.read(reinterpret_cast<char*>(&k), sizeof(int));
    file.read(reinterpret_cast<char*>(&b), sizeof(int));

    if (file.gcount() == 0) {
        std::cerr << "Error: File is empty or header is corrupted."
                  << std::endl;
        return 1;
    }

    if (show_specific_offset) {
        std::cout
            << "--- Ground Truth File Inspector (Specific Offset Mode) ---"
            << std::endl;
        std::cout << "Target Offset: " << target_offset << std::endl;
    } else {
        std::cout << "--- Ground Truth File Inspector (Summary Mode) ---"
                  << std::endl;
    }
    std::cout << "File: " << filename << std::endl;
    std::cout << "Header Info:" << std::endl;
    std::cout << "  Queries per Batch (n): " << n << std::endl;
    std::cout << "  Neighbors per Query (k): " << k << std::endl;
    std::cout << "  Batches in file (b): " << b << std::endl;
    std::cout << "----------------------------------------------------"
              << std::endl;

    if (b == 0) {
        std::cout << "No batches to inspect." << std::endl;
        return 0;
    }

    std::vector<BatchData> all_batches(b);
    for (int i = 0; i < b; ++i) {
        all_batches[i].absolute_query_ids.resize(n);
        all_batches[i].distances.resize(n * k);
        all_batches[i].neighbor_ids.resize(n * k);

        file.read(reinterpret_cast<char*>(&all_batches[i].offset),
                  sizeof(uint64_t));
        if (file.eof()) {
            std::cerr << "Warning: Reached end of file unexpectedly. Found "
                      << i << " batches instead of " << b << "." << std::endl;
            b = i;  // Adjust batch count
            all_batches.resize(b);
            break;
        }
        file.read(
            reinterpret_cast<char*>(all_batches[i].absolute_query_ids.data()),
            n * sizeof(int));
        file.read(reinterpret_cast<char*>(all_batches[i].distances.data()),
                  n * k * sizeof(float));
        file.read(reinterpret_cast<char*>(all_batches[i].neighbor_ids.data()),
                  n * k * sizeof(int));
    }
    file.close();

    if (show_specific_offset) {
        // Find batch with matching offset
        bool found = false;
        for (int i = 0; i < b; ++i) {
            if (all_batches[i].offset == target_offset) {
                print_batch_detailed(all_batches[i], i, n, k);
                found = true;
                break;
            }
        }

        if (!found) {
            std::cout << "\nError: No batch found with offset " << target_offset
                      << std::endl;
            std::cout << "Available offsets:" << std::endl;
            for (int i = 0; i < b; ++i) {
                std::cout << "  " << all_batches[i].offset << std::endl;
            }
            return 1;
        }
    } else {
        std::set<int> indices_to_print;
        if (b > 0) indices_to_print.insert(0);      // First batch
        if (b > 1) indices_to_print.insert(b - 1);  // Last batch
        if (b > 2) indices_to_print.insert(b / 2);  // Middle batch

        for (int index : indices_to_print) {
            print_batch_summary(all_batches[index], index, n, k);
        }
    }

    std::cout << "\n--- Inspection Complete ---" << std::endl;

    return 0;
}