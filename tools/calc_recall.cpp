#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

double calculate_recall(uint32_t num_queries, uint32_t *gold_std,
                        float *gs_dist, uint32_t dim_gs, uint32_t *our_results,
                        uint32_t dim_or, uint32_t recall_at) {
    double total_recall = 0;
    std::set<uint32_t> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
        gt.clear();
        res.clear();
        uint32_t *gt_vec = gold_std + dim_gs * i;
        uint32_t *res_vec = our_results + dim_or * i;
        size_t tie_breaker = recall_at;
        if (gs_dist != nullptr) {
            tie_breaker = recall_at - 1;
            float *gt_dist_vec = gs_dist + dim_gs * i;
            while (tie_breaker < dim_gs &&
                   gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
                tie_breaker++;
        }

        gt.insert(gt_vec, gt_vec + tie_breaker);
        res.insert(res_vec, res_vec + recall_at);
        uint32_t cur_recall = 0;
        for (auto &v : gt) {
            if (res.find(v) != res.end()) {
                cur_recall++;
            }
        }
        total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
}

int main(int argc, char **argv) {
    std::string gt_path, result_path;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--gt_path" && i + 1 < argc) {
            gt_path = argv[++i];
        } else if (arg == "--res_path" && i + 1 < argc) {
            result_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cerr << "Usage: " << argv[0]
                      << " --gt_path <gt_file> --res_path <result_file>"
                      << std::endl;
            return 0;
        }
    }

    if (gt_path.empty() || result_path.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --gt_path <gt_file> --res_path <result_file>"
                  << std::endl;
        std::cerr << "Error: Missing required arguments" << std::endl;
        return 1;
    }

    std::vector<uint32_t> gold_std_vec;
    std::vector<float> gold_dist_vec;
    std::vector<uint32_t> our_results_vec;
    uint32_t num_queries = 0, dim_gs = 0, dim_or = 0;

    {
        std::ifstream fin(gt_path, std::ios::binary);
        if (!fin) {
            std::cerr << "Cannot open file: " << gt_path << std::endl;
            return 1;
        }
        int32_t n, k;
        fin.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
        fin.read(reinterpret_cast<char *>(&k), sizeof(int32_t));
        gold_std_vec.resize(n * k);
        fin.read(reinterpret_cast<char *>(gold_std_vec.data()),
                 n * k * sizeof(int32_t));

        std::streampos current_pos = fin.tellg();
        fin.seekg(0, std::ios::end);
        std::streampos file_size = fin.tellg();
        fin.seekg(current_pos);

        if (file_size - current_pos >= n * k * sizeof(float)) {
            gold_dist_vec.resize(n * k);
            fin.read(reinterpret_cast<char *>(gold_dist_vec.data()),
                     n * k * sizeof(float));
            std::cout << "Ground truth file contains distance data"
                      << std::endl;
        } else {
            std::cout << "Ground truth file contains only ID data" << std::endl;
        }

        if (dim_gs == 0)
            dim_gs = k;
        else if (k != dim_gs) {
            std::cerr << "Inconsistent number of elements per line in "
                         "ground truth file"
                      << std::endl;
            return 1;
        }
        num_queries = n;
    }

    {
        std::ifstream fin(result_path, std::ios::binary);
        if (!fin) {
            std::cerr << "Cannot open file: " << result_path << std::endl;
            return 1;
        }
        int32_t n, k;
        fin.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
        fin.read(reinterpret_cast<char *>(&k), sizeof(int32_t));
        our_results_vec.resize(n * k);
        fin.read(reinterpret_cast<char *>(our_results_vec.data()),
                 n * k * sizeof(int32_t));
        if (dim_or == 0)
            dim_or = k;
        else if (k != dim_or) {
            std::cerr
                << "Inconsistent number of elements per line in result file"
                << std::endl;
            return 1;
        }
        if (n != num_queries) {
            std::cerr << "Number of queries mismatch: gt " << num_queries
                      << ", result " << n << std::endl;
            return 1;
        }
    }
    uint32_t recall_at = dim_or;
    float *gs_dist = gold_dist_vec.empty() ? nullptr : gold_dist_vec.data();
    double recall =
        calculate_recall(num_queries, gold_std_vec.data(), gs_dist, dim_gs,
                         our_results_vec.data(), dim_or, recall_at);
    std::cout << "recall@" << recall_at << " = " << recall << "%" << std::endl;
    return 0;
}
