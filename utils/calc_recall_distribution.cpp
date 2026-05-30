#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <gt_file> <res_file>" << std::endl;
        return 1;
    }
    std::string gt_path = argv[1], res_path = argv[2];

    // Load GT
    std::ifstream gt_in(gt_path, std::ios::binary);
    int32_t nq, k_gt;
    gt_in.read((char*)&nq, 4); gt_in.read((char*)&k_gt, 4);
    std::vector<uint32_t> gt(nq * k_gt);
    gt_in.read((char*)gt.data(), nq * k_gt * 4);

    // Load results
    std::ifstream res_in(res_path, std::ios::binary);
    int32_t nr, k_res;
    res_in.read((char*)&nr, 4); res_in.read((char*)&k_res, 4);
    std::vector<uint32_t> res(nr * k_res);
    res_in.read((char*)res.data(), nr * k_res * 4);

    if (nq != nr) { std::cerr << "Query count mismatch\n"; return 1; }

    uint32_t recall_at = k_res;
    std::vector<double> per_query_recall(nq);

    for (int i = 0; i < nq; i++) {
        std::set<uint32_t> gt_set(gt.data() + i*k_gt, gt.data() + i*k_gt + recall_at);
        std::set<uint32_t> res_set(res.data() + i*k_res, res.data() + i*k_res + recall_at);
        int hit = 0;
        for (auto& v : res_set) if (gt_set.count(v)) hit++;
        per_query_recall[i] = (double)hit / recall_at;
    }

    std::sort(per_query_recall.begin(), per_query_recall.end());

    double mean = std::accumulate(per_query_recall.begin(), per_query_recall.end(), 0.0) / nq;
    double var = 0;
    for (auto r : per_query_recall) var += (r - mean) * (r - mean);
    var /= nq;

    auto pct = [&](double p) { return per_query_recall[(int)(p * nq)]; };

    std::cout << "queries=" << nq << " recall_at=" << recall_at << std::endl;
    std::cout << "mean=" << mean << " std=" << std::sqrt(var) << std::endl;
    std::cout << "min=" << per_query_recall[0]
              << " p1=" << pct(0.01)
              << " p5=" << pct(0.05)
              << " p10=" << pct(0.10)
              << " p50=" << pct(0.50)
              << " p90=" << pct(0.90)
              << " p95=" << pct(0.95)
              << " p99=" << pct(0.99)
              << " max=" << per_query_recall[nq-1] << std::endl;

    // Count queries with recall < thresholds
    int below50 = 0, below70 = 0, below90 = 0;
    for (auto r : per_query_recall) {
        if (r < 0.5) below50++;
        if (r < 0.7) below70++;
        if (r < 0.9) below90++;
    }
    std::cout << "below_50%=" << below50 << " below_70%=" << below70 << " below_90%=" << below90 << std::endl;

    return 0;
}
