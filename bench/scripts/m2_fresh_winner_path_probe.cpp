#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../algorithms/annchor-m2/annchor_m2.hpp"

namespace {

struct DatasetSpec {
    std::string name;
    std::string data_path;
    std::string query_path;
    size_t dim;
    size_t m;
    size_t efc;
    size_t efs;
};

struct QueryRecord {
    std::string mode;
    size_t gap;
    uint32_t query_id;
    std::vector<float> query;
    std::vector<uint32_t> top_tags;
    std::vector<float> top_dists;
    std::vector<uint32_t> path_tags;
    std::vector<float> path_dists;
};

struct Summary {
    std::string dataset;
    std::string mode;
    size_t gap = 0;
    size_t queries = 0;
    size_t fresh_winners = 0;
    size_t top_hop1 = 0;
    size_t top_hop2 = 0;
    size_t top_hop3 = 0;
    size_t path_hop1 = 0;
    size_t path_hop2 = 0;
    size_t path_hop3 = 0;
    size_t top_plus_path_hop1 = 0;
    size_t top_plus_path_hop2 = 0;
    size_t top_plus_path_hop3 = 0;
    size_t top_miss_path2 = 0;
    size_t top_miss_path3 = 0;
    size_t neither_path3 = 0;
    size_t fresh_degree_zero = 0;
    size_t path_count_sum = 0;
};

struct PolicySummary {
    std::string dataset;
    std::string mode;
    size_t gap = 0;
    std::string policy;
    size_t budget = 0;
    size_t queries = 0;
    size_t fresh_winners = 0;
    size_t selected_sum = 0;
    size_t selected_hop1 = 0;
    size_t selected_hop2 = 0;
    size_t top_plus_selected_hop1 = 0;
    size_t top_plus_selected_hop2 = 0;
};

std::vector<float> read_bin_prefix(const std::string& path, size_t want,
                                   size_t expected_dim, size_t* total_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open failed: " + path);
    int32_t n = 0;
    int32_t d = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    if (!in || n < 0 || d <= 0) {
        throw std::runtime_error("bad bin header: " + path);
    }
    if (static_cast<size_t>(d) != expected_dim) {
        throw std::runtime_error("dim mismatch for " + path);
    }
    const size_t take = std::min<size_t>(want, static_cast<size_t>(n));
    std::vector<float> data(take * expected_dim);
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!in) throw std::runtime_error("short bin read: " + path);
    if (total_out) *total_out = static_cast<size_t>(n);
    return data;
}

std::vector<float> read_bin_indices(const std::string& path,
                                    const std::vector<uint32_t>& indices,
                                    size_t expected_dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open failed: " + path);
    int32_t n = 0;
    int32_t d = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    if (!in || n < 0 || d <= 0) {
        throw std::runtime_error("bad bin header: " + path);
    }
    if (static_cast<size_t>(d) != expected_dim) {
        throw std::runtime_error("dim mismatch for " + path);
    }
    std::vector<float> out(indices.size() * expected_dim);
    const std::streamoff base = sizeof(int32_t) * 2;
    const std::streamoff row_bytes =
        static_cast<std::streamoff>(expected_dim * sizeof(float));
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= static_cast<uint32_t>(n)) {
            throw std::runtime_error("query index out of range");
        }
        in.seekg(base + static_cast<std::streamoff>(indices[i]) * row_bytes);
        in.read(reinterpret_cast<char*>(out.data() + i * expected_dim),
                row_bytes);
        if (!in) throw std::runtime_error("short indexed read: " + path);
    }
    return out;
}

float l2(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

std::vector<uint32_t> query_ids_for_mode(const std::string& mode,
                                         size_t initial, size_t gap,
                                         size_t count, size_t stage_window,
                                         size_t total_queries) {
    std::vector<uint32_t> ids;
    ids.reserve(count);
    if (mode == "round_robin") {
        for (size_t i = 0; i < count; ++i) {
            ids.push_back(static_cast<uint32_t>(i % total_queries));
        }
        return ids;
    }

    size_t start = 0;
    size_t window = 0;
    if (mode == "chasing") {
        const size_t end = initial + gap;
        start = end > stage_window ? end - stage_window : 0;
        window = end - start;
    } else if (mode == "peeking") {
        start = initial + gap;
        window = start < total_queries
                     ? std::min(stage_window, total_queries - start)
                     : 0;
    } else {
        throw std::runtime_error("bad mode: " + mode);
    }
    if (window == 0) return ids;
    for (size_t i = 0; i < count; ++i) {
        ids.push_back(static_cast<uint32_t>(start + (i % window)));
    }
    return ids;
}

struct AdjCache {
    ANNchorM2<float>* index = nullptr;
    size_t max_neighbors = 128;
    std::unordered_map<uint32_t, std::vector<uint32_t>> cache;

    const std::vector<uint32_t>& neighbors(uint32_t label) {
        auto it = cache.find(label);
        if (it != cache.end()) return it->second;
        std::vector<uint32_t> out(max_neighbors);
        size_t count = 0;
        const uint32_t in_label = label;
        index->batch_base_neighbors(&in_label, 1, max_neighbors, out.data(),
                                    &count);
        out.resize(count);
        auto inserted = cache.emplace(label, std::move(out));
        return inserted.first->second;
    }
};

int min_hop(AdjCache& adj, const std::vector<uint32_t>& sources,
            uint32_t target, int max_depth) {
    if (sources.empty()) return 0;
    std::unordered_set<uint32_t> seen;
    std::vector<uint32_t> frontier;
    frontier.reserve(sources.size());
    for (uint32_t s : sources) {
        if (s == target) return 0;
        if (seen.insert(s).second) frontier.push_back(s);
    }
    for (int depth = 1; depth <= max_depth; ++depth) {
        std::vector<uint32_t> next;
        for (uint32_t node : frontier) {
            for (uint32_t nb : adj.neighbors(node)) {
                if (nb == target) return depth;
                if (seen.insert(nb).second) next.push_back(nb);
            }
        }
        frontier.swap(next);
        if (frontier.empty()) break;
    }
    return 0;
}

std::vector<uint32_t> unique_prefix(const std::vector<uint32_t>& in,
                                    size_t cap) {
    std::vector<uint32_t> out;
    out.reserve(std::min(cap, in.size()));
    std::unordered_set<uint32_t> seen;
    for (uint32_t x : in) {
        if (seen.insert(x).second) out.push_back(x);
        if (out.size() >= cap) break;
    }
    return out;
}

std::vector<uint32_t> unique_from_pairs(
    std::vector<std::pair<float, uint32_t>> pairs, size_t cap) {
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });
    std::vector<uint32_t> out;
    out.reserve(std::min(cap, pairs.size()));
    std::unordered_set<uint32_t> seen;
    for (const auto& entry : pairs) {
        if (seen.insert(entry.second).second) out.push_back(entry.second);
        if (out.size() >= cap) break;
    }
    return out;
}

std::vector<uint32_t> concat_unique(const std::vector<uint32_t>& a,
                                    const std::vector<uint32_t>& b) {
    std::vector<uint32_t> out;
    out.reserve(a.size() + b.size());
    std::unordered_set<uint32_t> seen;
    for (uint32_t x : a) {
        if (seen.insert(x).second) out.push_back(x);
    }
    for (uint32_t x : b) {
        if (seen.insert(x).second) out.push_back(x);
    }
    return out;
}

std::vector<uint32_t> select_path_sources(const QueryRecord& rec,
                                          const std::string& policy,
                                          size_t budget) {
    if (policy == "topk") return unique_prefix(rec.top_tags, budget);
    if (policy == "path_prefix") return unique_prefix(rec.path_tags, budget);
    if (policy == "path_tail") {
        std::vector<uint32_t> reversed;
        reversed.reserve(rec.path_tags.size());
        for (auto it = rec.path_tags.rbegin(); it != rec.path_tags.rend(); ++it) {
            reversed.push_back(*it);
        }
        return unique_prefix(reversed, budget);
    }
    if (policy == "path_closest") {
        std::vector<std::pair<float, uint32_t>> pairs;
        const size_t n = std::min(rec.path_tags.size(), rec.path_dists.size());
        pairs.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            pairs.emplace_back(rec.path_dists[i], rec.path_tags[i]);
        }
        return unique_from_pairs(std::move(pairs), budget);
    }
    auto tau_ratio = [&]() -> float {
        if (policy == "path_tau100") return 1.00f;
        if (policy == "path_tau105") return 1.05f;
        if (policy == "path_tau110") return 1.10f;
        if (policy == "path_tau120") return 1.20f;
        if (policy == "path_tau150") return 1.50f;
        return -1.0f;
    }();
    if (tau_ratio > 0.0f) {
        if (rec.top_dists.empty()) return {};
        const float tau = rec.top_dists.back();
        const float limit = tau * tau_ratio;
        std::vector<uint32_t> out;
        out.reserve(std::min(budget, rec.path_tags.size()));
        std::unordered_set<uint32_t> seen;
        const size_t n = std::min(rec.path_tags.size(), rec.path_dists.size());
        for (size_t i = 0; i < n; ++i) {
            if (rec.path_dists[i] > limit) continue;
            const uint32_t tag = rec.path_tags[i];
            if (seen.insert(tag).second) out.push_back(tag);
            if (out.size() >= budget) break;
        }
        return out;
    }
    throw std::runtime_error("bad path policy: " + policy);
}

std::vector<uint32_t> fresh_winner_targets(const DatasetSpec& ds,
                                           const std::vector<float>& data,
                                           const QueryRecord& rec,
                                           size_t initial, size_t k) {
    if (rec.top_tags.empty()) return {};
    struct Entry {
        float dist;
        uint32_t tag;
        bool fresh;
    };
    std::vector<Entry> combined;
    combined.reserve(k + rec.gap);
    for (size_t i = 0; i < rec.top_tags.size(); ++i) {
        combined.push_back({rec.top_dists[i], rec.top_tags[i], false});
    }
    const float* q = rec.query.data();
    for (size_t i = 0; i < rec.gap; ++i) {
        const uint32_t tag = static_cast<uint32_t>(initial + i);
        const float* x = data.data() + static_cast<size_t>(tag) * ds.dim;
        combined.push_back({l2(q, x, ds.dim), tag, true});
    }
    std::sort(combined.begin(), combined.end(), [](const Entry& a,
                                                   const Entry& b) {
        if (a.dist != b.dist) return a.dist < b.dist;
        return a.tag < b.tag;
    });
    std::vector<uint32_t> targets;
    const size_t limit = std::min(k, combined.size());
    for (size_t i = 0; i < limit; ++i) {
        if (combined[i].fresh) targets.push_back(combined[i].tag);
    }
    return targets;
}

void update_summary_for_record(const DatasetSpec& ds, ANNchorM2<float>& index,
                               AdjCache& adj,
                               const std::vector<float>& data,
                               const QueryRecord& rec, size_t initial,
                               size_t k, Summary& s) {
    s.queries++;
    s.path_count_sum += rec.path_tags.size();
    if (rec.top_tags.empty()) return;

    struct Entry {
        float dist;
        uint32_t tag;
        bool fresh;
    };
    std::vector<Entry> combined;
    combined.reserve(k + rec.gap);
    for (size_t i = 0; i < rec.top_tags.size(); ++i) {
        combined.push_back({rec.top_dists[i], rec.top_tags[i], false});
    }
    const float* q = rec.query.data();
    for (size_t i = 0; i < rec.gap; ++i) {
        const uint32_t tag = static_cast<uint32_t>(initial + i);
        const float* x = data.data() + static_cast<size_t>(tag) * ds.dim;
        combined.push_back({l2(q, x, ds.dim), tag, true});
    }
    std::sort(combined.begin(), combined.end(), [](const Entry& a,
                                                   const Entry& b) {
        if (a.dist != b.dist) return a.dist < b.dist;
        return a.tag < b.tag;
    });

    const auto top_sources = unique_prefix(rec.top_tags, k);
    const auto path_sources = unique_prefix(rec.path_tags, 128);
    const auto top_plus_path_sources = concat_unique(top_sources, path_sources);
    const size_t limit = std::min(k, combined.size());
    for (size_t i = 0; i < limit; ++i) {
        if (!combined[i].fresh) continue;
        s.fresh_winners++;
        const uint32_t target = combined[i].tag;
        const int top_hop = min_hop(adj, top_sources, target, 3);
        const int path_hop = min_hop(adj, path_sources, target, 3);
        const int top_plus_path_hop =
            min_hop(adj, top_plus_path_sources, target, 3);
        if (top_hop == 1) s.top_hop1++;
        if (top_hop > 0 && top_hop <= 2) s.top_hop2++;
        if (top_hop > 0 && top_hop <= 3) s.top_hop3++;
        if (path_hop == 1) s.path_hop1++;
        if (path_hop > 0 && path_hop <= 2) s.path_hop2++;
        if (path_hop > 0 && path_hop <= 3) s.path_hop3++;
        if (top_plus_path_hop == 1) s.top_plus_path_hop1++;
        if (top_plus_path_hop > 0 && top_plus_path_hop <= 2) {
            s.top_plus_path_hop2++;
        }
        if (top_plus_path_hop > 0 && top_plus_path_hop <= 3) {
            s.top_plus_path_hop3++;
        }
        if (!(top_hop > 0 && top_hop <= 2) &&
            (path_hop > 0 && path_hop <= 2)) {
            s.top_miss_path2++;
        }
        if (!(top_hop > 0 && top_hop <= 2) &&
            (path_hop > 0 && path_hop <= 3)) {
            s.top_miss_path3++;
        }
        if (!(path_hop > 0 && path_hop <= 3)) {
            s.neither_path3++;
        }
        if (adj.neighbors(target).empty()) {
            s.fresh_degree_zero++;
        }
    }
}

void update_policy_for_record(const DatasetSpec& ds, AdjCache& adj,
                              const std::vector<float>& data,
                              const QueryRecord& rec, size_t initial,
                              size_t k, PolicySummary& s) {
    s.queries++;
    std::vector<uint32_t> selected =
        select_path_sources(rec, s.policy, s.budget);
    const std::vector<uint32_t> top_sources = unique_prefix(rec.top_tags, k);
    const std::vector<uint32_t> union_sources =
        concat_unique(top_sources, selected);
    s.selected_sum += selected.size();

    const std::vector<uint32_t> targets =
        fresh_winner_targets(ds, data, rec, initial, k);
    for (uint32_t target : targets) {
        s.fresh_winners++;
        const int selected_hop = min_hop(adj, selected, target, 2);
        const int union_hop = min_hop(adj, union_sources, target, 2);
        if (selected_hop == 1) s.selected_hop1++;
        if (selected_hop > 0 && selected_hop <= 2) s.selected_hop2++;
        if (union_hop == 1) s.top_plus_selected_hop1++;
        if (union_hop > 0 && union_hop <= 2) s.top_plus_selected_hop2++;
    }
}

double pct(size_t num, size_t den) {
    return den == 0 ? 0.0 : 100.0 * static_cast<double>(num) /
                                static_cast<double>(den);
}

void write_summary_csv(const std::string& path,
                       const std::vector<Summary>& summaries) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("open output failed: " + path);
    out << "dataset,mode,gap,queries,fresh_winners,avg_path_count,"
           "top_1hop_pct,top_2hop_pct,top_3hop_pct,"
           "path_1hop_pct,path_2hop_pct,path_3hop_pct,"
           "top_plus_path_1hop_pct,top_plus_path_2hop_pct,"
           "top_plus_path_3hop_pct,"
           "top2_miss_path2_pct,top2_miss_path3_pct,neither_path3_pct,"
           "fresh_degree_zero\n";
    out << std::fixed << std::setprecision(4);
    for (const auto& s : summaries) {
        out << s.dataset << ',' << s.mode << ',' << s.gap << ','
            << s.queries << ',' << s.fresh_winners << ','
            << (s.queries == 0 ? 0.0
                                : static_cast<double>(s.path_count_sum) /
                                      static_cast<double>(s.queries))
            << ',' << pct(s.top_hop1, s.fresh_winners) << ','
            << pct(s.top_hop2, s.fresh_winners) << ','
            << pct(s.top_hop3, s.fresh_winners) << ','
            << pct(s.path_hop1, s.fresh_winners) << ','
            << pct(s.path_hop2, s.fresh_winners) << ','
            << pct(s.path_hop3, s.fresh_winners) << ','
            << pct(s.top_plus_path_hop1, s.fresh_winners) << ','
            << pct(s.top_plus_path_hop2, s.fresh_winners) << ','
            << pct(s.top_plus_path_hop3, s.fresh_winners) << ','
            << pct(s.top_miss_path2, s.fresh_winners) << ','
            << pct(s.top_miss_path3, s.fresh_winners) << ','
            << pct(s.neither_path3, s.fresh_winners) << ','
            << s.fresh_degree_zero << '\n';
    }
}

void write_policy_csv(const std::string& path,
                      const std::vector<PolicySummary>& summaries) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("open output failed: " + path);
    out << "dataset,mode,gap,policy,budget,queries,fresh_winners,"
           "avg_selected,path_1hop_pct,path_2hop_pct,"
           "top_plus_path_1hop_pct,top_plus_path_2hop_pct\n";
    out << std::fixed << std::setprecision(4);
    for (const auto& s : summaries) {
        out << s.dataset << ',' << s.mode << ',' << s.gap << ','
            << s.policy << ',' << s.budget << ',' << s.queries << ','
            << s.fresh_winners << ','
            << (s.queries == 0 ? 0.0
                                : static_cast<double>(s.selected_sum) /
                                      static_cast<double>(s.queries))
            << ',' << pct(s.selected_hop1, s.fresh_winners) << ','
            << pct(s.selected_hop2, s.fresh_winners) << ','
            << pct(s.top_plus_selected_hop1, s.fresh_winners) << ','
            << pct(s.top_plus_selected_hop2, s.fresh_winners) << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    size_t query_count = 200;
    if (argc >= 2) {
        query_count = static_cast<size_t>(std::stoul(argv[1]));
    }
    const size_t initial = 10000;
    const size_t max_gap = 1000;
    const size_t k = 10;
    const size_t threads = 8;
    const size_t stage_window = 1000;
    const std::vector<size_t> gaps = {400, 800, 1000};
    const std::vector<std::string> modes = {"round_robin", "chasing",
                                            "peeking"};
    const std::vector<DatasetSpec> datasets = {
        {"sift", "../data/sift/sift_base.bin",
         "../data/sift/sift_query_stream.bin", 128, 16, 100, 35},
        {"deep1m", "../data/deep1M/deep1M_base.bin",
         "../data/deep1M/deep1M_query_stream.bin", 256, 24, 200, 40},
        {"gist", "../data/gist/gist_base.bin",
         "../data/gist/gist_query_stream.bin", 960, 24, 500, 80},
    };

    std::vector<Summary> summaries;
    std::vector<PolicySummary> policy_summaries;
    const std::vector<std::string> policies = {
        "topk", "path_prefix", "path_closest", "path_tail",
        "path_tau100", "path_tau105", "path_tau110", "path_tau120",
        "path_tau150"};
    const std::vector<size_t> budgets = {4, 8, 16, 32, 64};
    for (const auto& ds : datasets) {
        std::cerr << "[dataset] " << ds.name << "\n";
        size_t total_data = 0;
        std::vector<float> data =
            read_bin_prefix(ds.data_path, initial + max_gap, ds.dim,
                            &total_data);
        size_t total_queries = 0;
        {
            size_t ignored = 0;
            (void)read_bin_prefix(ds.query_path, 1, ds.dim, &ignored);
            total_queries = ignored;
        }

        ANNchorM2<float> index(initial + max_gap + 16, ds.dim, threads, ds.m,
                               ds.efc, true, METRIC_L2);
        index.set_enable_mvcc(true);
        index.set_visibility_mode(0);
        QParams qp{};
        qp.ef_search = ds.efs;
        index.set_query_params(qp);

        std::vector<uint32_t> initial_tags(initial);
        std::iota(initial_tags.begin(), initial_tags.end(), 0);
        index.build(data.data(), initial_tags.data(), initial);

        std::vector<QueryRecord> records;
        records.reserve(gaps.size() * modes.size() * query_count);
        for (size_t gap : gaps) {
            for (const auto& mode : modes) {
                std::vector<uint32_t> qids = query_ids_for_mode(
                    mode, initial, gap, query_count, stage_window,
                    total_queries);
                std::vector<float> queries =
                    read_bin_indices(ds.query_path, qids, ds.dim);
                for (size_t qi = 0; qi < qids.size(); ++qi) {
                    QueryRecord rec;
                    rec.mode = mode;
                    rec.gap = gap;
                    rec.query_id = qids[qi];
                    rec.query.assign(queries.data() + qi * ds.dim,
                                     queries.data() + (qi + 1) * ds.dim);
                    index.diagnostic_search_path(
                        rec.query.data(), k, std::numeric_limits<size_t>::max(),
                        rec.top_tags, rec.top_dists, rec.path_tags,
                        rec.path_dists);
                    records.push_back(std::move(rec));
                }
            }
        }

        std::vector<uint32_t> fresh_tags(max_gap);
        std::iota(fresh_tags.begin(), fresh_tags.end(),
                  static_cast<uint32_t>(initial));
        index.batch_insert(data.data() + initial * ds.dim, fresh_tags.data(),
                           max_gap);
        AdjCache adj{&index, 128, {}};

        for (size_t gap : gaps) {
            for (const auto& mode : modes) {
                Summary s;
                s.dataset = ds.name;
                s.mode = mode;
                s.gap = gap;
                for (const auto& rec : records) {
                    if (rec.gap == gap && rec.mode == mode) {
                        update_summary_for_record(ds, index, adj, data, rec,
                                                  initial, k, s);
                    }
                }
                std::cerr << "  " << mode << " gap=" << gap
                          << " winners=" << s.fresh_winners
                          << " top2=" << pct(s.top_hop2, s.fresh_winners)
                          << " path2=" << pct(s.path_hop2, s.fresh_winners)
                          << "\n";
                summaries.push_back(s);

                for (const std::string& policy : policies) {
                    for (size_t budget : budgets) {
                        if (policy == "topk" && budget != budgets.front()) {
                            continue;
                        }
                        const bool tau_policy =
                            policy.rfind("path_tau", 0) == 0;
                        if (tau_policy && budget != 32 && budget != 64) {
                            continue;
                        }
                        PolicySummary ps;
                        ps.dataset = ds.name;
                        ps.mode = mode;
                        ps.gap = gap;
                        ps.policy = policy;
                        ps.budget = policy == "topk" ? k : budget;
                        for (const auto& rec : records) {
                            if (rec.gap == gap && rec.mode == mode) {
                                update_policy_for_record(ds, adj, data, rec,
                                                         initial, k, ps);
                            }
                        }
                        policy_summaries.push_back(ps);
                    }
                }
            }
        }
    }

    write_summary_csv("result/m2_relation_trial_20260518/"
                      "fresh_winner_path_probe_10k.csv",
                      summaries);
    write_policy_csv("result/m2_relation_trial_20260518/"
                     "fresh_winner_path_policy_10k.csv",
                     policy_summaries);
    return 0;
}
