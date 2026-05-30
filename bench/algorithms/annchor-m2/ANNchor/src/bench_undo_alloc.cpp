#include <algorithm>
// Microbenchmark: undo log allocation strategies
// Compare: (A) raw vector push_back, (B) reserve(32) on first use, (C) arena allocator
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static constexpr int NUM_NODES = 1000000;
static constexpr int PRUNE_RATIO = 30;  // ~30% of nodes get pruned
static constexpr int AVG_PRUNES_PER_NODE = 8;  // average prunes per modified node
static constexpr int TRIALS = 5;

// Strategy A: raw vector push_back (current)
struct LogA {
    std::vector<uint32_t> edges;
    std::vector<uint32_t> causes;
    void append(uint32_t edge, uint32_t cause) {
        edges.push_back(edge);
        causes.push_back(cause);
    }
};

// Strategy B: reserve on first use
struct LogB {
    std::vector<uint32_t> edges;
    std::vector<uint32_t> causes;
    void append(uint32_t edge, uint32_t cause) {
        if (__builtin_expect(edges.empty(), 0)) {
            edges.reserve(32);
            causes.reserve(32);
        }
        edges.push_back(edge);
        causes.push_back(cause);
    }
};

// Strategy C: arena allocator (flat arrays, no per-node vector)
struct ArenaLog {
    // All edges/causes in one big array, indexed by offset
    std::vector<uint32_t> all_edges;
    std::vector<uint32_t> all_causes;
    std::vector<uint32_t> node_offset;  // start offset per node
    std::vector<uint32_t> node_count;   // count per node

    ArenaLog(int max_nodes) : node_offset(max_nodes, 0), node_count(max_nodes, 0) {
        all_edges.reserve(max_nodes * AVG_PRUNES_PER_NODE / 3);
        all_causes.reserve(max_nodes * AVG_PRUNES_PER_NODE / 3);
    }

    void append(uint32_t node, uint32_t edge, uint32_t cause) {
        if (node_count[node] == 0) {
            node_offset[node] = all_edges.size();
        }
        // Note: this only works if appends for same node are contiguous
        // In practice they may not be — but for this benchmark we simulate that
        all_edges.push_back(edge);
        all_causes.push_back(cause);
        node_count[node]++;
    }
};

// Strategy D: pre-allocated fixed-size per node (like VersionMemoryPool)
struct LogD {
    static constexpr int MAX_PER_NODE = 64;
    struct Entry { uint32_t edge; uint32_t cause; };
    Entry* pool;       // one big allocation
    uint8_t* counts;   // count per node
    int max_nodes;

    LogD(int n) : max_nodes(n) {
        pool = new Entry[n * MAX_PER_NODE];
        counts = new uint8_t[n]();
    }
    ~LogD() { delete[] pool; delete[] counts; }

    void append(uint32_t node, uint32_t edge, uint32_t cause) {
        int idx = counts[node]++;
        if (idx < MAX_PER_NODE) {
            pool[node * MAX_PER_NODE + idx] = {edge, cause};
        }
    }
};

template<typename F>
double bench(const char* name, F fn) {
    double best = 1e18;
    for (int t = 0; t < TRIALS; t++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    printf("%-35s %8.2f ms\n", name, best);
    return best;
}

int main() {
    // Generate prune events
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> node_dist(0, NUM_NODES - 1);

    int num_modified = NUM_NODES * PRUNE_RATIO / 100;
    std::vector<uint32_t> modified_nodes(num_modified);
    for (int i = 0; i < num_modified; i++) modified_nodes[i] = node_dist(rng);

    // Each modified node gets AVG_PRUNES_PER_NODE prune events
    struct Event { uint32_t node, edge, cause; };
    std::vector<Event> events;
    events.reserve(num_modified * AVG_PRUNES_PER_NODE);
    for (auto node : modified_nodes) {
        int n = AVG_PRUNES_PER_NODE;
        for (int j = 0; j < n; j++) {
            events.push_back({node, node_dist(rng), node_dist(rng)});
        }
    }
    // Shuffle to simulate interleaved inserts
    std::shuffle(events.begin(), events.end(), rng);

    printf("Events: %zu prunes across %d modified nodes (of %d total)\n\n",
           events.size(), num_modified, NUM_NODES);

    // A: raw vector
    bench("A: raw vector push_back", [&]() {
        std::vector<LogA> logs(NUM_NODES);
        for (auto& e : events) logs[e.node].append(e.edge, e.cause);
    });

    // B: reserve(32) on first use
    bench("B: reserve(32) on first use", [&]() {
        std::vector<LogB> logs(NUM_NODES);
        for (auto& e : events) logs[e.node].append(e.edge, e.cause);
    });

    // C: arena (flat arrays)
    // Note: doesn't handle interleaved appends correctly, but shows alloc cost
    bench("C: arena (flat arrays)", [&]() {
        ArenaLog arena(NUM_NODES);
        // Sort by node for contiguous appends (best case for arena)
        std::vector<Event> sorted = events;
        std::sort(sorted.begin(), sorted.end(), [](const Event& a, const Event& b) { return a.node < b.node; });
        for (auto& e : sorted) arena.append(e.node, e.edge, e.cause);
    });

    // D: pre-allocated fixed-size pool
    bench("D: pre-alloc fixed pool (64/node)", [&]() {
        LogD pool(NUM_NODES);
        for (auto& e : events) pool.append(e.node, e.edge, e.cause);
    });

    // Also measure read (lookup) latency for each strategy
    printf("\n--- Read (lookup) latency ---\n");

    // Build logs for read test
    std::vector<LogA> logsA(NUM_NODES);
    for (auto& e : events) logsA[e.node].append(e.edge, e.cause);

    std::vector<LogB> logsB(NUM_NODES);
    for (auto& e : events) logsB[e.node].append(e.edge, e.cause);

    LogD poolD(NUM_NODES);
    for (auto& e : events) poolD.append(e.node, e.edge, e.cause);

    // Generate random lookups
    std::vector<uint32_t> lookups(2000000);
    for (auto& l : lookups) l = node_dist(rng);

    volatile uint32_t sink = 0;

    bench("A: read (vector)", [&]() {
        uint32_t s = 0;
        for (auto node : lookups) {
            if (!logsA[node].edges.empty()) s += logsA[node].edges[0];
        }
        sink = s;
    });

    bench("B: read (vector+reserve)", [&]() {
        uint32_t s = 0;
        for (auto node : lookups) {
            if (!logsB[node].edges.empty()) s += logsB[node].edges[0];
        }
        sink = s;
    });

    bench("D: read (fixed pool)", [&]() {
        uint32_t s = 0;
        for (auto node : lookups) {
            if (poolD.counts[node] > 0) s += poolD.pool[node * LogD::MAX_PER_NODE].edge;
        }
        sink = s;
    });

    return 0;
}
