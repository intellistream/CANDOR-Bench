// Python bindings for the concurrent index and the native benchmark
// driver. One call runs a whole concurrent benchmark with the GIL
// released.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "driver/concurrent_driver.hpp"
#include "factory.hpp"
#include "hnsw/hnsw.hpp"
#include "index.hpp"
#include "index_types.hpp"

namespace py = pybind11;
using candor::driver::DriverConfig;
using candor::driver::RunResult;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using TagArray = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

// Validates an (n, dim) float array and returns n.
size_t rows_2d(const py::buffer_info& info, size_t dim, const char* what) {
    if (info.ndim != 2 || static_cast<size_t>(info.shape[1]) != dim) {
        throw std::invalid_argument(std::string(what) +
                                    " must have shape (n, dim)");
    }
    return static_cast<size_t>(info.shape[0]);
}

void require_len(size_t got, size_t want, const char* what) {
    if (got != want) {
        throw std::invalid_argument(std::string(what) +
                                    " must have length n");
    }
}

class PyIndex {
   public:
    PyIndex(const std::string& type, size_t dim, size_t max_elements,
            size_t M, size_t ef_construction, size_t num_threads,
            float level_m, float alpha, size_t visit_limit,
            bool use_node_lock, size_t worker_scheduler,
            size_t seal_threshold, const std::string& sealed_type)
        : dim_(dim) {
        IndexParams params{};
        params.dim = dim;
        params.max_elements = max_elements;
        params.M = M;
        params.ef_construction = ef_construction;
        params.level_m = level_m;
        params.alpha = alpha;
        params.visit_limit = visit_limit;
        params.num_threads = num_threads;
        params.data_type = DATA_TYPE_FLOAT;
        params.use_node_lock = use_node_lock;
        params.worker_scheduler = worker_scheduler;
        params.seal_threshold = seal_threshold;
        params.sealed_index_type =
            candor::index_type_from_name(sealed_type.empty() ? "hnsw"
                                                            : sealed_type);
        index_ = candor::make_index(candor::index_type_from_name(type), params);
    }

    // Search-time features of the modified hnswlib; only meaningful for
    // the hnsw type, silently ignored elsewhere.
    void set_search_features(bool enable_s3, bool enable_search_sharing,
                             bool enable_path_skip,
                             bool enable_candidate_injection) {
        auto* hnsw = dynamic_cast<HNSW<float>*>(index_.get());
        if (!hnsw) return;
        hnsw->set_enable_s3(enable_s3);
        hnsw->set_enable_search_sharing(enable_search_sharing);
        hnsw->set_enable_path_skip(enable_path_skip);
        hnsw->set_enable_candidate_injection(enable_candidate_injection);
    }

    void build(FloatArray data, py::object tags_obj) {
        auto info = data.request();
        size_t n = rows_2d(info, dim_, "data");
        std::vector<uint32_t> tags;
        const uint32_t* tag_ptr = nullptr;
        TagArray tags_arr;
        if (tags_obj.is_none()) {
            tags.resize(n);
            for (size_t i = 0; i < n; ++i) tags[i] = static_cast<uint32_t>(i);
            tag_ptr = tags.data();
        } else {
            tags_arr = tags_obj.cast<TagArray>();
            if (static_cast<size_t>(tags_arr.size()) != n) {
                throw std::invalid_argument("tags must have length n");
            }
            tag_ptr = tags_arr.data();
        }
        const float* ptr = static_cast<const float*>(info.ptr);
        py::gil_scoped_release release;
        index_->build(ptr, tag_ptr, n);
    }

    void set_query_params(size_t ef_search, size_t beam_width, float alpha,
                          size_t visit_limit) {
        index_->set_query_params(
            QParams(ef_search, beam_width, alpha, visit_limit));
    }

    py::array_t<uint32_t> search(FloatArray query, size_t k) {
        auto info = query.request();
        if (static_cast<size_t>(info.size) != dim_) {
            throw std::invalid_argument("query must have dim elements");
        }
        std::vector<uint32_t> res;
        {
            py::gil_scoped_release release;
            index_->search(static_cast<const float*>(info.ptr), k, res);
        }
        return py::array_t<uint32_t>(static_cast<py::ssize_t>(res.size()),
                                     res.data());
    }

    void batch_insert(FloatArray data, TagArray tags) {
        auto info = data.request();
        size_t n = rows_2d(info, dim_, "data");
        require_len(static_cast<size_t>(tags.size()), n, "tags");
        int rc;
        {
            py::gil_scoped_release release;
            rc = index_->batch_insert(static_cast<const float*>(info.ptr),
                                      tags.data(), n);
        }
        if (rc != 0) throw std::runtime_error("batch_insert failed");
    }

    py::array_t<uint32_t> batch_search(FloatArray queries, size_t k,
                                       uint64_t visible_ts) {
        auto info = queries.request();
        size_t n = rows_2d(info, dim_, "queries");
        std::vector<uint32_t> res(n * k, 0);
        std::vector<uint32_t*> ptrs(n);
        for (size_t i = 0; i < n; ++i) ptrs[i] = res.data() + i * k;
        int rc;
        {
            py::gil_scoped_release release;
            size_t watermark = 0;
            rc = index_->batch_search(static_cast<const float*>(info.ptr), k,
                                      n, ptrs.data(), &watermark, visible_ts);
        }
        if (rc != 0) throw std::runtime_error("batch_search failed");
        py::array_t<uint32_t> out({static_cast<py::ssize_t>(n),
                                   static_cast<py::ssize_t>(k)});
        std::memcpy(out.mutable_data(), res.data(),
                    res.size() * sizeof(uint32_t));
        return out;
    }

    // Insert with optional per-point neighbor-update limits; returns the
    // per-point update counts when collect_stats is set.
    py::object batch_partial_insert(FloatArray data, TagArray tags,
                                    py::object limits_obj,
                                    bool collect_stats) {
        if (!index_->supports_partial_insert()) {
            throw std::runtime_error(
                "this index type does not support batch_partial_insert");
        }
        auto info = data.request();
        size_t n = rows_2d(info, dim_, "data");
        require_len(static_cast<size_t>(tags.size()), n, "tags");
        std::vector<size_t> limits;
        const size_t* limits_ptr = nullptr;
        if (!limits_obj.is_none()) {
            auto arr =
                limits_obj.cast<py::array_t<uint64_t,
                                            py::array::c_style |
                                                py::array::forcecast>>();
            require_len(static_cast<size_t>(arr.size()), n, "limits");
            limits.assign(arr.data(), arr.data() + n);
            limits_ptr = limits.data();
        }
        std::vector<size_t> stats(collect_stats ? n : 0);
        int rc;
        {
            py::gil_scoped_release release;
            rc = index_->batch_partial_insert(
                static_cast<const float*>(info.ptr), tags.data(), n,
                limits_ptr, collect_stats ? stats.data() : nullptr);
        }
        if (rc != 0) throw std::runtime_error("batch_partial_insert failed");
        if (!collect_stats) return py::none();
        py::array_t<uint64_t> out(static_cast<py::ssize_t>(n));
        for (size_t i = 0; i < n; ++i) {
            out.mutable_data()[i] = static_cast<uint64_t>(stats[i]);
        }
        return out;
    }

    void enable_insert_telemetry(bool enabled) {
        index_->enable_insert_telemetry(enabled);
    }

    std::string dump_stats() {
        std::string s;
        index_->dump_stats(s);
        return s;
    }

    py::bytes snapshot() {
        if (!index_->supports_snapshot()) {
            throw std::runtime_error("index type does not support snapshot");
        }
        std::vector<uint8_t> buf;
        {
            py::gil_scoped_release release;
            if (index_->snapshot(buf) != 0) {
                buf.clear();
            }
        }
        if (buf.empty()) throw std::runtime_error("snapshot failed");
        return py::bytes(reinterpret_cast<const char*>(buf.data()),
                         buf.size());
    }

    void restore(const py::bytes& blob) {
        if (!index_->supports_snapshot()) {
            throw std::runtime_error("index type does not support restore");
        }
        std::string_view view = blob;
        int rc;
        {
            py::gil_scoped_release release;
            rc = index_->restore(
                reinterpret_cast<const uint8_t*>(view.data()), view.size());
        }
        if (rc != 0) throw std::runtime_error("restore failed");
    }

    IndexBase<float>& index() { return *index_; }
    size_t dim() const { return dim_; }

   private:
    size_t dim_;
    std::unique_ptr<IndexBase<float>> index_;
};

// The driver's stage-query generator, exposed so offline pipelines
// (simulate) use the exact same windowing and RNG as concurrent runs.
class PyQuerySource {
   public:
    PyQuerySource(const std::string& mode, FloatArray workload,
                  size_t batch_size, size_t stage_window, double zipfian_skew,
                  uint64_t seed)
        : workload_(std::move(workload)) {
        auto info = workload_.request();
        if (info.ndim != 2) {
            throw std::invalid_argument("workload must have shape (n, dim)");
        }
        dim_ = static_cast<size_t>(info.shape[1]);
        source_ = std::make_unique<candor::driver::QuerySource>(
            candor::driver::parse_query_mode(mode),
            static_cast<const float*>(info.ptr),
            static_cast<size_t>(info.shape[0]), dim_, batch_size,
            stage_window, zipfian_skew, seed);
    }

    py::tuple next_batch(uint64_t end_insert_offset) {
        std::vector<float> q;
        std::vector<uint32_t> tags;
        size_t n = source_->next_batch(end_insert_offset, q, tags);
        py::array_t<float> queries({static_cast<py::ssize_t>(n),
                                    static_cast<py::ssize_t>(dim_)});
        if (n) {
            std::memcpy(queries.mutable_data(), q.data(),
                        q.size() * sizeof(float));
        }
        py::array_t<uint32_t> tag_arr(
            static_cast<py::ssize_t>(tags.size()), tags.data());
        return py::make_tuple(queries, tag_arr);
    }

   private:
    FloatArray workload_;  // keeps the buffer alive for the source
    size_t dim_ = 0;
    std::unique_ptr<candor::driver::QuerySource> source_;
};

DriverConfig config_from_dict(const py::dict& d) {
    // Absent keys keep the DriverConfig member defaults, so the struct is
    // the single source of truth for fallback values.
    DriverConfig cfg;
    auto get = [&](const char* key, auto def) {
        using T = decltype(def);
        if (d.contains(key)) return d[key].cast<T>();
        return def;
    };
    cfg.begin_num = get("begin_num", cfg.begin_num);
    cfg.max_elements = get("max_elements", cfg.max_elements);
    cfg.batch_size = get("batch_size", cfg.batch_size);
    cfg.queue_size = get("queue_size", cfg.queue_size);
    cfg.num_threads = get("num_threads", cfg.num_threads);
    cfg.insert_event_rate = get("insert_event_rate", cfg.insert_event_rate);
    cfg.search_event_rate = get("search_event_rate", cfg.search_event_rate);
    cfg.recall_at = get("recall_at", cfg.recall_at);
    cfg.query_mode = candor::driver::parse_query_mode(
        get("query_mode", std::string("round_robin")));
    cfg.stage_query_window =
        get("stage_query_window", cfg.stage_query_window);
    cfg.zipfian_skew = get("zipfian_skew", cfg.zipfian_skew);
    cfg.per_query_latency = get("per_query_latency", cfg.per_query_latency);
    cfg.external_rw_lock = get("external_rw_lock", cfg.external_rw_lock);
    cfg.mem_sample_interval_ms =
        get("mem_sample_interval_ms", cfg.mem_sample_interval_ms);
    cfg.seed = get("seed", cfg.seed);
    cfg.progress_log = get("progress_log", cfg.progress_log);
    cfg.warmup_points = get("warmup_points", cfg.warmup_points);
    cfg.record_schedule = get("record_schedule", false);
    if (d.contains("replay_schedule")) {
        for (auto entry : d["replay_schedule"].cast<py::list>()) {
            auto pair = entry.cast<py::tuple>();
            candor::driver::ScheduleEntry se;
            se.insert_offset = pair[0].cast<uint64_t>();
            auto tags = pair[1].cast<TagArray>();
            se.query_tags.assign(tags.data(), tags.data() + tags.size());
            cfg.replay_schedule.push_back(std::move(se));
        }
    }
    return cfg;
}

py::dict stats_to_dict(const candor::driver::DriverStats& s) {
    py::dict d;
    d["elapsed_sec"] = s.elapsed_sec;
    d["insert_qps"] = s.insert_qps;
    d["search_qps"] = s.search_qps;
    d["insert_points"] = s.insert_points;
    d["search_points"] = s.search_points;
    d["mean_insert_op_latency_ms"] = s.mean_insert_op;
    d["p95_insert_op_latency_ms"] = s.p95_insert_op;
    d["p99_insert_op_latency_ms"] = s.p99_insert_op;
    d["mean_search_op_latency_ms"] = s.mean_search_op;
    d["p95_search_op_latency_ms"] = s.p95_search_op;
    d["p99_search_op_latency_ms"] = s.p99_search_op;
    d["mean_insert_e2e_latency_ms"] = s.mean_insert_e2e;
    d["p95_insert_e2e_latency_ms"] = s.p95_insert_e2e;
    d["p99_insert_e2e_latency_ms"] = s.p99_insert_e2e;
    d["mean_search_e2e_latency_ms"] = s.mean_search_e2e;
    d["p95_search_e2e_latency_ms"] = s.p95_search_e2e;
    d["p99_search_e2e_latency_ms"] = s.p99_search_e2e;
    d["search_lag_avg"] = s.search_lag_avg;
    d["search_lag_max"] = s.search_lag_max;
    d["peak_memory_mb"] = s.peak_memory_mb;
    d["avg_memory_mb"] = s.avg_memory_mb;
    return d;
}

py::dict run_benchmark(PyIndex& index, FloatArray data, py::object queries_obj,
                       const py::dict& config) {
    auto data_info = data.request();
    size_t n = rows_2d(data_info, index.dim(), "data");
    DriverConfig cfg = config_from_dict(config);
    if (cfg.max_elements == 0 || cfg.max_elements > n) cfg.max_elements = n;

    const float* qptr = nullptr;
    size_t num_queries = 0;
    FloatArray queries;
    if (!queries_obj.is_none()) {
        queries = queries_obj.cast<FloatArray>();
        auto qinfo = queries.request();
        num_queries = rows_2d(qinfo, index.dim(), "queries");
        qptr = static_cast<const float*>(qinfo.ptr);
    }

    // The log callback fires from C++ worker threads, so it must not touch
    // Python; write straight to stderr.
    auto log = [](const std::string& msg) {
        std::fprintf(stderr, "[concurrency_native] %s\n", msg.c_str());
        std::fflush(stderr);
    };

    RunResult result;
    {
        py::gil_scoped_release release;
        result = candor::driver::run_concurrent_benchmark(
            index.index(), static_cast<const float*>(data_info.ptr),
            index.dim(), qptr, num_queries, cfg, log);
    }

    py::dict out;
    out["stats"] = stats_to_dict(result.stats);
    py::dict records;
    size_t num_records = result.rec_offsets.size();
    records["insert_offsets"] = py::array_t<uint64_t>(
        static_cast<py::ssize_t>(num_records), result.rec_offsets.data());
    records["query_tags"] = py::array_t<uint32_t>(
        static_cast<py::ssize_t>(num_records), result.rec_query_tags.data());
    py::array_t<uint32_t> result_tags(
        {static_cast<py::ssize_t>(num_records),
         static_cast<py::ssize_t>(result.k)});
    std::memcpy(result_tags.mutable_data(), result.rec_result_tags.data(),
                result.rec_result_tags.size() * sizeof(uint32_t));
    records["result_tags"] = result_tags;
    records["k"] = result.k;
    out["records"] = records;
    if (!result.recorded_schedule.empty()) {
        py::list schedule;
        for (const auto& entry : result.recorded_schedule) {
            schedule.append(py::make_tuple(
                entry.insert_offset,
                py::array_t<uint32_t>(
                    static_cast<py::ssize_t>(entry.query_tags.size()),
                    entry.query_tags.data())));
        }
        out["schedule"] = schedule;
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(concurrency_native, m) {
    m.doc() =
        "CANDOR concurrent index benchmark — native driver with Python "
        "orchestration";

    // The vendored hnswlib carries the S3 / search-sharing / path-skip
    // features, so the search-feature config flags can take effect.
    m.attr("experimental") = true;

    py::class_<PyIndex>(m, "Index")
        .def(py::init<const std::string&, size_t, size_t, size_t, size_t,
                      size_t, float, float, size_t, bool, size_t, size_t,
                      const std::string&>(),
             py::arg("index_type"), py::arg("dim"), py::arg("max_elements"),
             py::arg("M") = 16, py::arg("ef_construction") = 200,
             py::arg("num_threads") = 4, py::arg("level_m") = 0.0f,
             py::arg("alpha") = 1.2f, py::arg("visit_limit") = 0,
             py::arg("use_node_lock") = true,
             py::arg("worker_scheduler") = 0,
             py::arg("seal_threshold") = 0,
             py::arg("sealed_type") = "hnsw")
        .def("set_search_features", &PyIndex::set_search_features,
             py::arg("enable_s3") = false,
             py::arg("enable_search_sharing") = false,
             py::arg("enable_path_skip") = false,
             py::arg("enable_candidate_injection") = false)
        .def("build", &PyIndex::build, py::arg("data"),
             py::arg("tags") = py::none())
        .def("set_query_params", &PyIndex::set_query_params,
             py::arg("ef_search"), py::arg("beam_width") = 4,
             py::arg("alpha") = 1.2f, py::arg("visit_limit") = 0)
        .def("search", &PyIndex::search, py::arg("query"), py::arg("k"))
        .def("batch_insert", &PyIndex::batch_insert, py::arg("data"),
             py::arg("tags"))
        .def("batch_search", &PyIndex::batch_search, py::arg("queries"),
             py::arg("k"),
             py::arg("visible_ts") = std::numeric_limits<uint64_t>::max())
        .def("batch_partial_insert", &PyIndex::batch_partial_insert,
             py::arg("data"), py::arg("tags"), py::arg("limits") = py::none(),
             py::arg("collect_stats") = false)
        .def("enable_insert_telemetry", &PyIndex::enable_insert_telemetry,
             py::arg("enabled"))
        .def("dump_stats", &PyIndex::dump_stats)
        .def("snapshot", &PyIndex::snapshot)
        .def("restore", &PyIndex::restore, py::arg("blob"));

    py::class_<PyQuerySource>(m, "QuerySource")
        .def(py::init<const std::string&, FloatArray, size_t, size_t, double,
                      uint64_t>(),
             py::arg("mode"), py::arg("workload"), py::arg("batch_size"),
             py::arg("stage_window"), py::arg("zipfian_skew") = 0.99,
             py::arg("seed") = 0)
        .def("next_batch", &PyQuerySource::next_batch,
             py::arg("end_insert_offset"));

    m.def("run_benchmark", &run_benchmark, py::arg("index"), py::arg("data"),
          py::arg("queries") = py::none(), py::arg("config") = py::dict(),
          R"(Run the concurrent insert+search benchmark.

The index should already be built with data[:begin_num]. Inserts stream
data[begin_num:max_elements] while searches run against the committed
snapshot. Returns {"stats": ...,
"records": {"insert_offsets", "query_tags", "result_tags", "k"}} for
recall computation against (incremental) ground truth.)");
}
