#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "qg/qg.hpp"
#include "qg/qg_builder.hpp"

namespace py = pybind11;
using py_float_array = py::array_t<float, py::array::c_style | py::array::forcecast>;
using py_uint_array = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
using py_int64_array = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

namespace {
void get_arr_shape(const py::buffer_info& buffer, size_t& rows, size_t& cols) {
    if (buffer.ndim != 2 && buffer.ndim != 1) {
        std::cerr << "Input data has an incorrect shape. Data must be a 1D or 2D array.\n";
        return;
    }
    if (buffer.ndim == 2) {
        rows = buffer.shape[0];
        cols = buffer.shape[1];
    } else {
        rows = 1;
        cols = buffer.shape[0];
    }
}
}  // namespace

struct Index {
    std::unique_ptr<symqg::QuantizedGraph> index = nullptr;

    explicit Index(
        const std::string& index_type,
        const std::string& metric,
        size_t num_points,
        size_t dim,
        size_t degree
    ) {
        if (metric != "L2") {
            std::cerr << "Only L2 distance supported currently\n";
            return;
        }

        if (degree < 32 || degree % 32 != 0) {
            std::cerr << "The degree bound must be a multiple of 32\n";
            return;
        }

        if (index_type == "QG") {
            index = std::make_unique<symqg::QuantizedGraph>(num_points, degree, dim);
        } else {
            std::cerr << "Index type [" << index_type << "] not supported\n";
            return;
        }
    }

    void load(const std::string& filename) const { index->load_index(filename.c_str()); }

    void save(const std::string& filename) const { index->save_index(filename.c_str()); }

    void set_ef(size_t ef_search) const { index->set_ef(ef_search); }

    void build_index(
        const py::object& data,
        size_t ef_indexing,
        size_t num_iter = 3,
        size_t num_threads = UINT_MAX
    ) {
        py::array_t<float, py::array::c_style | py::array::forcecast> items(data);
        auto buffer = items.request();
        size_t num = 0;
        size_t dim = 0;
        get_arr_shape(buffer, num, dim);
        if (num != index->num_vertices() || dim != index->dimension()) {
            std::cerr
                << "The shape of data is different with initialization! Expected shape: ("
                << index->num_vertices() << ", " << index->dimension() << "), but got: ("
                << num << ", " << dim << ")\n";
            return;
        }

        symqg::QGBuilder builder(*index, ef_indexing, items.data(), num_threads);
        builder.build(num_iter);
        std::cout << "\tQuantizedGraph created\n";
    }

    void insert(const py::object& data, const py::object& ids, size_t ef_insert = 128) {
        py::array_t<float, py::array::c_style | py::array::forcecast> items(data);
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> id_arr(ids);

        auto item_buf = items.request();
        auto id_buf = id_arr.request();

        size_t rows = 0;
        size_t dim = 0;
        get_arr_shape(item_buf, rows, dim);
        if (dim != index->dimension()) {
            std::cerr << "insert vector dim mismatch\n";
            return;
        }

        const size_t expected_num_points = index->num_vertices();
        const size_t expected_dim = index->dimension();

        if (static_cast<size_t>(id_buf.size) != rows) {
            std::cerr << "insert ids size must equal number of vectors\n";
            return;
        }

        auto* src = static_cast<const float*>(item_buf.ptr);
        auto* id_ptr = static_cast<const int64_t*>(id_buf.ptr);
        std::vector<uint32_t> valid_ids;
        valid_ids.reserve(rows);
        std::vector<float> valid_vecs;
        valid_vecs.reserve(rows * expected_dim);

        for (size_t i = 0; i < rows; ++i) {
            int64_t id64 = id_ptr[i];
            if (id64 < 0 || static_cast<size_t>(id64) >= expected_num_points) {
                std::cerr << "insert id out of index range: " << id64 << "\n";
                continue;
            }
            valid_ids.push_back(static_cast<uint32_t>(id64));
            const float* vec = src + (i * expected_dim);
            valid_vecs.insert(valid_vecs.end(), vec, vec + expected_dim);
        }

        if (!valid_ids.empty()) {
            index->insert_batch(
                valid_vecs.data(),
                valid_ids.data(),
                valid_ids.size(),
                ef_insert
            );
        }
    }

    auto search(py_float_array& query, uint32_t knn) const {
        py_uint_array result(knn);
        auto* result_ptr = static_cast<uint32_t*>(result.request().ptr);
        index->search(query.data(0), knn, result_ptr);

        return result;
    }

    py_uint_array search_batch(py_float_array& queries, uint32_t knn) const {
        auto query_buf = queries.request();
        size_t rows = 0;
        size_t cols = 0;
        get_arr_shape(query_buf, rows, cols);

        if (cols != index->dimension()) {
            std::cerr << "Query dim mismatch. Expected " << index->dimension() << ", got "
                      << cols << "\n";
            return py_uint_array(py::array::ShapeContainer({0, 0}));
        }

        py_uint_array result({rows, static_cast<size_t>(knn)});
        auto* query_ptr = static_cast<float*>(query_buf.ptr);
        auto* result_ptr = static_cast<uint32_t*>(result.request().ptr);

        for (size_t i = 0; i < rows; ++i) {
            index->search(query_ptr + i * cols, knn, result_ptr + i * knn);
        }

        return result;
    }

    py_uint_array search_warm(
        size_t nq,
        py_float_array& queries,
        uint32_t knn,
        size_t ef_search,
        int streamseed_mode,
        int hint_level1_only,
        int hint_adaptive_gate_mode,
        int hint_hops,
        int hint_max_candidates,
        float hint_gate,
        float hint_qual_gate,
        float hint_cons_gate,
        float hint_gate_m_quantile,
        float hint_gate_o_quantile,
        int hint_gate_min_samples,
        int hint_table_slots,
        int hint_slot_capacity,
        py::object query_ids = py::none()
    ) const {
        (void)hint_adaptive_gate_mode;
        (void)hint_gate_m_quantile;
        (void)hint_gate_o_quantile;
        (void)hint_gate_min_samples;

        auto query_buf = queries.request();
        size_t rows = 0;
        size_t cols = 0;

        if (query_buf.ndim == 1) {
            cols = index->dimension();
            if (cols == 0 || nq == 0) {
                return py_uint_array(py::array::ShapeContainer({0}));
            }
            if (query_buf.shape[0] != static_cast<ssize_t>(nq * cols)) {
                std::cerr << "Query shape mismatch. Expected flat array of size "
                          << (nq * cols) << ", got " << query_buf.shape[0] << "\n";
                return py_uint_array(py::array::ShapeContainer({0}));
            }
            rows = nq;
        } else {
            get_arr_shape(query_buf, rows, cols);
            if (nq != rows) {
                std::cerr << "Query batch mismatch. Expected nq=" << nq << ", got "
                          << rows << "\n";
            }
        }

        if (cols != index->dimension()) {
            std::cerr << "Query dim mismatch. Expected " << index->dimension() << ", got "
                      << cols << "\n";
            return py_uint_array(py::array::ShapeContainer({0}));
        }

        index->set_ef(ef_search);

        const int64_t* query_id_ptr = nullptr;
        py_int64_array query_id_array;
        if (!query_ids.is_none()) {
            query_id_array = py_int64_array(query_ids);
            auto id_buf = query_id_array.request();
            if (id_buf.ndim != 1 || id_buf.shape[0] != static_cast<ssize_t>(rows)) {
                std::cerr << "query_ids shape mismatch. Expected " << rows
                          << " ids, got " << id_buf.shape[0] << "\n";
                return py_uint_array(py::array::ShapeContainer({0}));
            }
            query_id_ptr = static_cast<const int64_t*>(id_buf.ptr);
        }

        py_uint_array result(rows * static_cast<size_t>(knn));
        auto* query_ptr = static_cast<float*>(query_buf.ptr);
        auto* result_ptr = static_cast<uint32_t*>(result.request().ptr);

        for (size_t i = 0; i < rows; ++i) {
            const int64_t query_id = query_id_ptr ? query_id_ptr[i] : static_cast<int64_t>(i);
            index->search_warm(
                query_ptr + i * cols,
                knn,
                result_ptr + i * knn,
                query_id,
                streamseed_mode,
                hint_level1_only,
                hint_hops,
                hint_max_candidates,
                hint_gate,
                hint_qual_gate,
                hint_cons_gate,
                static_cast<size_t>(std::max(hint_table_slots, 0)),
                static_cast<size_t>(std::max(hint_slot_capacity, 0))
            );
        }

        return result;
    }
};

PYBIND11_MODULE(symphonyqg, m) {
    m.doc() = R"pbdoc(Towards Symphonious Integration of Graph and Quantization)pbdoc";

    py::class_<Index>(m, "Index")
        .def(
            py::init<const std::string&, const std::string&, size_t, size_t, size_t>(),
            py::arg("index_type"),
            py::arg("metric"),
            py::arg("num_elements"),
            py::arg("dimension"),
            py::arg("degree_bound") = 32
        )
        .def("load", &Index::load, py::arg("filename"))
        .def("save", &Index::save, py::arg("filename"))
        .def("set_ef", &Index::set_ef, py::arg("EF"))
        .def(
            "build_index",
            &Index::build_index,
            py::arg("data"),
            py::arg("EF"),
            py::arg("num_iter") = 3,
            py::arg("num_thread") = UINT_MAX
        )
        .def(
            "insert",
            &Index::insert,
            py::arg("data"),
            py::arg("ids"),
            py::arg("ef_insert") = 128
        )
        .def("search", &Index::search, py::arg("query"), py::arg("k"))
        .def("search_batch", &Index::search_batch, py::arg("queries"), py::arg("k"))
        .def(
            "search_warm",
            &Index::search_warm,
            py::arg("nq"),
            py::arg("queries"),
            py::arg("k"),
            py::arg("ef_search"),
            py::arg("streamseed_mode"),
            py::arg("hint_level1_only"),
            py::arg("hint_adaptive_gate_mode"),
            py::arg("hint_hops"),
            py::arg("hint_max_candidates"),
            py::arg("hint_gate"),
            py::arg("hint_qual_gate"),
            py::arg("hint_cons_gate"),
            py::arg("hint_gate_m_quantile"),
            py::arg("hint_gate_o_quantile"),
            py::arg("hint_gate_min_samples"),
            py::arg("hint_table_slots"),
            py::arg("hint_slot_capacity"),
            py::arg("query_ids") = py::none()
        );
}
