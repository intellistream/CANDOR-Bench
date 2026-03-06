package internal

// #cgo CXXFLAGS: -I${SRCDIR}/../../../src/CANDY/ConcurrentIndex -I${SRCDIR}/../../../include/CANDY/ConcurrentIndex -std=c++17
// #cgo LDFLAGS: -L${SRCDIR}/../build/lib -lindex
// #include <stdlib.h>
// #include "index_cgo.hpp"
import "C"
import (
	"fmt"
	"log"
	"math"
	"strconv"
	"time"
	"unsafe"
)

type IndexType int

const (
	IndexTypeHNSW IndexType = iota
	IndexTypeParlayHNSW
	IndexTypeParlayVamana
	IndexTypeVamana
	_ // reserved
	IndexTypeSegment
	_ // reserved
)

type Metric int

const (
	MetricL2           Metric = 0
	MetricInnerProduct Metric = 1
	MetricCosine       Metric = 2
)

type DataType int

const (
	DataTypeFloat DataType = iota
	DataTypeInt8
	DataTypeUint8
)

type IndexParams struct {
	Dim            int
	MaxElements    uint64
	M              int
	EfConstruction int
	LevelM         float32
	Alpha          float32
	VisitLimit     int

	Threads     int
	DataType    DataType
	UseNodeLock bool
	Metric      Metric

	SealThreshold int
	SealedType    IndexType
}

type QueryParams struct {
	EfSearch   uint
	BeamWidth  uint
	Alpha      float32
	VisitLimit uint
}

func BuildQueryParams(config *Config) QueryParams {
	return QueryParams{
		EfSearch:   uint(config.Search.EfSearch),
		BeamWidth:  uint(config.Search.BeamWidth),
		Alpha:      config.Search.Alpha,
		VisitLimit: uint(config.Search.VisitLimit),
	}
}

type Index struct {
	ptr         unsafe.Pointer
	indexType   IndexType
	config      IndexParams
	queryParams QueryParams
}

func NewIndex(indexType IndexType, params IndexParams) *Index {
	indexTypeNames := map[IndexType]string{
		IndexTypeHNSW:         "HNSW",
		IndexTypeParlayHNSW:   "ParlayHNSW",
		IndexTypeParlayVamana: "ParlayVamana",
		IndexTypeVamana:       "Vamana",
		IndexTypeSegment:      "Segmented",
	}

	indexName := indexTypeNames[indexType]
	if indexName == "" {
		indexName = "Unknown"
	}

	log.Printf("Creating %s index with params: dim=%d, max_elements=%d, threads=%d, metric=%d",
		indexName, params.Dim, params.MaxElements, params.Threads, params.Metric)

	cParams := C.IndexParams{
		dim:               C.size_t(params.Dim),
		max_elements:      C.size_t(params.MaxElements),
		M:                 C.size_t(params.M),
		ef_construction:   C.size_t(params.EfConstruction),
		level_m:           C.float(params.LevelM),
		alpha:             C.float(params.Alpha),
		visit_limit:       C.size_t(params.VisitLimit),
		seal_threshold:    C.size_t(params.SealThreshold),
		num_threads:       C.size_t(params.Threads),
		data_type:         C.DataType(params.DataType),
		use_node_lock:     C.bool(params.UseNodeLock),
		metric:            C.int(params.Metric),
		worker_scheduler:  C.size_t(0),
		sealed_index_type: C.IndexType(params.SealedType),
	}
	return &Index{
		ptr:         C.create_index(C.IndexType(indexType), cParams),
		indexType:   indexType,
		config:      params,
		queryParams: QueryParams{},
	}
}

func (i *Index) Close() {
	if i.ptr != nil {
		C.destroy_index(i.ptr)
		i.ptr = nil
	}
}

func (i *Index) Build(data [][]float32, tags []uint32) error {
	if len(data) == 0 || len(tags) == 0 {
		return nil
	}

	startTime := time.Now()

	numPoints := len(data)
	dim := len(data[0])
	flatData := make([]float32, numPoints*dim)
	for j, vec := range data {
		copy(flatData[j*dim:], vec)
	}

	result := C.build(
		i.ptr,
		(*C.float)(&flatData[0]),
		(*C.uint32_t)(&tags[0]),
		C.size_t(len(tags)),
	)

	buildTime := time.Since(startTime)
	qps := float64(numPoints) / buildTime.Seconds()

	log.Printf("Build completed: %d points in %v (%.2f QPS)\n", numPoints, buildTime, qps)

	if result != 0 {
		return fmt.Errorf("build index failed with code: %d", result)
	}
	return nil
}

func (i *Index) SetQueryParams(params QueryParams) {
	cParams := C.C_QueryParams{
		ef_search:   C.size_t(params.EfSearch),
		beam_width:  C.size_t(params.BeamWidth),
		alpha:       C.float(params.Alpha),
		visit_limit: C.size_t(params.VisitLimit),
	}
	C.set_query_params(i.ptr, cParams)

	i.queryParams = params
}

func (i *Index) Search(query []float32, k uint) ([]uint32, error) {
	results := make([]uint32, k)
	result := C.search(
		i.ptr,
		(*C.float)(&query[0]),
		C.size_t(k),
		(*C.uint32_t)(&results[0]),
	)
	if result != 0 {
		return nil, fmt.Errorf("search failed with code: %d", result)
	}
	return results, nil
}

func (i *Index) BatchInsert(batchData [][]float32, batchTags []uint32) error {
	if len(batchData) == 0 || len(batchTags) == 0 {
		return nil
	}

	numPoints := len(batchData)
	dim := len(batchData[0])
	flatData := make([]float32, numPoints*dim)
	for j, vec := range batchData {
		copy(flatData[j*dim:], vec)
	}

	result := C.batch_insert(
		i.ptr,
		(*C.float)(&flatData[0]),
		(*C.uint32_t)(&batchTags[0]),
		C.size_t(numPoints),
	)

	if result != 0 {
		return fmt.Errorf("batch insert failed with code: %d", result)
	}
	return nil
}

func (i *Index) BatchPartialInsert(batchData [][]float32, batchTags []uint32, partialLimits []uint64, updates []uint64) error {
	if len(batchData) == 0 || len(batchTags) == 0 {
		return nil
	}

	numPoints := len(batchData)
	if len(batchTags) != numPoints {
		return fmt.Errorf("batchInsertWithControl: tags length mismatch")
	}

	dim := len(batchData[0])
	flatData := make([]float32, numPoints*dim)
	for j, vec := range batchData {
		copy(flatData[j*dim:], vec)
	}

	var limitsPtr *C.size_t
	var limitsBuf []C.size_t
	if partialLimits != nil {
		if len(partialLimits) != numPoints {
			return fmt.Errorf("batchInsertWithControl: partialLimits length mismatch")
		}
		limitsBuf = make([]C.size_t, numPoints)
		for j, v := range partialLimits {
			limitsBuf[j] = C.size_t(v)
		}
		limitsPtr = (*C.size_t)(unsafe.Pointer(&limitsBuf[0]))
	}

	var updatesPtr *C.size_t
	var updatesBuf []C.size_t
	if updates != nil {
		if len(updates) != numPoints {
			return fmt.Errorf("batchInsertWithControl: updates length mismatch")
		}
		updatesBuf = make([]C.size_t, numPoints)
		updatesPtr = (*C.size_t)(unsafe.Pointer(&updatesBuf[0]))
	}

	res := C.batch_partial_insert(
		i.ptr,
		(*C.float)(&flatData[0]),
		(*C.uint32_t)(&batchTags[0]),
		C.size_t(numPoints),
		limitsPtr,
		updatesPtr,
	)

	for j := range updates {
		updates[j] = uint64(updatesBuf[j])
	}

	if res != 0 {
		return fmt.Errorf("batch insert with control failed: %d", res)
	}
	return nil
}

func (i *Index) BatchSearch(queries [][]float32, k uint32, visibleTs uint64) ([][]uint32, uint64, error) {
	if len(queries) == 0 {
		return nil, 0, nil
	}

	dim := len(queries[0])
	flatQueries := make([]float32, len(queries)*dim)
	for j, q := range queries {
		copy(flatQueries[j*dim:], q)
	}

	results := make([][]uint32, len(queries))
	for j := range results {
		results[j] = make([]uint32, k)
	}

	resultPtrs := make([]*C.uint32_t, len(queries))
	for j := range results {
		resultPtrs[j] = (*C.uint32_t)(&results[j][0])
	}
	resultPtrsC := C.malloc(C.size_t(len(queries)) * C.size_t(unsafe.Sizeof(uintptr(0))))
	defer C.free(resultPtrsC)
	ptrSlice := (*[1 << 30]*C.uint32_t)(resultPtrsC)[:len(queries):len(queries)]
	copy(ptrSlice, resultPtrs)

	var watermark C.size_t

	result := C.batch_search(
		i.ptr,
		(*C.float)(&flatQueries[0]),
		C.size_t(k),
		C.size_t(len(queries)),
		(**C.uint32_t)(resultPtrsC),
		&watermark,
		C.size_t(visibleTs),
	)

	if result != 0 {
		return nil, 0, fmt.Errorf("batch search failed with code: %d", result)
	}
	return results, uint64(watermark), nil
}

func (i *Index) DumpIndexParams() string {
	switch i.indexType {
	case IndexTypeHNSW:
		return fmt.Sprintf("M:%d, ef_c:%d", i.config.M, i.config.EfConstruction)
	case IndexTypeVamana:
		return fmt.Sprintf("alpha:%.2f", i.config.Alpha)
	case IndexTypeParlayHNSW, IndexTypeParlayVamana:
		return fmt.Sprintf("M:%d, ef_c:%d, alpha:%.2f", i.config.M, i.config.EfConstruction, i.config.Alpha)
	case IndexTypeSegment:
		return fmt.Sprintf("M:%d, ef_c:%d, seal:%d", i.config.M, i.config.EfConstruction, i.config.SealThreshold)
	default:
		return fmt.Sprintf("M:%d, ef_c:%d", i.config.M, i.config.EfConstruction)
	}
}

func (i *Index) DumpQueryParams() string {
	switch i.indexType {
	case IndexTypeHNSW:
		return fmt.Sprintf("ef_s:%d", i.queryParams.EfSearch)
	case IndexTypeVamana:
		return fmt.Sprintf("alpha:%.2f", i.queryParams.Alpha)
	case IndexTypeParlayHNSW, IndexTypeParlayVamana:
		return fmt.Sprintf("alpha:%.2f", i.queryParams.Alpha)
	case IndexTypeSegment:
		return fmt.Sprintf("ef_s:%d", i.queryParams.EfSearch)
	default:
		return ""
	}
}

func (i *Index) DumpStats() string {
	buffer := make([]byte, 2048)
	C.dump_stats(
		i.ptr,
		(*C.char)(unsafe.Pointer(&buffer[0])),
	)

	return C.GoString((*C.char)(unsafe.Pointer(&buffer[0])))
}

type IndexConfig struct {
	IndexType      string
	M              int
	EfConstruction int
	EfSearch       uint32
	SealedType     string
	SealThreshold  int
}

func BuildAlgorithmSuffix(config IndexConfig) string {
	var base string
	switch config.IndexType {
	case "hnsw", "cchnsw":
		base = fmt.Sprintf("m%d_efc%d_efs%d",
			config.M,
			config.EfConstruction,
			config.EfSearch)
	default:
		base = ""
	}

	if base == "" {
		return base
	}
	if config.SealedType != "" && config.SealThreshold > 0 {
		base = fmt.Sprintf("%s_s%s", base, formatSealThreshold(config.SealThreshold))
	}
	return base
}

func formatSealThreshold(th int) string {
	if th%1_000_000 == 0 {
		return fmt.Sprintf("%dm", th/1_000_000)
	}
	if th%1000 == 0 {
		return fmt.Sprintf("%dk", th/1000)
	}
	return strconv.Itoa(th)
}

func (i *Index) Snapshot() ([]byte, error) {
	if !i.SupportsSnapshot() {
		return nil, fmt.Errorf("snapshot not supported for index type: %d", i.indexType)
	}
	var cBuf *C.uint8_t
	var cSize C.size_t
	if rc := C.snapshot_index(i.ptr, &cBuf, &cSize); rc != 0 {
		return nil, fmt.Errorf("snapshot failed: %d", rc)
	}
	defer C.free_snapshot_buffer(cBuf)

	if cSize == 0 {
		return make([]byte, 0), nil
	}
	if cSize > math.MaxInt64 {
		return nil, fmt.Errorf("snapshot size too large: %d", cSize)
	}
	size := int(cSize)
	if cSize > math.MaxInt32 {
		buf := make([]byte, size)
		src := unsafe.Slice((*byte)(unsafe.Pointer(cBuf)), size)
		copy(buf, src)
		return buf, nil
	}
	return C.GoBytes(unsafe.Pointer(cBuf), C.int(size)), nil
}

func (i *Index) Restore(snapshot []byte) error {
	if !i.SupportsSnapshot() {
		return fmt.Errorf("restore not supported for index type: %d", i.indexType)
	}
	if len(snapshot) == 0 {
		return fmt.Errorf("empty snapshot")
	}
	rc := C.restore_index(i.ptr, (*C.uint8_t)(unsafe.Pointer(&snapshot[0])), C.size_t(len(snapshot)))
	if rc != 0 {
		return fmt.Errorf("restore failed: %d", rc)
	}
	return nil
}

func (i *Index) SupportsSnapshot() bool {
	return i.indexType == IndexTypeHNSW
}

func (i *Index) EnableInsertTelemetry(enable bool) {
	if i.indexType == IndexTypeHNSW {
		C.hnsw_enable_insert_telemetry(i.ptr, C.bool(enable))
	}
}
