package internal

// #cgo CXXFLAGS: -I${SRCDIR}/../algorithms -std=c++17
// #cgo LDFLAGS: -L${SRCDIR}/../build/lib -lindex
// #include <stdlib.h>
// #include "../algorithms/index_cgo.hpp"
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
	IndexTypeANNchor
	IndexTypeSegment
	IndexTypeANNchorDev
	IndexTypeANNchorM1
	IndexTypeANNchorPreempt
	IndexTypeANNchorM3
	IndexTypeANNchorTrim
	IndexTypeANNchorM2
	IndexTypeHNSWVisible
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

type GraphMutationStats struct {
	ConnectCalls                         uint64
	ConnectNs                            uint64
	LinkCriticalNs                       uint64
	UniqueLockWaitNs                     uint64
	SearchUniqueLockWaitNs               uint64
	SearchUniqueLockAcqs                 uint64
	SearchCriticalNs                     uint64
	LinkUpdates                          uint64
	UpperSearchNs                        uint64
	UpperSearchDistComps                 uint64
	UpperSearchEdgesScanned              uint64
	BaseSearchNs                         uint64
	BaseSearchExpansions                 uint64
	BaseSearchEdgesScanned               uint64
	BaseSearchDistComps                  uint64
	SelectNewNeighborsNs                 uint64
	SelectNewNeighborsInput              uint64
	SelectNewNeighborsSelected           uint64
	SelectNewNeighborsHeuristicDistComps uint64
	InsertedNodeLinkNs                   uint64
	InsertedNodeEdgesWritten             uint64
	ExistingNeighborUpdateLoopNs         uint64
	ExistingNeighborLoadScanNs           uint64
	ExistingNeighborLoadedEdges          uint64
	ExistingNeighborAppendNs             uint64
	ExistingNeighborPruneNs              uint64
	ExistingNeighborPruneCandidates      uint64
	ExistingNeighborPruneDistComps       uint64
	ExistingNeighborUndoRecordNs         uint64
	ExistingNeighborRewriteNs            uint64
	ExistingNeighborEdgesWritten         uint64
	ExistingNeighborEdgesPruned          uint64
	ExistingNeighborVisits               uint64
	ExistingNeighborAppends              uint64
	ExistingNeighborPrunes               uint64
	ExistingNeighborPrunedEdgesRecorded  uint64
}

type SearchWorkStats struct {
	SearchKnnNs                 uint64
	SearchKnnThreadCpuNs        uint64
	WorkStartCPU                int32
	WorkEndCPU                  int32
	ResultCopyNs                uint64
	EntryNs                     uint64
	UpperSearchNs               uint64
	BaseSearchNs                uint64
	ResultMaterializeNs         uint64
	SnapshotGuardNs             uint64
	VisitedListGetNs            uint64
	VisitedListReleaseNs        uint64
	UpperLockWaitNs             uint64
	Level0LockWaitNs            uint64
	DistanceComputations        uint64
	UpperDistanceComputations   uint64
	Level0DistanceComputations  uint64
	DistanceComputeNs           uint64
	UpperDistanceComputeNs      uint64
	Level0DistanceComputeNs     uint64
	Level0QueuePopNs            uint64
	Level0AdjFetchNs            uint64
	Level0LocalityCaptureNs     uint64
	Level0CandidateLoopNs       uint64
	Level0VisitedCheckNs        uint64
	Level0VisibilityCheckNs     uint64
	Level0CandidateAcceptNs     uint64
	UpperHops                   uint64
	UpperEdgesScanned           uint64
	Level0Expansions            uint64
	Level0EdgesScanned          uint64
	CandidatePops               uint64
	CandidatePushes             uint64
	VisitedNodes                uint64
	ResultPushes                uint64
	InvisibleExpansions         uint64
	InvisibleEdges              uint64
	InvisibleCandidateDistComps uint64
	InvisibleCandidateEnqueues  uint64
	FutureSkipHops              uint64
	RewriteActiveExpansions     uint64
	RewriteRecentExpansions     uint64
	RewritePeriodExpansions     uint64
	RewritePeriodActiveSum      uint64
	RewritePeriodActiveMax      uint64
	ExpandVisibleCount          uint64
	ExpandRecent1KHits          uint64
	ExpandRecent4KHits          uint64
	ExpandRecent16KHits         uint64
	ExpandLabelGapSum           uint64
	ExpandLabelSpan             uint64
	ExpandUniqueLabel4KBuckets  uint64
	ExpandUniqueData4KPages     uint64
	ExpandUniqueData2MPages     uint64
	ExpandUniqueAdj4KPages      uint64
	ExpandUniqueAdj2MPages      uint64
	ExpandUniqueOverflow        uint64
	PathCount                   uint32
	PathLabels                  [128]uint32
	PathDists                   [128]float32
}

func MonotonicRawNs() uint64 {
	return uint64(C.annbench_monotonic_raw_ns())
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
		IndexTypeHNSW:           "HNSW",
		IndexTypeParlayHNSW:     "ParlayHNSW",
		IndexTypeParlayVamana:   "ParlayVamana",
		IndexTypeVamana:         "Vamana",
		IndexTypeANNchor:        "ANNchor",
		IndexTypeSegment:        "Segmented",
		IndexTypeANNchorDev:     "ANNchorDev",
		IndexTypeANNchorM1:      "ANNchorM1",
		IndexTypeANNchorPreempt: "ANNchorPreempt",
		IndexTypeANNchorM3:      "ANNchorM3",
		IndexTypeANNchorTrim:    "ANNchorTrim",
		IndexTypeANNchorM2:      "ANNchorM2",
		IndexTypeHNSWVisible:    "HNSWVisible",
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

func (i *Index) SetEnableMvcc(enable bool) {
	if i.ptr != nil {
		C.annchor_set_enable_mvcc(i.ptr, C.bool(enable))
	}
}

func (i *Index) SetEnableUndoRecovery(enable bool) {
	if i.ptr != nil {
		C.annchor_set_enable_undo_recovery(i.ptr, C.bool(enable))
	}
}

func (i *Index) SetVisibilityMode(mode int) {
	if i.ptr != nil {
		C.hnsw_visible_set_visibility_mode(i.ptr, C.int(mode))
	}
}

func (i *Index) SetEnableTrimRecoveryFilter(enable bool) {
	if i.ptr != nil {
		C.annchor_trim_set_enable_recovery_filter(i.ptr, C.bool(enable))
	}
}

func (i *Index) SetTrimRecoveryRelaxFactor(factor float64) {
	if i.ptr != nil {
		C.annchor_trim_set_recovery_relax_factor(i.ptr, C.float(factor))
	}
}

func (i *Index) SetTrimRecoveryMarginRatio(ratio float64) {
	if i.ptr != nil {
		C.annchor_trim_set_recovery_margin_ratio(i.ptr, C.float(ratio))
	}
}

func (i *Index) SetEnableS3(enable bool) {
	if i.ptr != nil {
		if i.indexType == IndexTypeHNSW {
			C.hnsw_set_enable_s3(i.ptr, C.bool(enable))
		} else {
			C.annchor_set_enable_s3(i.ptr, C.bool(enable))
		}
	}
}

func (i *Index) SetS3ProximityThreshold(threshold float32) {
	if i.ptr != nil {
		if i.indexType == IndexTypeHNSW {
			C.hnsw_set_s3_proximity_threshold(i.ptr, C.float(threshold))
		} else {
			C.annchor_set_s3_proximity_threshold(i.ptr, C.float(threshold))
		}
	}
}

func (i *Index) SetEnablePathSkip(enable bool) {
	if i.ptr != nil && i.indexType == IndexTypeHNSW {
		C.hnsw_set_enable_path_skip(i.ptr, C.bool(enable))
	}
}

func (i *Index) SetEnableCandidateInjection(enable bool) {
	if i.ptr != nil && i.indexType == IndexTypeHNSW {
		C.hnsw_set_enable_candidate_injection(i.ptr, C.bool(enable))
	}
}

func (i *Index) SetEnableSearchSharing(enable bool) {
	if i.ptr != nil && i.indexType == IndexTypeHNSW {
		C.hnsw_set_enable_search_sharing(i.ptr, C.bool(enable))
	}
}

func (i *Index) SetSearchSharingCheckInterval(interval int) {
	if i.ptr != nil && i.indexType == IndexTypeHNSW {
		C.hnsw_set_search_sharing_check_interval(i.ptr, C.int(interval))
	}
}

func (i *Index) SetEnableWarmStart(enable bool) {
	if i.ptr != nil && i.indexType == IndexTypeHNSW {
		C.hnsw_set_enable_warm_start(i.ptr, C.bool(enable))
	}
}

func (i *Index) SetEnableSlipstream(enable bool) {
	if i.ptr != nil {
		C.annchor_set_enable_slipstream(i.ptr, C.bool(enable))
	}
}

func (i *Index) SetSlipstreamTTL(ttlNs uint64) {
	if i.ptr != nil {
		C.annchor_set_slipstream_ttl(i.ptr, C.uint64_t(ttlNs))
	}
}

func (i *Index) SetSlipstreamMode(mode int) {
	if i.ptr != nil {
		C.annchor_m3_set_slipstream_mode(i.ptr, C.int(mode))
	}
}

func (i *Index) SetEnablePruneOnlyNodeLock(enable bool) {
	if i.ptr != nil {
		C.annchor_m3_set_prune_only_node_lock(i.ptr, C.bool(enable))
	}
}

func (i *Index) SetSlipstreamQualityGate(enable bool) {
	if i.ptr != nil {
		C.annchor_set_slipstream_quality_gate(i.ptr, C.bool(enable))
	}
}

func (i *Index) SetSlipstreamSkipRatio(ratio float32) {
	if i.ptr != nil {
		C.annchor_set_slipstream_skip_ratio(i.ptr, C.float(ratio))
	}
}

func (i *Index) SetEnableCnr(enable bool) {
	if i.ptr != nil {
		C.annchor_dev_set_enable_cnr(i.ptr, C.bool(enable))
	}
}

func (i *Index) SetCnrDegreeThreshold(threshold float32) {
	if i.ptr != nil {
		C.annchor_dev_set_cnr_degree_threshold(i.ptr, C.float(threshold))
	}
}

func (i *Index) SetCnrMaxRecover(maxRecover int) {
	if i.ptr != nil {
		C.annchor_dev_set_cnr_max_recover(i.ptr, C.int(maxRecover))
	}
}

func (i *Index) SetCnrStagnationHops(hops int) {
	if i.ptr != nil {
		C.annchor_dev_set_cnr_stagnation_hops(i.ptr, C.int(hops))
	}
}

func (i *Index) SetEnableM2DualPath(enable bool) {
	if i.ptr != nil {
		C.annchor_dev_set_enable_m2_dual_path(i.ptr, C.bool(enable))
	}
}

func (i *Index) SetM2RiskHops(hops int) {
	if i.ptr != nil {
		C.annchor_dev_set_m2_risk_hops(i.ptr, C.int(hops))
	}
}

func (i *Index) SetM2AssistBudget(budget int) {
	if i.ptr != nil {
		C.annchor_dev_set_m2_assist_budget(i.ptr, C.int(budget))
	}
}

func (i *Index) SetPreemptEnableM2(enable bool) {
	if i.ptr != nil {
		C.annchor_preempt_set_enable_m2(i.ptr, C.bool(enable))
	}
}

func (i *Index) SetPreemptQuantumPoints(quantumPoints int) {
	if i.ptr != nil {
		C.annchor_preempt_set_quantum_points(i.ptr, C.int(quantumPoints))
	}
}

func (i *Index) SetPreemptSearchBacklogThreshold(threshold int) {
	if i.ptr != nil {
		C.annchor_preempt_set_search_backlog_threshold(i.ptr, C.int(threshold))
	}
}

func (i *Index) SetPreemptMaxYieldsPerBatch(maxYieldsPerBatch int) {
	if i.ptr != nil {
		C.annchor_preempt_set_max_yields_per_batch(i.ptr, C.int(maxYieldsPerBatch))
	}
}

func (i *Index) SetPreemptBudgetWindowUs(windowUs int) {
	if i.ptr != nil {
		C.annchor_preempt_set_budget_window_us(i.ptr, C.int(windowUs))
	}
}

func (i *Index) SetPreemptBudgetPct(budgetPct float64) {
	if i.ptr != nil {
		C.annchor_preempt_set_budget_pct(i.ptr, C.float(budgetPct))
	}
}

func (i *Index) SetPreemptPriorityCap(priorityCap int) {
	if i.ptr != nil {
		C.annchor_preempt_set_priority_cap(i.ptr, C.int(priorityCap))
	}
}

func (i *Index) SetPreemptRuntimeSearchBacklog(backlog int) {
	C.annchor_preempt_set_runtime_search_backlog(C.int(backlog))
}

func (i *Index) SetPreemptRuntimePrioritySearches(prioritySearches int) {
	C.annchor_preempt_set_runtime_priority_searches(C.int(prioritySearches))
}

// GetInflightPoints returns the number of point IDs that have entered
// addPoint (incremented cur_element_count) but are not yet committed to
// the query-visible watermark. Bounded by ANNchor's internal thread-pool
// depth. Returns -1 if unsupported.
func (i *Index) GetInflightPoints() int64 {
	return int64(C.annchor_get_inflight_points(i.ptr))
}

func (i *Index) InflightLabels(maxLabels int) []uint32 {
	if i.ptr == nil || maxLabels <= 0 {
		return nil
	}
	out := make([]uint32, maxLabels)
	n := int(C.annchor_get_inflight_labels(
		i.ptr,
		(*C.uint32_t)(unsafe.Pointer(&out[0])),
		C.size_t(maxLabels),
	))
	if n <= 0 {
		return nil
	}
	if n > len(out) {
		n = len(out)
	}
	return out[:n]
}

func (i *Index) InflightLabelsBefore(maxLabels int, snapshotRawNs uint64) []uint32 {
	if i.ptr == nil || maxLabels <= 0 {
		return nil
	}
	out := make([]uint32, maxLabels)
	n := int(C.annchor_get_inflight_labels_before(
		i.ptr,
		(*C.uint32_t)(unsafe.Pointer(&out[0])),
		C.size_t(maxLabels),
		C.uint64_t(snapshotRawNs),
	))
	if n <= 0 {
		return nil
	}
	if n > len(out) {
		n = len(out)
	}
	return out[:n]
}

func (i *Index) BaseNeighbors(labels []uint32, maxNeighbors int) ([][]uint32, error) {
	if i.ptr == nil || len(labels) == 0 || maxNeighbors <= 0 {
		return nil, nil
	}
	out := make([]uint32, len(labels)*maxNeighbors)
	counts := make([]C.size_t, len(labels))
	rc := C.annchor_batch_base_neighbors(
		i.ptr,
		(*C.uint32_t)(unsafe.Pointer(&labels[0])),
		C.size_t(len(labels)),
		C.size_t(maxNeighbors),
		(*C.uint32_t)(unsafe.Pointer(&out[0])),
		(*C.size_t)(unsafe.Pointer(&counts[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("base neighbor export unsupported")
	}
	neighbors := make([][]uint32, len(labels))
	for i := range labels {
		n := int(counts[i])
		if n > maxNeighbors {
			n = maxNeighbors
		}
		start := i * maxNeighbors
		neighbors[i] = append([]uint32(nil), out[start:start+n]...)
	}
	return neighbors, nil
}

type FreshJoinParams struct {
	QueryResultPrefix   int
	SmallFreshThreshold int
	CandidateFloorPct   float64
}

type FreshJoinStats struct {
	Queries                uint64
	FreshVectors           uint64
	ActivePlacementCenters uint64
	ActivePlacementRegions uint64
	SelectedCandidates     uint64
	ExactedCandidates      uint64
	ExactFallbackQueries   uint64
	FloorFallbackQueries   uint64
}

func (i *Index) FreshJoinMerge(queries [][]float32, committed [][]uint32, k uint32, freshData [][]float32, freshTags []uint32, committedViewLimit uint64, params FreshJoinParams) ([][]uint32, FreshJoinStats, error) {
	stats := FreshJoinStats{}
	if i.ptr == nil {
		return nil, stats, fmt.Errorf("fresh join merge: nil index")
	}
	if len(queries) == 0 {
		return nil, stats, nil
	}
	if len(committed) != len(queries) {
		return nil, stats, fmt.Errorf("fresh join merge: committed result count mismatch")
	}
	if len(freshData) != len(freshTags) {
		return nil, stats, fmt.Errorf("fresh join merge: fresh data/tag count mismatch")
	}

	dim := len(queries[0])
	flatQueries := make([]float32, len(queries)*dim)
	for j, q := range queries {
		if len(q) < dim {
			return nil, stats, fmt.Errorf("fresh join merge: query dimension mismatch")
		}
		copy(flatQueries[j*dim:], q)
	}

	flatCommitted := make([]uint32, len(queries)*int(k))
	for qi, row := range committed {
		for j := 0; j < int(k) && j < len(row); j++ {
			flatCommitted[qi*int(k)+j] = row[j]
		}
	}

	var flatFresh []float32
	var freshPtr *C.float
	var freshTagsPtr *C.uint32_t
	if len(freshData) > 0 {
		flatFresh = make([]float32, len(freshData)*dim)
		for j, point := range freshData {
			if len(point) < dim {
				return nil, stats, fmt.Errorf("fresh join merge: fresh dimension mismatch")
			}
			copy(flatFresh[j*dim:], point)
		}
		freshPtr = (*C.float)(unsafe.Pointer(&flatFresh[0]))
		freshTagsPtr = (*C.uint32_t)(unsafe.Pointer(&freshTags[0]))
	}

	results := make([][]uint32, len(queries))
	for j := range results {
		results[j] = make([]uint32, k)
	}
	resultPtrs := make([]*C.uint32_t, len(queries))
	for j := range results {
		resultPtrs[j] = (*C.uint32_t)(unsafe.Pointer(&results[j][0]))
	}
	resultPtrsC := C.malloc(C.size_t(len(queries)) * C.size_t(unsafe.Sizeof(uintptr(0))))
	defer C.free(resultPtrsC)
	ptrSlice := (*[1 << 30]*C.uint32_t)(resultPtrsC)[:len(queries):len(queries)]
	copy(ptrSlice, resultPtrs)

	cParams := C.C_FreshJoinParams{
		query_result_prefix:   C.size_t(maxInt(params.QueryResultPrefix, 1)),
		small_fresh_threshold: C.size_t(maxInt(params.SmallFreshThreshold, 0)),
		candidate_floor_pct:   C.double(params.CandidateFloorPct),
	}
	cStats := C.C_FreshJoinStats{}
	rc := C.annchor_fresh_join_merge(
		i.ptr,
		(*C.float)(unsafe.Pointer(&flatQueries[0])),
		C.size_t(k),
		C.size_t(len(queries)),
		freshPtr,
		freshTagsPtr,
		C.size_t(len(freshData)),
		(*C.uint32_t)(unsafe.Pointer(&flatCommitted[0])),
		(**C.uint32_t)(resultPtrsC),
		C.size_t(committedViewLimit),
		cParams,
		&cStats,
	)
	if rc != 0 {
		return nil, stats, fmt.Errorf("fresh join merge failed with code: %d", int(rc))
	}
	stats = FreshJoinStats{
		Queries:                uint64(cStats.queries),
		FreshVectors:           uint64(cStats.fresh_vectors),
		ActivePlacementCenters: uint64(cStats.active_placement_centers),
		ActivePlacementRegions: uint64(cStats.active_placement_regions),
		SelectedCandidates:     uint64(cStats.selected_candidates),
		ExactedCandidates:      uint64(cStats.exacted_candidates),
		ExactFallbackQueries:   uint64(cStats.exact_fallback_queries),
		FloorFallbackQueries:   uint64(cStats.floor_fallback_queries),
	}
	return results, stats, nil
}

func (i *Index) FreshJoinMergeLabels(queries [][]float32, committed [][]uint32, k uint32, freshTags []uint32, params FreshJoinParams) ([][]uint32, FreshJoinStats, error) {
	stats := FreshJoinStats{}
	if i.ptr == nil {
		return nil, stats, fmt.Errorf("fresh join merge labels: nil index")
	}
	if len(queries) == 0 {
		return nil, stats, nil
	}
	if len(committed) != len(queries) {
		return nil, stats, fmt.Errorf("fresh join merge labels: committed result count mismatch")
	}

	dim := len(queries[0])
	flatQueries := make([]float32, len(queries)*dim)
	for j, q := range queries {
		if len(q) < dim {
			return nil, stats, fmt.Errorf("fresh join merge labels: query dimension mismatch")
		}
		copy(flatQueries[j*dim:], q)
	}

	flatCommitted := make([]uint32, len(queries)*int(k))
	for qi, row := range committed {
		for j := 0; j < int(k) && j < len(row); j++ {
			flatCommitted[qi*int(k)+j] = row[j]
		}
	}

	results := make([][]uint32, len(queries))
	for j := range results {
		results[j] = make([]uint32, k)
	}
	resultPtrs := make([]*C.uint32_t, len(queries))
	for j := range results {
		resultPtrs[j] = (*C.uint32_t)(unsafe.Pointer(&results[j][0]))
	}
	resultPtrsC := C.malloc(C.size_t(len(queries)) * C.size_t(unsafe.Sizeof(uintptr(0))))
	defer C.free(resultPtrsC)
	ptrSlice := (*[1 << 30]*C.uint32_t)(resultPtrsC)[:len(queries):len(queries)]
	copy(ptrSlice, resultPtrs)

	var freshTagsPtr *C.uint32_t
	if len(freshTags) > 0 {
		freshTagsPtr = (*C.uint32_t)(unsafe.Pointer(&freshTags[0]))
	}

	cParams := C.C_FreshJoinParams{
		query_result_prefix:   C.size_t(maxInt(params.QueryResultPrefix, 1)),
		small_fresh_threshold: C.size_t(maxInt(params.SmallFreshThreshold, 0)),
		candidate_floor_pct:   C.double(params.CandidateFloorPct),
	}
	cStats := C.C_FreshJoinStats{}
	rc := C.annchor_fresh_join_merge_labels(
		i.ptr,
		(*C.float)(unsafe.Pointer(&flatQueries[0])),
		C.size_t(k),
		C.size_t(len(queries)),
		freshTagsPtr,
		C.size_t(len(freshTags)),
		(*C.uint32_t)(unsafe.Pointer(&flatCommitted[0])),
		(**C.uint32_t)(resultPtrsC),
		cParams,
		&cStats,
	)
	if rc != 0 {
		return nil, stats, fmt.Errorf("fresh join merge labels failed with code: %d", int(rc))
	}
	stats = FreshJoinStats{
		Queries:                uint64(cStats.queries),
		FreshVectors:           uint64(cStats.fresh_vectors),
		ActivePlacementCenters: uint64(cStats.active_placement_centers),
		ActivePlacementRegions: uint64(cStats.active_placement_regions),
		SelectedCandidates:     uint64(cStats.selected_candidates),
		ExactedCandidates:      uint64(cStats.exacted_candidates),
		ExactFallbackQueries:   uint64(cStats.exact_fallback_queries),
		FloorFallbackQueries:   uint64(cStats.floor_fallback_queries),
	}
	return results, stats, nil
}

func maxInt(v int, floor int) int {
	if v < floor {
		return floor
	}
	return v
}

func (i *Index) GraphLinkWriteProbe(labels []uint32, loops int, maxEdges int) uint64 {
	if i.ptr == nil || len(labels) == 0 || loops <= 0 {
		return 0
	}
	if maxEdges < 0 {
		maxEdges = 0
	}
	return uint64(C.graph_link_write_probe(
		i.ptr,
		(*C.uint32_t)(unsafe.Pointer(&labels[0])),
		C.size_t(len(labels)),
		C.size_t(loops),
		C.size_t(maxEdges),
	))
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
	results, watermark, _, err := i.batchSearch(queries, k, visibleTs, false, false)
	return results, watermark, err
}

func (i *Index) BatchSearchMeasured(queries [][]float32, k uint32, visibleTs uint64) ([][]uint32, uint64, error) {
	results, watermark, _, err := i.batchSearch(queries, k, visibleTs, true, false)
	return results, watermark, err
}

func (i *Index) BatchSearchMeasuredWithWork(queries [][]float32, k uint32, visibleTs uint64) ([][]uint32, uint64, []SearchWorkStats, error) {
	return i.batchSearch(queries, k, visibleTs, true, true)
}

func (i *Index) BatchSearchPathWork(queries [][]float32, k uint32, visibleTs uint64) ([][]uint32, uint64, []SearchWorkStats, error) {
	return i.batchSearch(queries, k, visibleTs, false, true)
}

func (i *Index) batchSearch(queries [][]float32, k uint32, visibleTs uint64, measured bool, collectWork bool) ([][]uint32, uint64, []SearchWorkStats, error) {
	if len(queries) == 0 {
		return nil, 0, nil, nil
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

	var result C.int
	var work []SearchWorkStats
	if collectWork {
		workBuf := make([]C.C_SearchWorkStats, len(queries))
		if measured {
			result = C.batch_search_measured_work(
				i.ptr,
				(*C.float)(&flatQueries[0]),
				C.size_t(k),
				C.size_t(len(queries)),
				(**C.uint32_t)(resultPtrsC),
				(*C.C_SearchWorkStats)(unsafe.Pointer(&workBuf[0])),
				&watermark,
				C.size_t(visibleTs),
			)
		} else {
			result = C.batch_search_path_work(
				i.ptr,
				(*C.float)(&flatQueries[0]),
				C.size_t(k),
				C.size_t(len(queries)),
				(**C.uint32_t)(resultPtrsC),
				(*C.C_SearchWorkStats)(unsafe.Pointer(&workBuf[0])),
				&watermark,
				C.size_t(visibleTs),
			)
		}
		work = make([]SearchWorkStats, len(queries))
		for j, s := range workBuf {
			work[j] = SearchWorkStats{
				SearchKnnNs:                 uint64(s.searchknn_ns),
				SearchKnnThreadCpuNs:        uint64(s.searchknn_thread_cpu_ns),
				WorkStartCPU:                int32(s.work_start_cpu),
				WorkEndCPU:                  int32(s.work_end_cpu),
				ResultCopyNs:                uint64(s.result_copy_ns),
				EntryNs:                     uint64(s.entry_ns),
				UpperSearchNs:               uint64(s.upper_search_ns),
				BaseSearchNs:                uint64(s.base_search_ns),
				ResultMaterializeNs:         uint64(s.result_materialize_ns),
				SnapshotGuardNs:             uint64(s.snapshot_guard_ns),
				VisitedListGetNs:            uint64(s.visited_list_get_ns),
				VisitedListReleaseNs:        uint64(s.visited_list_release_ns),
				UpperLockWaitNs:             uint64(s.upper_lock_wait_ns),
				Level0LockWaitNs:            uint64(s.level0_lock_wait_ns),
				DistanceComputations:        uint64(s.distance_computations),
				UpperDistanceComputations:   uint64(s.upper_distance_computations),
				Level0DistanceComputations:  uint64(s.level0_distance_computations),
				DistanceComputeNs:           uint64(s.distance_compute_ns),
				UpperDistanceComputeNs:      uint64(s.upper_distance_compute_ns),
				Level0DistanceComputeNs:     uint64(s.level0_distance_compute_ns),
				Level0QueuePopNs:            uint64(s.level0_queue_pop_ns),
				Level0AdjFetchNs:            uint64(s.level0_adj_fetch_ns),
				Level0LocalityCaptureNs:     uint64(s.level0_locality_capture_ns),
				Level0CandidateLoopNs:       uint64(s.level0_candidate_loop_ns),
				Level0VisitedCheckNs:        uint64(s.level0_visited_check_ns),
				Level0VisibilityCheckNs:     uint64(s.level0_visibility_check_ns),
				Level0CandidateAcceptNs:     uint64(s.level0_candidate_accept_ns),
				UpperHops:                   uint64(s.upper_hops),
				UpperEdgesScanned:           uint64(s.upper_edges_scanned),
				Level0Expansions:            uint64(s.level0_expansions),
				Level0EdgesScanned:          uint64(s.level0_edges_scanned),
				CandidatePops:               uint64(s.candidate_pops),
				CandidatePushes:             uint64(s.candidate_pushes),
				VisitedNodes:                uint64(s.visited_nodes),
				ResultPushes:                uint64(s.result_pushes),
				InvisibleExpansions:         uint64(s.invisible_expansions),
				InvisibleEdges:              uint64(s.invisible_edges),
				InvisibleCandidateDistComps: uint64(s.invisible_candidate_dist_comps),
				InvisibleCandidateEnqueues:  uint64(s.invisible_candidate_enqueues),
				FutureSkipHops:              uint64(s.future_skip_hops),
				RewriteActiveExpansions:     uint64(s.rewrite_active_expansions),
				RewriteRecentExpansions:     uint64(s.rewrite_recent_expansions),
				RewritePeriodExpansions:     uint64(s.rewrite_period_expansions),
				RewritePeriodActiveSum:      uint64(s.rewrite_period_active_sum),
				RewritePeriodActiveMax:      uint64(s.rewrite_period_active_max),
				ExpandVisibleCount:          uint64(s.expand_visible_count),
				ExpandRecent1KHits:          uint64(s.expand_recent_1k_hits),
				ExpandRecent4KHits:          uint64(s.expand_recent_4k_hits),
				ExpandRecent16KHits:         uint64(s.expand_recent_16k_hits),
				ExpandLabelGapSum:           uint64(s.expand_label_gap_sum),
				ExpandLabelSpan:             uint64(s.expand_label_span),
				ExpandUniqueLabel4KBuckets:  uint64(s.expand_unique_label_4k_buckets),
				ExpandUniqueData4KPages:     uint64(s.expand_unique_data_4k_pages),
				ExpandUniqueData2MPages:     uint64(s.expand_unique_data_2m_pages),
				ExpandUniqueAdj4KPages:      uint64(s.expand_unique_adj_4k_pages),
				ExpandUniqueAdj2MPages:      uint64(s.expand_unique_adj_2m_pages),
				ExpandUniqueOverflow:        uint64(s.expand_unique_overflow),
				PathCount:                   uint32(s.path_count),
			}
			for p := 0; p < int(work[j].PathCount) && p < len(work[j].PathLabels); p++ {
				work[j].PathLabels[p] = uint32(s.path_labels[p])
				work[j].PathDists[p] = float32(s.path_dists[p])
			}
		}
	} else if measured {
		result = C.batch_search_measured(
			i.ptr,
			(*C.float)(&flatQueries[0]),
			C.size_t(k),
			C.size_t(len(queries)),
			(**C.uint32_t)(resultPtrsC),
			&watermark,
			C.size_t(visibleTs),
		)
	} else {
		result = C.batch_search(
			i.ptr,
			(*C.float)(&flatQueries[0]),
			C.size_t(k),
			C.size_t(len(queries)),
			(**C.uint32_t)(resultPtrsC),
			&watermark,
			C.size_t(visibleTs),
		)
	}

	if result != 0 {
		return nil, 0, nil, fmt.Errorf("batch search failed with code: %d", result)
	}
	return results, uint64(watermark), work, nil
}

func (i *Index) DumpIndexParams() string {
	switch i.indexType {
	case IndexTypeHNSW, IndexTypeANNchor, IndexTypeANNchorM1, IndexTypeANNchorPreempt, IndexTypeANNchorTrim, IndexTypeANNchorM2, IndexTypeHNSWVisible:
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
	case IndexTypeHNSW, IndexTypeANNchor, IndexTypeANNchorM1, IndexTypeANNchorPreempt, IndexTypeANNchorTrim, IndexTypeANNchorM2, IndexTypeHNSWVisible:
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
	bufLen := C.dump_stats_len(i.ptr)
	if bufLen == 0 {
		return ""
	}

	buffer := make([]byte, int(bufLen))
	C.dump_stats_copy(
		i.ptr,
		(*C.char)(unsafe.Pointer(&buffer[0])),
		bufLen,
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
	case "hnsw", "hnsw-visible", "replica_snapshot", "replica_refresh", "cchnsw", "annchor", "annchor-m1", "annchor-m2", "annchor-m3", "annchor-preempt", "annchor-trim":
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
	return i.indexType == IndexTypeHNSW || i.indexType == IndexTypeANNchor || i.indexType == IndexTypeANNchorM1 || i.indexType == IndexTypeANNchorPreempt || i.indexType == IndexTypeANNchorTrim || i.indexType == IndexTypeANNchorM2 || i.indexType == IndexTypeHNSWVisible
}

func (i *Index) EnableInsertTelemetry(enable bool) {
	if i.indexType == IndexTypeHNSW || i.indexType == IndexTypeANNchor || i.indexType == IndexTypeANNchorM1 || i.indexType == IndexTypeANNchorPreempt || i.indexType == IndexTypeANNchorTrim || i.indexType == IndexTypeANNchorM2 || i.indexType == IndexTypeHNSWVisible {
		C.hnsw_enable_insert_telemetry(i.ptr, C.bool(enable))
	}
}

func (i *Index) GraphMutationStats() (GraphMutationStats, bool) {
	var cstats C.C_GraphMutationStats
	if !bool(C.graph_mutation_stats(i.ptr, &cstats)) {
		return GraphMutationStats{}, false
	}
	return GraphMutationStats{
		ConnectCalls:                         uint64(cstats.connect_calls),
		ConnectNs:                            uint64(cstats.connect_ns),
		LinkCriticalNs:                       uint64(cstats.link_critical_ns),
		UniqueLockWaitNs:                     uint64(cstats.unique_lock_wait_ns),
		SearchUniqueLockWaitNs:               uint64(cstats.search_unique_lock_wait_ns),
		SearchUniqueLockAcqs:                 uint64(cstats.search_unique_lock_acqs),
		SearchCriticalNs:                     uint64(cstats.search_critical_ns),
		LinkUpdates:                          uint64(cstats.link_updates),
		UpperSearchNs:                        uint64(cstats.upper_search_ns),
		UpperSearchDistComps:                 uint64(cstats.upper_search_dist_comps),
		UpperSearchEdgesScanned:              uint64(cstats.upper_search_edges_scanned),
		BaseSearchNs:                         uint64(cstats.base_search_ns),
		BaseSearchExpansions:                 uint64(cstats.base_search_expansions),
		BaseSearchEdgesScanned:               uint64(cstats.base_search_edges_scanned),
		BaseSearchDistComps:                  uint64(cstats.base_search_dist_comps),
		SelectNewNeighborsNs:                 uint64(cstats.select_new_neighbors_ns),
		SelectNewNeighborsInput:              uint64(cstats.select_new_neighbors_input),
		SelectNewNeighborsSelected:           uint64(cstats.select_new_neighbors_selected),
		SelectNewNeighborsHeuristicDistComps: uint64(cstats.select_new_neighbors_heuristic_dist_comps),
		InsertedNodeLinkNs:                   uint64(cstats.inserted_node_link_ns),
		InsertedNodeEdgesWritten:             uint64(cstats.inserted_node_edges_written),
		ExistingNeighborUpdateLoopNs:         uint64(cstats.existing_neighbor_update_loop_ns),
		ExistingNeighborLoadScanNs:           uint64(cstats.existing_neighbor_load_scan_ns),
		ExistingNeighborLoadedEdges:          uint64(cstats.existing_neighbor_loaded_edges),
		ExistingNeighborAppendNs:             uint64(cstats.existing_neighbor_append_ns),
		ExistingNeighborPruneNs:              uint64(cstats.existing_neighbor_prune_ns),
		ExistingNeighborPruneCandidates:      uint64(cstats.existing_neighbor_prune_candidates),
		ExistingNeighborPruneDistComps:       uint64(cstats.existing_neighbor_prune_dist_comps),
		ExistingNeighborUndoRecordNs:         uint64(cstats.existing_neighbor_undo_record_ns),
		ExistingNeighborRewriteNs:            uint64(cstats.existing_neighbor_rewrite_ns),
		ExistingNeighborEdgesWritten:         uint64(cstats.existing_neighbor_edges_written),
		ExistingNeighborEdgesPruned:          uint64(cstats.existing_neighbor_edges_pruned),
		ExistingNeighborVisits:               uint64(cstats.existing_neighbor_visits),
		ExistingNeighborAppends:              uint64(cstats.existing_neighbor_appends),
		ExistingNeighborPrunes:               uint64(cstats.existing_neighbor_prunes),
		ExistingNeighborPrunedEdgesRecorded:  uint64(cstats.existing_neighbor_pruned_edges_recorded),
	}, true
}
