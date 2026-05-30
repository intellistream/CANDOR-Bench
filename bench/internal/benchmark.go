package internal

import (
	"bufio"
	"context"
	"fmt"
	"golang.org/x/sys/unix"
	"golang.org/x/time/rate"
	"log"
	"math"
	mathbits "math/bits"
	mathrand "math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type TaskType int

const (
	InsertTask TaskType = iota
	SearchTask
)

type Task struct {
	Type          TaskType
	Data          [][]float32
	Tags          []uint32
	RecallAt      uint32
	InsertOffset  uint64
	CreateTime    time.Time
	CreateRawNs   uint64
	WaitForInsert bool
	Measured      bool
}

type Stat struct {
	InsertQPS                   float64
	SearchQPS                   float64
	MeanInsertOpLatency         float64
	MeanSearchOpLatency         float64
	P95InsertOpLatency          float64
	P99InsertOpLatency          float64
	P95SearchOpLatency          float64
	P99SearchOpLatency          float64
	MeanInsertE2ELatency        float64
	P95InsertE2ELatency         float64
	P99InsertE2ELatency         float64
	MeanSearchE2ELatency        float64
	P95SearchE2ELatency         float64
	P99SearchE2ELatency         float64
	MeanFreshnessTotalPts       float64
	P95FreshnessTotalPts        float64
	P99FreshnessTotalPts        float64
	MeanFreshnessContiguityPts  float64
	P95FreshnessContiguityPts   float64
	P99FreshnessContiguityPts   float64
	MeanFreshnessQueuePts       float64
	P95FreshnessQueuePts        float64
	P99FreshnessQueuePts        float64
	MeanFreshnessActiveFrontPts float64
	P95FreshnessActiveFrontPts  float64
	P99FreshnessActiveFrontPts  float64
	MeanFreshnessMissedE2EPts   float64
	P95FreshnessMissedE2EPts    float64
	P99FreshnessMissedE2EPts    float64
	InflightBruteforceEnabled   bool
	InflightJoinEnabled         bool
	MeanInflightBruteforcePts   float64
	P95InflightBruteforcePts    float64
	P99InflightBruteforcePts    float64
	MeanInflightOccMergeLatency float64
	P95InflightOccMergeLatency  float64
	P99InflightOccMergeLatency  float64
	MeanInflightOccShare        float64
	MeanInflightOccL2Dists      float64
	MeanInflightOccMergeComps   float64
	MeanInflightOccFilterTests  float64
	MeanInflightOccCandidates   float64
	ResultBelowKCount           int
	PeakMemoryMB                float64
	AvgMemoryMB                 float64
	ExtraStats                  string
}

type Bench struct {
	insertQueue                                     chan Task
	searchQueue                                     chan Task
	measuredSearchQueue                             chan Task
	index                                           *Index
	searchIndex                                     *Index
	replicaRefresh                                  *replicaRefreshManager
	rwMu                                            sync.RWMutex
	wg                                              sync.WaitGroup
	insertOpLatencies                               []float64
	searchOpLatencies                               []float64
	insertE2ELatencies                              []float64
	searchE2ELatencies                              []float64
	searchIndexMs                                   []float64
	searchPostIndexMs                               []float64
	searchResultsLockWaitMs                         []float64
	searchResultsLockHoldMs                         []float64
	searchRecordLockWaitMs                          []float64
	insertFinishMs                                  []float64
	searchFinishMs                                  []float64
	searchMeasured                                  []float64
	searchQueryTags                                 []float64
	insertCreateRawNs                               []uint64
	searchCreateRawNs                               []uint64
	insertStartRawNs                                []uint64
	searchStartRawNs                                []uint64
	insertFinishRawNs                               []uint64
	searchFinishRawNs                               []uint64
	searchTaskInsertOffset                          []float64
	searchCommittedAtStart                          []float64
	searchViewOffset                                []float64
	searchPhysicalHead                              []float64
	insertGraphConnectMs                            []float64
	insertGraphLinkCriticalMs                       []float64
	insertGraphUniqueLockWaitMs                     []float64
	insertGraphSearchUniqueLockWaitMs               []float64
	insertGraphSearchUniqueLockAcqs                 []float64
	insertGraphSearchCriticalMs                     []float64
	insertBatchInsertWallMs                         []float64 // narrow: only the index call
	insertGraphLinkUpdates                          []float64
	insertGraphConnectCalls                         []float64
	insertGraphUpperSearchMs                        []float64
	insertGraphUpperSearchDistComps                 []float64
	insertGraphUpperSearchEdgesScanned              []float64
	insertGraphBaseSearchMs                         []float64
	insertGraphBaseSearchExpansions                 []float64
	insertGraphBaseSearchEdgesScanned               []float64
	insertGraphBaseSearchDistComps                  []float64
	insertGraphSelectNewNeighborsMs                 []float64
	insertGraphSelectNewNeighborsInput              []float64
	insertGraphSelectNewNeighborsSelected           []float64
	insertGraphSelectNewNeighborsHeuristicDistComps []float64
	insertGraphInsertedNodeLinkMs                   []float64
	insertGraphInsertedNodeEdgesWritten             []float64
	insertGraphExistingNeighborUpdateLoopMs         []float64
	insertGraphExistingNeighborLoadScanMs           []float64
	insertGraphExistingNeighborLoadedEdges          []float64
	insertGraphExistingNeighborAppendMs             []float64
	insertGraphExistingNeighborPruneMs              []float64
	insertGraphExistingNeighborPruneCandidates      []float64
	insertGraphExistingNeighborPruneDistComps       []float64
	insertGraphExistingNeighborUndoRecordMs         []float64
	insertGraphExistingNeighborRewriteMs            []float64
	insertGraphExistingNeighborEdgesWritten         []float64
	insertGraphExistingNeighborEdgesPruned          []float64
	insertGraphExistingNeighborVisits               []float64
	insertGraphExistingNeighborAppends              []float64
	insertGraphExistingNeighborPrunes               []float64
	insertGraphExistingNeighborPrunedEdgesRecorded  []float64
	searchFreshnessTotalPts                         []float64
	searchFreshnessContiguityPts                    []float64
	searchFreshnessQueuePts                         []float64
	searchFreshnessActiveFrontPts                   []float64
	searchFreshnessMissedE2EPts                     []float64
	searchInflightBruteforcePts                     []float64
	searchInflightOccLatencies                      []float64
	searchInflightOccL2Dists                        []float64
	searchInflightOccMergeComps                     []float64
	searchInflightOccFilterTests                    []float64
	searchInflightOccCandidates                     []float64
	searchDiagDistComps                             []float64
	searchDiagHops                                  []float64
	searchDiagFutureSkipQueries                     []float64
	searchDiagFutureSkipHops                        []float64
	searchDiagRecoveryTriggers                      []float64
	searchDiagRecoveredEdges                        []float64
	searchDiagUsefulRecovered                       []float64
	searchDiagModifiedExpansions                    []float64
	searchDiagRecoveryAttempts                      []float64
	searchDiagRecoveryCandidates                    []float64
	searchDiagRecoveryGetMs                         []float64
	searchDiagRecoveryLoopMs                        []float64
	searchDiagRewriteActiveExp                      []float64
	searchDiagRewriteActiveQuery                    []float64
	searchDiagRewriteRecentExp                      []float64
	searchDiagRewriteRecentQuery                    []float64
	searchDiagRewritePeriodExp                      []float64
	searchDiagRewritePeriodQuery                    []float64
	searchDiagRewritePeriodActiveSum                []float64
	searchDiagRewritePeriodActiveMax                []float64
	searchDiagBatchSearchCalls                      []float64
	searchDiagBatchSearchQueries                    []float64
	searchDiagBatchSearchArenaMs                    []float64
	searchDiagBatchSearchKnnMs                      []float64
	searchDiagBatchSearchCopyMs                     []float64
	searchDiagInvisibleExpansions                   []float64
	searchDiagInvisibleExpansionEdges               []float64
	searchDiagInvisibleCandidateDistComps           []float64
	searchDiagInvisibleCandidateEnqueues            []float64
	searchDiagPhaseOverlapSamples                   []float64
	searchDiagPhaseExistingUpdateSamples            []float64
	searchDiagPhaseExistingUpdateQueries            []float64
	searchDiagPhaseLinkCriticalSamples              []float64
	searchDiagPhaseLinkCriticalQueries              []float64
	searchDiagPhaseLoadScanSamples                  []float64
	searchDiagPhaseLoadScanQueries                  []float64
	searchDiagPhaseAppendSamples                    []float64
	searchDiagPhaseAppendQueries                    []float64
	searchDiagPhasePruneSamples                     []float64
	searchDiagPhasePruneQueries                     []float64
	searchDiagPhaseUndoRecordSamples                []float64
	searchDiagPhaseUndoRecordQueries                []float64
	searchDiagPhaseRewriteSamples                   []float64
	searchDiagPhaseRewriteQueries                   []float64
	searchWorkSearchKnnMs                           []float64
	searchWorkSearchKnnThreadCpuMs                  []float64
	searchWorkStartCPU                              []float64
	searchWorkEndCPU                                []float64
	searchWorkResultCopyMs                          []float64
	searchWorkEntryMs                               []float64
	searchWorkUpperSearchMs                         []float64
	searchWorkBaseSearchMs                          []float64
	searchWorkResultMaterializeMs                   []float64
	searchWorkSnapshotGuardMs                       []float64
	searchWorkVisitedListGetMs                      []float64
	searchWorkVisitedListReleaseMs                  []float64
	searchWorkUpperLockWaitMs                       []float64
	searchWorkLevel0LockWaitMs                      []float64
	searchWorkDistanceComputations                  []float64
	searchWorkUpperDistanceComputations             []float64
	searchWorkLevel0DistanceComputations            []float64
	searchWorkDistanceComputeMs                     []float64
	searchWorkUpperDistanceComputeMs                []float64
	searchWorkLevel0DistanceComputeMs               []float64
	searchWorkLevel0QueuePopMs                      []float64
	searchWorkLevel0AdjFetchMs                      []float64
	searchWorkLevel0LocalityCaptureMs               []float64
	searchWorkLevel0CandidateLoopMs                 []float64
	searchWorkLevel0VisitedCheckMs                  []float64
	searchWorkLevel0VisibilityCheckMs               []float64
	searchWorkLevel0CandidateAcceptMs               []float64
	searchWorkUpperHops                             []float64
	searchWorkUpperEdgesScanned                     []float64
	searchWorkLevel0Expansions                      []float64
	searchWorkLevel0EdgesScanned                    []float64
	searchWorkCandidatePops                         []float64
	searchWorkCandidatePushes                       []float64
	searchWorkVisitedNodes                          []float64
	searchWorkResultPushes                          []float64
	searchWorkInvisibleExpansions                   []float64
	searchWorkInvisibleEdges                        []float64
	searchWorkInvisibleCandidateDistComps           []float64
	searchWorkInvisibleCandidateEnqueues            []float64
	searchWorkFutureSkipHops                        []float64
	searchWorkRewriteActiveExpansions               []float64
	searchWorkRewriteRecentExpansions               []float64
	searchWorkRewritePeriodExpansions               []float64
	searchWorkRewritePeriodActiveSum                []float64
	searchWorkRewritePeriodActiveMax                []float64
	searchWorkExpandVisibleCount                    []float64
	searchWorkExpandRecent1KHits                    []float64
	searchWorkExpandRecent4KHits                    []float64
	searchWorkExpandRecent16KHits                   []float64
	searchWorkExpandLabelGapSum                     []float64
	searchWorkExpandLabelSpan                       []float64
	searchWorkExpandUniqueLabel4KBuckets            []float64
	searchWorkExpandUniqueData4KPages               []float64
	searchWorkExpandUniqueData2MPages               []float64
	searchWorkExpandUniqueAdj4KPages                []float64
	searchWorkExpandUniqueAdj2MPages                []float64
	searchWorkExpandUniqueOverflow                  []float64
	searchWorkPathCount                             []float64
	searchWorkPathFirstLabel                        []float64
	searchWorkPathLastLabel                         []float64
	searchWorkPathMinLabel                          []float64
	searchWorkPathMaxLabel                          []float64
	searchWorkPathLabelSpan                         []float64
	searchWorkPathMeanAbsGap                        []float64
	searchWorkPathUnique4KBuckets                   []float64
	searchWorkPathMinVisibleAge                     []float64
	searchWorkPathMeanVisibleAge                    []float64
	searchWorkPathRecent1KHits                      []float64
	searchWorkPathRecent4KHits                      []float64
	searchWorkPathRecent16KHits                     []float64
	insertLimiter                                   *rate.Limiter
	searchLimiter                                   *rate.Limiter
	searchResults                                   []*SearchResult
	resultsMu                                       sync.Mutex
	insertMu                                        sync.Mutex
	searchMu                                        sync.Mutex
	config                                          *Config
	insertPointCnt                                  int
	searchPointCnt                                  int
	globalInsertCnt                                 int64
	committedOffset                                 atomic.Uint64
	latestInsertFront                               atomic.Uint64
	completionMu                                    sync.Mutex
	completedOffsets                                map[uint64]bool
	pendingOffsets                                  []uint64
	pendingIdx                                      int
	commitEventOffset                               []uint64
	commitEventMs                                   []float64
	searchLagSum                                    int64
	searchLagCount                                  int64
	searchLagMax                                    int64
	searchLagContiguitySum                          int64
	searchLagContiguityMax                          int64
	searchLagQueueSum                               int64
	searchLagQueueMax                               int64
	startTime                                       time.Time
	startRawNs                                      uint64
	resultBelowKCount                               int
	data                                            []float32
	dataDim                                         int
	graphTouchHotLabels                             []uint32
	graphTouchColdLabels                            []uint32
	closedLoopWorkloadQueries                       []float32
	closedLoopDataDim                               int
	closedLoopTotalQueries                          int
	directionSigMu                                  sync.Mutex
	directionSigBits                                int
	directionSigPlanes                              []float32
	directionSigLo                                  []uint64
	directionSigHi                                  []uint64
	directionSigReady                               []atomic.Uint32
	directionSigBucketed                            []atomic.Uint32
	directionSigBucketMu                            sync.RWMutex
	directionSigBuckets                             map[uint64][]uint32

	// Lock wait instrumentation
	searchLockMu                              sync.Mutex
	searchLockWaits                           []float64
	searchLockWaitCount                       int64
	activeSearchTasks                         atomic.Int64
	activeInsertMu                            sync.Mutex
	activeInsertRanges                        map[uint64]insertRange
	activeInsertMax                           uint64
	m2FreshNeighborMu                         sync.RWMutex
	m2FreshNeighborCache                      map[uint32][]uint32
	m3BatchDiagEnabled                        bool
	searchWorkCountersEnabled                 bool
	graphMutationStatsEnabled                 bool
	m2VValidateSeen                           atomic.Uint64
	m2VValidateRuns                           atomic.Uint64
	m2VValidateFresh                          atomic.Uint64
	m2VValidateWins                           atomic.Uint64
	m2VValidateMiss                           atomic.Uint64
	m2VValidatePendingWinner                  atomic.Uint64
	m2VValidateActiveRegionWinner             atomic.Uint64
	m2VValidateOpenedRegionWinner             atomic.Uint64
	m2VValidateTrianglePrunedWinner           atomic.Uint64
	m2VRouteDiagSeenQueries                   atomic.Uint64
	m2VRouteDiagNoRouteQueries                atomic.Uint64
	m2VRouteDiagCandidateRegions              atomic.Uint64
	m2VRouteDiagOpenedRegions                 atomic.Uint64
	m2VRouteDiagOpenedPostings                atomic.Uint64
	m2VRouteDiagMemberBoundChecks             atomic.Uint64
	m2VRouteDiagMemberBoundSkips              atomic.Uint64
	m2VRouteDiagHubSuppressed                 atomic.Uint64
	m2VRouteDiagWideSuppressed                atomic.Uint64
	m2VRouteDiagWeakSuppressed                atomic.Uint64
	m2VRouteDiagStrongRegionHits              atomic.Uint64
	m2VRouteDiagMediumRegionHits              atomic.Uint64
	m2VRouteDiagWeakRegionHits                atomic.Uint64
	m2PathOneHopValidateSeen                  atomic.Uint64
	m2PathOneHopValidateRuns                  atomic.Uint64
	m2PathOneHopValidateFresh                 atomic.Uint64
	m2PathOneHopValidateWins                  atomic.Uint64
	m2PathOneHopValidateMiss                  atomic.Uint64
	m2PathOneHopValidatePendingWinner         atomic.Uint64
	m2PathOneHopValidateSelectedOneHop        atomic.Uint64
	m2PathOneHopValidateAllPathOneHop         atomic.Uint64
	m2PathOneHopValidateMissAllPathOneHop     atomic.Uint64
	m2PathOneHopValidateMissPathEventSelf     atomic.Uint64
	m2PathOneHopValidateMissSelectedTwoHop    atomic.Uint64
	m2PathOneHopValidateMissAllPathTwoHop     atomic.Uint64
	m2PathOneHopValidateMissMonotoneTwoHop    atomic.Uint64
	m2PathOneHopValidateMissRadiusTwoHop      atomic.Uint64
	m2PathOneHopValidateMissSupportedTwoHop   atomic.Uint64
	m2PathOneHopValidateMissRecipSourceTwoHop atomic.Uint64
	m2PathOneHopValidateMissRecipTargetTwoHop atomic.Uint64
}

type insertRange struct {
	start uint64
	end   uint64
}

type memResult struct {
	peakUsage    uint64
	avgUsage     float64
	samplesCount int
}

type indexDiagStats map[string]float64

func parseIndexDiagStats(raw string) indexDiagStats {
	out := make(indexDiagStats)
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		kv := strings.SplitN(part, ":", 2)
		if len(kv) != 2 {
			continue
		}
		key := strings.TrimSpace(kv[0])
		value, err := strconv.ParseFloat(strings.TrimSpace(kv[1]), 64)
		if err != nil {
			continue
		}
		out[key] = value
	}
	return out
}

func indexDiagDelta(before, after indexDiagStats, key string) float64 {
	return after[key] - before[key]
}

func concurrentBench(index *Index, searchIndex *Index, replicaRefresh *replicaRefreshManager, config Config, data []float32, dataDim int) *Bench {
	batchSize := config.Workload.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}
	expectedBatches := (config.Data.MaxElements - config.Data.BeginNum) / batchSize
	if expectedBatches <= 0 {
		expectedBatches = 1000
	}

	b := &Bench{
		insertQueue:                                     make(chan Task, config.Workload.QueueSize),
		searchQueue:                                     make(chan Task, config.Workload.QueueSize),
		measuredSearchQueue:                             make(chan Task, config.Workload.QueueSize),
		index:                                           index,
		searchIndex:                                     searchIndex,
		replicaRefresh:                                  replicaRefresh,
		insertOpLatencies:                               make([]float64, 0, expectedBatches),
		searchOpLatencies:                               make([]float64, 0, expectedBatches),
		insertE2ELatencies:                              make([]float64, 0, expectedBatches),
		searchE2ELatencies:                              make([]float64, 0, expectedBatches),
		searchIndexMs:                                   make([]float64, 0, expectedBatches),
		searchPostIndexMs:                               make([]float64, 0, expectedBatches),
		searchResultsLockWaitMs:                         make([]float64, 0, expectedBatches),
		searchResultsLockHoldMs:                         make([]float64, 0, expectedBatches),
		searchRecordLockWaitMs:                          make([]float64, 0, expectedBatches),
		searchFreshnessTotalPts:                         make([]float64, 0, expectedBatches),
		searchFreshnessContiguityPts:                    make([]float64, 0, expectedBatches),
		searchFreshnessQueuePts:                         make([]float64, 0, expectedBatches),
		searchFreshnessActiveFrontPts:                   make([]float64, 0, expectedBatches),
		searchFreshnessMissedE2EPts:                     make([]float64, 0, expectedBatches),
		searchInflightBruteforcePts:                     make([]float64, 0, expectedBatches),
		searchInflightOccLatencies:                      make([]float64, 0, expectedBatches),
		searchInflightOccL2Dists:                        make([]float64, 0, expectedBatches),
		searchInflightOccMergeComps:                     make([]float64, 0, expectedBatches),
		searchInflightOccFilterTests:                    make([]float64, 0, expectedBatches),
		searchInflightOccCandidates:                     make([]float64, 0, expectedBatches),
		searchTaskInsertOffset:                          make([]float64, 0, expectedBatches),
		searchCommittedAtStart:                          make([]float64, 0, expectedBatches),
		searchViewOffset:                                make([]float64, 0, expectedBatches),
		searchPhysicalHead:                              make([]float64, 0, expectedBatches),
		searchDiagDistComps:                             make([]float64, 0, expectedBatches),
		searchDiagHops:                                  make([]float64, 0, expectedBatches),
		searchDiagFutureSkipQueries:                     make([]float64, 0, expectedBatches),
		searchDiagFutureSkipHops:                        make([]float64, 0, expectedBatches),
		searchDiagRecoveryTriggers:                      make([]float64, 0, expectedBatches),
		searchDiagRecoveredEdges:                        make([]float64, 0, expectedBatches),
		searchDiagUsefulRecovered:                       make([]float64, 0, expectedBatches),
		searchDiagModifiedExpansions:                    make([]float64, 0, expectedBatches),
		searchDiagRecoveryAttempts:                      make([]float64, 0, expectedBatches),
		searchDiagRecoveryCandidates:                    make([]float64, 0, expectedBatches),
		searchDiagRecoveryGetMs:                         make([]float64, 0, expectedBatches),
		searchDiagRecoveryLoopMs:                        make([]float64, 0, expectedBatches),
		searchDiagRewriteActiveExp:                      make([]float64, 0, expectedBatches),
		searchDiagRewriteActiveQuery:                    make([]float64, 0, expectedBatches),
		searchDiagRewriteRecentExp:                      make([]float64, 0, expectedBatches),
		searchDiagRewriteRecentQuery:                    make([]float64, 0, expectedBatches),
		searchDiagRewritePeriodExp:                      make([]float64, 0, expectedBatches),
		searchDiagRewritePeriodQuery:                    make([]float64, 0, expectedBatches),
		searchDiagRewritePeriodActiveSum:                make([]float64, 0, expectedBatches),
		searchDiagRewritePeriodActiveMax:                make([]float64, 0, expectedBatches),
		searchDiagBatchSearchCalls:                      make([]float64, 0, expectedBatches),
		searchDiagBatchSearchQueries:                    make([]float64, 0, expectedBatches),
		searchDiagBatchSearchArenaMs:                    make([]float64, 0, expectedBatches),
		searchDiagBatchSearchKnnMs:                      make([]float64, 0, expectedBatches),
		searchDiagBatchSearchCopyMs:                     make([]float64, 0, expectedBatches),
		searchDiagInvisibleExpansions:                   make([]float64, 0, expectedBatches),
		searchDiagInvisibleExpansionEdges:               make([]float64, 0, expectedBatches),
		searchDiagInvisibleCandidateDistComps:           make([]float64, 0, expectedBatches),
		searchDiagInvisibleCandidateEnqueues:            make([]float64, 0, expectedBatches),
		searchDiagPhaseOverlapSamples:                   make([]float64, 0, expectedBatches),
		searchDiagPhaseExistingUpdateSamples:            make([]float64, 0, expectedBatches),
		searchDiagPhaseExistingUpdateQueries:            make([]float64, 0, expectedBatches),
		searchDiagPhaseLinkCriticalSamples:              make([]float64, 0, expectedBatches),
		searchDiagPhaseLinkCriticalQueries:              make([]float64, 0, expectedBatches),
		searchDiagPhaseLoadScanSamples:                  make([]float64, 0, expectedBatches),
		searchDiagPhaseLoadScanQueries:                  make([]float64, 0, expectedBatches),
		searchDiagPhaseAppendSamples:                    make([]float64, 0, expectedBatches),
		searchDiagPhaseAppendQueries:                    make([]float64, 0, expectedBatches),
		searchDiagPhasePruneSamples:                     make([]float64, 0, expectedBatches),
		searchDiagPhasePruneQueries:                     make([]float64, 0, expectedBatches),
		searchDiagPhaseUndoRecordSamples:                make([]float64, 0, expectedBatches),
		searchDiagPhaseUndoRecordQueries:                make([]float64, 0, expectedBatches),
		searchDiagPhaseRewriteSamples:                   make([]float64, 0, expectedBatches),
		searchDiagPhaseRewriteQueries:                   make([]float64, 0, expectedBatches),
		searchWorkSearchKnnMs:                           make([]float64, 0, expectedBatches),
		searchWorkSearchKnnThreadCpuMs:                  make([]float64, 0, expectedBatches),
		searchWorkStartCPU:                              make([]float64, 0, expectedBatches),
		searchWorkEndCPU:                                make([]float64, 0, expectedBatches),
		searchWorkResultCopyMs:                          make([]float64, 0, expectedBatches),
		searchWorkEntryMs:                               make([]float64, 0, expectedBatches),
		searchWorkUpperSearchMs:                         make([]float64, 0, expectedBatches),
		searchWorkBaseSearchMs:                          make([]float64, 0, expectedBatches),
		searchWorkResultMaterializeMs:                   make([]float64, 0, expectedBatches),
		searchWorkSnapshotGuardMs:                       make([]float64, 0, expectedBatches),
		searchWorkVisitedListGetMs:                      make([]float64, 0, expectedBatches),
		searchWorkVisitedListReleaseMs:                  make([]float64, 0, expectedBatches),
		searchWorkUpperLockWaitMs:                       make([]float64, 0, expectedBatches),
		searchWorkLevel0LockWaitMs:                      make([]float64, 0, expectedBatches),
		searchWorkDistanceComputations:                  make([]float64, 0, expectedBatches),
		searchWorkUpperDistanceComputations:             make([]float64, 0, expectedBatches),
		searchWorkLevel0DistanceComputations:            make([]float64, 0, expectedBatches),
		searchWorkDistanceComputeMs:                     make([]float64, 0, expectedBatches),
		searchWorkUpperDistanceComputeMs:                make([]float64, 0, expectedBatches),
		searchWorkLevel0DistanceComputeMs:               make([]float64, 0, expectedBatches),
		searchWorkLevel0QueuePopMs:                      make([]float64, 0, expectedBatches),
		searchWorkLevel0AdjFetchMs:                      make([]float64, 0, expectedBatches),
		searchWorkLevel0LocalityCaptureMs:               make([]float64, 0, expectedBatches),
		searchWorkLevel0CandidateLoopMs:                 make([]float64, 0, expectedBatches),
		searchWorkLevel0VisitedCheckMs:                  make([]float64, 0, expectedBatches),
		searchWorkLevel0VisibilityCheckMs:               make([]float64, 0, expectedBatches),
		searchWorkLevel0CandidateAcceptMs:               make([]float64, 0, expectedBatches),
		searchWorkUpperHops:                             make([]float64, 0, expectedBatches),
		searchWorkUpperEdgesScanned:                     make([]float64, 0, expectedBatches),
		searchWorkLevel0Expansions:                      make([]float64, 0, expectedBatches),
		searchWorkLevel0EdgesScanned:                    make([]float64, 0, expectedBatches),
		searchWorkCandidatePops:                         make([]float64, 0, expectedBatches),
		searchWorkCandidatePushes:                       make([]float64, 0, expectedBatches),
		searchWorkVisitedNodes:                          make([]float64, 0, expectedBatches),
		searchWorkResultPushes:                          make([]float64, 0, expectedBatches),
		searchWorkInvisibleExpansions:                   make([]float64, 0, expectedBatches),
		searchWorkInvisibleEdges:                        make([]float64, 0, expectedBatches),
		searchWorkInvisibleCandidateDistComps:           make([]float64, 0, expectedBatches),
		searchWorkInvisibleCandidateEnqueues:            make([]float64, 0, expectedBatches),
		searchWorkFutureSkipHops:                        make([]float64, 0, expectedBatches),
		searchWorkRewriteActiveExpansions:               make([]float64, 0, expectedBatches),
		searchWorkRewriteRecentExpansions:               make([]float64, 0, expectedBatches),
		searchWorkRewritePeriodExpansions:               make([]float64, 0, expectedBatches),
		searchWorkRewritePeriodActiveSum:                make([]float64, 0, expectedBatches),
		searchWorkRewritePeriodActiveMax:                make([]float64, 0, expectedBatches),
		searchWorkExpandVisibleCount:                    make([]float64, 0, expectedBatches),
		searchWorkExpandRecent1KHits:                    make([]float64, 0, expectedBatches),
		searchWorkExpandRecent4KHits:                    make([]float64, 0, expectedBatches),
		searchWorkExpandRecent16KHits:                   make([]float64, 0, expectedBatches),
		searchWorkExpandLabelGapSum:                     make([]float64, 0, expectedBatches),
		searchWorkExpandLabelSpan:                       make([]float64, 0, expectedBatches),
		searchWorkExpandUniqueLabel4KBuckets:            make([]float64, 0, expectedBatches),
		searchWorkExpandUniqueData4KPages:               make([]float64, 0, expectedBatches),
		searchWorkExpandUniqueData2MPages:               make([]float64, 0, expectedBatches),
		searchWorkExpandUniqueAdj4KPages:                make([]float64, 0, expectedBatches),
		searchWorkExpandUniqueAdj2MPages:                make([]float64, 0, expectedBatches),
		searchWorkExpandUniqueOverflow:                  make([]float64, 0, expectedBatches),
		searchWorkPathCount:                             make([]float64, 0, expectedBatches),
		searchWorkPathFirstLabel:                        make([]float64, 0, expectedBatches),
		searchWorkPathLastLabel:                         make([]float64, 0, expectedBatches),
		searchWorkPathMinLabel:                          make([]float64, 0, expectedBatches),
		searchWorkPathMaxLabel:                          make([]float64, 0, expectedBatches),
		searchWorkPathLabelSpan:                         make([]float64, 0, expectedBatches),
		searchWorkPathMeanAbsGap:                        make([]float64, 0, expectedBatches),
		searchWorkPathUnique4KBuckets:                   make([]float64, 0, expectedBatches),
		searchWorkPathMinVisibleAge:                     make([]float64, 0, expectedBatches),
		searchWorkPathMeanVisibleAge:                    make([]float64, 0, expectedBatches),
		searchWorkPathRecent1KHits:                      make([]float64, 0, expectedBatches),
		searchWorkPathRecent4KHits:                      make([]float64, 0, expectedBatches),
		searchWorkPathRecent16KHits:                     make([]float64, 0, expectedBatches),
		insertFinishMs:                                  make([]float64, 0, expectedBatches),
		searchFinishMs:                                  make([]float64, 0, expectedBatches),
		searchMeasured:                                  make([]float64, 0, expectedBatches),
		searchQueryTags:                                 make([]float64, 0, expectedBatches),
		insertCreateRawNs:                               make([]uint64, 0, expectedBatches),
		searchCreateRawNs:                               make([]uint64, 0, expectedBatches),
		insertStartRawNs:                                make([]uint64, 0, expectedBatches),
		searchStartRawNs:                                make([]uint64, 0, expectedBatches),
		insertFinishRawNs:                               make([]uint64, 0, expectedBatches),
		searchFinishRawNs:                               make([]uint64, 0, expectedBatches),
		insertGraphConnectMs:                            make([]float64, 0, expectedBatches),
		insertGraphLinkCriticalMs:                       make([]float64, 0, expectedBatches),
		insertGraphUniqueLockWaitMs:                     make([]float64, 0, expectedBatches),
		insertGraphSearchUniqueLockWaitMs:               make([]float64, 0, expectedBatches),
		insertGraphSearchUniqueLockAcqs:                 make([]float64, 0, expectedBatches),
		insertGraphSearchCriticalMs:                     make([]float64, 0, expectedBatches),
		insertBatchInsertWallMs:                         make([]float64, 0, expectedBatches),
		insertGraphLinkUpdates:                          make([]float64, 0, expectedBatches),
		insertGraphConnectCalls:                         make([]float64, 0, expectedBatches),
		insertGraphUpperSearchMs:                        make([]float64, 0, expectedBatches),
		insertGraphUpperSearchDistComps:                 make([]float64, 0, expectedBatches),
		insertGraphUpperSearchEdgesScanned:              make([]float64, 0, expectedBatches),
		insertGraphBaseSearchMs:                         make([]float64, 0, expectedBatches),
		insertGraphBaseSearchExpansions:                 make([]float64, 0, expectedBatches),
		insertGraphBaseSearchEdgesScanned:               make([]float64, 0, expectedBatches),
		insertGraphBaseSearchDistComps:                  make([]float64, 0, expectedBatches),
		insertGraphSelectNewNeighborsMs:                 make([]float64, 0, expectedBatches),
		insertGraphSelectNewNeighborsInput:              make([]float64, 0, expectedBatches),
		insertGraphSelectNewNeighborsSelected:           make([]float64, 0, expectedBatches),
		insertGraphSelectNewNeighborsHeuristicDistComps: make([]float64, 0, expectedBatches),
		insertGraphInsertedNodeLinkMs:                   make([]float64, 0, expectedBatches),
		insertGraphInsertedNodeEdgesWritten:             make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborUpdateLoopMs:         make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborLoadScanMs:           make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborLoadedEdges:          make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborAppendMs:             make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborPruneMs:              make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborPruneCandidates:      make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborPruneDistComps:       make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborUndoRecordMs:         make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborRewriteMs:            make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborEdgesWritten:         make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborEdgesPruned:          make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborVisits:               make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborAppends:              make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborPrunes:               make([]float64, 0, expectedBatches),
		insertGraphExistingNeighborPrunedEdgesRecorded:  make([]float64, 0, expectedBatches),
		insertLimiter:                                   buildLimiter(config.Workload.InsertEventRate, config.Workload.BatchSize),
		searchLimiter:                                   buildLimiter(config.Workload.SearchEventRate, config.Workload.BatchSize),
		commitEventOffset:                               make([]uint64, 0, expectedBatches),
		commitEventMs:                                   make([]float64, 0, expectedBatches),
		config:                                          &config,
		activeInsertRanges:                              make(map[uint64]insertRange),
		m2FreshNeighborCache:                            make(map[uint32][]uint32),
		m3BatchDiagEnabled:                              os.Getenv("ANNCHOR_M3_BATCH_DIAG") == "1",
		searchWorkCountersEnabled:                       os.Getenv("ANNCHOR_SEARCH_WORK_COUNTERS") == "1",
		graphMutationStatsEnabled:                       !strings.EqualFold(os.Getenv("ANNCHOR_GRAPH_MUTATION_STATS"), "0") && !strings.EqualFold(os.Getenv("ANNCHOR_GRAPH_MUTATION_STATS"), "false"),
		data:                                            data,
		dataDim:                                         dataDim,
	}
	b.committedOffset.Store(uint64(config.Data.BeginNum))
	b.latestInsertFront.Store(uint64(config.Data.BeginNum))
	b.completedOffsets = make(map[uint64]bool)
	bs := config.Workload.BatchSize
	if bs <= 0 {
		bs = 1
	}
	for off := config.Data.BeginNum + bs; off <= config.Data.MaxElements; off += bs {
		actual := off
		if actual > config.Data.MaxElements {
			actual = config.Data.MaxElements
		}
		b.pendingOffsets = append(b.pendingOffsets, uint64(actual))
	}
	if len(b.pendingOffsets) == 0 || b.pendingOffsets[len(b.pendingOffsets)-1] != uint64(config.Data.MaxElements) {
		if config.Data.MaxElements > config.Data.BeginNum {
			b.pendingOffsets = append(b.pendingOffsets, uint64(config.Data.MaxElements))
		}
	}
	return b
}

func (b *Bench) servingIndex() *Index {
	if b.replicaRefresh != nil {
		return b.replicaRefresh.current()
	}
	if b.searchIndex != nil {
		return b.searchIndex
	}
	return b.index
}

func (b *Bench) replicaSnapshotMode() bool {
	return strings.EqualFold(b.config.Index.IndexType, "replica_snapshot")
}

func (b *Bench) replicaRefreshMode() bool {
	return b.replicaRefresh != nil || isReplicaRefreshIndex(b.config.Index.IndexType)
}

func execBenchmark(index *Index, config *Config, data []float32, workloadQueries []float32, dataDim int, overallQueries []float32) ([]*SearchResult, Stat, error) {
	return execBenchmarkWithServingIndex(index, nil, config, data, workloadQueries, dataDim, overallQueries)
}

func execBenchmarkWithServingIndex(index *Index, searchIndex *Index, config *Config, data []float32, workloadQueries []float32, dataDim int, overallQueries []float32) ([]*SearchResult, Stat, error) {
	return execBenchmarkWithReplica(index, searchIndex, nil, config, data, workloadQueries, dataDim, overallQueries)
}

func execBenchmarkWithReplica(index *Index, searchIndex *Index, replicaRefresh *replicaRefreshManager, config *Config, data []float32, workloadQueries []float32, dataDim int, overallQueries []float32) ([]*SearchResult, Stat, error) {
	var memResultChan <-chan memResult
	var doneChan chan struct{}

	if config.Profile.EnableMemoryProfile {
		doneChan = make(chan struct{})
		monitorInterval := config.Profile.MemoryMonitorInterval
		if monitorInterval == 0 {
			monitorInterval = 100
		}
		log.Printf("Executing benchmark with memory monitoring enabled, interval: %dms", monitorInterval)
		memResultChan = monitorMemUsage(time.Duration(monitorInterval)*time.Millisecond, doneChan)
	} else {
		log.Printf("Executing benchmark with memory monitoring disabled")
		dummyChan := make(chan memResult, 1)
		dummyChan <- memResult{}
		close(dummyChan)
		memResultChan = dummyChan
	}

	if config.Workload.WithExternalRWLock {
		log.Printf("External RW-Lock: ENABLED")
	} else {
		log.Printf("External RW-Lock: DISABLED")
	}

	bench := concurrentBench(index, searchIndex, replicaRefresh, *config, data, dataDim)
	expectedResults := config.Data.MaxElements * 2
	bench.searchResults = make([]*SearchResult, 0, expectedResults)

	log.Println("Start producing tasks and consuming tasks...")

	bench.startTime = time.Now()
	bench.startRawNs = MonotonicRawNs()
	totalQueries := len(workloadQueries) / dataDim

	if closedLoopEnabled() {
		log.Printf("Closed-loop worker mode enabled")
		if err := bench.runClosedLoopWorkers(data, dataDim, workloadQueries, totalQueries); err != nil {
			if doneChan != nil {
				close(doneChan)
			}
			<-memResultChan
			return nil, Stat{}, err
		}
		if doneChan != nil {
			close(doneChan)
		}
		memStats := <-memResultChan
		elapsedSec := time.Since(bench.startTime).Seconds()
		stats := bench.calcStats(elapsedSec, memStats)
		finishBench(bench.servingIndex(), overallQueries, dataDim, config, stats)
		return bench.searchResults, stats, nil
	}

	allowScheduledSearchOnly := config.Workload.PrecomputeSchedule &&
		config.Workload.SearchEventRate > 0 &&
		config.Workload.ScheduleHorizonMs > 0
	if config.Data.BeginNum >= config.Data.MaxElements && !allowScheduledSearchOnly {
		if doneChan != nil {
			close(doneChan)
		}
		<-memResultChan
		emptyStats := Stat{}
		finishBench(bench.servingIndex(), overallQueries, dataDim, config, emptyStats)
		return bench.searchResults, emptyStats, nil
	}

	insertRate := config.Workload.InsertEventRate
	searchRate := config.Workload.SearchEventRate
	batchSize := config.Workload.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}

	if insertRate <= 0 && searchRate <= 0 {
		log.Println("Limit test mode enabled - filling queue to capacity")
	} else {
		if insertRate > 0 {
			log.Printf("Insert rate limiter active: %.2f points/sec (≈ %.2f batches/sec @ batch_size=%d)", insertRate, insertRate/float64(batchSize), batchSize)
		}
		if searchRate > 0 {
			log.Printf("Search rate limiter active: %.2f points/sec (≈ %.2f batches/sec @ batch_size=%d)", searchRate, searchRate/float64(batchSize), batchSize)
		}
	}

	var err error
	if config.Workload.PrecomputeSchedule {
		schedule, scheduleErr := buildProducerSchedule(config, totalQueries)
		if scheduleErr != nil {
			return nil, Stat{}, fmt.Errorf("failed to build producer schedule: %w", scheduleErr)
		}
		log.Printf("Producer: precomputed schedule enabled (insert batches=%d, search tasks=%d, planned horizon=%s)", len(schedule.InsertTasks), len(schedule.SearchTasks), schedule.InsertHorizon)
		_, err = StartScheduledProducers(
			config,
			schedule,
			data,
			dataDim,
			workloadQueries,
			bench.insertQueue,
			bench.searchQueue,
			bench.measuredSearchQueue,
			&bench.committedOffset,
		)
	} else {
		_, err = StartProducers(
			config,
			data,
			dataDim,
			workloadQueries,
			config.Data.BeginNum,
			config.Data.MaxElements,
			totalQueries,
			bench.insertQueue,
			bench.searchQueue,
			bench.insertLimiter,
			bench.searchLimiter,
			&bench.committedOffset,
			nil,
			nil,
		)
	}
	if err != nil {
		return nil, Stat{}, fmt.Errorf("failed to start producers: %w", err)
	}

	progressDone := make(chan struct{})
	go func() {
		bench.printProgress(config.Data.MaxElements - config.Data.BeginNum)
		close(progressDone)
	}()

	bench.startDispatchers()
	bench.wg.Wait()

	if doneChan != nil {
		close(doneChan)
	}
	memStats := <-memResultChan

	<-progressDone

	elapsedSec := time.Since(bench.startTime).Seconds()
	stats := bench.calcStats(elapsedSec, memStats)

	finishBench(bench.servingIndex(), overallQueries, dataDim, config, stats)

	return bench.searchResults, stats, nil
}

func (b *Bench) startDispatchers() {
	b.servingIndex().SetQueryParams(BuildQueryParams(b.config))

	workers := b.config.Workload.NumThreads
	if workers <= 0 {
		workers = 1
	}
	measuredWorkers := measuredLanes()
	if measuredWorkers < 0 {
		measuredWorkers = 0
	}
	if !b.config.Workload.PrecomputeSchedule {
		measuredWorkers = 0
	}
	if measuredWorkers >= workers {
		measuredWorkers = workers - 1
	}
	sharedWorkers := workers - measuredWorkers
	log.Printf("Dispatchers: shared worker pool, %d workers; measured-search reserved workers=%d", sharedWorkers, measuredWorkers)

	for i := 0; i < measuredWorkers; i++ {
		b.wg.Add(1)
		go func() {
			runtime.LockOSThread()
			defer b.wg.Done()
			for task := range b.measuredSearchQueue {
				b.handleSearchTask(task, time.Now())
			}
		}()
	}

	for i := 0; i < sharedWorkers; i++ {
		b.wg.Add(1)
		go func() {
			runtime.LockOSThread()
			defer b.wg.Done()
			insertQueue := b.insertQueue
			searchQueue := b.searchQueue
			for insertQueue != nil || searchQueue != nil {
				select {
				case task, ok := <-insertQueue:
					if !ok {
						insertQueue = nil
						continue
					}
					b.handleInsertTask(task, time.Now())
				case task, ok := <-searchQueue:
					if !ok {
						searchQueue = nil
						continue
					}
					b.handleSearchTask(task, time.Now())
				}
			}
		}()
	}
}

func measuredLanes() int {
	raw := os.Getenv("MEASURED_LANES")
	if raw == "" {
		return 0
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return 0
	}
	return value
}

func pinThreadToCPUs(cpus []int) {
	var mask unix.CPUSet
	for _, c := range cpus {
		mask.Set(c)
	}
	err := unix.SchedSetaffinity(0, &mask)
	if err != nil {
		log.Printf("Warning: failed to set CPU affinity: %v", err)
	}
}

func (b *Bench) preemptSearchPressure() int {
	queued := len(b.searchQueue)
	active := int(b.activeSearchTasks.Load())
	if queued < 0 {
		queued = 0
	}
	if active < 0 {
		active = 0
	}
	return queued + active
}

func (b *Bench) handleInsertTask(task Task, start time.Time) {
	if len(task.Data) == 0 {
		return
	}
	rangeStart := task.InsertOffset - uint64(len(task.Data))
	if b.replicaRefresh != nil {
		b.replicaRefresh.beginInsert(rangeStart)
		defer b.replicaRefresh.finishInsert(task.InsertOffset)
	}
	insertStartRawNs := MonotonicRawNs()
	b.updateLatestInsertFront(task.InsertOffset)
	if strings.EqualFold(b.config.Index.IndexType, "annchor-preempt") {
		pressure := b.preemptSearchPressure()
		b.index.SetPreemptRuntimeSearchBacklog(pressure)
		b.index.SetPreemptRuntimePrioritySearches(pressure)
	}
	if b.config.Workload.WithExternalRWLock {
		b.rwMu.Lock()
	}
	b.markInsertActive(rangeStart, task.InsertOffset)
	if sigBits, _, ok := b.inflightSignatureConfig(); ok {
		b.materializeDirectionSignatures(task.Tags, sigBits)
	}
	var graphBefore GraphMutationStats
	graphStatsOK := false
	if b.graphMutationStatsEnabled {
		graphBefore, graphStatsOK = b.index.GraphMutationStats()
	}
	var err error
	dummyMode := os.Getenv("INSERT_DUMMY_MODE")
	var dummyGraphWrites uint64
	// Narrow wall-clock window: only the C++ index call itself.
	// Excludes Go-side bookkeeping (watermark, mutex, graph-stat snapshot
	// cgo calls before/after) and matches what an external user of the
	// index API would observe.
	batchInsertStart := time.Now()
	if strings.EqualFold(dummyMode, "noop") || strings.EqualFold(dummyMode, "none") {
		// Deliberately leave the index unchanged while the producer and
		// watermark path behave like an insert completed.
	} else if strings.EqualFold(dummyMode, "cpu") {
		b.dummyInsertCPU(task)
	} else if strings.EqualFold(dummyMode, "ann_search") {
		err = b.dummyInsertANNSearch(task)
	} else if strings.EqualFold(dummyMode, "ann_search_query") {
		err = b.dummyInsertANNQuerySearch(task)
	} else if strings.EqualFold(dummyMode, "graph_write_hot") {
		dummyGraphWrites = b.dummyInsertGraphWrite("hot", task)
	} else if strings.EqualFold(dummyMode, "graph_write_cold") {
		dummyGraphWrites = b.dummyInsertGraphWrite("cold", task)
	} else {
		err = b.index.BatchInsert(task.Data, task.Tags)
	}
	batchInsertWallMs := float64(time.Since(batchInsertStart).Microseconds()) / 1000.0
	b.postInsertCPUDelay(task)
	if err == nil && b.replicaRefresh != nil {
		err = b.replicaRefresh.maybeRefresh(b.index, task.InsertOffset)
	}
	var graphAfter GraphMutationStats
	graphStatsAfterOK := false
	if b.graphMutationStatsEnabled {
		graphAfter, graphStatsAfterOK = b.index.GraphMutationStats()
	}
	b.unmarkInsertActive(task.InsertOffset)
	if b.config.Workload.WithExternalRWLock {
		b.rwMu.Unlock()
	}
	if err != nil {
		log.Printf("Insert error: %v", err)
		return
	}
	insertOpLat := float64(time.Since(start).Microseconds()) / 1000.0
	insertE2ELat := float64(time.Since(task.CreateTime).Microseconds()) / 1000.0
	insertFinishMs := float64(time.Since(b.startTime).Microseconds()) / 1000.0
	insertFinishRawNs := MonotonicRawNs()
	b.insertMu.Lock()
	b.insertBatchInsertWallMs = append(b.insertBatchInsertWallMs, batchInsertWallMs)
	b.insertOpLatencies = append(b.insertOpLatencies, insertOpLat)
	b.insertE2ELatencies = append(b.insertE2ELatencies, insertE2ELat)
	b.insertFinishMs = append(b.insertFinishMs, insertFinishMs)
	b.insertCreateRawNs = append(b.insertCreateRawNs, task.CreateRawNs)
	b.insertStartRawNs = append(b.insertStartRawNs, insertStartRawNs)
	b.insertFinishRawNs = append(b.insertFinishRawNs, insertFinishRawNs)
	if graphStatsOK && graphStatsAfterOK {
		b.insertGraphConnectMs = append(b.insertGraphConnectMs, float64(graphAfter.ConnectNs-graphBefore.ConnectNs)/1e6)
		b.insertGraphLinkCriticalMs = append(b.insertGraphLinkCriticalMs, float64(graphAfter.LinkCriticalNs-graphBefore.LinkCriticalNs)/1e6)
		b.insertGraphUniqueLockWaitMs = append(b.insertGraphUniqueLockWaitMs, float64(graphAfter.UniqueLockWaitNs-graphBefore.UniqueLockWaitNs)/1e6)
		b.insertGraphSearchUniqueLockWaitMs = append(b.insertGraphSearchUniqueLockWaitMs, float64(graphAfter.SearchUniqueLockWaitNs-graphBefore.SearchUniqueLockWaitNs)/1e6)
		b.insertGraphSearchUniqueLockAcqs = append(b.insertGraphSearchUniqueLockAcqs, float64(graphAfter.SearchUniqueLockAcqs-graphBefore.SearchUniqueLockAcqs))
		b.insertGraphSearchCriticalMs = append(b.insertGraphSearchCriticalMs, float64(graphAfter.SearchCriticalNs-graphBefore.SearchCriticalNs)/1e6)
		b.insertGraphLinkUpdates = append(b.insertGraphLinkUpdates, float64(graphAfter.LinkUpdates-graphBefore.LinkUpdates))
		b.insertGraphConnectCalls = append(b.insertGraphConnectCalls, float64(graphAfter.ConnectCalls-graphBefore.ConnectCalls))
		b.insertGraphUpperSearchMs = append(b.insertGraphUpperSearchMs, float64(graphAfter.UpperSearchNs-graphBefore.UpperSearchNs)/1e6)
		b.insertGraphUpperSearchDistComps = append(b.insertGraphUpperSearchDistComps, float64(graphAfter.UpperSearchDistComps-graphBefore.UpperSearchDistComps))
		b.insertGraphUpperSearchEdgesScanned = append(b.insertGraphUpperSearchEdgesScanned, float64(graphAfter.UpperSearchEdgesScanned-graphBefore.UpperSearchEdgesScanned))
		b.insertGraphBaseSearchMs = append(b.insertGraphBaseSearchMs, float64(graphAfter.BaseSearchNs-graphBefore.BaseSearchNs)/1e6)
		b.insertGraphBaseSearchExpansions = append(b.insertGraphBaseSearchExpansions, float64(graphAfter.BaseSearchExpansions-graphBefore.BaseSearchExpansions))
		b.insertGraphBaseSearchEdgesScanned = append(b.insertGraphBaseSearchEdgesScanned, float64(graphAfter.BaseSearchEdgesScanned-graphBefore.BaseSearchEdgesScanned))
		b.insertGraphBaseSearchDistComps = append(b.insertGraphBaseSearchDistComps, float64(graphAfter.BaseSearchDistComps-graphBefore.BaseSearchDistComps))
		b.insertGraphSelectNewNeighborsMs = append(b.insertGraphSelectNewNeighborsMs, float64(graphAfter.SelectNewNeighborsNs-graphBefore.SelectNewNeighborsNs)/1e6)
		b.insertGraphSelectNewNeighborsInput = append(b.insertGraphSelectNewNeighborsInput, float64(graphAfter.SelectNewNeighborsInput-graphBefore.SelectNewNeighborsInput))
		b.insertGraphSelectNewNeighborsSelected = append(b.insertGraphSelectNewNeighborsSelected, float64(graphAfter.SelectNewNeighborsSelected-graphBefore.SelectNewNeighborsSelected))
		b.insertGraphSelectNewNeighborsHeuristicDistComps = append(b.insertGraphSelectNewNeighborsHeuristicDistComps, float64(graphAfter.SelectNewNeighborsHeuristicDistComps-graphBefore.SelectNewNeighborsHeuristicDistComps))
		b.insertGraphInsertedNodeLinkMs = append(b.insertGraphInsertedNodeLinkMs, float64(graphAfter.InsertedNodeLinkNs-graphBefore.InsertedNodeLinkNs)/1e6)
		b.insertGraphInsertedNodeEdgesWritten = append(b.insertGraphInsertedNodeEdgesWritten, float64(graphAfter.InsertedNodeEdgesWritten-graphBefore.InsertedNodeEdgesWritten))
		b.insertGraphExistingNeighborUpdateLoopMs = append(b.insertGraphExistingNeighborUpdateLoopMs, float64(graphAfter.ExistingNeighborUpdateLoopNs-graphBefore.ExistingNeighborUpdateLoopNs)/1e6)
		b.insertGraphExistingNeighborLoadScanMs = append(b.insertGraphExistingNeighborLoadScanMs, float64(graphAfter.ExistingNeighborLoadScanNs-graphBefore.ExistingNeighborLoadScanNs)/1e6)
		b.insertGraphExistingNeighborLoadedEdges = append(b.insertGraphExistingNeighborLoadedEdges, float64(graphAfter.ExistingNeighborLoadedEdges-graphBefore.ExistingNeighborLoadedEdges))
		b.insertGraphExistingNeighborAppendMs = append(b.insertGraphExistingNeighborAppendMs, float64(graphAfter.ExistingNeighborAppendNs-graphBefore.ExistingNeighborAppendNs)/1e6)
		b.insertGraphExistingNeighborPruneMs = append(b.insertGraphExistingNeighborPruneMs, float64(graphAfter.ExistingNeighborPruneNs-graphBefore.ExistingNeighborPruneNs)/1e6)
		b.insertGraphExistingNeighborPruneCandidates = append(b.insertGraphExistingNeighborPruneCandidates, float64(graphAfter.ExistingNeighborPruneCandidates-graphBefore.ExistingNeighborPruneCandidates))
		b.insertGraphExistingNeighborPruneDistComps = append(b.insertGraphExistingNeighborPruneDistComps, float64(graphAfter.ExistingNeighborPruneDistComps-graphBefore.ExistingNeighborPruneDistComps))
		b.insertGraphExistingNeighborUndoRecordMs = append(b.insertGraphExistingNeighborUndoRecordMs, float64(graphAfter.ExistingNeighborUndoRecordNs-graphBefore.ExistingNeighborUndoRecordNs)/1e6)
		b.insertGraphExistingNeighborRewriteMs = append(b.insertGraphExistingNeighborRewriteMs, float64(graphAfter.ExistingNeighborRewriteNs-graphBefore.ExistingNeighborRewriteNs)/1e6)
		existingNeighborEdgesWritten := graphAfter.ExistingNeighborEdgesWritten - graphBefore.ExistingNeighborEdgesWritten
		b.insertGraphExistingNeighborEdgesWritten = append(b.insertGraphExistingNeighborEdgesWritten, float64(existingNeighborEdgesWritten+dummyGraphWrites))
		b.insertGraphExistingNeighborEdgesPruned = append(b.insertGraphExistingNeighborEdgesPruned, float64(graphAfter.ExistingNeighborEdgesPruned-graphBefore.ExistingNeighborEdgesPruned))
		b.insertGraphExistingNeighborVisits = append(b.insertGraphExistingNeighborVisits, float64(graphAfter.ExistingNeighborVisits-graphBefore.ExistingNeighborVisits))
		b.insertGraphExistingNeighborAppends = append(b.insertGraphExistingNeighborAppends, float64(graphAfter.ExistingNeighborAppends-graphBefore.ExistingNeighborAppends))
		b.insertGraphExistingNeighborPrunes = append(b.insertGraphExistingNeighborPrunes, float64(graphAfter.ExistingNeighborPrunes-graphBefore.ExistingNeighborPrunes))
		b.insertGraphExistingNeighborPrunedEdgesRecorded = append(b.insertGraphExistingNeighborPrunedEdgesRecorded, float64(graphAfter.ExistingNeighborPrunedEdgesRecorded-graphBefore.ExistingNeighborPrunedEdgesRecorded))
	}
	b.insertPointCnt += len(task.Data)
	b.insertMu.Unlock()
	atomic.AddInt64(&b.globalInsertCnt, int64(len(task.Data)))

	b.completionMu.Lock()
	b.completedOffsets[task.InsertOffset] = true
	for b.pendingIdx < len(b.pendingOffsets) {
		off := b.pendingOffsets[b.pendingIdx]
		if !b.completedOffsets[off] {
			break
		}
		delete(b.completedOffsets, off)
		b.committedOffset.Store(off)
		b.commitEventOffset = append(b.commitEventOffset, off)
		b.commitEventMs = append(b.commitEventMs, float64(time.Since(b.startTime).Microseconds())/1000.0)
		b.pendingIdx++
	}
	b.completionMu.Unlock()
}

func (b *Bench) dummyInsertANNSearch(task Task) error {
	loops := 1
	if r := os.Getenv("INSERT_DUMMY_LOOPS"); r != "" {
		var override int
		if n, err := fmt.Sscanf(r, "%d", &override); n == 1 && err == nil && override > 0 {
			loops = override
		}
	}
	k := b.config.Search.RecallAt
	if k == 0 {
		k = 10
	}
	view := b.committedOffset.Load()
	for i := 0; i < loops; i++ {
		if _, _, err := b.index.BatchSearch(task.Data, k, view); err != nil {
			return err
		}
	}
	return nil
}

func (b *Bench) dummyInsertANNQuerySearch(task Task) error {
	loops := 1
	if r := os.Getenv("INSERT_DUMMY_LOOPS"); r != "" {
		var override int
		if n, err := fmt.Sscanf(r, "%d", &override); n == 1 && err == nil && override > 0 {
			loops = override
		}
	}
	k := b.config.Search.RecallAt
	if k == 0 {
		k = 10
	}
	totalQueries := b.closedLoopTotalQueries
	if totalQueries <= 0 || b.closedLoopDataDim <= 0 || len(b.closedLoopWorkloadQueries) == 0 {
		return b.dummyInsertANNSearch(task)
	}
	batchSize := len(task.Data)
	if batchSize <= 0 {
		return nil
	}
	view := b.committedOffset.Load()
	tags := make([]uint32, batchSize)
	for loop := 0; loop < loops; loop++ {
		start := task.InsertOffset + uint64(loop*batchSize)
		for i := 0; i < batchSize; i++ {
			tags[i] = uint32((start + uint64(i)) % uint64(totalQueries))
		}
		queries := buildQueriesFromTags(b.closedLoopWorkloadQueries, b.closedLoopDataDim, tags)
		if _, _, err := b.index.BatchSearch(queries, k, view); err != nil {
			return err
		}
	}
	return nil
}

func (b *Bench) prepareGraphTouchLabels(workloadQueries []float32, dataDim int, totalQueries int) error {
	dummyMode := strings.ToLower(os.Getenv("INSERT_DUMMY_MODE"))
	if dummyMode != "graph_write_hot" && dummyMode != "graph_write_cold" {
		return nil
	}
	if totalQueries <= 0 || dataDim <= 0 {
		return fmt.Errorf("empty query workload")
	}

	sampleQueries := closedLoopEnvInt("GRAPH_TOUCH_SAMPLE_QUERIES", 512)
	if sampleQueries <= 0 {
		sampleQueries = 1
	}
	if sampleQueries > totalQueries {
		sampleQueries = totalQueries
	}
	searchK := closedLoopEnvInt("GRAPH_TOUCH_SEARCH_K", int(b.config.Search.RecallAt))
	if searchK <= 0 {
		searchK = 10
	}
	maxNeighbors := closedLoopEnvInt("GRAPH_TOUCH_MAX_NEIGHBORS", 32)
	if maxNeighbors < 0 {
		maxNeighbors = 0
	}
	labelLimit := closedLoopEnvInt("GRAPH_TOUCH_LABEL_LIMIT", 4096)
	if labelLimit <= 0 {
		labelLimit = 4096
	}

	tags := make([]uint32, sampleQueries)
	for i := 0; i < sampleQueries; i++ {
		tags[i] = uint32(i)
	}
	queries := buildQueriesFromTags(workloadQueries, dataDim, tags)
	results, _, err := b.index.BatchSearch(queries, uint32(searchK), b.committedOffset.Load())
	if err != nil {
		return err
	}

	hotSeen := make(map[uint32]struct{}, labelLimit)
	hot := make([]uint32, 0, labelLimit)
	flatResults := make([]uint32, 0, sampleQueries*searchK)
	addHot := func(label uint32) {
		if int(label) < 0 || int(label) >= b.config.Data.BeginNum {
			return
		}
		if _, ok := hotSeen[label]; ok {
			return
		}
		if len(hot) >= labelLimit {
			return
		}
		hotSeen[label] = struct{}{}
		hot = append(hot, label)
	}
	for _, row := range results {
		for _, label := range row {
			addHot(label)
			flatResults = append(flatResults, label)
		}
	}
	if maxNeighbors > 0 && len(flatResults) > 0 {
		neighbors, err := b.index.BaseNeighbors(flatResults, maxNeighbors)
		if err != nil {
			log.Printf("Graph write touch labels: base-neighbor expansion skipped: %v", err)
		} else {
			for _, row := range neighbors {
				for _, label := range row {
					addHot(label)
				}
			}
		}
	}
	if len(hot) == 0 {
		return fmt.Errorf("hot label set is empty")
	}

	cold := make([]uint32, 0, len(hot))
	for label := b.config.Data.BeginNum - 1; label >= 0 && len(cold) < len(hot); label-- {
		u := uint32(label)
		if _, ok := hotSeen[u]; ok {
			continue
		}
		cold = append(cold, u)
	}
	if len(cold) == 0 {
		return fmt.Errorf("cold label set is empty")
	}

	b.graphTouchHotLabels = hot
	b.graphTouchColdLabels = cold
	log.Printf(
		"Graph write touch labels: hot=%d cold=%d sample_queries=%d search_k=%d neighbor_cap=%d label_limit=%d",
		len(hot), len(cold), sampleQueries, searchK, maxNeighbors, labelLimit,
	)
	return nil
}

func (b *Bench) dummyInsertGraphWrite(kind string, task Task) uint64 {
	var labels []uint32
	if strings.EqualFold(kind, "cold") {
		labels = b.graphTouchColdLabels
	} else {
		labels = b.graphTouchHotLabels
	}
	if len(labels) == 0 {
		return 0
	}

	loops := insertCPULoopsFromEnv("INSERT_DUMMY_LOOPS")
	if loops <= 0 {
		loops = 1
	}
	maxEdges := closedLoopEnvInt("GRAPH_TOUCH_MAX_EDGES", 32)
	if maxEdges < 0 {
		maxEdges = 0
	}
	defaultLabelsPerTask := len(task.Data) * 64
	if defaultLabelsPerTask <= 0 {
		defaultLabelsPerTask = 1
	}
	labelsPerTask := closedLoopEnvInt("GRAPH_TOUCH_LABELS_PER_TASK", defaultLabelsPerTask)
	if labelsPerTask <= 0 {
		labelsPerTask = 1
	}

	batchLabels := make([]uint32, labelsPerTask)
	start := int(task.InsertOffset % uint64(len(labels)))
	for i := 0; i < labelsPerTask; i++ {
		batchLabels[i] = labels[(start+i)%len(labels)]
	}
	return b.index.GraphLinkWriteProbe(batchLabels, loops, maxEdges)
}

func (b *Bench) dummyInsertCPU(task Task) {
	loops := insertCPULoopsFromEnv("INSERT_DUMMY_LOOPS")
	if loops <= 0 {
		loops = 1
	}
	b.burnInsertCPU(task, loops)
}

func (b *Bench) postInsertCPUDelay(task Task) {
	loops := insertCPULoopsFromEnv("INSERT_POST_CPU_LOOPS")
	if loops <= 0 {
		return
	}
	b.burnInsertCPU(task, loops)
}

func insertCPULoopsFromEnv(name string) int {
	loops := 0
	if r := os.Getenv(name); r != "" {
		var override int
		if n, err := fmt.Sscanf(r, "%d", &override); n == 1 && err == nil && override > 0 {
			loops = override
		}
	}
	return loops
}

func closedLoopEnabled() bool {
	raw := os.Getenv("M3_CLOSED_LOOP")
	return raw == "1" || strings.EqualFold(raw, "true")
}

func closedLoopEnvInt(name string, fallback int) int {
	raw := os.Getenv(name)
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return value
}

func parseCPUList(raw string) []int {
	var cpus []int
	for _, token := range strings.Split(raw, ",") {
		token = strings.TrimSpace(token)
		if token == "" {
			continue
		}
		if strings.Contains(token, "-") {
			parts := strings.SplitN(token, "-", 2)
			begin, errBegin := strconv.Atoi(strings.TrimSpace(parts[0]))
			end, errEnd := strconv.Atoi(strings.TrimSpace(parts[1]))
			if errBegin != nil || errEnd != nil || begin < 0 || end < 0 {
				continue
			}
			if begin <= end {
				for cpu := begin; cpu <= end; cpu++ {
					cpus = append(cpus, cpu)
				}
			} else {
				for cpu := begin; cpu >= end; cpu-- {
					cpus = append(cpus, cpu)
				}
			}
			continue
		}
		cpu, err := strconv.Atoi(token)
		if err == nil && cpu >= 0 {
			cpus = append(cpus, cpu)
		}
	}
	return cpus
}

func pinCurrentThreadToEnvCPU(name string, ordinal int) {
	raw := os.Getenv(name)
	if raw == "" {
		return
	}
	cpus := parseCPUList(raw)
	if len(cpus) == 0 {
		return
	}
	var set unix.CPUSet
	set.Zero()
	set.Set(cpus[ordinal%len(cpus)])
	_ = unix.SchedSetaffinity(0, &set)
}

func (b *Bench) runClosedLoopWorkers(data []float32, dataDim int, workloadQueries []float32, totalQueries int) error {
	if totalQueries <= 0 {
		return fmt.Errorf("closed-loop mode requires workload queries")
	}
	b.index.SetQueryParams(BuildQueryParams(b.config))
	b.closedLoopWorkloadQueries = workloadQueries
	b.closedLoopDataDim = dataDim
	b.closedLoopTotalQueries = totalQueries
	if err := b.prepareGraphTouchLabels(workloadQueries, dataDim, totalQueries); err != nil {
		log.Printf("graph write touch labels unavailable: %v", err)
	}

	durationMs := closedLoopEnvInt("M3_CLOSED_LOOP_DURATION_MS", 8000)
	if durationMs <= 0 {
		durationMs = 8000
	}
	measuredWorkers := closedLoopEnvInt("M3_CLOSED_LOOP_MEASURED_WORKERS", measuredLanes())
	if measuredWorkers <= 0 {
		measuredWorkers = 1
	}
	competingWorkers := closedLoopEnvInt("M3_CLOSED_LOOP_COMPETING_WORKERS", b.config.Workload.NumThreads-measuredWorkers)
	if competingWorkers < 0 {
		competingWorkers = 0
	}
	mode := strings.ToLower(os.Getenv("M3_CLOSED_LOOP_COMPETING_MODE"))
	if mode == "" {
		mode = "idle"
	}
	batchSize := b.config.Workload.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}
	recallAt := b.config.Search.RecallAt
	if recallAt == 0 {
		recallAt = 10
	}
	log.Printf(
		"Closed-loop setup: duration=%dms measured_workers=%d competing_workers=%d mode=%s batch_size=%d burst_windows=%d",
		durationMs, measuredWorkers, competingWorkers, mode, batchSize, len(b.config.Workload.InsertBurstSchedule),
	)
	if b.insertLimiter != nil {
		log.Printf(
			"Closed-loop insert limiter active: %.2f points/sec (%.2f batches/sec @ batch_size=%d)",
			b.config.Workload.InsertEventRate,
			b.config.Workload.InsertEventRate/float64(batchSize),
			batchSize,
		)
	}
	bgSearchLimiter := buildLimiter(
		float64(closedLoopEnvInt("M3_CLOSED_LOOP_BG_SEARCH_EVENT_RATE", 0)),
		batchSize,
	)
	if bgSearchLimiter != nil {
		bgSearchRate := closedLoopEnvInt("M3_CLOSED_LOOP_BG_SEARCH_EVENT_RATE", 0)
		log.Printf(
			"Closed-loop background search limiter active: %d queries/sec (%.2f batches/sec @ batch_size=%d)",
			bgSearchRate,
			float64(bgSearchRate)/float64(batchSize),
			batchSize,
		)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(durationMs)*time.Millisecond)
	defer cancel()

	var wg sync.WaitGroup
	var querySeq atomic.Uint64
	var nextInsertOffset atomic.Uint64
	nextInsertOffset.Store(uint64(b.config.Data.BeginNum))
	startRawNs := MonotonicRawNs()

	inInsertWindow := func() bool {
		if len(b.config.Workload.InsertBurstSchedule) == 0 {
			return false
		}
		elapsedMs := float64(MonotonicRawNs()-startRawNs) / 1_000_000.0
		for _, window := range b.config.Workload.InsertBurstSchedule {
			startMs := window.StartMs
			endMs := startMs + window.DurationMs
			if elapsedMs >= startMs && elapsedMs < endMs {
				return true
			}
		}
		return false
	}

	makeSearchTask := func(measured bool) Task {
		start := querySeq.Add(uint64(batchSize)) - uint64(batchSize)
		tags := make([]uint32, batchSize)
		for i := 0; i < batchSize; i++ {
			tags[i] = uint32((start + uint64(i)) % uint64(totalQueries))
		}
		queries := buildQueriesFromTags(workloadQueries, dataDim, tags)
		now := time.Now()
		view := b.committedOffset.Load()
		return Task{
			Type:          SearchTask,
			Data:          queries,
			Tags:          tags,
			RecallAt:      recallAt,
			InsertOffset:  view,
			WaitForInsert: false,
			CreateTime:    now,
			CreateRawNs:   MonotonicRawNs(),
			Measured:      measured,
		}
	}

	makeInsertTask := func() (Task, bool) {
		begin := int(nextInsertOffset.Add(uint64(batchSize)) - uint64(batchSize))
		if begin >= b.config.Data.MaxElements {
			return Task{}, false
		}
		end := begin + batchSize
		if end > b.config.Data.MaxElements {
			end = b.config.Data.MaxElements
		}
		if end <= begin {
			return Task{}, false
		}
		now := time.Now()
		return Task{
			Type:         InsertTask,
			Data:         sliceVectorsRange(data, dataDim, begin, end),
			Tags:         makeSequentialTags(begin, end),
			InsertOffset: uint64(end),
			CreateTime:   now,
			CreateRawNs:  MonotonicRawNs(),
		}, true
	}

	startSearchWorkers := func(workers int, measured bool) {
		for i := 0; i < workers; i++ {
			wg.Add(1)
			go func(workerID int) {
				runtime.LockOSThread()
				if measured {
					pinCurrentThreadToEnvCPU("ANNCHOR_MEASURED_SEARCH_CPU_LIST", workerID)
				} else {
					pinCurrentThreadToEnvCPU("ANNCHOR_SEARCH_CPU_LIST", workerID)
				}
				defer wg.Done()
				for {
					select {
					case <-ctx.Done():
						return
					default:
					}
					if !measured && bgSearchLimiter != nil {
						if err := bgSearchLimiter.Wait(ctx); err != nil {
							return
						}
					}
					task := makeSearchTask(measured)
					b.handleSearchTask(task, time.Now())
				}
			}(i)
		}
	}

	startInsertWorkers := func(workers int) {
		for i := 0; i < workers; i++ {
			wg.Add(1)
			go func(workerID int) {
				runtime.LockOSThread()
				pinCurrentThreadToEnvCPU("ANNCHOR_INSERT_CPU_LIST", workerID)
				defer wg.Done()
				for {
					select {
					case <-ctx.Done():
						return
					default:
					}
					if b.insertLimiter != nil {
						if err := b.insertLimiter.Wait(ctx); err != nil {
							return
						}
					}
					task, ok := makeInsertTask()
					if !ok {
						return
					}
					b.handleInsertTask(task, task.CreateTime)
				}
			}(i)
		}
	}

	startPeriodicWorkers := func(workers int) {
		for i := 0; i < workers; i++ {
			wg.Add(1)
			go func(workerID int) {
				runtime.LockOSThread()
				pinCurrentThreadToEnvCPU("ANNCHOR_SEARCH_CPU_LIST", workerID)
				defer wg.Done()
				for {
					select {
					case <-ctx.Done():
						return
					default:
					}
					if inInsertWindow() {
						if b.insertLimiter != nil {
							if err := b.insertLimiter.Wait(ctx); err != nil {
								return
							}
						}
						task, ok := makeInsertTask()
						if ok {
							b.handleInsertTask(task, task.CreateTime)
							continue
						}
					}
					task := makeSearchTask(false)
					b.handleSearchTask(task, time.Now())
				}
			}(i)
		}
	}

	startPeriodicInsertWorkers := func(workers int) {
		for i := 0; i < workers; i++ {
			wg.Add(1)
			go func(workerID int) {
				runtime.LockOSThread()
				pinCurrentThreadToEnvCPU("ANNCHOR_INSERT_CPU_LIST", workerID)
				defer wg.Done()
				for {
					select {
					case <-ctx.Done():
						return
					default:
					}
					if !inInsertWindow() {
						time.Sleep(100 * time.Microsecond)
						continue
					}
					if b.insertLimiter != nil {
						if err := b.insertLimiter.Wait(ctx); err != nil {
							return
						}
					}
					task, ok := makeInsertTask()
					if !ok {
						return
					}
					b.handleInsertTask(task, task.CreateTime)
				}
			}(i)
		}
	}

	startSpinWorkers := func(workers int) {
		for i := 0; i < workers; i++ {
			wg.Add(1)
			go func(workerID int) {
				runtime.LockOSThread()
				pinCurrentThreadToEnvCPU("ANNCHOR_INSERT_CPU_LIST", workerID)
				defer wg.Done()
				var x uint64 = uint64(workerID) + 1
				for {
					select {
					case <-ctx.Done():
						return
					default:
					}
					for k := 0; k < 1000000; k++ {
						x = x*1103515245 + 12345
					}
					_ = x
				}
			}(i)
		}
	}

	startSearchWorkers(measuredWorkers, true)
	switch mode {
	case "idle", "none", "baseline":
	case "spin", "spin_idle", "noop":
		startSpinWorkers(competingWorkers)
	case "search", "bg_search", "background_search":
		startSearchWorkers(competingWorkers, false)
	case "insert", "write":
		startInsertWorkers(competingWorkers)
	case "periodic", "alternate", "alternating":
		startPeriodicWorkers(competingWorkers)
	case "search_plus_periodic_insert", "bg_search_periodic_insert", "matched_rw":
		insertWorkers := closedLoopEnvInt("M3_CLOSED_LOOP_INSERT_WORKERS", competingWorkers)
		if insertWorkers < 0 {
			insertWorkers = 0
		}
		log.Printf(
			"Closed-loop matched RW: background_search_workers=%d insert_workers=%d",
			competingWorkers, insertWorkers,
		)
		startSearchWorkers(competingWorkers, false)
		startPeriodicInsertWorkers(insertWorkers)
	default:
		return fmt.Errorf("unknown M3_CLOSED_LOOP_COMPETING_MODE=%q", mode)
	}
	wg.Wait()
	return nil
}

func (b *Bench) burnInsertCPU(task Task, loops int) {
	var sum float32
	for l := 0; l < loops; l++ {
		for _, vec := range task.Data {
			local := float32(0)
			for _, v := range vec {
				local += v * v
			}
			sum += local
		}
	}
	if sum < 0 {
		log.Printf("dummy insert impossible sink: %f", sum)
	}
}

func (b *Bench) updateLatestInsertFront(offset uint64) {
	for {
		old := b.latestInsertFront.Load()
		if offset <= old {
			return
		}
		if b.latestInsertFront.CompareAndSwap(old, offset) {
			return
		}
	}
}

func (b *Bench) markInsertActive(start, end uint64) {
	b.activeInsertMu.Lock()
	b.activeInsertRanges[end] = insertRange{start: start, end: end}
	if end > b.activeInsertMax {
		b.activeInsertMax = end
	}
	b.activeInsertMu.Unlock()
}

func (b *Bench) unmarkInsertActive(end uint64) {
	b.activeInsertMu.Lock()
	delete(b.activeInsertRanges, end)
	if end == b.activeInsertMax {
		var nextMax uint64
		for _, r := range b.activeInsertRanges {
			if r.end > nextMax {
				nextMax = r.end
			}
		}
		b.activeInsertMax = nextMax
	}
	b.activeInsertMu.Unlock()
}

func (b *Bench) activeInsertSnapshot(committed uint64) []insertRange {
	b.activeInsertMu.Lock()
	defer b.activeInsertMu.Unlock()

	ranges := make([]insertRange, 0, len(b.activeInsertRanges))
	for _, r := range b.activeInsertRanges {
		if r.end <= committed {
			continue
		}
		if r.start < committed {
			r.start = committed
		}
		if r.end > r.start {
			ranges = append(ranges, r)
		}
	}
	sort.Slice(ranges, func(i, j int) bool {
		return ranges[i].start < ranges[j].start
	})
	return ranges
}

func (b *Bench) activeInsertFront() uint64 {
	b.activeInsertMu.Lock()
	defer b.activeInsertMu.Unlock()
	return b.activeInsertMax
}

func (b *Bench) waitForInsertRangesDone(ranges []insertRange, maxWaitUs int) {
	if len(ranges) == 0 || maxWaitUs == 0 {
		return
	}
	deadline := time.Time{}
	if maxWaitUs > 0 {
		deadline = time.Now().Add(time.Duration(maxWaitUs) * time.Microsecond)
	}
	spin := 0
	for {
		pending := false
		b.activeInsertMu.Lock()
		for _, r := range ranges {
			if _, ok := b.activeInsertRanges[r.end]; ok {
				pending = true
				break
			}
		}
		b.activeInsertMu.Unlock()
		if !pending {
			return
		}
		if !deadline.IsZero() && time.Now().After(deadline) {
			return
		}
		runtime.Gosched()
		spin++
		if spin%16 == 0 {
			time.Sleep(50 * time.Microsecond)
		}
	}
}

type scoredTag struct {
	tag  uint32
	dist float32
}

type occMergeStats struct {
	diffPts                  int
	seedDists                int
	diffDists                int
	mergeComps               int
	totalL2Dists             int
	filterTests              int
	filterL2Dists            int
	candidates               int
	m2vSeenQueries           int
	m2vNoRouteQueries        int
	m2vCandidateRegions      int
	m2vOpenedRegions         int
	m2vOpenedPostings        int
	m2vMemberBoundChecks     int
	m2vMemberBoundSkips      int
	m2vHubSuppressedRegions  int
	m2vWideSuppressedRegions int
	m2vWeakSuppressedRegions int
	m2vStrongRegionHits      int
	m2vMediumRegionHits      int
	m2vWeakRegionHits        int
}

func (b *Bench) l2Squared(query []float32, pointID int) float32 {
	offset := pointID * b.dataDim
	point := b.data[offset : offset+b.dataDim]
	var sum float32
	for i := 0; i < b.dataDim; i++ {
		d := query[i] - point[i]
		sum += d * d
	}
	return sum
}

func (b *Bench) l2SquaredBounded(query []float32, pointID int, limit float32) (float32, bool) {
	offset := pointID * b.dataDim
	point := b.data[offset : offset+b.dataDim]
	var sum float32
	for i := 0; i < b.dataDim; i++ {
		d := query[i] - point[i]
		sum += d * d
		if sum > limit {
			return sum, false
		}
	}
	return sum, true
}

func (b *Bench) l2SquaredBetweenBounded(leftID int, rightID int, limit float32) (float32, bool) {
	leftOffset := leftID * b.dataDim
	rightOffset := rightID * b.dataDim
	if leftOffset < 0 || rightOffset < 0 ||
		leftOffset+b.dataDim > len(b.data) ||
		rightOffset+b.dataDim > len(b.data) {
		return 0, false
	}
	left := b.data[leftOffset : leftOffset+b.dataDim]
	right := b.data[rightOffset : rightOffset+b.dataDim]
	var sum float32
	for i := 0; i < b.dataDim; i++ {
		d := left[i] - right[i]
		sum += d * d
		if sum > limit {
			return sum, false
		}
	}
	return sum, true
}

func (b *Bench) inflightPathCorridorEnabled() bool {
	return b.config != nil &&
		b.config.Search.InflightJoinPathCorridor != nil &&
		*b.config.Search.InflightJoinPathCorridor
}

func (b *Bench) inflightPathCorridorScale() float64 {
	if b.config != nil && b.config.Search.InflightJoinPathCorridorScale != nil {
		scale := *b.config.Search.InflightJoinPathCorridorScale
		if scale >= 1 {
			return scale
		}
	}
	return 1
}

func (b *Bench) inflightPathCorridorWitnesses() int {
	if b.config != nil && b.config.Search.InflightJoinPathCorridorWitnesses != nil {
		witnesses := *b.config.Search.InflightJoinPathCorridorWitnesses
		if witnesses > 0 && witnesses <= 32 {
			return witnesses
		}
	}
	return 2
}

func (b *Bench) inflightPathCorridorMaxDim() int {
	if b.config != nil && b.config.Search.InflightJoinPathCorridorMaxDim != nil {
		maxDim := *b.config.Search.InflightJoinPathCorridorMaxDim
		if maxDim <= 0 {
			return 0
		}
		return maxDim
	}
	return 256
}

func (b *Bench) inflightPathCorridorAllowedForDim() bool {
	maxDim := b.inflightPathCorridorMaxDim()
	return maxDim <= 0 || b.dataDim <= maxDim
}

func (b *Bench) inflightPathOneHopEnabled() bool {
	if b.config == nil {
		return false
	}
	if b.config.Search.InflightJoinPathOneHop != nil {
		return *b.config.Search.InflightJoinPathOneHop
	}
	return false
}

func (b *Bench) inflightM2VEnabled() bool {
	return b.config != nil &&
		b.config.Search.InflightJoinM2V != nil &&
		*b.config.Search.InflightJoinM2V
}

func (b *Bench) inflightM2VEntryNeighborLimit(maxNeighbors int) int {
	limit := 4
	if b.dataDim > 0 && b.dataDim <= 256 {
		limit = 8
	}
	if raw := strings.TrimSpace(os.Getenv("ANNCHOR_M2V_ENTRY_NEIGHBORS")); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			limit = parsed
		}
	}
	if b.config != nil && b.config.Search.InflightJoinM2VEntryNeighbors != nil {
		if configured := *b.config.Search.InflightJoinM2VEntryNeighbors; configured > 0 {
			limit = configured
		}
	}
	if limit > maxNeighbors {
		limit = maxNeighbors
	}
	if limit < 1 {
		limit = 1
	}
	return limit
}

func (b *Bench) inflightM2VRegionCapacity(smallThreshold int) int {
	capacity := smallThreshold
	if capacity <= 0 {
		capacity = 256
	}
	if raw := strings.TrimSpace(os.Getenv("ANNCHOR_M2V_REGION_CAPACITY")); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			capacity = parsed
		}
	}
	if b.config != nil && b.config.Search.InflightJoinM2VRegionCapacity != nil {
		if configured := *b.config.Search.InflightJoinM2VRegionCapacity; configured > 0 {
			capacity = configured
		}
	}
	if capacity < 16 {
		capacity = 16
	}
	return capacity
}

func (b *Bench) inflightM2VRouteLabelCount() int {
	labels := 2
	if b.dataDim >= 512 {
		labels = 4
	} else if b.dataDim >= 256 {
		labels = 3
	}
	if raw := strings.TrimSpace(os.Getenv("ANNCHOR_M2V_ROUTE_LABELS_PER_NODE")); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			labels = parsed
		}
	}
	if labels < 1 {
		labels = 1
	}
	if labels > 8 {
		labels = 8
	}
	return labels
}

func (b *Bench) inflightM2VRouteLabelSpace(targetRegions int) uint32 {
	if targetRegions < 1 {
		targetRegions = 1
	}
	space := targetRegions * 8
	if space < 64 {
		space = 64
	}
	if b.dataDim >= 512 && space < 128 {
		space = 128
	}
	if raw := strings.TrimSpace(os.Getenv("ANNCHOR_M2V_ROUTE_LABEL_SPACE")); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			space = parsed
		}
	}
	if space < 16 {
		space = 16
	}
	if space > 16384 {
		space = 16384
	}
	return uint32(space)
}

const (
	m2vRouteSourceStrong uint8 = 1 << iota
	m2vRouteSourceMedium
	m2vRouteSourceWeak
)

func m2vEnvBool(name string) bool {
	raw := strings.TrimSpace(os.Getenv(name))
	return raw == "1" || strings.EqualFold(raw, "true") || strings.EqualFold(raw, "yes")
}

func m2vEnvBoolDefault(name string, fallback bool) bool {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	return raw == "1" || strings.EqualFold(raw, "true") || strings.EqualFold(raw, "yes")
}

func m2vMix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

func m2vDiscoveryLabels(id uint32, neighbors []uint32, labelSpace uint32, labelCount int) []uint32 {
	if labelSpace == 0 || labelCount <= 0 {
		return nil
	}
	labels := make([]uint32, 0, labelCount)
	seen := make(map[uint32]struct{}, labelCount)
	for salt := 0; salt < labelCount; salt++ {
		best := ^uint64(0)
		seenNeighbor := false
		saltSeed := uint64(salt+1) * 0x9e3779b97f4a7c15
		for _, nb := range neighbors {
			if nb == id {
				continue
			}
			h := m2vMix64(uint64(nb) ^ saltSeed)
			if h < best {
				best = h
			}
			seenNeighbor = true
		}
		if !seenNeighbor {
			best = m2vMix64(uint64(id) ^ saltSeed)
		}
		label := uint32(salt)*labelSpace + uint32(best%uint64(labelSpace))
		if _, ok := seen[label]; ok {
			continue
		}
		seen[label] = struct{}{}
		labels = append(labels, label)
	}
	return labels
}

func (b *Bench) inflightBaseNeighborLimit() (int, bool) {
	if b.config == nil || b.config.Index.M <= 0 {
		return 0, false
	}
	return b.config.Index.M * 2, true
}

func (b *Bench) inflightPathOneHopBridgeMinSupport() int {
	minSupport := 1
	if b.dataDim >= 512 {
		minSupport = 2
	}
	if raw := strings.TrimSpace(os.Getenv("ANNCHOR_M2_BRIDGE_MIN_QUERY_SUPPORT")); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil {
			minSupport = parsed
		}
	}
	if minSupport < 1 {
		minSupport = 1
	}
	return minSupport
}

func (b *Bench) inflightPathOneHopBridgeSelectivitySigma() float64 {
	sigma := 2.0
	if raw := strings.TrimSpace(os.Getenv("ANNCHOR_M2_BRIDGE_SELECTIVITY_SIGMA")); raw != "" {
		if parsed, err := strconv.ParseFloat(raw, 64); err == nil {
			sigma = parsed
		}
	}
	if sigma < 0 {
		return 0
	}
	return sigma
}

func (b *Bench) inflightPathOneHopFrontierRescueLimit(diffPts int, maxNeighbors int) int {
	if diffPts <= 0 || maxNeighbors <= 0 || b.dataDim > 128 {
		return 0
	}
	if raw := strings.TrimSpace(os.Getenv("ANNCHOR_M2_FRONTIER_RESCUE")); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil {
			if parsed < 0 {
				return 0
			}
			if parsed > diffPts {
				return diffPts
			}
			return parsed
		}
	}
	// Low-dimensional exact checks are cheap enough to cover the uncertain
	// publication frontier.  The budget grows sublinearly with the fresh gap,
	// and with the graph fanout because each frontier point can enter the query
	// corridor through several base-layer choices.
	fanoutChoices := math.Log2(float64(maxNeighbors) + 1.0)
	if fanoutChoices < 1 {
		fanoutChoices = 1
	}
	limit := int(math.Ceil(math.Sqrt(float64(diffPts*maxNeighbors) * fanoutChoices)))
	if limit > diffPts {
		return diffPts
	}
	return limit
}

func (b *Bench) cachedBaseNeighbors(labels []uint32, maxNeighbors int) ([][]uint32, error) {
	if len(labels) == 0 {
		return nil, nil
	}
	out := make([][]uint32, len(labels))
	missing := make([]uint32, 0)
	missingPos := make([]int, 0)

	b.m2FreshNeighborMu.RLock()
	for i, label := range labels {
		if cached, ok := b.m2FreshNeighborCache[label]; ok {
			out[i] = cached
			continue
		}
		missing = append(missing, label)
		missingPos = append(missingPos, i)
	}
	b.m2FreshNeighborMu.RUnlock()

	if len(missing) == 0 {
		return out, nil
	}
	fetched, err := b.index.BaseNeighbors(missing, maxNeighbors)
	if err != nil {
		return nil, err
	}

	b.m2FreshNeighborMu.Lock()
	for i, label := range missing {
		var neighbors []uint32
		if i < len(fetched) {
			neighbors = append([]uint32(nil), fetched[i]...)
		}
		b.m2FreshNeighborCache[label] = neighbors
		out[missingPos[i]] = neighbors
	}
	b.m2FreshNeighborMu.Unlock()

	return out, nil
}

func (b *Bench) inflightSignatureConfig() (int, int, bool) {
	if b.config == nil || b.config.Search.InflightJoinSignatureBits == nil {
		return 0, 0, false
	}
	sigBits := *b.config.Search.InflightJoinSignatureBits
	if sigBits <= 0 {
		return 0, 0, false
	}
	radius := sigBits / 4
	if b.config.Search.InflightJoinSignatureRadius != nil {
		radius = *b.config.Search.InflightJoinSignatureRadius
	}
	if radius < 0 {
		radius = 0
	}
	return sigBits, radius, true
}

func (b *Bench) ensureDirectionSignatures(sigBits int) {
	if sigBits <= 0 {
		return
	}
	if sigBits > 128 {
		sigBits = 128
	}
	totalPoints := 0
	if b.dataDim > 0 {
		totalPoints = len(b.data) / b.dataDim
	}
	b.directionSigMu.Lock()
	defer b.directionSigMu.Unlock()
	if b.directionSigBits == sigBits &&
		len(b.directionSigPlanes) == sigBits*b.dataDim &&
		len(b.directionSigLo) == totalPoints {
		return
	}
	rng := mathrand.New(mathrand.NewSource(20260507))
	b.directionSigBits = sigBits
	b.directionSigPlanes = make([]float32, sigBits*b.dataDim)
	for i := range b.directionSigPlanes {
		b.directionSigPlanes[i] = float32(rng.NormFloat64())
	}
	b.directionSigLo = make([]uint64, totalPoints)
	b.directionSigHi = make([]uint64, totalPoints)
	b.directionSigReady = make([]atomic.Uint32, totalPoints)
	b.directionSigBucketed = make([]atomic.Uint32, totalPoints)
	b.directionSigBucketMu.Lock()
	b.directionSigBuckets = make(map[uint64][]uint32)
	b.directionSigBucketMu.Unlock()
}

func encodeDirectionSignature(vec []float32, dim int, planes []float32, sigBits int) (uint64, uint64) {
	var lo, hi uint64
	for bit := 0; bit < sigBits; bit++ {
		plane := planes[bit*dim : (bit+1)*dim]
		var dot float32
		for d := 0; d < dim; d++ {
			dot += vec[d] * plane[d]
		}
		if dot >= 0 {
			if bit < 64 {
				lo |= uint64(1) << uint(bit)
			} else {
				hi |= uint64(1) << uint(bit-64)
			}
		}
	}
	return lo, hi
}

func directionSignatureBandCount(sigBits int) int {
	if sigBits <= 0 {
		return 0
	}
	return (sigBits + 15) / 16
}

func directionSignatureBandValue(lo, hi uint64, band int) uint16 {
	shift := uint(band * 16)
	if shift < 64 {
		return uint16((lo >> shift) & 0xffff)
	}
	return uint16((hi >> (shift - 64)) & 0xffff)
}

func directionSignatureBucketKey(band int, value uint16) uint64 {
	return (uint64(uint32(band)) << 32) | uint64(value)
}

func directionBandMasks(maxDist int) []uint16 {
	if maxDist < 0 {
		maxDist = 0
	}
	if maxDist > 16 {
		maxDist = 16
	}
	masks := make([]uint16, 0, 1)
	var rec func(start int, left int, mask uint16)
	rec = func(start int, left int, mask uint16) {
		if left == 0 {
			masks = append(masks, mask)
			return
		}
		for bit := start; bit <= 16-left; bit++ {
			rec(bit+1, left-1, mask^(uint16(1)<<uint(bit)))
		}
	}
	for dist := 0; dist <= maxDist; dist++ {
		rec(0, dist, 0)
	}
	return masks
}

func idInRanges(id int, ranges []insertRange) bool {
	for _, r := range ranges {
		if uint64(id) >= r.start && uint64(id) < r.end {
			return true
		}
	}
	return false
}

func (b *Bench) addDirectionSignatureBuckets(pointID int, lo uint64, hi uint64, sigBits int) {
	if pointID < 0 || pointID >= len(b.directionSigBucketed) {
		return
	}
	if !b.directionSigBucketed[pointID].CompareAndSwap(0, 1) {
		return
	}
	bands := directionSignatureBandCount(sigBits)
	b.directionSigBucketMu.Lock()
	if b.directionSigBuckets == nil {
		b.directionSigBuckets = make(map[uint64][]uint32)
	}
	for band := 0; band < bands; band++ {
		value := directionSignatureBandValue(lo, hi, band)
		key := directionSignatureBucketKey(band, value)
		b.directionSigBuckets[key] = append(b.directionSigBuckets[key], uint32(pointID))
	}
	b.directionSigBucketMu.Unlock()
}

func (b *Bench) pointDirectionSignature(pointID int, sigBits int) (uint64, uint64, bool) {
	if pointID < 0 || b.dataDim <= 0 {
		return 0, 0, false
	}
	b.ensureDirectionSignatures(sigBits)
	b.directionSigMu.Lock()
	if pointID >= len(b.directionSigReady) {
		b.directionSigMu.Unlock()
		return 0, 0, false
	}
	if b.directionSigReady[pointID].Load() != 0 {
		lo, hi := b.directionSigLo[pointID], b.directionSigHi[pointID]
		b.directionSigMu.Unlock()
		b.addDirectionSignatureBuckets(pointID, lo, hi, sigBits)
		return lo, hi, true
	}
	planes := b.directionSigPlanes
	dim := b.dataDim
	b.directionSigMu.Unlock()

	offset := pointID * dim
	if offset < 0 || offset+dim > len(b.data) {
		return 0, 0, false
	}
	lo, hi := encodeDirectionSignature(b.data[offset:offset+dim], dim, planes, sigBits)

	b.directionSigMu.Lock()
	if pointID < len(b.directionSigReady) && b.directionSigReady[pointID].Load() == 0 {
		b.directionSigLo[pointID] = lo
		b.directionSigHi[pointID] = hi
		b.directionSigReady[pointID].Store(1)
	}
	b.directionSigMu.Unlock()
	b.addDirectionSignatureBuckets(pointID, lo, hi, sigBits)
	return lo, hi, true
}

func (b *Bench) materializeDirectionSignatures(tags []uint32, sigBits int) {
	if sigBits <= 0 {
		return
	}
	for _, tag := range tags {
		b.pointDirectionSignature(int(tag), sigBits)
	}
}

func (b *Bench) queryDirectionSignature(query []float32, sigBits int) (uint64, uint64, bool) {
	if len(query) < b.dataDim || b.dataDim <= 0 {
		return 0, 0, false
	}
	b.ensureDirectionSignatures(sigBits)
	b.directionSigMu.Lock()
	planes := b.directionSigPlanes
	dim := b.dataDim
	b.directionSigMu.Unlock()
	lo, hi := encodeDirectionSignature(query[:dim], dim, planes, sigBits)
	return lo, hi, true
}

func worstScoredTag(tags []scoredTag) int {
	worst := 0
	for i := 1; i < len(tags); i++ {
		if tags[i].dist > tags[worst].dist {
			worst = i
		}
	}
	return worst
}

func sortScoredTags(tags []scoredTag) {
	for i := 1; i < len(tags); i++ {
		cur := tags[i]
		j := i - 1
		for j >= 0 && tags[j].dist > cur.dist {
			tags[j+1] = tags[j]
			j--
		}
		tags[j+1] = cur
	}
}

func (b *Bench) mergeInflightBruteforce(results [][]uint32, queries [][]float32, k uint32, ranges []insertRange) occMergeStats {
	stats := occMergeStats{}
	if len(results) == 0 || len(queries) == 0 || k == 0 || b.dataDim <= 0 {
		return stats
	}
	totalPoints := len(b.data) / b.dataDim
	if totalPoints <= 0 {
		return stats
	}
	if len(ranges) == 0 {
		return stats
	}

	kk := int(k)
	diffPts := 0
	clamped := make([]insertRange, 0, len(ranges))
	for _, r := range ranges {
		if r.start > uint64(totalPoints) {
			r.start = uint64(totalPoints)
		}
		if r.end > uint64(totalPoints) {
			r.end = uint64(totalPoints)
		}
		if r.end <= r.start {
			continue
		}
		diffPts += int(r.end - r.start)
		clamped = append(clamped, r)
	}
	if len(clamped) == 0 {
		return stats
	}

	for qi := range queries {
		if qi >= len(results) || len(queries[qi]) < b.dataDim {
			continue
		}
		var topBuf [128]scoredTag
		var top []scoredTag
		if kk <= len(topBuf) {
			top = topBuf[:0]
		} else {
			top = make([]scoredTag, 0, kk)
		}
		worstIdx := -1
		for _, tag := range results[qi] {
			id := int(tag)
			if id < 0 || id >= totalPoints {
				continue
			}
			stats.seedDists++
			top = append(top, scoredTag{
				tag:  tag,
				dist: b.l2Squared(queries[qi], id),
			})
			if len(top) >= kk {
				break
			}
		}
		if len(top) == kk {
			worstIdx = worstScoredTag(top)
		}

		for _, r := range clamped {
			for id := int(r.start); id < int(r.end); id++ {
				stats.diffDists++
				dist := b.l2Squared(queries[qi], id)
				if len(top) < kk {
					top = append(top, scoredTag{tag: uint32(id), dist: dist})
					if len(top) == kk {
						worstIdx = worstScoredTag(top)
					}
					continue
				}
				stats.mergeComps += len(top)
				if worstIdx < 0 {
					worstIdx = worstScoredTag(top)
				}
				if dist < top[worstIdx].dist {
					top[worstIdx] = scoredTag{tag: uint32(id), dist: dist}
					worstIdx = worstScoredTag(top)
				}
			}
		}

		sortScoredTags(top)
		for j := 0; j < len(results[qi]) && j < len(top); j++ {
			results[qi][j] = top[j].tag
		}
	}

	stats.diffPts = diffPts
	stats.totalL2Dists = stats.seedDists + stats.diffDists
	return stats
}

func (b *Bench) mergeInflightBruteforceLabels(results [][]uint32, queries [][]float32, k uint32, labels []uint32) occMergeStats {
	stats := occMergeStats{}
	if len(results) == 0 || len(queries) == 0 || k == 0 || b.dataDim <= 0 || len(labels) == 0 {
		return stats
	}
	totalPoints := len(b.data) / b.dataDim
	if totalPoints <= 0 {
		return stats
	}
	ids := make([]int, 0, len(labels))
	seen := make(map[uint32]struct{}, len(labels))
	for _, tag := range labels {
		id := int(tag)
		if id < 0 || id >= totalPoints {
			continue
		}
		if _, ok := seen[tag]; ok {
			continue
		}
		seen[tag] = struct{}{}
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return stats
	}

	kk := int(k)
	for qi := range queries {
		if qi >= len(results) || len(queries[qi]) < b.dataDim {
			continue
		}
		var topBuf [128]scoredTag
		var top []scoredTag
		if kk <= len(topBuf) {
			top = topBuf[:0]
		} else {
			top = make([]scoredTag, 0, kk)
		}
		worstIdx := -1
		for _, tag := range results[qi] {
			id := int(tag)
			if id < 0 || id >= totalPoints {
				continue
			}
			stats.seedDists++
			top = append(top, scoredTag{
				tag:  tag,
				dist: b.l2Squared(queries[qi], id),
			})
			if len(top) >= kk {
				break
			}
		}
		if len(top) == kk {
			worstIdx = worstScoredTag(top)
		}

		for _, id := range ids {
			stats.diffDists++
			dist := b.l2Squared(queries[qi], id)
			if len(top) < kk {
				top = append(top, scoredTag{tag: uint32(id), dist: dist})
				if len(top) == kk {
					worstIdx = worstScoredTag(top)
				}
				continue
			}
			stats.mergeComps += len(top)
			if worstIdx < 0 {
				worstIdx = worstScoredTag(top)
			}
			if dist < top[worstIdx].dist {
				top[worstIdx] = scoredTag{tag: uint32(id), dist: dist}
				worstIdx = worstScoredTag(top)
			}
		}

		sortScoredTags(top)
		for j := 0; j < len(results[qi]) && j < len(top); j++ {
			results[qi][j] = top[j].tag
		}
	}

	stats.diffPts = len(ids)
	stats.totalL2Dists = stats.seedDists + stats.diffDists
	return stats
}

func (b *Bench) mergeInflightM2V(results [][]uint32, queries [][]float32, k uint32, ranges []insertRange, work []SearchWorkStats) occMergeStats {
	stats := occMergeStats{}
	if len(results) == 0 || len(queries) == 0 || k == 0 || b.dataDim <= 0 {
		return stats
	}
	totalPoints := len(b.data) / b.dataDim
	if totalPoints <= 0 || len(ranges) == 0 {
		return stats
	}

	kk := int(k)
	diffPts := 0
	committedLimit := totalPoints
	clamped := make([]insertRange, 0, len(ranges))
	for _, r := range ranges {
		if r.start > uint64(totalPoints) {
			r.start = uint64(totalPoints)
		}
		if r.end > uint64(totalPoints) {
			r.end = uint64(totalPoints)
		}
		if r.end <= r.start {
			continue
		}
		if int(r.start) < committedLimit {
			committedLimit = int(r.start)
		}
		diffPts += int(r.end - r.start)
		clamped = append(clamped, r)
	}
	if len(clamped) == 0 {
		return stats
	}

	pendingRanges := make([]insertRange, 0, len(clamped))
	publishedRanges := make([]insertRange, 0, len(clamped))
	b.activeInsertMu.Lock()
	for _, r := range clamped {
		if _, active := b.activeInsertRanges[r.end]; active {
			pendingRanges = append(pendingRanges, r)
		} else {
			publishedRanges = append(publishedRanges, r)
		}
	}
	b.activeInsertMu.Unlock()

	smallThreshold := 200
	if b.config != nil && b.config.Search.InflightJoinSmallFreshThreshold != nil {
		smallThreshold = *b.config.Search.InflightJoinSmallFreshThreshold
	}
	if diffPts <= smallThreshold || len(work) == 0 || len(publishedRanges) == 0 || committedLimit <= 0 {
		return b.mergeInflightBruteforce(results, queries, k, clamped)
	}

	maxNeighbors, ok := b.inflightBaseNeighborLimit()
	if !ok {
		return b.mergeInflightBruteforce(results, queries, k, clamped)
	}
	entryLimit := b.inflightM2VEntryNeighborLimit(maxNeighbors)
	regionCapacity := b.inflightM2VRegionCapacity(smallThreshold)
	routeLabelCount := b.inflightM2VRouteLabelCount()
	memberBoundEnabled := m2vEnvBoolDefault("ANNCHOR_M2V_MEMBER_BOUND", true)
	hubGateEnabled := m2vEnvBool("ANNCHOR_M2V_HUB_GATE")
	sourceTierEnabled := m2vEnvBoolDefault("ANNCHOR_M2V_SOURCE_TIER", true) || hubGateEnabled
	weakSupportGateEnabled := m2vEnvBool("ANNCHOR_M2V_WEAK_SUPPORT_GATE")
	wideGateEnabled := m2vEnvBool("ANNCHOR_M2V_WIDE_GATE")
	shortEdgeRouteOnly := m2vEnvBoolDefault("ANNCHOR_M2V_ROUTE_SHORT_EDGES", true)

	validateFn := false
	if raw := strings.TrimSpace(os.Getenv("ANNCHOR_M2_VALIDATE_FN")); raw != "" {
		validateFn = raw == "1" || strings.EqualFold(raw, "true") || strings.EqualFold(raw, "yes")
	}
	validateStride := uint64(1)
	if raw := strings.TrimSpace(os.Getenv("ANNCHOR_M2_VALIDATE_FN_STRIDE")); raw != "" {
		if parsed, err := strconv.ParseUint(raw, 10, 64); err == nil && parsed > 0 {
			validateStride = parsed
		}
	}

	allFreshIDs := make([]int, 0, diffPts)
	freshIndex := make(map[uint32]int, diffPts)
	for _, r := range clamped {
		for id := int(r.start); id < int(r.end); id++ {
			freshIndex[uint32(id)] = len(allFreshIDs)
			allFreshIDs = append(allFreshIDs, id)
		}
	}
	pendingIDs := make([]int, 0)
	pendingFreshSet := make(map[uint32]struct{}, diffPts)
	for _, r := range pendingRanges {
		for id := int(r.start); id < int(r.end); id++ {
			pendingIDs = append(pendingIDs, id)
			pendingFreshSet[uint32(id)] = struct{}{}
		}
	}
	publishedFreshLabels := make([]uint32, 0, diffPts-len(pendingIDs))
	publishedFreshIDs := make([]int, 0, diffPts-len(pendingIDs))
	for _, r := range publishedRanges {
		for id := int(r.start); id < int(r.end); id++ {
			publishedFreshLabels = append(publishedFreshLabels, uint32(id))
			publishedFreshIDs = append(publishedFreshIDs, id)
		}
	}
	publishedFreshNeighbors, err := b.cachedBaseNeighbors(publishedFreshLabels, maxNeighbors)
	if err != nil {
		log.Printf("M2-V could not export fresh graph neighbors: %v; falling back to brute OCC", err)
		return b.mergeInflightBruteforce(results, queries, k, clamped)
	}

	addIntUnique := func(values *[]int, seen map[int]struct{}, value int) {
		if value < 0 {
			return
		}
		if _, ok := seen[value]; ok {
			return
		}
		seen[value] = struct{}{}
		*values = append(*values, value)
	}
	edgePercentile := func(samples []float32, q float64) float32 {
		if len(samples) == 0 {
			return float32(math.Inf(1))
		}
		sorted := append([]float32(nil), samples...)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
		idx := int(math.Ceil(q*float64(len(sorted)))) - 1
		if idx < 0 {
			idx = 0
		}
		if idx >= len(sorted) {
			idx = len(sorted) - 1
		}
		return sorted[idx]
	}
	edgeScale := func(samples []float32) float32 {
		return edgePercentile(samples, 0.99)
	}

	type freshRegionInfo struct {
		id            int
		neighbors     []uint32
		neighborDists []float32
		scale         float32
		routeLimit    float32
	}
	freshInfos := make([]freshRegionInfo, 0, len(publishedFreshIDs))
	allEdgeSamples := make([]float32, 0, len(publishedFreshIDs)*entryLimit)
	for i, id := range publishedFreshIDs {
		neighbors := make([]uint32, 0, entryLimit)
		edgeSamples := make([]float32, 0, entryLimit)
		if i < len(publishedFreshNeighbors) {
			nbs := publishedFreshNeighbors[i]
			if len(nbs) > entryLimit {
				nbs = nbs[:entryLimit]
			}
			for _, nb := range nbs {
				nbID := int(nb)
				if nbID < 0 || nbID >= totalPoints || nbID == id {
					continue
				}
				neighbors = append(neighbors, nb)
				stats.filterL2Dists++
				edgeDist, _ := b.l2SquaredBetweenBounded(id, nbID, float32(math.Inf(1)))
				edgeSamples = append(edgeSamples, edgeDist)
				allEdgeSamples = append(allEdgeSamples, edgeDist)
			}
		}
		freshInfos = append(freshInfos, freshRegionInfo{
			id:            id,
			neighbors:     neighbors,
			neighborDists: edgeSamples,
			scale:         edgeScale(edgeSamples),
			routeLimit:    edgePercentile(edgeSamples, 0.50),
		})
	}
	globalScale := edgeScale(allEdgeSamples)

	type dynamicRegion struct {
		anchorID    int
		posting     []int
		anchorDist2 []float32
		radius      float32
		scale       float32
		wide        bool
	}
	regionEstimate := 1
	if regionCapacity > 0 {
		regionEstimate += len(publishedFreshIDs) / regionCapacity
	}
	dynamicRegions := make([]dynamicRegion, 0, regionEstimate)
	freshRegions := make(map[int][]int, len(publishedFreshIDs))
	nodeRegion := make(map[int]int, len(publishedFreshIDs))

	allowRegionCount := func(assignments int) int {
		if assignments <= 0 {
			return 1
		}
		target := (assignments + regionCapacity - 1) / regionCapacity
		allow := target + int(math.Ceil(2*math.Sqrt(float64(target+1))))
		if allow < target {
			return target
		}
		return allow
	}
	createRegion := func(id int, scale float32) int {
		if math.IsInf(float64(scale), 1) {
			scale = globalScale
		}
		regionID := len(dynamicRegions)
		dynamicRegions = append(dynamicRegions, dynamicRegion{
			anchorID:    id,
			posting:     []int{id},
			anchorDist2: []float32{0},
			scale:       scale,
		})
		return regionID
	}
	attachRegion := func(regionID int, id int, dist float32, scale float32, wide bool) {
		region := &dynamicRegions[regionID]
		region.posting = append(region.posting, id)
		region.anchorDist2 = append(region.anchorDist2, dist)
		if dist > region.radius {
			region.radius = dist
		}
		if !math.IsInf(float64(scale), 1) && (math.IsInf(float64(region.scale), 1) || scale > region.scale) {
			region.scale = scale
		}
		if wide {
			region.wide = true
		}
	}
	assignments := 0
	insertIntoRegion := func(info freshRegionInfo) int {
		assignments++
		scale := info.scale
		if math.IsInf(float64(scale), 1) {
			scale = globalScale
		}
		if len(dynamicRegions) == 0 {
			return createRegion(info.id, scale)
		}
		seenCandidates := make(map[int]struct{}, len(info.neighbors))
		candidates := make([]int, 0, len(info.neighbors))
		for _, nb := range info.neighbors {
			if regionID, ok := nodeRegion[int(nb)]; ok {
				addIntUnique(&candidates, seenCandidates, regionID)
			}
		}
		if len(candidates) == 0 {
			candidates = make([]int, 0, len(dynamicRegions))
			for regionID := range dynamicRegions {
				candidates = append(candidates, regionID)
			}
		}
		bestRegion := -1
		bestDist := float32(math.Inf(1))
		for _, regionID := range candidates {
			if regionID < 0 || regionID >= len(dynamicRegions) {
				continue
			}
			region := &dynamicRegions[regionID]
			if len(region.posting) >= regionCapacity {
				continue
			}
			stats.filterL2Dists++
			dist, _ := b.l2SquaredBetweenBounded(info.id, region.anchorID, bestDist)
			if dist < bestDist {
				bestDist = dist
				bestRegion = regionID
			}
		}
		if bestRegion < 0 {
			return createRegion(info.id, scale)
		}
		region := &dynamicRegions[bestRegion]
		limit := region.scale
		if math.IsInf(float64(limit), 1) || (!math.IsInf(float64(scale), 1) && scale > limit) {
			limit = scale
		}
		newRadius := region.radius
		if bestDist > newRadius {
			newRadius = bestDist
		}
		if newRadius <= limit {
			attachRegion(bestRegion, info.id, bestDist, scale, false)
			return bestRegion
		}
		if len(dynamicRegions) < allowRegionCount(assignments) {
			return createRegion(info.id, scale)
		}
		attachRegion(bestRegion, info.id, bestDist, scale, true)
		return bestRegion
	}
	for _, info := range freshInfos {
		regionID := insertIntoRegion(info)
		if regionID >= 0 {
			freshRegions[info.id] = []int{regionID}
			nodeRegion[info.id] = regionID
		}
	}

	routeLabelSpace := b.inflightM2VRouteLabelSpace(allowRegionCount(assignments))
	routeNodeSet := make(map[uint32]struct{}, len(freshInfos)*(entryLimit+1))
	routeNodeLabels := make([]uint32, 0, len(freshInfos)*(entryLimit+1))
	addRouteNode := func(label uint32) {
		id := int(label)
		if id < 0 || id >= totalPoints {
			return
		}
		if _, ok := routeNodeSet[label]; ok {
			return
		}
		routeNodeSet[label] = struct{}{}
		routeNodeLabels = append(routeNodeLabels, label)
	}
	for _, info := range freshInfos {
		addRouteNode(uint32(info.id))
		for nbIdx, nb := range info.neighbors {
			if shortEdgeRouteOnly && nbIdx < len(info.neighborDists) && info.neighborDists[nbIdx] > info.routeLimit {
				continue
			}
			addRouteNode(nb)
		}
	}

	type routeNode struct {
		label  uint32
		source uint8
	}
	type preparedQuery struct {
		top        []scoredTag
		worstIdx   int
		radius     float32
		routeNodes []routeNode
	}
	prepared := make([]preparedQuery, len(queries))
	for qi := 0; qi < len(queries); qi++ {
		if qi >= len(results) || len(queries[qi]) < b.dataDim {
			continue
		}
		top := make([]scoredTag, 0, kk)
		for _, tag := range results[qi] {
			id := int(tag)
			if id < 0 || id >= totalPoints {
				continue
			}
			stats.seedDists++
			top = append(top, scoredTag{tag: tag, dist: b.l2Squared(queries[qi], id)})
			if len(top) >= kk {
				break
			}
		}
		if len(top) != kk {
			continue
		}
		worstIdx := worstScoredTag(top)
		radius := top[worstIdx].dist
		queryRouteIndex := make(map[uint32]int, kk+8)
		queryRouteNodes := make([]routeNode, 0, kk+8)
		addQueryRouteNode := func(label uint32, source uint8) {
			id := int(label)
			if id < 0 || id >= totalPoints {
				return
			}
			if idx, ok := queryRouteIndex[label]; ok {
				queryRouteNodes[idx].source |= source
				return
			}
			queryRouteIndex[label] = len(queryRouteNodes)
			queryRouteNodes = append(queryRouteNodes, routeNode{label: label, source: source})
			addRouteNode(label)
		}
		for _, entry := range top {
			addQueryRouteNode(entry.tag, m2vRouteSourceStrong)
		}
		if qi < len(work) {
			pathCount := int(work[qi].PathCount)
			if pathCount > len(work[qi].PathLabels) {
				pathCount = len(work[qi].PathLabels)
			}
			bestPathLabel := uint32(0)
			bestPathDist := float32(math.Inf(1))
			bestPathOK := false
			bestSoFar := float32(math.Inf(1))
			for p := 0; p < pathCount; p++ {
				pathID := int(work[qi].PathLabels[p])
				pathDist := work[qi].PathDists[p]
				if pathID < 0 || pathID >= totalPoints || !(pathDist >= 0) {
					continue
				}
				if pathDist < bestPathDist {
					bestPathDist = pathDist
					bestPathLabel = work[qi].PathLabels[p]
					bestPathOK = true
				}
				if sourceTierEnabled && pathDist < bestSoFar {
					addQueryRouteNode(work[qi].PathLabels[p], m2vRouteSourceMedium)
					bestSoFar = pathDist
				}
				if pathDist <= radius {
					addQueryRouteNode(work[qi].PathLabels[p], m2vRouteSourceWeak)
				}
			}
			if bestPathOK {
				addQueryRouteNode(bestPathLabel, m2vRouteSourceStrong)
			}
		}
		prepared[qi] = preparedQuery{
			top:        top,
			worstIdx:   worstIdx,
			radius:     radius,
			routeNodes: queryRouteNodes,
		}
	}

	routeNeighbors, err := b.cachedBaseNeighbors(routeNodeLabels, maxNeighbors)
	if err != nil {
		log.Printf("M2-V could not export route graph neighbors: %v; falling back to brute OCC", err)
		return b.mergeInflightBruteforce(results, queries, k, clamped)
	}
	routeLabelsByNode := make(map[uint32][]uint32, len(routeNodeLabels))
	for i, label := range routeNodeLabels {
		var neighbors []uint32
		if i < len(routeNeighbors) {
			neighbors = routeNeighbors[i]
		}
		routeLabelsByNode[label] = m2vDiscoveryLabels(label, neighbors, routeLabelSpace, routeLabelCount)
	}
	routeRegionByLabel := make(map[uint32][]int, len(routeNodeLabels))
	routePairSeen := make(map[uint64]struct{}, len(routeNodeLabels))
	addRouteRegion := func(label uint32, regionID int) {
		if regionID < 0 || regionID >= len(dynamicRegions) {
			return
		}
		key := (uint64(label) << 32) | uint64(uint32(regionID))
		if _, ok := routePairSeen[key]; ok {
			return
		}
		routePairSeen[key] = struct{}{}
		routeRegionByLabel[label] = append(routeRegionByLabel[label], regionID)
	}
	for _, info := range freshInfos {
		regionIDs := freshRegions[info.id]
		if len(regionIDs) == 0 {
			continue
		}
		registerNode := func(node uint32) {
			for _, routeLabel := range routeLabelsByNode[node] {
				for _, regionID := range regionIDs {
					addRouteRegion(routeLabel, regionID)
				}
			}
		}
		registerNode(uint32(info.id))
		for nbIdx, nb := range info.neighbors {
			if shortEdgeRouteOnly && nbIdx < len(info.neighborDists) && info.neighborDists[nbIdx] > info.routeLimit {
				continue
			}
			registerNode(nb)
		}
	}
	if len(routeRegionByLabel) == 0 {
		return b.mergeInflightBruteforce(results, queries, k, clamped)
	}
	hubRouteLabel := make(map[uint32]struct{})
	if hubGateEnabled {
		totalBuckets := float64(int(routeLabelSpace) * routeLabelCount)
		if totalBuckets < 1 {
			totalBuckets = 1
		}
		lambda := float64(len(routePairSeen)) / totalBuckets
		hubLimit := lambda + 2*math.Sqrt(lambda+1)
		for label, regions := range routeRegionByLabel {
			if float64(len(regions)) > hubLimit {
				hubRouteLabel[label] = struct{}{}
			}
		}
	}

	addCandidate := func(id int, marks []uint32, gen uint32, candidateIDs *[]int) {
		idx, ok := freshIndex[uint32(id)]
		if !ok {
			return
		}
		if marks[idx] == gen {
			return
		}
		marks[idx] = gen
		*candidateIDs = append(*candidateIDs, id)
	}
	resetMarks := func(marks []uint32, gen uint32) uint32 {
		gen++
		if gen == 0 {
			for i := range marks {
				marks[i] = 0
			}
			gen = 1
		}
		return gen
	}

	processRange := func(start, end int) occMergeStats {
		local := occMergeStats{}
		marks := make([]uint32, len(allFreshIDs))
		regionHitMarks := make([]uint32, len(dynamicRegions))
		regionHitSupport := make([]uint8, len(dynamicRegions))
		regionHitHubSupport := make([]uint8, len(dynamicRegions))
		regionHitNonHubSupport := make([]uint8, len(dynamicRegions))
		regionHitSources := make([]uint8, len(dynamicRegions))
		var gen uint32
		var regionHitGen uint32
		candidateIDs := make([]int, 0, len(pendingIDs)+32)
		candidateRegions := make([]int, 0, 32)
		hitRegions := make([]int, 0, 32)
		type openedRegion struct {
			id         int
			lb         float32
			anchorNorm float64
		}
		opened := make([]openedRegion, 0, 32)
		for qi := start; qi < end && qi < len(queries); qi++ {
			pq := prepared[qi]
			if len(pq.top) != kk || len(queries[qi]) < b.dataDim {
				continue
			}
			local.m2vSeenQueries++
			top := make([]scoredTag, len(pq.top))
			copy(top, pq.top)
			worstIdx := pq.worstIdx

			gen = resetMarks(marks, gen)
			candidateIDs = candidateIDs[:0]
			for _, id := range pendingIDs {
				addCandidate(id, marks, gen, &candidateIDs)
			}

			regionHitGen = resetMarks(regionHitMarks, regionHitGen)
			candidateRegions = candidateRegions[:0]
			hitRegions = hitRegions[:0]
			for _, routeNode := range pq.routeNodes {
				for _, routeLabel := range routeLabelsByNode[routeNode.label] {
					_, isHub := hubRouteLabel[routeLabel]
					for _, regionID := range routeRegionByLabel[routeLabel] {
						if regionID < 0 || regionID >= len(dynamicRegions) {
							continue
						}
						if regionHitMarks[regionID] != regionHitGen {
							regionHitMarks[regionID] = regionHitGen
							regionHitSupport[regionID] = 0
							regionHitHubSupport[regionID] = 0
							regionHitNonHubSupport[regionID] = 0
							regionHitSources[regionID] = 0
							hitRegions = append(hitRegions, regionID)
						}
						if regionHitSupport[regionID] < 255 {
							regionHitSupport[regionID]++
						}
						if isHub {
							if regionHitHubSupport[regionID] < 255 {
								regionHitHubSupport[regionID]++
							}
						} else if regionHitNonHubSupport[regionID] < 255 {
							regionHitNonHubSupport[regionID]++
						}
						regionHitSources[regionID] |= routeNode.source
					}
				}
			}
			for _, regionID := range hitRegions {
				region := &dynamicRegions[regionID]
				sources := regionHitSources[regionID]
				support := int(regionHitSupport[regionID])
				allow := true
				if hubGateEnabled && regionHitNonHubSupport[regionID] == 0 &&
					regionHitHubSupport[regionID] > 0 &&
					(sources&m2vRouteSourceStrong) == 0 &&
					support < 2 {
					local.m2vHubSuppressedRegions++
					allow = false
				}
				if allow && weakSupportGateEnabled &&
					sources == m2vRouteSourceWeak &&
					support < 2 {
					local.m2vWeakSuppressedRegions++
					allow = false
				}
				if allow && wideGateEnabled && region.wide &&
					(sources&m2vRouteSourceStrong) == 0 &&
					support < 2 {
					local.m2vWideSuppressedRegions++
					allow = false
				}
				if !allow {
					continue
				}
				if sources&m2vRouteSourceStrong != 0 {
					local.m2vStrongRegionHits++
				}
				if sources&m2vRouteSourceMedium != 0 {
					local.m2vMediumRegionHits++
				}
				if sources&m2vRouteSourceWeak != 0 {
					local.m2vWeakRegionHits++
				}
				local.m2vCandidateRegions++
				candidateRegions = append(candidateRegions, regionID)
			}
			if len(candidateRegions) == 0 {
				local.m2vNoRouteQueries++
			}
			opened = opened[:0]
			for _, regionID := range candidateRegions {
				region := &dynamicRegions[regionID]
				local.filterL2Dists++
				anchorDist := b.l2Squared(queries[qi], region.anchorID)
				anchorNorm := math.Sqrt(float64(anchorDist))
				regionNorm := math.Sqrt(float64(region.radius))
				lbNorm := anchorNorm - regionNorm
				lb := float32(0)
				if lbNorm > 0 {
					lb = float32(lbNorm * lbNorm)
				}
				if lb > pq.radius {
					continue
				}
				opened = append(opened, openedRegion{id: regionID, lb: lb, anchorNorm: anchorNorm})
			}
			openedSeen := make(map[int]struct{}, len(opened))
			for _, item := range opened {
				region := &dynamicRegions[item.id]
				openedSeen[item.id] = struct{}{}
				local.m2vOpenedRegions++
				local.m2vOpenedPostings += len(region.posting)
				for pos, id := range region.posting {
					local.filterTests++
					if memberBoundEnabled && pos < len(region.anchorDist2) {
						local.m2vMemberBoundChecks++
						memberNorm := math.Sqrt(float64(region.anchorDist2[pos]))
						lbNorm := item.anchorNorm - memberNorm
						if lbNorm > 0 && float32(lbNorm*lbNorm) > pq.radius {
							local.m2vMemberBoundSkips++
							continue
						}
					}
					addCandidate(id, marks, gen, &candidateIDs)
				}
			}

			if validateFn {
				seq := b.m2VValidateSeen.Add(1)
				if validateStride <= 1 || seq%validateStride == 0 {
					b.m2VValidateRuns.Add(1)
					b.m2VValidateFresh.Add(uint64(len(allFreshIDs)))
					bestFresh := -1
					bestFreshDist := top[worstIdx].dist
					for _, id := range allFreshIDs {
						dist := b.l2Squared(queries[qi], id)
						if dist < bestFreshDist {
							bestFreshDist = dist
							bestFresh = id
						}
					}
					if bestFresh >= 0 {
						b.m2VValidateWins.Add(1)
						if _, ok := pendingFreshSet[uint32(bestFresh)]; ok {
							b.m2VValidatePendingWinner.Add(1)
						}
						regionRegistered := false
						regionOpened := false
						for _, regionID := range freshRegions[bestFresh] {
							regionRegistered = true
							if _, ok := openedSeen[regionID]; ok {
								regionOpened = true
							}
						}
						if regionRegistered {
							b.m2VValidateActiveRegionWinner.Add(1)
						}
						if regionOpened {
							b.m2VValidateOpenedRegionWinner.Add(1)
						}
						if regionRegistered && !regionOpened {
							b.m2VValidateTrianglePrunedWinner.Add(1)
						}
						if idx, ok := freshIndex[uint32(bestFresh)]; !ok || marks[idx] != gen {
							b.m2VValidateMiss.Add(1)
						}
					}
				}
			}

			local.candidates += len(candidateIDs)
			for _, id := range candidateIDs {
				if worstIdx < 0 {
					worstIdx = worstScoredTag(top)
				}
				local.mergeComps += len(top)
				dist, ok := b.l2SquaredBounded(queries[qi], id, top[worstIdx].dist)
				local.diffDists++
				if !ok {
					continue
				}
				if dist < top[worstIdx].dist {
					top[worstIdx] = scoredTag{tag: uint32(id), dist: dist}
					worstIdx = worstScoredTag(top)
				}
			}

			sortScoredTags(top)
			for j := 0; j < len(results[qi]) && j < len(top); j++ {
				results[qi][j] = top[j].tag
			}
		}
		return local
	}

	workers := 1
	if b.config != nil && b.config.Search.InflightJoinSignatureWorkers != nil {
		workers = *b.config.Search.InflightJoinSignatureWorkers
	}
	if workers < 1 {
		workers = 1
	}
	if workers > len(queries) {
		workers = len(queries)
	}
	if workers <= 1 {
		local := processRange(0, len(queries))
		stats.diffDists += local.diffDists
		stats.mergeComps += local.mergeComps
		stats.filterTests += local.filterTests
		stats.filterL2Dists += local.filterL2Dists
		stats.candidates += local.candidates
		stats.m2vSeenQueries += local.m2vSeenQueries
		stats.m2vNoRouteQueries += local.m2vNoRouteQueries
		stats.m2vCandidateRegions += local.m2vCandidateRegions
		stats.m2vOpenedRegions += local.m2vOpenedRegions
		stats.m2vOpenedPostings += local.m2vOpenedPostings
		stats.m2vMemberBoundChecks += local.m2vMemberBoundChecks
		stats.m2vMemberBoundSkips += local.m2vMemberBoundSkips
		stats.m2vHubSuppressedRegions += local.m2vHubSuppressedRegions
		stats.m2vWideSuppressedRegions += local.m2vWideSuppressedRegions
		stats.m2vWeakSuppressedRegions += local.m2vWeakSuppressedRegions
		stats.m2vStrongRegionHits += local.m2vStrongRegionHits
		stats.m2vMediumRegionHits += local.m2vMediumRegionHits
		stats.m2vWeakRegionHits += local.m2vWeakRegionHits
	} else {
		partials := make([]occMergeStats, workers)
		chunk := (len(queries) + workers - 1) / workers
		var wg sync.WaitGroup
		for worker := 0; worker < workers; worker++ {
			start := worker * chunk
			end := start + chunk
			if start >= len(queries) {
				break
			}
			wg.Add(1)
			go func(worker, start, end int) {
				defer wg.Done()
				partials[worker] = processRange(start, end)
			}(worker, start, end)
		}
		wg.Wait()
		for _, local := range partials {
			stats.diffDists += local.diffDists
			stats.mergeComps += local.mergeComps
			stats.filterTests += local.filterTests
			stats.filterL2Dists += local.filterL2Dists
			stats.candidates += local.candidates
			stats.m2vSeenQueries += local.m2vSeenQueries
			stats.m2vNoRouteQueries += local.m2vNoRouteQueries
			stats.m2vCandidateRegions += local.m2vCandidateRegions
			stats.m2vOpenedRegions += local.m2vOpenedRegions
			stats.m2vOpenedPostings += local.m2vOpenedPostings
			stats.m2vMemberBoundChecks += local.m2vMemberBoundChecks
			stats.m2vMemberBoundSkips += local.m2vMemberBoundSkips
			stats.m2vHubSuppressedRegions += local.m2vHubSuppressedRegions
			stats.m2vWideSuppressedRegions += local.m2vWideSuppressedRegions
			stats.m2vWeakSuppressedRegions += local.m2vWeakSuppressedRegions
			stats.m2vStrongRegionHits += local.m2vStrongRegionHits
			stats.m2vMediumRegionHits += local.m2vMediumRegionHits
			stats.m2vWeakRegionHits += local.m2vWeakRegionHits
		}
	}

	if stats.m2vSeenQueries > 0 {
		b.m2VRouteDiagSeenQueries.Add(uint64(stats.m2vSeenQueries))
		b.m2VRouteDiagNoRouteQueries.Add(uint64(stats.m2vNoRouteQueries))
		b.m2VRouteDiagCandidateRegions.Add(uint64(stats.m2vCandidateRegions))
		b.m2VRouteDiagOpenedRegions.Add(uint64(stats.m2vOpenedRegions))
		b.m2VRouteDiagOpenedPostings.Add(uint64(stats.m2vOpenedPostings))
		b.m2VRouteDiagMemberBoundChecks.Add(uint64(stats.m2vMemberBoundChecks))
		b.m2VRouteDiagMemberBoundSkips.Add(uint64(stats.m2vMemberBoundSkips))
		b.m2VRouteDiagHubSuppressed.Add(uint64(stats.m2vHubSuppressedRegions))
		b.m2VRouteDiagWideSuppressed.Add(uint64(stats.m2vWideSuppressedRegions))
		b.m2VRouteDiagWeakSuppressed.Add(uint64(stats.m2vWeakSuppressedRegions))
		b.m2VRouteDiagStrongRegionHits.Add(uint64(stats.m2vStrongRegionHits))
		b.m2VRouteDiagMediumRegionHits.Add(uint64(stats.m2vMediumRegionHits))
		b.m2VRouteDiagWeakRegionHits.Add(uint64(stats.m2vWeakRegionHits))
	}

	stats.diffPts = diffPts
	stats.totalL2Dists = stats.seedDists + stats.diffDists + stats.filterL2Dists
	return stats
}

func (b *Bench) mergeInflightPathOneHop(results [][]uint32, queries [][]float32, k uint32, ranges []insertRange, work []SearchWorkStats) occMergeStats {
	stats := occMergeStats{}
	if len(results) == 0 || len(queries) == 0 || k == 0 || b.dataDim <= 0 {
		return stats
	}
	totalPoints := len(b.data) / b.dataDim
	if totalPoints <= 0 || len(ranges) == 0 {
		return stats
	}

	kk := int(k)
	diffPts := 0
	clamped := make([]insertRange, 0, len(ranges))
	for _, r := range ranges {
		if r.start > uint64(totalPoints) {
			r.start = uint64(totalPoints)
		}
		if r.end > uint64(totalPoints) {
			r.end = uint64(totalPoints)
		}
		if r.end <= r.start {
			continue
		}
		diffPts += int(r.end - r.start)
		clamped = append(clamped, r)
	}
	if len(clamped) == 0 {
		return stats
	}
	pendingRanges := make([]insertRange, 0, len(clamped))
	publishedRanges := make([]insertRange, 0, len(clamped))
	b.activeInsertMu.Lock()
	for _, r := range clamped {
		if _, active := b.activeInsertRanges[r.end]; active {
			pendingRanges = append(pendingRanges, r)
		} else {
			publishedRanges = append(publishedRanges, r)
		}
	}
	b.activeInsertMu.Unlock()

	smallThreshold := 200
	if b.config != nil && b.config.Search.InflightJoinSmallFreshThreshold != nil {
		smallThreshold = *b.config.Search.InflightJoinSmallFreshThreshold
	}
	if diffPts <= smallThreshold || len(work) == 0 || len(publishedRanges) == 0 {
		return b.mergeInflightBruteforce(results, queries, k, clamped)
	}

	maxNeighbors, ok := b.inflightBaseNeighborLimit()
	if !ok {
		return b.mergeInflightBruteforce(results, queries, k, clamped)
	}
	minBridgeSupport := b.inflightPathOneHopBridgeMinSupport()
	bridgeSelectivitySigma := b.inflightPathOneHopBridgeSelectivitySigma()

	validateFn := false
	if raw := strings.TrimSpace(os.Getenv("ANNCHOR_M2_VALIDATE_FN")); raw != "" {
		validateFn = raw == "1" || strings.EqualFold(raw, "true") || strings.EqualFold(raw, "yes")
	}
	validateStride := uint64(1)
	if raw := strings.TrimSpace(os.Getenv("ANNCHOR_M2_VALIDATE_FN_STRIDE")); raw != "" {
		if parsed, err := strconv.ParseUint(raw, 10, 64); err == nil && parsed > 0 {
			validateStride = parsed
		}
	}

	type preparedQuery struct {
		top            []scoredTag
		worstIdx       int
		witnesses      []uint32
		bridgeAnchors  []uint32
		bridgeSupport  map[uint32]int
		pathEvents     []uint32
		pathEventDists []float32
	}
	prepared := make([]preparedQuery, len(queries))
	labelIndex := make(map[uint32]int, len(queries)*kk)
	uniqueLabels := make([]uint32, 0, len(queries)*kk)
	addGlobalLabel := func(label uint32) {
		if _, ok := labelIndex[label]; ok {
			return
		}
		labelIndex[label] = len(uniqueLabels)
		uniqueLabels = append(uniqueLabels, label)
	}

	for qi := 0; qi < len(queries); qi++ {
		if qi >= len(results) || len(queries[qi]) < b.dataDim {
			continue
		}
		top := make([]scoredTag, 0, kk)
		for _, tag := range results[qi] {
			id := int(tag)
			if id < 0 || id >= totalPoints {
				continue
			}
			stats.seedDists++
			top = append(top, scoredTag{
				tag:  tag,
				dist: b.l2Squared(queries[qi], id),
			})
			if len(top) >= kk {
				break
			}
		}
		if len(top) != kk {
			continue
		}
		worstIdx := worstScoredTag(top)
		radius := top[worstIdx].dist
		seen := make(map[uint32]struct{}, kk+8)
		witnesses := make([]uint32, 0, kk+8)
		addWitness := func(label uint32) {
			if _, ok := seen[label]; ok {
				return
			}
			seen[label] = struct{}{}
			witnesses = append(witnesses, label)
			addGlobalLabel(label)
		}
		pathSeen := make(map[uint32]struct{}, kk+8)
		pathIndex := make(map[uint32]int, kk+8)
		pathEvents := make([]uint32, 0, kk+8)
		pathEventDists := make([]float32, 0, kk+8)
		addPathEvent := func(label uint32, dist float32) {
			if !validateFn {
				return
			}
			if _, ok := pathSeen[label]; ok {
				if idx, exists := pathIndex[label]; exists && dist < pathEventDists[idx] {
					pathEventDists[idx] = dist
				}
				return
			}
			pathSeen[label] = struct{}{}
			pathIndex[label] = len(pathEvents)
			pathEvents = append(pathEvents, label)
			pathEventDists = append(pathEventDists, dist)
			addGlobalLabel(label)
		}
		for _, entry := range top {
			addWitness(entry.tag)
			addPathEvent(entry.tag, entry.dist)
		}
		if qi < len(work) {
			pathCount := int(work[qi].PathCount)
			if pathCount > len(work[qi].PathLabels) {
				pathCount = len(work[qi].PathLabels)
			}
			bestPathDist := float32(math.Inf(1))
			for p := 0; p < pathCount; p++ {
				dist := work[qi].PathDists[p]
				if !(dist >= 0) {
					continue
				}
				label := work[qi].PathLabels[p]
				addPathEvent(label, dist)
				insideRadius := dist <= radius
				if dist < bestPathDist {
					bestPathDist = dist
					addWitness(label)
				}
				if insideRadius {
					addWitness(label)
				}
			}
		}
		prepared[qi] = preparedQuery{
			top:            top,
			worstIdx:       worstIdx,
			witnesses:      witnesses,
			pathEvents:     pathEvents,
			pathEventDists: pathEventDists,
		}
	}
	if len(uniqueLabels) == 0 {
		return b.mergeInflightBruteforce(results, queries, k, clamped)
	}

	neighbors, err := b.cachedBaseNeighbors(uniqueLabels, maxNeighbors)
	if err != nil {
		log.Printf("Path 1-hop inflight join could not export graph neighbors: %v; falling back to brute OCC", err)
		return b.mergeInflightBruteforce(results, queries, k, clamped)
	}
	allFreshIDs := make([]int, 0, diffPts)
	freshIndex := make(map[uint32]int, diffPts)
	for _, r := range clamped {
		for id := int(r.start); id < int(r.end); id++ {
			freshIndex[uint32(id)] = len(allFreshIDs)
			allFreshIDs = append(allFreshIDs, id)
		}
	}
	pendingIDs := make([]int, 0)
	pendingFreshSet := make(map[uint32]struct{}, diffPts)
	for _, r := range pendingRanges {
		for id := int(r.start); id < int(r.end); id++ {
			pendingIDs = append(pendingIDs, id)
			pendingFreshSet[uint32(id)] = struct{}{}
		}
	}
	frontierRescueLimit := b.inflightPathOneHopFrontierRescueLimit(diffPts, maxNeighbors)
	frontierRescueIDs := make([]int, 0, frontierRescueLimit)
	for ri := len(publishedRanges) - 1; ri >= 0 && len(frontierRescueIDs) < frontierRescueLimit; ri-- {
		r := publishedRanges[ri]
		for id := int(r.end) - 1; id >= int(r.start) && len(frontierRescueIDs) < frontierRescueLimit; id-- {
			frontierRescueIDs = append(frontierRescueIDs, id)
		}
	}
	publishedFreshIndex := make(map[uint32]int, diffPts-len(pendingIDs))
	for _, r := range publishedRanges {
		for id := int(r.start); id < int(r.end); id++ {
			publishedFreshIndex[uint32(id)] = id
		}
	}

	neighborByLabel := make(map[uint32][]int, len(uniqueLabels))
	allNeighborByLabel := make(map[uint32][]uint32, len(uniqueLabels))
	neighborScanCount := make(map[uint32]int, len(uniqueLabels))
	for i, label := range uniqueLabels {
		if i >= len(neighbors) {
			continue
		}
		allNeighborByLabel[label] = neighbors[i]
		neighborScanCount[label] = len(neighbors[i])
		for _, nb := range neighbors[i] {
			if id, ok := publishedFreshIndex[nb]; ok {
				neighborByLabel[label] = append(neighborByLabel[label], id)
			}
		}
	}
	queryJoinKeySet := make(map[uint32]struct{}, len(uniqueLabels))
	for qi := range prepared {
		pq := &prepared[qi]
		if len(pq.witnesses) == 0 {
			continue
		}
		for _, witness := range pq.witnesses {
			queryJoinKeySet[witness] = struct{}{}
		}
		if validateFn {
			for _, event := range pq.pathEvents {
				queryJoinKeySet[event] = struct{}{}
			}
		}
		bridgeSupport := make(map[uint32]int)
		for _, witness := range pq.witnesses {
			for _, bridge := range allNeighborByLabel[witness] {
				if bridge == witness {
					continue
				}
				if _, isFresh := freshIndex[bridge]; isFresh {
					continue
				}
				bridgeID := int(bridge)
				if bridgeID < 0 || bridgeID >= totalPoints {
					continue
				}
				bridgeSupport[bridge]++
			}
		}
		for bridge := range bridgeSupport {
			pq.bridgeAnchors = append(pq.bridgeAnchors, bridge)
			queryJoinKeySet[bridge] = struct{}{}
		}
		pq.bridgeSupport = bridgeSupport
	}
	publishedFreshLabels := make([]uint32, 0, len(publishedFreshIndex))
	for label := range publishedFreshIndex {
		publishedFreshLabels = append(publishedFreshLabels, label)
	}
	publishedFreshNeighbors, err := b.cachedBaseNeighbors(publishedFreshLabels, maxNeighbors)
	if err != nil {
		log.Printf("Path reciprocal-bridge join could not export fresh neighbors: %v; falling back to brute OCC", err)
		return b.mergeInflightBruteforce(results, queries, k, clamped)
	}
	freshByJoinKey := make(map[uint32][]int, len(queryJoinKeySet))
	for i, label := range publishedFreshLabels {
		id, ok := publishedFreshIndex[label]
		if !ok || i >= len(publishedFreshNeighbors) {
			continue
		}
		for _, key := range publishedFreshNeighbors[i] {
			if _, needed := queryJoinKeySet[key]; !needed {
				continue
			}
			freshByJoinKey[key] = append(freshByJoinKey[key], id)
		}
	}
	bridgeFreshLambda := float64(diffPts*maxNeighbors) / float64(totalPoints)
	bridgeSelectivityLimit := int(math.Ceil(bridgeFreshLambda + bridgeSelectivitySigma*math.Sqrt(bridgeFreshLambda+1.0)))
	if bridgeSelectivityLimit < 1 {
		bridgeSelectivityLimit = 1
	}
	addCandidate := func(id int, marks []uint32, gen uint32, candidateIDs *[]int) {
		idx, ok := freshIndex[uint32(id)]
		if !ok {
			return
		}
		if marks[idx] == gen {
			return
		}
		marks[idx] = gen
		*candidateIDs = append(*candidateIDs, id)
	}
	resetMarks := func(marks []uint32, gen uint32) uint32 {
		gen++
		if gen == 0 {
			for i := range marks {
				marks[i] = 0
			}
			gen = 1
		}
		return gen
	}
	type twoHopDiagnostic struct {
		hit              bool
		monotoneBridge   bool
		radiusBridge     bool
		multiSource      bool
		reciprocalSource bool
		reciprocalTarget bool
	}
	diagnoseTwoHop := func(query []float32, target uint32, targetDist float32, radius float32, sources []uint32, sourceDists []float32) twoHopDiagnostic {
		diag := twoHopDiagnostic{}
		if len(query) < b.dataDim || len(sources) == 0 || len(sources) != len(sourceDists) {
			return diag
		}
		bridgeSourceDist := make(map[uint32]float32)
		bridgeSources := make(map[uint32][]uint32)
		for i, source := range sources {
			srcDist := sourceDists[i]
			for _, bridge := range allNeighborByLabel[source] {
				if bridge == target {
					continue
				}
				if _, isFresh := freshIndex[bridge]; isFresh {
					continue
				}
				bridgeID := int(bridge)
				if bridgeID < 0 || bridgeID >= totalPoints {
					continue
				}
				if prev, ok := bridgeSourceDist[bridge]; !ok || srcDist < prev {
					bridgeSourceDist[bridge] = srcDist
				}
				bridgeSources[bridge] = append(bridgeSources[bridge], source)
			}
		}
		if len(bridgeSourceDist) == 0 {
			return diag
		}
		bridgeLabels := make([]uint32, 0, len(bridgeSourceDist))
		bridgeSourceDists := make([]float32, 0, len(bridgeSourceDist))
		for bridge, srcDist := range bridgeSourceDist {
			bridgeLabels = append(bridgeLabels, bridge)
			bridgeSourceDists = append(bridgeSourceDists, srcDist)
		}
		bridgeNeighbors, err := b.index.BaseNeighbors(bridgeLabels, maxNeighbors)
		if err != nil {
			return diag
		}
		targetNeighborSet := make(map[uint32]struct{})
		if targetNeighbors, err := b.index.BaseNeighbors([]uint32{target}, maxNeighbors); err == nil && len(targetNeighbors) > 0 {
			for _, nb := range targetNeighbors[0] {
				targetNeighborSet[nb] = struct{}{}
			}
		}
		for i, nbs := range bridgeNeighbors {
			bridge := bridgeLabels[i]
			for _, nb := range nbs {
				if nb != target {
					continue
				}
				diag.hit = true
				bridgeID := int(bridge)
				bridgeDist := b.l2Squared(query, bridgeID)
				if bridgeDist <= bridgeSourceDists[i] && targetDist <= bridgeDist {
					diag.monotoneBridge = true
				}
				if bridgeDist <= radius {
					diag.radiusBridge = true
				}
				if len(bridgeSources[bridge]) >= 2 {
					diag.multiSource = true
				}
				for _, source := range bridgeSources[bridge] {
					for _, bridgeNb := range nbs {
						if bridgeNb == source {
							diag.reciprocalSource = true
							break
						}
					}
					if diag.reciprocalSource {
						break
					}
				}
				if _, ok := targetNeighborSet[bridge]; ok {
					diag.reciprocalTarget = true
				}
			}
		}
		return diag
	}
	processRange := func(start, end int) occMergeStats {
		local := occMergeStats{}
		marks := make([]uint32, len(allFreshIDs))
		var gen uint32
		candidateIDs := make([]int, 0, len(pendingIDs)+len(uniqueLabels))
		for qi := start; qi < end && qi < len(queries); qi++ {
			pq := prepared[qi]
			if len(pq.top) != kk || len(pq.witnesses) == 0 || len(queries[qi]) < b.dataDim {
				continue
			}
			top := make([]scoredTag, len(pq.top))
			copy(top, pq.top)
			worstIdx := pq.worstIdx

			gen = resetMarks(marks, gen)
			candidateIDs = candidateIDs[:0]
			for _, id := range pendingIDs {
				addCandidate(id, marks, gen, &candidateIDs)
			}
			for _, id := range frontierRescueIDs {
				addCandidate(id, marks, gen, &candidateIDs)
			}
			for _, witness := range pq.witnesses {
				local.filterTests += neighborScanCount[witness]
				for _, nb := range neighborByLabel[witness] {
					addCandidate(nb, marks, gen, &candidateIDs)
				}
				local.filterTests += len(freshByJoinKey[witness])
				for _, nb := range freshByJoinKey[witness] {
					addCandidate(nb, marks, gen, &candidateIDs)
				}
			}
			for _, bridge := range pq.bridgeAnchors {
				bridgeFresh := freshByJoinKey[bridge]
				if minBridgeSupport > 1 && pq.bridgeSupport[bridge] < minBridgeSupport && len(bridgeFresh) > bridgeSelectivityLimit {
					continue
				}
				local.filterTests += len(bridgeFresh)
				for _, nb := range bridgeFresh {
					addCandidate(nb, marks, gen, &candidateIDs)
				}
			}

			if validateFn {
				seq := b.m2PathOneHopValidateSeen.Add(1)
				if validateStride <= 1 || seq%validateStride == 0 {
					b.m2PathOneHopValidateRuns.Add(1)
					b.m2PathOneHopValidateFresh.Add(uint64(len(allFreshIDs)))
					bestFresh := -1
					bestFreshDist := top[worstIdx].dist
					for _, id := range allFreshIDs {
						dist := b.l2Squared(queries[qi], id)
						if dist < bestFreshDist {
							bestFreshDist = dist
							bestFresh = id
						}
					}
					if bestFresh >= 0 {
						target := uint32(bestFresh)
						b.m2PathOneHopValidateWins.Add(1)
						if _, ok := pendingFreshSet[target]; ok {
							b.m2PathOneHopValidatePendingWinner.Add(1)
						}
						pathSelf := false
						for _, label := range pq.pathEvents {
							if label == target {
								pathSelf = true
								break
							}
						}
						selectedOneHop := false
						for _, witness := range pq.witnesses {
							for _, nb := range neighborByLabel[witness] {
								if uint32(nb) == target {
									selectedOneHop = true
									break
								}
							}
							if selectedOneHop {
								break
							}
							for _, nb := range freshByJoinKey[witness] {
								if uint32(nb) == target {
									selectedOneHop = true
									break
								}
							}
							if selectedOneHop {
								break
							}
						}
						allPathOneHop := false
						for _, event := range pq.pathEvents {
							for _, nb := range neighborByLabel[event] {
								if uint32(nb) == target {
									allPathOneHop = true
									break
								}
							}
							if allPathOneHop {
								break
							}
							for _, nb := range freshByJoinKey[event] {
								if uint32(nb) == target {
									allPathOneHop = true
									break
								}
							}
							if allPathOneHop {
								break
							}
						}
						if selectedOneHop {
							b.m2PathOneHopValidateSelectedOneHop.Add(1)
						}
						if allPathOneHop {
							b.m2PathOneHopValidateAllPathOneHop.Add(1)
						}
						missed := false
						if idx, ok := freshIndex[target]; !ok || marks[idx] != gen {
							missed = true
							b.m2PathOneHopValidateMiss.Add(1)
						}
						if missed && allPathOneHop {
							b.m2PathOneHopValidateMissAllPathOneHop.Add(1)
						}
						if missed && pathSelf {
							b.m2PathOneHopValidateMissPathEventSelf.Add(1)
						}
						if missed {
							selectedDists := make([]float32, 0, len(pq.witnesses))
							selectedSources := make([]uint32, 0, len(pq.witnesses))
							for _, witness := range pq.witnesses {
								witnessID := int(witness)
								if witnessID < 0 || witnessID >= totalPoints {
									continue
								}
								selectedSources = append(selectedSources, witness)
								selectedDists = append(selectedDists, b.l2Squared(queries[qi], witnessID))
							}
							selectedDiag := diagnoseTwoHop(
								queries[qi],
								target,
								bestFreshDist,
								top[worstIdx].dist,
								selectedSources,
								selectedDists,
							)
							if selectedDiag.hit {
								b.m2PathOneHopValidateMissSelectedTwoHop.Add(1)
							}
							allPathDiag := diagnoseTwoHop(
								queries[qi],
								target,
								bestFreshDist,
								top[worstIdx].dist,
								pq.pathEvents,
								pq.pathEventDists,
							)
							if allPathDiag.hit {
								b.m2PathOneHopValidateMissAllPathTwoHop.Add(1)
							}
							if allPathDiag.monotoneBridge {
								b.m2PathOneHopValidateMissMonotoneTwoHop.Add(1)
							}
							if allPathDiag.radiusBridge {
								b.m2PathOneHopValidateMissRadiusTwoHop.Add(1)
							}
							if allPathDiag.multiSource {
								b.m2PathOneHopValidateMissSupportedTwoHop.Add(1)
							}
							if allPathDiag.reciprocalSource {
								b.m2PathOneHopValidateMissRecipSourceTwoHop.Add(1)
							}
							if allPathDiag.reciprocalTarget {
								b.m2PathOneHopValidateMissRecipTargetTwoHop.Add(1)
							}
						}
					}
				}
			}

			local.candidates += len(candidateIDs)
			for _, id := range candidateIDs {
				if worstIdx < 0 {
					worstIdx = worstScoredTag(top)
				}
				local.mergeComps += len(top)
				dist, ok := b.l2SquaredBounded(queries[qi], id, top[worstIdx].dist)
				local.diffDists++
				if !ok {
					continue
				}
				if dist < top[worstIdx].dist {
					top[worstIdx] = scoredTag{tag: uint32(id), dist: dist}
					worstIdx = worstScoredTag(top)
				}
			}

			sortScoredTags(top)
			for j := 0; j < len(results[qi]) && j < len(top); j++ {
				results[qi][j] = top[j].tag
			}
		}
		return local
	}

	workers := 1
	if b.config != nil && b.config.Search.InflightJoinSignatureWorkers != nil {
		workers = *b.config.Search.InflightJoinSignatureWorkers
	}
	if workers < 1 {
		workers = 1
	}
	if workers > len(queries) {
		workers = len(queries)
	}
	if workers <= 1 {
		local := processRange(0, len(queries))
		stats.diffDists += local.diffDists
		stats.mergeComps += local.mergeComps
		stats.filterTests += local.filterTests
		stats.candidates += local.candidates
	} else {
		partials := make([]occMergeStats, workers)
		chunk := (len(queries) + workers - 1) / workers
		var wg sync.WaitGroup
		for worker := 0; worker < workers; worker++ {
			start := worker * chunk
			end := start + chunk
			if start >= len(queries) {
				break
			}
			wg.Add(1)
			go func(worker, start, end int) {
				defer wg.Done()
				partials[worker] = processRange(start, end)
			}(worker, start, end)
		}
		wg.Wait()
		for _, local := range partials {
			stats.diffDists += local.diffDists
			stats.mergeComps += local.mergeComps
			stats.filterTests += local.filterTests
			stats.candidates += local.candidates
		}
	}

	stats.diffPts = diffPts
	stats.totalL2Dists = stats.seedDists + stats.diffDists
	return stats
}

func (b *Bench) mergeInflightPathCorridor(results [][]uint32, queries [][]float32, k uint32, ranges []insertRange, work []SearchWorkStats) occMergeStats {
	stats := occMergeStats{}
	if len(results) == 0 || len(queries) == 0 || k == 0 || b.dataDim <= 0 {
		return stats
	}
	totalPoints := len(b.data) / b.dataDim
	if totalPoints <= 0 || len(ranges) == 0 {
		return stats
	}

	kk := int(k)
	diffPts := 0
	clamped := make([]insertRange, 0, len(ranges))
	for _, r := range ranges {
		if r.start > uint64(totalPoints) {
			r.start = uint64(totalPoints)
		}
		if r.end > uint64(totalPoints) {
			r.end = uint64(totalPoints)
		}
		if r.end <= r.start {
			continue
		}
		diffPts += int(r.end - r.start)
		clamped = append(clamped, r)
	}
	if len(clamped) == 0 {
		return stats
	}

	scale := float32(b.inflightPathCorridorScale())
	maxWitnesses := b.inflightPathCorridorWitnesses()
	sigBits, sigRadius, useSignatureGate := b.inflightSignatureConfig()
	if useSignatureGate && !b.inflightPathCorridorAllowedForDim() {
		return b.mergeInflightSignature(results, queries, k, clamped, sigBits, sigRadius)
	}
	if useSignatureGate {
		b.ensureDirectionSignatures(sigBits)
	}
	processRange := func(start, end int) occMergeStats {
		local := occMergeStats{}
		for qi := start; qi < end && qi < len(queries); qi++ {
			if qi >= len(results) || len(queries[qi]) < b.dataDim {
				continue
			}
			var topBuf [128]scoredTag
			var top []scoredTag
			if kk <= len(topBuf) {
				top = topBuf[:0]
			} else {
				top = make([]scoredTag, 0, kk)
			}
			worstIdx := -1
			for _, tag := range results[qi] {
				id := int(tag)
				if id < 0 || id >= totalPoints {
					continue
				}
				local.seedDists++
				top = append(top, scoredTag{
					tag:  tag,
					dist: b.l2Squared(queries[qi], id),
				})
				if len(top) >= kk {
					break
				}
			}
			if len(top) == kk {
				worstIdx = worstScoredTag(top)
			}

			var witnessIDs [32]int
			var witnessRadiusSq [32]float32
			witnessCount := 0
			var qlo, qhi uint64
			querySigOK := false
			if useSignatureGate {
				qlo, qhi, querySigOK = b.queryDirectionSignature(queries[qi], sigBits)
			}
			if qi < len(work) && len(top) == kk && worstIdx >= 0 {
				tauRoot := float32(math.Sqrt(float64(top[worstIdx].dist)))
				pathCount := int(work[qi].PathCount)
				if pathCount > len(work[qi].PathLabels) {
					pathCount = len(work[qi].PathLabels)
				}
				selectedLimit := maxWitnesses
				if selectedLimit > len(witnessIDs) {
					selectedLimit = len(witnessIDs)
				}
				var witnessPathDists [32]float32
				for p := 0; p < pathCount && selectedLimit > 0; p++ {
					witnessID := int(work[qi].PathLabels[p])
					pathDist := work[qi].PathDists[p]
					if witnessID < 0 || witnessID >= totalPoints || !(pathDist >= 0) {
						continue
					}
					radius := scale * (tauRoot + float32(math.Sqrt(float64(pathDist))))
					insertAt := witnessCount
					if insertAt > selectedLimit {
						insertAt = selectedLimit
					}
					for insertAt > 0 && pathDist < witnessPathDists[insertAt-1] {
						insertAt--
					}
					if insertAt >= selectedLimit {
						continue
					}
					if witnessCount < selectedLimit {
						witnessCount++
					}
					for s := witnessCount - 1; s > insertAt; s-- {
						witnessPathDists[s] = witnessPathDists[s-1]
						witnessIDs[s] = witnessIDs[s-1]
						witnessRadiusSq[s] = witnessRadiusSq[s-1]
					}
					witnessPathDists[insertAt] = pathDist
					witnessIDs[insertAt] = witnessID
					witnessRadiusSq[insertAt] = radius * radius
				}
			}
			pathActive := witnessCount > 0 && len(top) == kk

			for _, r := range clamped {
				for id := int(r.start); id < int(r.end); id++ {
					if useSignatureGate {
						if !querySigOK ||
							id < 0 || id >= len(b.directionSigReady) ||
							b.directionSigReady[id].Load() == 0 {
							continue
						}
						local.filterTests++
						lo := b.directionSigLo[id]
						hi := b.directionSigHi[id]
						h := mathbits.OnesCount64(qlo^lo) + mathbits.OnesCount64(qhi^hi)
						if h > sigRadius {
							continue
						}
					}
					selected := !pathActive
					if pathActive {
						selected = true
						for p := 0; p < witnessCount; p++ {
							local.filterTests++
							local.filterL2Dists++
							if _, ok := b.l2SquaredBetweenBounded(id, witnessIDs[p], witnessRadiusSq[p]); !ok {
								selected = false
								break
							}
						}
					}
					if !selected {
						continue
					}
					local.candidates++
					local.diffDists++
					if len(top) < kk {
						dist := b.l2Squared(queries[qi], id)
						top = append(top, scoredTag{tag: uint32(id), dist: dist})
						if len(top) == kk {
							worstIdx = worstScoredTag(top)
						}
						continue
					}
					local.mergeComps += len(top)
					if worstIdx < 0 {
						worstIdx = worstScoredTag(top)
					}
					dist, ok := b.l2SquaredBounded(queries[qi], id, top[worstIdx].dist)
					if !ok {
						continue
					}
					if dist < top[worstIdx].dist {
						top[worstIdx] = scoredTag{tag: uint32(id), dist: dist}
						worstIdx = worstScoredTag(top)
					}
				}
			}

			sortScoredTags(top)
			for j := 0; j < len(results[qi]) && j < len(top); j++ {
				results[qi][j] = top[j].tag
			}
		}
		return local
	}

	workers := 1
	if b.config != nil && b.config.Search.InflightJoinSignatureWorkers != nil {
		workers = *b.config.Search.InflightJoinSignatureWorkers
	}
	if workers < 1 {
		workers = 1
	}
	if workers > len(queries) {
		workers = len(queries)
	}
	if workers <= 1 {
		stats = processRange(0, len(queries))
	} else {
		partials := make([]occMergeStats, workers)
		chunk := (len(queries) + workers - 1) / workers
		var wg sync.WaitGroup
		for worker := 0; worker < workers; worker++ {
			start := worker * chunk
			end := start + chunk
			if start >= len(queries) {
				break
			}
			wg.Add(1)
			go func(worker, start, end int) {
				defer wg.Done()
				partials[worker] = processRange(start, end)
			}(worker, start, end)
		}
		wg.Wait()
		for _, local := range partials {
			stats.seedDists += local.seedDists
			stats.diffDists += local.diffDists
			stats.mergeComps += local.mergeComps
			stats.filterTests += local.filterTests
			stats.filterL2Dists += local.filterL2Dists
			stats.candidates += local.candidates
		}
	}

	stats.diffPts = diffPts
	stats.totalL2Dists = stats.seedDists + stats.diffDists + stats.filterL2Dists
	return stats
}

func (b *Bench) mergeInflightSignatureScan(results [][]uint32, queries [][]float32, k uint32, ranges []insertRange, sigBits int, radius int) occMergeStats {
	stats := occMergeStats{}
	if len(results) == 0 || len(queries) == 0 || k == 0 || b.dataDim <= 0 {
		return stats
	}
	totalPoints := len(b.data) / b.dataDim
	if totalPoints <= 0 || len(ranges) == 0 {
		return stats
	}

	kk := int(k)
	diffPts := 0
	clamped := make([]insertRange, 0, len(ranges))
	for _, r := range ranges {
		if r.start > uint64(totalPoints) {
			r.start = uint64(totalPoints)
		}
		if r.end > uint64(totalPoints) {
			r.end = uint64(totalPoints)
		}
		if r.end <= r.start {
			continue
		}
		diffPts += int(r.end - r.start)
		clamped = append(clamped, r)
	}
	if len(clamped) == 0 {
		return stats
	}

	b.ensureDirectionSignatures(sigBits)
	fallbackRadius := 0
	fallbackFloor := 0
	if b.config != nil {
		if b.config.Search.InflightJoinSignatureFallbackRadius != nil {
			fallbackRadius = *b.config.Search.InflightJoinSignatureFallbackRadius
		}
		if b.config.Search.InflightJoinSignatureFallbackFloor != nil {
			fallbackFloor = *b.config.Search.InflightJoinSignatureFallbackFloor
		}
	}
	useFallback := fallbackRadius > radius && fallbackFloor > 0
	processRange := func(start, end int) occMergeStats {
		local := occMergeStats{}
		for qi := start; qi < end && qi < len(queries); qi++ {
			if qi >= len(results) || len(queries[qi]) < b.dataDim {
				continue
			}
			qlo, qhi, ok := b.queryDirectionSignature(queries[qi], sigBits)
			if !ok {
				continue
			}

			var topBuf [128]scoredTag
			var top []scoredTag
			if kk <= len(topBuf) {
				top = topBuf[:0]
			} else {
				top = make([]scoredTag, 0, kk)
			}
			worstIdx := -1
			for _, tag := range results[qi] {
				id := int(tag)
				if id < 0 || id >= totalPoints {
					continue
				}
				local.seedDists++
				top = append(top, scoredTag{
					tag:  tag,
					dist: b.l2Squared(queries[qi], id),
				})
				if len(top) >= kk {
					break
				}
			}
			if len(top) == kk {
				worstIdx = worstScoredTag(top)
			}

			selectedThisQuery := 0
			for _, r := range clamped {
				for id := int(r.start); id < int(r.end); id++ {
					if id < 0 || id >= len(b.directionSigReady) ||
						b.directionSigReady[id].Load() == 0 {
						continue
					}
					local.filterTests++
					lo := b.directionSigLo[id]
					hi := b.directionSigHi[id]
					h := mathbits.OnesCount64(qlo^lo) + mathbits.OnesCount64(qhi^hi)
					if h > radius {
						continue
					}
					local.candidates++
					local.diffDists++
					selectedThisQuery++
					if len(top) < kk {
						dist := b.l2Squared(queries[qi], id)
						top = append(top, scoredTag{tag: uint32(id), dist: dist})
						if len(top) == kk {
							worstIdx = worstScoredTag(top)
						}
						continue
					}
					local.mergeComps += len(top)
					if worstIdx < 0 {
						worstIdx = worstScoredTag(top)
					}
					dist, ok := b.l2SquaredBounded(queries[qi], id, top[worstIdx].dist)
					if !ok {
						continue
					}
					if dist < top[worstIdx].dist {
						top[worstIdx] = scoredTag{tag: uint32(id), dist: dist}
						worstIdx = worstScoredTag(top)
					}
				}
			}
			if useFallback && selectedThisQuery < fallbackFloor {
				for _, r := range clamped {
					for id := int(r.start); id < int(r.end); id++ {
						if id < 0 || id >= len(b.directionSigReady) ||
							b.directionSigReady[id].Load() == 0 {
							continue
						}
						local.filterTests++
						lo := b.directionSigLo[id]
						hi := b.directionSigHi[id]
						h := mathbits.OnesCount64(qlo^lo) + mathbits.OnesCount64(qhi^hi)
						if h <= radius || h > fallbackRadius {
							continue
						}
						local.candidates++
						local.diffDists++
						if len(top) < kk {
							dist := b.l2Squared(queries[qi], id)
							top = append(top, scoredTag{tag: uint32(id), dist: dist})
							if len(top) == kk {
								worstIdx = worstScoredTag(top)
							}
							continue
						}
						local.mergeComps += len(top)
						if worstIdx < 0 {
							worstIdx = worstScoredTag(top)
						}
						dist, ok := b.l2SquaredBounded(queries[qi], id, top[worstIdx].dist)
						if !ok {
							continue
						}
						if dist < top[worstIdx].dist {
							top[worstIdx] = scoredTag{tag: uint32(id), dist: dist}
							worstIdx = worstScoredTag(top)
						}
					}
				}
			}

			sortScoredTags(top)
			for j := 0; j < len(results[qi]) && j < len(top); j++ {
				results[qi][j] = top[j].tag
			}
		}
		return local
	}

	workers := 1
	if b.config != nil && b.config.Search.InflightJoinSignatureWorkers != nil {
		workers = *b.config.Search.InflightJoinSignatureWorkers
	}
	if workers < 1 {
		workers = 1
	}
	if workers > len(queries) {
		workers = len(queries)
	}
	if workers <= 1 {
		stats = processRange(0, len(queries))
	} else {
		partials := make([]occMergeStats, workers)
		chunk := (len(queries) + workers - 1) / workers
		var wg sync.WaitGroup
		for worker := 0; worker < workers; worker++ {
			start := worker * chunk
			end := start + chunk
			if start >= len(queries) {
				break
			}
			wg.Add(1)
			go func(worker, start, end int) {
				defer wg.Done()
				partials[worker] = processRange(start, end)
			}(worker, start, end)
		}
		wg.Wait()
		for _, local := range partials {
			stats.seedDists += local.seedDists
			stats.diffDists += local.diffDists
			stats.mergeComps += local.mergeComps
			stats.filterTests += local.filterTests
			stats.candidates += local.candidates
		}
	}

	stats.diffPts = diffPts
	stats.totalL2Dists = stats.seedDists + stats.diffDists
	return stats
}

func (b *Bench) mergeInflightSignatureBucket(results [][]uint32, queries [][]float32, k uint32, ranges []insertRange, sigBits int, radius int) occMergeStats {
	stats := occMergeStats{}
	if len(results) == 0 || len(queries) == 0 || k == 0 || b.dataDim <= 0 {
		return stats
	}
	totalPoints := len(b.data) / b.dataDim
	if totalPoints <= 0 || len(ranges) == 0 {
		return stats
	}

	kk := int(k)
	diffPts := 0
	clamped := make([]insertRange, 0, len(ranges))
	for _, r := range ranges {
		if r.start > uint64(totalPoints) {
			r.start = uint64(totalPoints)
		}
		if r.end > uint64(totalPoints) {
			r.end = uint64(totalPoints)
		}
		if r.end <= r.start {
			continue
		}
		diffPts += int(r.end - r.start)
		clamped = append(clamped, r)
	}
	if len(clamped) == 0 {
		return stats
	}

	b.ensureDirectionSignatures(sigBits)
	bands := directionSignatureBandCount(sigBits)
	if bands == 0 {
		return stats
	}
	bandRadius := 0
	if b.config != nil && b.config.Search.InflightJoinSignatureBandRadius != nil {
		bandRadius = *b.config.Search.InflightJoinSignatureBandRadius
	}
	masks := directionBandMasks(bandRadius)
	for qi := range queries {
		if qi >= len(results) || len(queries[qi]) < b.dataDim {
			continue
		}
		qlo, qhi, ok := b.queryDirectionSignature(queries[qi], sigBits)
		if !ok {
			continue
		}

		top := make([]scoredTag, 0, kk)
		for _, tag := range results[qi] {
			id := int(tag)
			if id < 0 || id >= totalPoints {
				continue
			}
			stats.seedDists++
			top = append(top, scoredTag{
				tag:  tag,
				dist: b.l2Squared(queries[qi], id),
			})
			if len(top) >= kk {
				break
			}
		}

		seen := make(map[int]struct{}, 256)
		candidateIDs := make([]int, 0, 256)
		b.directionSigBucketMu.RLock()
		for band := 0; band < bands; band++ {
			qvalue := directionSignatureBandValue(qlo, qhi, band)
			for _, mask := range masks {
				key := directionSignatureBucketKey(band, qvalue^mask)
				for _, tag := range b.directionSigBuckets[key] {
					id := int(tag)
					if _, ok := seen[id]; ok {
						continue
					}
					seen[id] = struct{}{}
					if !idInRanges(id, clamped) {
						continue
					}
					candidateIDs = append(candidateIDs, id)
				}
			}
		}
		b.directionSigBucketMu.RUnlock()

		for _, id := range candidateIDs {
			stats.filterTests++
			if id < 0 || id >= len(b.directionSigReady) ||
				b.directionSigReady[id].Load() == 0 {
				continue
			}
			lo := b.directionSigLo[id]
			hi := b.directionSigHi[id]
			h := mathbits.OnesCount64(qlo^lo) + mathbits.OnesCount64(qhi^hi)
			if h > radius {
				continue
			}
			stats.candidates++
			stats.diffDists++
			dist := b.l2Squared(queries[qi], id)
			if len(top) < kk {
				top = append(top, scoredTag{tag: uint32(id), dist: dist})
				continue
			}
			stats.mergeComps += len(top)
			worst := worstScoredTag(top)
			if dist < top[worst].dist {
				top[worst] = scoredTag{tag: uint32(id), dist: dist}
			}
		}

		sort.Slice(top, func(i, j int) bool {
			return top[i].dist < top[j].dist
		})
		for j := 0; j < len(results[qi]) && j < len(top); j++ {
			results[qi][j] = top[j].tag
		}
	}

	stats.diffPts = diffPts
	stats.totalL2Dists = stats.seedDists + stats.diffDists
	return stats
}

func (b *Bench) mergeInflightSignature(results [][]uint32, queries [][]float32, k uint32, ranges []insertRange, sigBits int, radius int) occMergeStats {
	if b.config != nil && b.config.Search.InflightJoinSignatureBucket != nil &&
		*b.config.Search.InflightJoinSignatureBucket {
		return b.mergeInflightSignatureBucket(results, queries, k, ranges, sigBits, radius)
	}
	return b.mergeInflightSignatureScan(results, queries, k, ranges, sigBits, radius)
}

func (b *Bench) inflightJoinParams() FreshJoinParams {
	params := FreshJoinParams{
		QueryResultPrefix:   10,
		SmallFreshThreshold: 200,
		CandidateFloorPct:   0,
	}
	if b.config.Search.InflightJoinQueryResultPrefix != nil {
		params.QueryResultPrefix = *b.config.Search.InflightJoinQueryResultPrefix
	}
	if b.config.Search.InflightJoinSmallFreshThreshold != nil {
		params.SmallFreshThreshold = *b.config.Search.InflightJoinSmallFreshThreshold
	}
	if b.config.Search.InflightJoinCandidateFloorPct != nil {
		params.CandidateFloorPct = *b.config.Search.InflightJoinCandidateFloorPct
	}
	return params
}

func (b *Bench) collectFreshRanges(ranges []insertRange) ([][]float32, []uint32, uint64) {
	if b.dataDim <= 0 || len(ranges) == 0 {
		return nil, nil, 0
	}
	totalPoints := len(b.data) / b.dataDim
	if totalPoints <= 0 {
		return nil, nil, 0
	}
	viewLimit := uint64(totalPoints)
	fresh := make([][]float32, 0)
	tags := make([]uint32, 0)
	for _, r := range ranges {
		if r.start > uint64(totalPoints) {
			r.start = uint64(totalPoints)
		}
		if r.end > uint64(totalPoints) {
			r.end = uint64(totalPoints)
		}
		if r.end <= r.start {
			continue
		}
		if r.start < viewLimit {
			viewLimit = r.start
		}
		for id := int(r.start); id < int(r.end); id++ {
			offset := id * b.dataDim
			fresh = append(fresh, b.data[offset:offset+b.dataDim])
			tags = append(tags, uint32(id))
		}
	}
	if len(fresh) == 0 {
		return nil, nil, 0
	}
	return fresh, tags, viewLimit
}

func (b *Bench) mergeInflightJoinLabels(results [][]uint32, queries [][]float32, k uint32, labels []uint32) occMergeStats {
	stats := occMergeStats{}
	if len(results) == 0 || len(queries) == 0 || k == 0 || len(labels) == 0 {
		return stats
	}
	joined, joinStats, err := b.index.FreshJoinMergeLabels(
		queries, results, k, labels, b.inflightJoinParams())
	if err != nil {
		log.Printf("Inflight join failed for ANNchor inflight labels; falling back to brute OCC: %v", err)
		return b.mergeInflightBruteforceLabels(results, queries, k, labels)
	}
	for qi := range results {
		copy(results[qi], joined[qi])
	}
	stats.diffPts = int(joinStats.FreshVectors)
	stats.seedDists = len(queries) * int(k)
	stats.diffDists = int(joinStats.ExactedCandidates)
	stats.totalL2Dists = stats.seedDists + stats.diffDists
	stats.filterTests = int(joinStats.ActivePlacementRegions)
	stats.candidates = int(joinStats.SelectedCandidates)
	stats.mergeComps = int(joinStats.SelectedCandidates)
	return stats
}

func (b *Bench) handleSearchTask(task Task, start time.Time) {
	if len(task.Data) == 0 || task.RecallAt == 0 {
		return
	}
	searchStartRawNs := MonotonicRawNs()

	if strings.EqualFold(b.config.Index.IndexType, "annchor-preempt") {
		b.activeSearchTasks.Add(1)
		pressure := b.preemptSearchPressure()
		b.index.SetPreemptRuntimeSearchBacklog(pressure)
		b.index.SetPreemptRuntimePrioritySearches(pressure)
		defer func() {
			b.activeSearchTasks.Add(-1)
			p := b.preemptSearchPressure()
			b.index.SetPreemptRuntimeSearchBacklog(p)
			b.index.SetPreemptRuntimePrioritySearches(p)
		}()
	}

	committedAtStart := b.committedOffset.Load()
	viewOffset := committedAtStart
	joinEnabled := b.config.Search.EnableInflightJoin != nil && *b.config.Search.EnableInflightJoin
	occEnabled := joinEnabled || (b.config.Search.EnableInflightBruteforce != nil && *b.config.Search.EnableInflightBruteforce)
	isAnnchorM2 := strings.EqualFold(b.config.Index.IndexType, "annchor-m2")
	if !occEnabled && viewOffset > task.InsertOffset {
		viewOffset = task.InsertOffset
	}
	activeRangesAtQueryEnter := b.activeInsertSnapshot(viewOffset)
	// ANNchor-M2 snapshots C++-side insert batches that had already started
	// when this search entered the benchmark and had not completed at that
	// timestamp. The legacy Go active range is still collected for other OCC
	// paths and lag telemetry, but it must not drive ANNchor-M2 validation.
	var releaseReplica func()
	searchIndex := b.servingIndex()
	if b.replicaRefresh != nil {
		searchIndex, releaseReplica = b.replicaRefresh.acquire()
		defer releaseReplica()
	}
	annchorInflightPts := searchIndex.GetInflightPoints()
	if b.replicaSnapshotMode() {
		viewOffset = uint64(b.config.Data.BeginNum)
	} else if b.replicaRefresh != nil {
		viewOffset = b.replicaRefresh.visible()
	}
	var annchorInflightLabels []uint32
	if occEnabled && isAnnchorM2 {
		maxLabels := b.config.Workload.BatchSize * b.config.Workload.NumThreads
		if maxLabels < 128 {
			maxLabels = 128
		}
		annchorInflightLabels = searchIndex.InflightLabelsBefore(
			maxLabels,
			searchStartRawNs,
		)
		annchorInflightPts = int64(len(annchorInflightLabels))
	}

	currentInserted := atomic.LoadInt64(&b.globalInsertCnt)
	beginNum := int64(b.config.Data.BeginNum)
	physicalHead := currentInserted + beginNum
	totalLagPts := physicalHead - int64(viewOffset)
	if totalLagPts < 0 {
		totalLagPts = 0
	}
	contiguityLagPts := physicalHead - int64(committedAtStart)
	if contiguityLagPts < 0 {
		contiguityLagPts = 0
	}
	queueLagPts := int64(committedAtStart) - int64(viewOffset)
	if queueLagPts < 0 {
		queueLagPts = 0
	}

	atomic.AddInt64(&b.searchLagSum, totalLagPts)
	atomic.AddInt64(&b.searchLagCount, 1)
	atomic.AddInt64(&b.searchLagContiguitySum, contiguityLagPts)
	atomic.AddInt64(&b.searchLagQueueSum, queueLagPts)
	updateMax := func(addr *int64, v int64) {
		for {
			old := atomic.LoadInt64(addr)
			if v <= old {
				return
			}
			if atomic.CompareAndSwapInt64(addr, old, v) {
				return
			}
		}
	}
	updateMax(&b.searchLagMax, totalLagPts)
	updateMax(&b.searchLagContiguityMax, contiguityLagPts)
	updateMax(&b.searchLagQueueMax, queueLagPts)

	var lockWaitStart time.Time
	if b.config.Workload.WithExternalRWLock {
		lockWaitStart = time.Now()
		b.rwMu.RLock()
		lockWait := float64(time.Since(lockWaitStart).Microseconds()) / 1000.0
		if lockWait > 1.0 {
			atomic.AddInt64(&b.searchLockWaitCount, 1)
		}
		b.searchLockMu.Lock()
		b.searchLockWaits = append(b.searchLockWaits, lockWait)
		b.searchLockMu.Unlock()
	}
	activeFrontOffset := viewOffset
	for _, r := range activeRangesAtQueryEnter {
		if r.end > activeFrontOffset {
			activeFrontOffset = r.end
		}
	}
	activeFrontLagPts := int64(activeFrontOffset) - int64(b.committedOffset.Load())
	if activeFrontLagPts < 0 {
		activeFrontLagPts = 0
	}
	// OCC freshness here means: run MVCC at the committed watermark observed
	// just before the query enters the index, then brute-force the latest
	// insert front beyond that watermark. latestOffsetAtQueryEnter may include
	// insert batches that have started but are not yet part of the contiguous
	// committed watermark.
	var diagBefore indexDiagStats
	if b.m3BatchDiagEnabled {
		diagBefore = parseIndexDiagStats(searchIndex.DumpStats())
	}
	var results [][]uint32
	var searchWork []SearchWorkStats
	var err error
	indexStart := time.Now()
	m2VEnabled := joinEnabled && b.inflightM2VEnabled()
	pathOneHopEnabled := joinEnabled && b.inflightPathOneHopEnabled()
	pathCorridorEnabled := joinEnabled &&
		!isAnnchorM2 &&
		b.inflightPathCorridorEnabled() &&
		b.inflightPathCorridorAllowedForDim()
	searchVisibleOffset := uint64(viewOffset)
	if os.Getenv("ANNCHOR_FORCE_VISIBLE_TS_MAX") == "1" {
		searchVisibleOffset = ^uint64(0)
	}
	if m2VEnabled || pathOneHopEnabled || pathCorridorEnabled {
		results, _, searchWork, err = searchIndex.BatchSearchPathWork(task.Data, uint32(task.RecallAt), searchVisibleOffset)
	} else if task.Measured && b.searchWorkCountersEnabled {
		results, _, searchWork, err = searchIndex.BatchSearchMeasuredWithWork(task.Data, uint32(task.RecallAt), searchVisibleOffset)
	} else if task.Measured {
		results, _, err = searchIndex.BatchSearchMeasured(task.Data, uint32(task.RecallAt), searchVisibleOffset)
	} else {
		results, _, err = searchIndex.BatchSearch(task.Data, uint32(task.RecallAt), searchVisibleOffset)
	}
	indexMs := float64(time.Since(indexStart).Microseconds()) / 1000.0
	var diagAfter indexDiagStats
	if b.m3BatchDiagEnabled {
		diagAfter = parseIndexDiagStats(searchIndex.DumpStats())
	}
	inflightBruteforcePts := 0
	inflightOccMs := 0.0
	inflightOccL2Dists := 0
	inflightOccMergeComps := 0
	inflightOccFilterTests := 0
	inflightOccCandidates := 0
	if err == nil && occEnabled {
		if b.config.Search.DeferInflightOccAfterInsert != nil &&
			*b.config.Search.DeferInflightOccAfterInsert &&
			len(activeRangesAtQueryEnter) > 0 {
			maxWaitUs := -1
			if b.config.Search.DeferInflightOccMaxWaitUs != nil {
				maxWaitUs = *b.config.Search.DeferInflightOccMaxWaitUs
			}
			b.waitForInsertRangesDone(activeRangesAtQueryEnter, maxWaitUs)
			runtime.Gosched()
		}
		occStart := time.Now()
		var occStats occMergeStats
		if joinEnabled {
			if m2VEnabled {
				occStats = b.mergeInflightM2V(
					results,
					task.Data,
					uint32(task.RecallAt),
					activeRangesAtQueryEnter,
					searchWork,
				)
			} else if pathOneHopEnabled {
				occStats = b.mergeInflightPathOneHop(
					results,
					task.Data,
					uint32(task.RecallAt),
					activeRangesAtQueryEnter,
					searchWork,
				)
			} else if isAnnchorM2 {
				occStats = b.mergeInflightJoinLabels(
					results,
					task.Data,
					uint32(task.RecallAt),
					annchorInflightLabels,
				)
			} else if pathCorridorEnabled {
				occStats = b.mergeInflightPathCorridor(
					results,
					task.Data,
					uint32(task.RecallAt),
					activeRangesAtQueryEnter,
					searchWork,
				)
			} else if sigBits, radius, ok := b.inflightSignatureConfig(); ok {
				occStats = b.mergeInflightSignature(
					results,
					task.Data,
					uint32(task.RecallAt),
					activeRangesAtQueryEnter,
					sigBits,
					radius,
				)
			} else {
				log.Printf("Inflight join has no active signature/path corridor; falling back to brute OCC")
				occStats = b.mergeInflightBruteforce(
					results,
					task.Data,
					uint32(task.RecallAt),
					activeRangesAtQueryEnter,
				)
			}
		} else {
			if isAnnchorM2 {
				occStats = b.mergeInflightBruteforceLabels(
					results,
					task.Data,
					uint32(task.RecallAt),
					annchorInflightLabels,
				)
			} else {
				occStats = b.mergeInflightBruteforce(
					results,
					task.Data,
					uint32(task.RecallAt),
					activeRangesAtQueryEnter,
				)
			}
		}
		inflightOccMs = float64(time.Since(occStart).Microseconds()) / 1000.0
		_ = occStats.diffPts // legacy bench-layer diff, replaced by ANNchor-internal metric below
		inflightOccL2Dists = occStats.totalL2Dists
		inflightOccMergeComps = occStats.mergeComps
		inflightOccFilterTests = occStats.filterTests
		inflightOccCandidates = occStats.candidates
	}
	// ANNchor-M2 reports real C++-side inflight labels. Other paths keep the
	// legacy coarse point count.
	if annchorInflightPts >= 0 {
		inflightBruteforcePts = int(annchorInflightPts)
	}
	if b.config.Workload.WithExternalRWLock {
		b.rwMu.RUnlock()
	}
	if err != nil {
		log.Printf("Search error: %v", err)
		return
	}
	committedAtFinish := b.committedOffset.Load()
	missedE2EPts := int64(committedAtFinish) - int64(viewOffset)
	if missedE2EPts < 0 {
		missedE2EPts = 0
	}
	diagDistComps := 0.0
	diagHops := 0.0
	diagFutureSkipQueries := 0.0
	diagFutureSkipHops := 0.0
	diagRecoveryTriggers := 0.0
	diagRecoveredEdges := 0.0
	diagUsefulRecovered := 0.0
	diagModifiedExpansions := 0.0
	diagRecoveryAttempts := 0.0
	diagRecoveryCandidates := 0.0
	diagRecoveryGetMs := 0.0
	diagRecoveryLoopMs := 0.0
	diagRewriteActiveExp := 0.0
	diagRewriteActiveQuery := 0.0
	diagRewriteRecentExp := 0.0
	diagRewriteRecentQuery := 0.0
	diagRewritePeriodExp := 0.0
	diagRewritePeriodQuery := 0.0
	diagRewritePeriodActiveSum := 0.0
	diagRewritePeriodActiveMax := 0.0
	diagBatchSearchCalls := 0.0
	diagBatchSearchQueries := 0.0
	diagBatchSearchArenaMs := 0.0
	diagBatchSearchKnnMs := 0.0
	diagBatchSearchCopyMs := 0.0
	diagInvisibleExpansions := 0.0
	diagInvisibleExpansionEdges := 0.0
	diagInvisibleCandidateDistComps := 0.0
	diagInvisibleCandidateEnqueues := 0.0
	diagPhaseOverlapSamples := 0.0
	diagPhaseExistingUpdateSamples := 0.0
	diagPhaseExistingUpdateQueries := 0.0
	diagPhaseLinkCriticalSamples := 0.0
	diagPhaseLinkCriticalQueries := 0.0
	diagPhaseLoadScanSamples := 0.0
	diagPhaseLoadScanQueries := 0.0
	diagPhaseAppendSamples := 0.0
	diagPhaseAppendQueries := 0.0
	diagPhasePruneSamples := 0.0
	diagPhasePruneQueries := 0.0
	diagPhaseUndoRecordSamples := 0.0
	diagPhaseUndoRecordQueries := 0.0
	diagPhaseRewriteSamples := 0.0
	diagPhaseRewriteQueries := 0.0
	if b.m3BatchDiagEnabled {
		diagDistComps = indexDiagDelta(diagBefore, diagAfter, "dist_computations")
		diagHops = indexDiagDelta(diagBefore, diagAfter, "hops")
		diagFutureSkipQueries = indexDiagDelta(diagBefore, diagAfter, "future_skip_queries")
		diagFutureSkipHops = indexDiagDelta(diagBefore, diagAfter, "future_skip_hops_total")
		diagRecoveryTriggers = indexDiagDelta(diagBefore, diagAfter, "recovery_triggers")
		diagRecoveredEdges = indexDiagDelta(diagBefore, diagAfter, "recovered_edges_total")
		diagUsefulRecovered = indexDiagDelta(diagBefore, diagAfter, "recovered_edges_useful")
		diagModifiedExpansions = indexDiagDelta(diagBefore, diagAfter, "recovery_modified_expansions")
		diagRecoveryAttempts = indexDiagDelta(diagBefore, diagAfter, "recovery_attempts")
		diagRecoveryCandidates = indexDiagDelta(diagBefore, diagAfter, "recovery_candidate_edges")
		diagRecoveryGetMs = indexDiagDelta(diagBefore, diagAfter, "recovery_get_ns") / 1e6
		diagRecoveryLoopMs = indexDiagDelta(diagBefore, diagAfter, "recovery_loop_ns") / 1e6
		diagRewriteActiveExp = indexDiagDelta(diagBefore, diagAfter, "rewrite_active_expansions")
		diagRewriteActiveQuery = indexDiagDelta(diagBefore, diagAfter, "rewrite_active_queries")
		diagRewriteRecentExp = indexDiagDelta(diagBefore, diagAfter, "rewrite_recent_expansions")
		diagRewriteRecentQuery = indexDiagDelta(diagBefore, diagAfter, "rewrite_recent_queries")
		diagRewritePeriodExp = indexDiagDelta(diagBefore, diagAfter, "rewrite_period_expansions")
		diagRewritePeriodQuery = indexDiagDelta(diagBefore, diagAfter, "rewrite_period_queries")
		diagRewritePeriodActiveSum = indexDiagDelta(diagBefore, diagAfter, "rewrite_period_active_sum")
		diagRewritePeriodActiveMax = indexDiagDelta(diagBefore, diagAfter, "rewrite_period_active_max")
		diagBatchSearchCalls = indexDiagDelta(diagBefore, diagAfter, "batch_search_calls")
		diagBatchSearchQueries = indexDiagDelta(diagBefore, diagAfter, "batch_search_queries")
		diagBatchSearchArenaMs = indexDiagDelta(diagBefore, diagAfter, "batch_search_arena_ns") / 1e6
		diagBatchSearchKnnMs = indexDiagDelta(diagBefore, diagAfter, "batch_search_searchknn_ns") / 1e6
		diagBatchSearchCopyMs = indexDiagDelta(diagBefore, diagAfter, "batch_search_result_copy_ns") / 1e6
		diagInvisibleExpansions = indexDiagDelta(diagBefore, diagAfter, "invisible_expansions")
		diagInvisibleExpansionEdges = indexDiagDelta(diagBefore, diagAfter, "invisible_expansion_edges")
		diagInvisibleCandidateDistComps = indexDiagDelta(diagBefore, diagAfter, "invisible_candidate_dist_comps")
		diagInvisibleCandidateEnqueues = indexDiagDelta(diagBefore, diagAfter, "invisible_candidate_enqueues")
		diagPhaseOverlapSamples = indexDiagDelta(diagBefore, diagAfter, "phase_overlap_samples")
		diagPhaseExistingUpdateSamples = indexDiagDelta(diagBefore, diagAfter, "phase_existing_update_samples")
		diagPhaseExistingUpdateQueries = indexDiagDelta(diagBefore, diagAfter, "phase_existing_update_queries")
		diagPhaseLinkCriticalSamples = indexDiagDelta(diagBefore, diagAfter, "phase_link_critical_samples")
		diagPhaseLinkCriticalQueries = indexDiagDelta(diagBefore, diagAfter, "phase_link_critical_queries")
		diagPhaseLoadScanSamples = indexDiagDelta(diagBefore, diagAfter, "phase_load_scan_samples")
		diagPhaseLoadScanQueries = indexDiagDelta(diagBefore, diagAfter, "phase_load_scan_queries")
		diagPhaseAppendSamples = indexDiagDelta(diagBefore, diagAfter, "phase_append_samples")
		diagPhaseAppendQueries = indexDiagDelta(diagBefore, diagAfter, "phase_append_queries")
		diagPhasePruneSamples = indexDiagDelta(diagBefore, diagAfter, "phase_prune_samples")
		diagPhasePruneQueries = indexDiagDelta(diagBefore, diagAfter, "phase_prune_queries")
		diagPhaseUndoRecordSamples = indexDiagDelta(diagBefore, diagAfter, "phase_undo_record_samples")
		diagPhaseUndoRecordQueries = indexDiagDelta(diagBefore, diagAfter, "phase_undo_record_queries")
		diagPhaseRewriteSamples = indexDiagDelta(diagBefore, diagAfter, "phase_rewrite_samples")
		diagPhaseRewriteQueries = indexDiagDelta(diagBefore, diagAfter, "phase_rewrite_queries")
	}

	postIndexMs := float64(time.Since(indexStart).Microseconds())/1000.0 - indexMs
	searchOpMs := float64(time.Since(start).Microseconds()) / 1000.0
	searchE2EMs := float64(time.Since(task.CreateTime).Microseconds()) / 1000.0
	searchFinishMs := float64(time.Since(b.startTime).Microseconds()) / 1000.0
	searchFinishRawNs := MonotonicRawNs()

	resultsLockWaitStart := time.Now()
	b.resultsMu.Lock()
	resultsLockWaitMs := float64(time.Since(resultsLockWaitStart).Microseconds()) / 1000.0
	resultsLockHoldStart := time.Now()
	for i, tags := range results {
		result := NewSearchResult(
			task.InsertOffset,
			uint64(task.Tags[i]),
			tags,
		)
		b.searchResults = append(b.searchResults, result)
	}
	if strings.EqualFold(b.config.Index.IndexType, "annchor") || strings.EqualFold(b.config.Index.IndexType, "annchor-m1") || strings.EqualFold(b.config.Index.IndexType, "annchor-m2") || strings.EqualFold(b.config.Index.IndexType, "annchor-preempt") || strings.EqualFold(b.config.Index.IndexType, "annchor-trim") || strings.EqualFold(b.config.Index.IndexType, "hnsw-visible") {
		short := 0
		for _, tags := range results {
			if len(tags) < int(task.RecallAt) {
				short++
			}
		}
		b.resultBelowKCount += short
	}
	b.resultsMu.Unlock()
	resultsLockHoldMs := float64(time.Since(resultsLockHoldStart).Microseconds()) / 1000.0
	recordLockWaitStart := time.Now()
	b.searchMu.Lock()
	recordLockWaitMs := float64(time.Since(recordLockWaitStart).Microseconds()) / 1000.0
	measured := 0.0
	if task.Measured {
		measured = 1.0
	}
	workAt := func(i int) SearchWorkStats {
		if i >= 0 && i < len(searchWork) {
			return searchWork[i]
		}
		return SearchWorkStats{}
	}
	appendSearchWork := func(work SearchWorkStats, viewOffset uint64) {
		b.searchWorkSearchKnnMs = append(b.searchWorkSearchKnnMs, float64(work.SearchKnnNs)/1e6)
		b.searchWorkSearchKnnThreadCpuMs = append(b.searchWorkSearchKnnThreadCpuMs, float64(work.SearchKnnThreadCpuNs)/1e6)
		b.searchWorkStartCPU = append(b.searchWorkStartCPU, float64(work.WorkStartCPU))
		b.searchWorkEndCPU = append(b.searchWorkEndCPU, float64(work.WorkEndCPU))
		b.searchWorkResultCopyMs = append(b.searchWorkResultCopyMs, float64(work.ResultCopyNs)/1e6)
		b.searchWorkEntryMs = append(b.searchWorkEntryMs, float64(work.EntryNs)/1e6)
		b.searchWorkUpperSearchMs = append(b.searchWorkUpperSearchMs, float64(work.UpperSearchNs)/1e6)
		b.searchWorkBaseSearchMs = append(b.searchWorkBaseSearchMs, float64(work.BaseSearchNs)/1e6)
		b.searchWorkResultMaterializeMs = append(b.searchWorkResultMaterializeMs, float64(work.ResultMaterializeNs)/1e6)
		b.searchWorkSnapshotGuardMs = append(b.searchWorkSnapshotGuardMs, float64(work.SnapshotGuardNs)/1e6)
		b.searchWorkVisitedListGetMs = append(b.searchWorkVisitedListGetMs, float64(work.VisitedListGetNs)/1e6)
		b.searchWorkVisitedListReleaseMs = append(b.searchWorkVisitedListReleaseMs, float64(work.VisitedListReleaseNs)/1e6)
		b.searchWorkUpperLockWaitMs = append(b.searchWorkUpperLockWaitMs, float64(work.UpperLockWaitNs)/1e6)
		b.searchWorkLevel0LockWaitMs = append(b.searchWorkLevel0LockWaitMs, float64(work.Level0LockWaitNs)/1e6)
		b.searchWorkDistanceComputations = append(b.searchWorkDistanceComputations, float64(work.DistanceComputations))
		b.searchWorkUpperDistanceComputations = append(b.searchWorkUpperDistanceComputations, float64(work.UpperDistanceComputations))
		b.searchWorkLevel0DistanceComputations = append(b.searchWorkLevel0DistanceComputations, float64(work.Level0DistanceComputations))
		b.searchWorkDistanceComputeMs = append(b.searchWorkDistanceComputeMs, float64(work.DistanceComputeNs)/1e6)
		b.searchWorkUpperDistanceComputeMs = append(b.searchWorkUpperDistanceComputeMs, float64(work.UpperDistanceComputeNs)/1e6)
		b.searchWorkLevel0DistanceComputeMs = append(b.searchWorkLevel0DistanceComputeMs, float64(work.Level0DistanceComputeNs)/1e6)
		b.searchWorkLevel0QueuePopMs = append(b.searchWorkLevel0QueuePopMs, float64(work.Level0QueuePopNs)/1e6)
		b.searchWorkLevel0AdjFetchMs = append(b.searchWorkLevel0AdjFetchMs, float64(work.Level0AdjFetchNs)/1e6)
		b.searchWorkLevel0LocalityCaptureMs = append(b.searchWorkLevel0LocalityCaptureMs, float64(work.Level0LocalityCaptureNs)/1e6)
		b.searchWorkLevel0CandidateLoopMs = append(b.searchWorkLevel0CandidateLoopMs, float64(work.Level0CandidateLoopNs)/1e6)
		b.searchWorkLevel0VisitedCheckMs = append(b.searchWorkLevel0VisitedCheckMs, float64(work.Level0VisitedCheckNs)/1e6)
		b.searchWorkLevel0VisibilityCheckMs = append(b.searchWorkLevel0VisibilityCheckMs, float64(work.Level0VisibilityCheckNs)/1e6)
		b.searchWorkLevel0CandidateAcceptMs = append(b.searchWorkLevel0CandidateAcceptMs, float64(work.Level0CandidateAcceptNs)/1e6)
		b.searchWorkUpperHops = append(b.searchWorkUpperHops, float64(work.UpperHops))
		b.searchWorkUpperEdgesScanned = append(b.searchWorkUpperEdgesScanned, float64(work.UpperEdgesScanned))
		b.searchWorkLevel0Expansions = append(b.searchWorkLevel0Expansions, float64(work.Level0Expansions))
		b.searchWorkLevel0EdgesScanned = append(b.searchWorkLevel0EdgesScanned, float64(work.Level0EdgesScanned))
		b.searchWorkCandidatePops = append(b.searchWorkCandidatePops, float64(work.CandidatePops))
		b.searchWorkCandidatePushes = append(b.searchWorkCandidatePushes, float64(work.CandidatePushes))
		b.searchWorkVisitedNodes = append(b.searchWorkVisitedNodes, float64(work.VisitedNodes))
		b.searchWorkResultPushes = append(b.searchWorkResultPushes, float64(work.ResultPushes))
		b.searchWorkInvisibleExpansions = append(b.searchWorkInvisibleExpansions, float64(work.InvisibleExpansions))
		b.searchWorkInvisibleEdges = append(b.searchWorkInvisibleEdges, float64(work.InvisibleEdges))
		b.searchWorkInvisibleCandidateDistComps = append(b.searchWorkInvisibleCandidateDistComps, float64(work.InvisibleCandidateDistComps))
		b.searchWorkInvisibleCandidateEnqueues = append(b.searchWorkInvisibleCandidateEnqueues, float64(work.InvisibleCandidateEnqueues))
		b.searchWorkFutureSkipHops = append(b.searchWorkFutureSkipHops, float64(work.FutureSkipHops))
		b.searchWorkRewriteActiveExpansions = append(b.searchWorkRewriteActiveExpansions, float64(work.RewriteActiveExpansions))
		b.searchWorkRewriteRecentExpansions = append(b.searchWorkRewriteRecentExpansions, float64(work.RewriteRecentExpansions))
		b.searchWorkRewritePeriodExpansions = append(b.searchWorkRewritePeriodExpansions, float64(work.RewritePeriodExpansions))
		b.searchWorkRewritePeriodActiveSum = append(b.searchWorkRewritePeriodActiveSum, float64(work.RewritePeriodActiveSum))
		b.searchWorkRewritePeriodActiveMax = append(b.searchWorkRewritePeriodActiveMax, float64(work.RewritePeriodActiveMax))
		b.searchWorkExpandVisibleCount = append(b.searchWorkExpandVisibleCount, float64(work.ExpandVisibleCount))
		b.searchWorkExpandRecent1KHits = append(b.searchWorkExpandRecent1KHits, float64(work.ExpandRecent1KHits))
		b.searchWorkExpandRecent4KHits = append(b.searchWorkExpandRecent4KHits, float64(work.ExpandRecent4KHits))
		b.searchWorkExpandRecent16KHits = append(b.searchWorkExpandRecent16KHits, float64(work.ExpandRecent16KHits))
		b.searchWorkExpandLabelGapSum = append(b.searchWorkExpandLabelGapSum, float64(work.ExpandLabelGapSum))
		b.searchWorkExpandLabelSpan = append(b.searchWorkExpandLabelSpan, float64(work.ExpandLabelSpan))
		b.searchWorkExpandUniqueLabel4KBuckets = append(b.searchWorkExpandUniqueLabel4KBuckets, float64(work.ExpandUniqueLabel4KBuckets))
		b.searchWorkExpandUniqueData4KPages = append(b.searchWorkExpandUniqueData4KPages, float64(work.ExpandUniqueData4KPages))
		b.searchWorkExpandUniqueData2MPages = append(b.searchWorkExpandUniqueData2MPages, float64(work.ExpandUniqueData2MPages))
		b.searchWorkExpandUniqueAdj4KPages = append(b.searchWorkExpandUniqueAdj4KPages, float64(work.ExpandUniqueAdj4KPages))
		b.searchWorkExpandUniqueAdj2MPages = append(b.searchWorkExpandUniqueAdj2MPages, float64(work.ExpandUniqueAdj2MPages))
		b.searchWorkExpandUniqueOverflow = append(b.searchWorkExpandUniqueOverflow, float64(work.ExpandUniqueOverflow))

		count := int(work.PathCount)
		if count > len(work.PathLabels) {
			count = len(work.PathLabels)
		}
		b.searchWorkPathCount = append(b.searchWorkPathCount, float64(count))
		if count == 0 {
			b.searchWorkPathFirstLabel = append(b.searchWorkPathFirstLabel, 0)
			b.searchWorkPathLastLabel = append(b.searchWorkPathLastLabel, 0)
			b.searchWorkPathMinLabel = append(b.searchWorkPathMinLabel, 0)
			b.searchWorkPathMaxLabel = append(b.searchWorkPathMaxLabel, 0)
			b.searchWorkPathLabelSpan = append(b.searchWorkPathLabelSpan, 0)
			b.searchWorkPathMeanAbsGap = append(b.searchWorkPathMeanAbsGap, 0)
			b.searchWorkPathUnique4KBuckets = append(b.searchWorkPathUnique4KBuckets, 0)
			b.searchWorkPathMinVisibleAge = append(b.searchWorkPathMinVisibleAge, 0)
			b.searchWorkPathMeanVisibleAge = append(b.searchWorkPathMeanVisibleAge, 0)
			b.searchWorkPathRecent1KHits = append(b.searchWorkPathRecent1KHits, 0)
			b.searchWorkPathRecent4KHits = append(b.searchWorkPathRecent4KHits, 0)
			b.searchWorkPathRecent16KHits = append(b.searchWorkPathRecent16KHits, 0)
			return
		}

		minLabel := work.PathLabels[0]
		maxLabel := work.PathLabels[0]
		gapSum := uint64(0)
		unique4K := 0
		visibleAgeCount := 0
		visibleAgeSum := uint64(0)
		minVisibleAge := uint64(0)
		recent1KHits := 0
		recent4KHits := 0
		recent16KHits := 0
		for i := 0; i < count; i++ {
			label := work.PathLabels[i]
			if label < minLabel {
				minLabel = label
			}
			if label > maxLabel {
				maxLabel = label
			}
			if i > 0 {
				prev := work.PathLabels[i-1]
				if label > prev {
					gapSum += uint64(label - prev)
				} else {
					gapSum += uint64(prev - label)
				}
			}
			bucket := label / 4096
			seen := false
			for j := 0; j < i; j++ {
				if work.PathLabels[j]/4096 == bucket {
					seen = true
					break
				}
			}
			if !seen {
				unique4K++
			}
			if viewOffset > 0 && uint64(label) <= viewOffset {
				age := viewOffset - uint64(label)
				if visibleAgeCount == 0 || age < minVisibleAge {
					minVisibleAge = age
				}
				visibleAgeCount++
				visibleAgeSum += age
				if age < 1024 {
					recent1KHits++
				}
				if age < 4096 {
					recent4KHits++
				}
				if age < 16384 {
					recent16KHits++
				}
			}
		}
		meanGap := 0.0
		if count > 1 {
			meanGap = float64(gapSum) / float64(count-1)
		}
		meanVisibleAge := 0.0
		if visibleAgeCount > 0 {
			meanVisibleAge = float64(visibleAgeSum) / float64(visibleAgeCount)
		}
		b.searchWorkPathFirstLabel = append(b.searchWorkPathFirstLabel, float64(work.PathLabels[0]))
		b.searchWorkPathLastLabel = append(b.searchWorkPathLastLabel, float64(work.PathLabels[count-1]))
		b.searchWorkPathMinLabel = append(b.searchWorkPathMinLabel, float64(minLabel))
		b.searchWorkPathMaxLabel = append(b.searchWorkPathMaxLabel, float64(maxLabel))
		b.searchWorkPathLabelSpan = append(b.searchWorkPathLabelSpan, float64(maxLabel-minLabel))
		b.searchWorkPathMeanAbsGap = append(b.searchWorkPathMeanAbsGap, meanGap)
		b.searchWorkPathUnique4KBuckets = append(b.searchWorkPathUnique4KBuckets, float64(unique4K))
		b.searchWorkPathMinVisibleAge = append(b.searchWorkPathMinVisibleAge, float64(minVisibleAge))
		b.searchWorkPathMeanVisibleAge = append(b.searchWorkPathMeanVisibleAge, meanVisibleAge)
		b.searchWorkPathRecent1KHits = append(b.searchWorkPathRecent1KHits, float64(recent1KHits))
		b.searchWorkPathRecent4KHits = append(b.searchWorkPathRecent4KHits, float64(recent4KHits))
		b.searchWorkPathRecent16KHits = append(b.searchWorkPathRecent16KHits, float64(recent16KHits))
	}
	if b.config.Workload.PerQueryLatency != nil && *b.config.Workload.PerQueryLatency && len(task.Tags) > 0 {
		queryCount := len(task.Tags)
		perQueryOp := searchOpMs / float64(queryCount)
		perQueryE2E := searchE2EMs / float64(queryCount)
		for i := 0; i < queryCount; i++ {
			b.searchOpLatencies = append(b.searchOpLatencies, perQueryOp)
			b.searchE2ELatencies = append(b.searchE2ELatencies, perQueryE2E)
			b.searchIndexMs = append(b.searchIndexMs, indexMs/float64(queryCount))
			b.searchPostIndexMs = append(b.searchPostIndexMs, postIndexMs/float64(queryCount))
			b.searchResultsLockWaitMs = append(b.searchResultsLockWaitMs, resultsLockWaitMs/float64(queryCount))
			b.searchResultsLockHoldMs = append(b.searchResultsLockHoldMs, resultsLockHoldMs/float64(queryCount))
			b.searchRecordLockWaitMs = append(b.searchRecordLockWaitMs, recordLockWaitMs/float64(queryCount))
			b.searchFinishMs = append(b.searchFinishMs, searchFinishMs)
			b.searchMeasured = append(b.searchMeasured, measured)
			b.searchQueryTags = append(b.searchQueryTags, float64(task.Tags[i]))
			b.searchCreateRawNs = append(b.searchCreateRawNs, task.CreateRawNs)
			b.searchStartRawNs = append(b.searchStartRawNs, searchStartRawNs)
			b.searchFinishRawNs = append(b.searchFinishRawNs, searchFinishRawNs)
			b.searchFreshnessTotalPts = append(b.searchFreshnessTotalPts, float64(totalLagPts))
			b.searchFreshnessContiguityPts = append(b.searchFreshnessContiguityPts, float64(contiguityLagPts))
			b.searchFreshnessQueuePts = append(b.searchFreshnessQueuePts, float64(queueLagPts))
			b.searchFreshnessActiveFrontPts = append(b.searchFreshnessActiveFrontPts, float64(activeFrontLagPts))
			b.searchFreshnessMissedE2EPts = append(b.searchFreshnessMissedE2EPts, float64(missedE2EPts))
			b.searchInflightBruteforcePts = append(b.searchInflightBruteforcePts, float64(inflightBruteforcePts))
			b.searchTaskInsertOffset = append(b.searchTaskInsertOffset, float64(task.InsertOffset))
			b.searchCommittedAtStart = append(b.searchCommittedAtStart, float64(committedAtStart))
			b.searchViewOffset = append(b.searchViewOffset, float64(viewOffset))
			b.searchPhysicalHead = append(b.searchPhysicalHead, float64(physicalHead))
			b.searchInflightOccLatencies = append(b.searchInflightOccLatencies, inflightOccMs/float64(len(task.Data)))
			b.searchInflightOccL2Dists = append(b.searchInflightOccL2Dists, float64(inflightOccL2Dists)/float64(len(task.Data)))
			b.searchInflightOccMergeComps = append(b.searchInflightOccMergeComps, float64(inflightOccMergeComps)/float64(len(task.Data)))
			b.searchInflightOccFilterTests = append(b.searchInflightOccFilterTests, float64(inflightOccFilterTests)/float64(len(task.Data)))
			b.searchInflightOccCandidates = append(b.searchInflightOccCandidates, float64(inflightOccCandidates)/float64(len(task.Data)))
			b.searchDiagDistComps = append(b.searchDiagDistComps, diagDistComps/float64(len(task.Data)))
			b.searchDiagHops = append(b.searchDiagHops, diagHops/float64(len(task.Data)))
			b.searchDiagFutureSkipQueries = append(b.searchDiagFutureSkipQueries, diagFutureSkipQueries/float64(len(task.Data)))
			b.searchDiagFutureSkipHops = append(b.searchDiagFutureSkipHops, diagFutureSkipHops/float64(len(task.Data)))
			b.searchDiagRecoveryTriggers = append(b.searchDiagRecoveryTriggers, diagRecoveryTriggers/float64(len(task.Data)))
			b.searchDiagRecoveredEdges = append(b.searchDiagRecoveredEdges, diagRecoveredEdges/float64(len(task.Data)))
			b.searchDiagUsefulRecovered = append(b.searchDiagUsefulRecovered, diagUsefulRecovered/float64(len(task.Data)))
			b.searchDiagModifiedExpansions = append(b.searchDiagModifiedExpansions, diagModifiedExpansions/float64(len(task.Data)))
			b.searchDiagRecoveryAttempts = append(b.searchDiagRecoveryAttempts, diagRecoveryAttempts/float64(len(task.Data)))
			b.searchDiagRecoveryCandidates = append(b.searchDiagRecoveryCandidates, diagRecoveryCandidates/float64(len(task.Data)))
			b.searchDiagRecoveryGetMs = append(b.searchDiagRecoveryGetMs, diagRecoveryGetMs/float64(len(task.Data)))
			b.searchDiagRecoveryLoopMs = append(b.searchDiagRecoveryLoopMs, diagRecoveryLoopMs/float64(len(task.Data)))
			b.searchDiagRewriteActiveExp = append(b.searchDiagRewriteActiveExp, diagRewriteActiveExp/float64(len(task.Data)))
			b.searchDiagRewriteActiveQuery = append(b.searchDiagRewriteActiveQuery, diagRewriteActiveQuery/float64(len(task.Data)))
			b.searchDiagRewriteRecentExp = append(b.searchDiagRewriteRecentExp, diagRewriteRecentExp/float64(len(task.Data)))
			b.searchDiagRewriteRecentQuery = append(b.searchDiagRewriteRecentQuery, diagRewriteRecentQuery/float64(len(task.Data)))
			b.searchDiagRewritePeriodExp = append(b.searchDiagRewritePeriodExp, diagRewritePeriodExp/float64(len(task.Data)))
			b.searchDiagRewritePeriodQuery = append(b.searchDiagRewritePeriodQuery, diagRewritePeriodQuery/float64(len(task.Data)))
			b.searchDiagRewritePeriodActiveSum = append(b.searchDiagRewritePeriodActiveSum, diagRewritePeriodActiveSum/float64(len(task.Data)))
			b.searchDiagRewritePeriodActiveMax = append(b.searchDiagRewritePeriodActiveMax, diagRewritePeriodActiveMax/float64(len(task.Data)))
			b.searchDiagBatchSearchCalls = append(b.searchDiagBatchSearchCalls, diagBatchSearchCalls/float64(len(task.Data)))
			b.searchDiagBatchSearchQueries = append(b.searchDiagBatchSearchQueries, diagBatchSearchQueries/float64(len(task.Data)))
			b.searchDiagBatchSearchArenaMs = append(b.searchDiagBatchSearchArenaMs, diagBatchSearchArenaMs/float64(len(task.Data)))
			b.searchDiagBatchSearchKnnMs = append(b.searchDiagBatchSearchKnnMs, diagBatchSearchKnnMs/float64(len(task.Data)))
			b.searchDiagBatchSearchCopyMs = append(b.searchDiagBatchSearchCopyMs, diagBatchSearchCopyMs/float64(len(task.Data)))
			b.searchDiagInvisibleExpansions = append(b.searchDiagInvisibleExpansions, diagInvisibleExpansions/float64(len(task.Data)))
			b.searchDiagInvisibleExpansionEdges = append(b.searchDiagInvisibleExpansionEdges, diagInvisibleExpansionEdges/float64(len(task.Data)))
			b.searchDiagInvisibleCandidateDistComps = append(b.searchDiagInvisibleCandidateDistComps, diagInvisibleCandidateDistComps/float64(len(task.Data)))
			b.searchDiagInvisibleCandidateEnqueues = append(b.searchDiagInvisibleCandidateEnqueues, diagInvisibleCandidateEnqueues/float64(len(task.Data)))
			b.searchDiagPhaseOverlapSamples = append(b.searchDiagPhaseOverlapSamples, diagPhaseOverlapSamples/float64(len(task.Data)))
			b.searchDiagPhaseExistingUpdateSamples = append(b.searchDiagPhaseExistingUpdateSamples, diagPhaseExistingUpdateSamples/float64(len(task.Data)))
			b.searchDiagPhaseExistingUpdateQueries = append(b.searchDiagPhaseExistingUpdateQueries, diagPhaseExistingUpdateQueries/float64(len(task.Data)))
			b.searchDiagPhaseLinkCriticalSamples = append(b.searchDiagPhaseLinkCriticalSamples, diagPhaseLinkCriticalSamples/float64(len(task.Data)))
			b.searchDiagPhaseLinkCriticalQueries = append(b.searchDiagPhaseLinkCriticalQueries, diagPhaseLinkCriticalQueries/float64(len(task.Data)))
			b.searchDiagPhaseLoadScanSamples = append(b.searchDiagPhaseLoadScanSamples, diagPhaseLoadScanSamples/float64(len(task.Data)))
			b.searchDiagPhaseLoadScanQueries = append(b.searchDiagPhaseLoadScanQueries, diagPhaseLoadScanQueries/float64(len(task.Data)))
			b.searchDiagPhaseAppendSamples = append(b.searchDiagPhaseAppendSamples, diagPhaseAppendSamples/float64(len(task.Data)))
			b.searchDiagPhaseAppendQueries = append(b.searchDiagPhaseAppendQueries, diagPhaseAppendQueries/float64(len(task.Data)))
			b.searchDiagPhasePruneSamples = append(b.searchDiagPhasePruneSamples, diagPhasePruneSamples/float64(len(task.Data)))
			b.searchDiagPhasePruneQueries = append(b.searchDiagPhasePruneQueries, diagPhasePruneQueries/float64(len(task.Data)))
			b.searchDiagPhaseUndoRecordSamples = append(b.searchDiagPhaseUndoRecordSamples, diagPhaseUndoRecordSamples/float64(len(task.Data)))
			b.searchDiagPhaseUndoRecordQueries = append(b.searchDiagPhaseUndoRecordQueries, diagPhaseUndoRecordQueries/float64(len(task.Data)))
			b.searchDiagPhaseRewriteSamples = append(b.searchDiagPhaseRewriteSamples, diagPhaseRewriteSamples/float64(len(task.Data)))
			b.searchDiagPhaseRewriteQueries = append(b.searchDiagPhaseRewriteQueries, diagPhaseRewriteQueries/float64(len(task.Data)))
			appendSearchWork(workAt(i), viewOffset)
		}
	} else {
		batchWork := SearchWorkStats{}
		for _, work := range searchWork {
			batchWork.SearchKnnNs += work.SearchKnnNs
			batchWork.SearchKnnThreadCpuNs += work.SearchKnnThreadCpuNs
			batchWork.ResultCopyNs += work.ResultCopyNs
			batchWork.EntryNs += work.EntryNs
			batchWork.UpperSearchNs += work.UpperSearchNs
			batchWork.BaseSearchNs += work.BaseSearchNs
			batchWork.ResultMaterializeNs += work.ResultMaterializeNs
			batchWork.SnapshotGuardNs += work.SnapshotGuardNs
			batchWork.VisitedListGetNs += work.VisitedListGetNs
			batchWork.VisitedListReleaseNs += work.VisitedListReleaseNs
			batchWork.UpperLockWaitNs += work.UpperLockWaitNs
			batchWork.Level0LockWaitNs += work.Level0LockWaitNs
			batchWork.DistanceComputations += work.DistanceComputations
			batchWork.UpperDistanceComputations += work.UpperDistanceComputations
			batchWork.Level0DistanceComputations += work.Level0DistanceComputations
			batchWork.UpperHops += work.UpperHops
			batchWork.UpperEdgesScanned += work.UpperEdgesScanned
			batchWork.Level0Expansions += work.Level0Expansions
			batchWork.Level0EdgesScanned += work.Level0EdgesScanned
			batchWork.CandidatePops += work.CandidatePops
			batchWork.CandidatePushes += work.CandidatePushes
			batchWork.VisitedNodes += work.VisitedNodes
			batchWork.ResultPushes += work.ResultPushes
			batchWork.InvisibleExpansions += work.InvisibleExpansions
			batchWork.InvisibleEdges += work.InvisibleEdges
			batchWork.InvisibleCandidateDistComps += work.InvisibleCandidateDistComps
			batchWork.InvisibleCandidateEnqueues += work.InvisibleCandidateEnqueues
			batchWork.FutureSkipHops += work.FutureSkipHops
			batchWork.RewriteActiveExpansions += work.RewriteActiveExpansions
			batchWork.RewriteRecentExpansions += work.RewriteRecentExpansions
			batchWork.RewritePeriodExpansions += work.RewritePeriodExpansions
			batchWork.RewritePeriodActiveSum += work.RewritePeriodActiveSum
			if work.RewritePeriodActiveMax > batchWork.RewritePeriodActiveMax {
				batchWork.RewritePeriodActiveMax = work.RewritePeriodActiveMax
			}
			batchWork.ExpandVisibleCount += work.ExpandVisibleCount
			batchWork.ExpandRecent1KHits += work.ExpandRecent1KHits
			batchWork.ExpandRecent4KHits += work.ExpandRecent4KHits
			batchWork.ExpandRecent16KHits += work.ExpandRecent16KHits
			batchWork.ExpandLabelGapSum += work.ExpandLabelGapSum
			batchWork.ExpandLabelSpan += work.ExpandLabelSpan
			batchWork.ExpandUniqueLabel4KBuckets += work.ExpandUniqueLabel4KBuckets
			batchWork.ExpandUniqueData4KPages += work.ExpandUniqueData4KPages
			batchWork.ExpandUniqueData2MPages += work.ExpandUniqueData2MPages
			batchWork.ExpandUniqueAdj4KPages += work.ExpandUniqueAdj4KPages
			batchWork.ExpandUniqueAdj2MPages += work.ExpandUniqueAdj2MPages
			batchWork.ExpandUniqueOverflow += work.ExpandUniqueOverflow
		}
		b.searchOpLatencies = append(b.searchOpLatencies, searchOpMs)
		b.searchE2ELatencies = append(b.searchE2ELatencies, searchE2EMs)
		b.searchIndexMs = append(b.searchIndexMs, indexMs)
		b.searchPostIndexMs = append(b.searchPostIndexMs, postIndexMs)
		b.searchResultsLockWaitMs = append(b.searchResultsLockWaitMs, resultsLockWaitMs)
		b.searchResultsLockHoldMs = append(b.searchResultsLockHoldMs, resultsLockHoldMs)
		b.searchRecordLockWaitMs = append(b.searchRecordLockWaitMs, recordLockWaitMs)
		b.searchFinishMs = append(b.searchFinishMs, searchFinishMs)
		b.searchMeasured = append(b.searchMeasured, measured)
		queryTag := float64(0)
		if len(task.Tags) > 0 {
			queryTag = float64(task.Tags[0])
		}
		b.searchQueryTags = append(b.searchQueryTags, queryTag)
		b.searchCreateRawNs = append(b.searchCreateRawNs, task.CreateRawNs)
		b.searchStartRawNs = append(b.searchStartRawNs, searchStartRawNs)
		b.searchFinishRawNs = append(b.searchFinishRawNs, searchFinishRawNs)
		b.searchFreshnessTotalPts = append(b.searchFreshnessTotalPts, float64(totalLagPts))
		b.searchFreshnessContiguityPts = append(b.searchFreshnessContiguityPts, float64(contiguityLagPts))
		b.searchFreshnessQueuePts = append(b.searchFreshnessQueuePts, float64(queueLagPts))
		b.searchFreshnessActiveFrontPts = append(b.searchFreshnessActiveFrontPts, float64(activeFrontLagPts))
		b.searchFreshnessMissedE2EPts = append(b.searchFreshnessMissedE2EPts, float64(missedE2EPts))
		b.searchInflightBruteforcePts = append(b.searchInflightBruteforcePts, float64(inflightBruteforcePts))
		b.searchTaskInsertOffset = append(b.searchTaskInsertOffset, float64(task.InsertOffset))
		b.searchCommittedAtStart = append(b.searchCommittedAtStart, float64(committedAtStart))
		b.searchViewOffset = append(b.searchViewOffset, float64(viewOffset))
		b.searchPhysicalHead = append(b.searchPhysicalHead, float64(physicalHead))
		b.searchInflightOccLatencies = append(b.searchInflightOccLatencies, inflightOccMs)
		b.searchInflightOccL2Dists = append(b.searchInflightOccL2Dists, float64(inflightOccL2Dists))
		b.searchInflightOccMergeComps = append(b.searchInflightOccMergeComps, float64(inflightOccMergeComps))
		b.searchInflightOccFilterTests = append(b.searchInflightOccFilterTests, float64(inflightOccFilterTests))
		b.searchInflightOccCandidates = append(b.searchInflightOccCandidates, float64(inflightOccCandidates))
		b.searchDiagDistComps = append(b.searchDiagDistComps, diagDistComps)
		b.searchDiagHops = append(b.searchDiagHops, diagHops)
		b.searchDiagFutureSkipQueries = append(b.searchDiagFutureSkipQueries, diagFutureSkipQueries)
		b.searchDiagFutureSkipHops = append(b.searchDiagFutureSkipHops, diagFutureSkipHops)
		b.searchDiagRecoveryTriggers = append(b.searchDiagRecoveryTriggers, diagRecoveryTriggers)
		b.searchDiagRecoveredEdges = append(b.searchDiagRecoveredEdges, diagRecoveredEdges)
		b.searchDiagUsefulRecovered = append(b.searchDiagUsefulRecovered, diagUsefulRecovered)
		b.searchDiagModifiedExpansions = append(b.searchDiagModifiedExpansions, diagModifiedExpansions)
		b.searchDiagRecoveryAttempts = append(b.searchDiagRecoveryAttempts, diagRecoveryAttempts)
		b.searchDiagRecoveryCandidates = append(b.searchDiagRecoveryCandidates, diagRecoveryCandidates)
		b.searchDiagRecoveryGetMs = append(b.searchDiagRecoveryGetMs, diagRecoveryGetMs)
		b.searchDiagRecoveryLoopMs = append(b.searchDiagRecoveryLoopMs, diagRecoveryLoopMs)
		b.searchDiagRewriteActiveExp = append(b.searchDiagRewriteActiveExp, diagRewriteActiveExp)
		b.searchDiagRewriteActiveQuery = append(b.searchDiagRewriteActiveQuery, diagRewriteActiveQuery)
		b.searchDiagRewriteRecentExp = append(b.searchDiagRewriteRecentExp, diagRewriteRecentExp)
		b.searchDiagRewriteRecentQuery = append(b.searchDiagRewriteRecentQuery, diagRewriteRecentQuery)
		b.searchDiagRewritePeriodExp = append(b.searchDiagRewritePeriodExp, diagRewritePeriodExp)
		b.searchDiagRewritePeriodQuery = append(b.searchDiagRewritePeriodQuery, diagRewritePeriodQuery)
		b.searchDiagRewritePeriodActiveSum = append(b.searchDiagRewritePeriodActiveSum, diagRewritePeriodActiveSum)
		b.searchDiagRewritePeriodActiveMax = append(b.searchDiagRewritePeriodActiveMax, diagRewritePeriodActiveMax)
		b.searchDiagBatchSearchCalls = append(b.searchDiagBatchSearchCalls, diagBatchSearchCalls)
		b.searchDiagBatchSearchQueries = append(b.searchDiagBatchSearchQueries, diagBatchSearchQueries)
		b.searchDiagBatchSearchArenaMs = append(b.searchDiagBatchSearchArenaMs, diagBatchSearchArenaMs)
		b.searchDiagBatchSearchKnnMs = append(b.searchDiagBatchSearchKnnMs, diagBatchSearchKnnMs)
		b.searchDiagBatchSearchCopyMs = append(b.searchDiagBatchSearchCopyMs, diagBatchSearchCopyMs)
		b.searchDiagInvisibleExpansions = append(b.searchDiagInvisibleExpansions, diagInvisibleExpansions)
		b.searchDiagInvisibleExpansionEdges = append(b.searchDiagInvisibleExpansionEdges, diagInvisibleExpansionEdges)
		b.searchDiagInvisibleCandidateDistComps = append(b.searchDiagInvisibleCandidateDistComps, diagInvisibleCandidateDistComps)
		b.searchDiagInvisibleCandidateEnqueues = append(b.searchDiagInvisibleCandidateEnqueues, diagInvisibleCandidateEnqueues)
		b.searchDiagPhaseOverlapSamples = append(b.searchDiagPhaseOverlapSamples, diagPhaseOverlapSamples)
		b.searchDiagPhaseExistingUpdateSamples = append(b.searchDiagPhaseExistingUpdateSamples, diagPhaseExistingUpdateSamples)
		b.searchDiagPhaseExistingUpdateQueries = append(b.searchDiagPhaseExistingUpdateQueries, diagPhaseExistingUpdateQueries)
		b.searchDiagPhaseLinkCriticalSamples = append(b.searchDiagPhaseLinkCriticalSamples, diagPhaseLinkCriticalSamples)
		b.searchDiagPhaseLinkCriticalQueries = append(b.searchDiagPhaseLinkCriticalQueries, diagPhaseLinkCriticalQueries)
		b.searchDiagPhaseLoadScanSamples = append(b.searchDiagPhaseLoadScanSamples, diagPhaseLoadScanSamples)
		b.searchDiagPhaseLoadScanQueries = append(b.searchDiagPhaseLoadScanQueries, diagPhaseLoadScanQueries)
		b.searchDiagPhaseAppendSamples = append(b.searchDiagPhaseAppendSamples, diagPhaseAppendSamples)
		b.searchDiagPhaseAppendQueries = append(b.searchDiagPhaseAppendQueries, diagPhaseAppendQueries)
		b.searchDiagPhasePruneSamples = append(b.searchDiagPhasePruneSamples, diagPhasePruneSamples)
		b.searchDiagPhasePruneQueries = append(b.searchDiagPhasePruneQueries, diagPhasePruneQueries)
		b.searchDiagPhaseUndoRecordSamples = append(b.searchDiagPhaseUndoRecordSamples, diagPhaseUndoRecordSamples)
		b.searchDiagPhaseUndoRecordQueries = append(b.searchDiagPhaseUndoRecordQueries, diagPhaseUndoRecordQueries)
		b.searchDiagPhaseRewriteSamples = append(b.searchDiagPhaseRewriteSamples, diagPhaseRewriteSamples)
		b.searchDiagPhaseRewriteQueries = append(b.searchDiagPhaseRewriteQueries, diagPhaseRewriteQueries)
		appendSearchWork(batchWork, viewOffset)
	}
	b.searchPointCnt += len(task.Data)
	b.searchMu.Unlock()
}

func (b *Bench) printProgress(totalInsert int) {
	if totalInsert <= 0 {
		log.Printf("Progress: no insert workload; tracking search-only schedule")
		return
	}
	lastDecade := -1
	for {
		current := int(atomic.LoadInt64(&b.globalInsertCnt))
		percent := current * 100 / totalInsert
		decade := percent / 10
		if decade != lastDecade {
			elapsed := time.Since(b.startTime).Seconds()
			insertQPS := float64(current) / elapsed
			b.searchMu.Lock()
			searchCnt := b.searchPointCnt
			b.searchMu.Unlock()
			searchQPS := float64(searchCnt) / elapsed
			committed := b.committedOffset.Load()
			offsetLag := int64(current) - int64(committed) + int64(b.config.Data.BeginNum)
			sLagCount := atomic.LoadInt64(&b.searchLagCount)
			sLagMax := atomic.LoadInt64(&b.searchLagMax)
			sLagContigMax := atomic.LoadInt64(&b.searchLagContiguityMax)
			sLagQueueMax := atomic.LoadInt64(&b.searchLagQueueMax)
			var sLagAvg float64
			var sLagContigAvg float64
			var sLagQueueAvg float64
			if sLagCount > 0 {
				sLagAvg = float64(atomic.LoadInt64(&b.searchLagSum)) / float64(sLagCount)
				sLagContigAvg = float64(atomic.LoadInt64(&b.searchLagContiguitySum)) / float64(sLagCount)
				sLagQueueAvg = float64(atomic.LoadInt64(&b.searchLagQueueSum)) / float64(sLagCount)
			}
			log.Printf("Progress: %d/%d (%d%%), Insert QPS: %.2f, Search QPS: %.2f, offset_lag: %d, freshness_total(avg/max): %.0f/%d pts, contiguity(avg/max): %.0f/%d pts, queue(avg/max): %.0f/%d pts",
				current, totalInsert, percent, insertQPS, searchQPS, offsetLag, sLagAvg, sLagMax, sLagContigAvg, sLagContigMax, sLagQueueAvg, sLagQueueMax)
			lastDecade = decade
		}
		if current >= totalInsert {
			break
		}
		time.Sleep(1 * time.Second)
	}
}

func (b *Bench) calcStats(elapsedSec float64, memStats memResult) Stat {
	s := Stat{}

	// OCC progress 2026-04-22: optional raw latency dump for fluctuation analysis.
	// Enable by setting env var OCC_RAW_LATENCY_DIR=<dir>. Writes per-sample arrays
	// as CSV: insert_op.csv, insert_e2e.csv, search_op.csv, search_e2e.csv plus a
	// combined time-indexed search.csv when PerQueryLatency was on.
	if rawDir := os.Getenv("OCC_RAW_LATENCY_DIR"); rawDir != "" {
		if err := b.dumpRawLatencies(rawDir); err != nil {
			log.Printf("[raw-latency] dump failed: %v", err)
		} else {
			log.Printf("[raw-latency] dumped to %s", rawDir)
		}
	}

	if b.insertPointCnt > 0 {
		s.InsertQPS = float64(b.insertPointCnt) / elapsedSec
		s.MeanInsertOpLatency = mean(b.insertOpLatencies)
		s.P95InsertOpLatency = percentile(b.insertOpLatencies, 0.95)
		s.P99InsertOpLatency = percentile(b.insertOpLatencies, 0.99)
		s.MeanInsertE2ELatency = mean(b.insertE2ELatencies)
		s.P95InsertE2ELatency = percentile(b.insertE2ELatencies, 0.95)
		s.P99InsertE2ELatency = percentile(b.insertE2ELatencies, 0.99)
	}

	if b.searchPointCnt > 0 {
		s.SearchQPS = float64(b.searchPointCnt) / elapsedSec
		s.MeanSearchOpLatency = mean(b.searchOpLatencies)
		s.P95SearchOpLatency = percentile(b.searchOpLatencies, 0.95)
		s.P99SearchOpLatency = percentile(b.searchOpLatencies, 0.99)
		s.MeanSearchE2ELatency = mean(b.searchE2ELatencies)
		s.P95SearchE2ELatency = percentile(b.searchE2ELatencies, 0.95)
		s.P99SearchE2ELatency = percentile(b.searchE2ELatencies, 0.99)
		s.MeanFreshnessTotalPts = mean(b.searchFreshnessTotalPts)
		s.P95FreshnessTotalPts = percentile(b.searchFreshnessTotalPts, 0.95)
		s.P99FreshnessTotalPts = percentile(b.searchFreshnessTotalPts, 0.99)
		s.MeanFreshnessContiguityPts = mean(b.searchFreshnessContiguityPts)
		s.P95FreshnessContiguityPts = percentile(b.searchFreshnessContiguityPts, 0.95)
		s.P99FreshnessContiguityPts = percentile(b.searchFreshnessContiguityPts, 0.99)
		s.MeanFreshnessQueuePts = mean(b.searchFreshnessQueuePts)
		s.P95FreshnessQueuePts = percentile(b.searchFreshnessQueuePts, 0.95)
		s.P99FreshnessQueuePts = percentile(b.searchFreshnessQueuePts, 0.99)
		s.MeanFreshnessActiveFrontPts = mean(b.searchFreshnessActiveFrontPts)
		s.P95FreshnessActiveFrontPts = percentile(b.searchFreshnessActiveFrontPts, 0.95)
		s.P99FreshnessActiveFrontPts = percentile(b.searchFreshnessActiveFrontPts, 0.99)
		s.MeanFreshnessMissedE2EPts = mean(b.searchFreshnessMissedE2EPts)
		s.P95FreshnessMissedE2EPts = percentile(b.searchFreshnessMissedE2EPts, 0.95)
		s.P99FreshnessMissedE2EPts = percentile(b.searchFreshnessMissedE2EPts, 0.99)
		s.InflightBruteforceEnabled = b.config.Search.EnableInflightBruteforce != nil && *b.config.Search.EnableInflightBruteforce
		s.InflightJoinEnabled = b.config.Search.EnableInflightJoin != nil && *b.config.Search.EnableInflightJoin
		s.MeanInflightBruteforcePts = mean(b.searchInflightBruteforcePts)
		s.P95InflightBruteforcePts = percentile(b.searchInflightBruteforcePts, 0.95)
		s.P99InflightBruteforcePts = percentile(b.searchInflightBruteforcePts, 0.99)
		s.MeanInflightOccMergeLatency = mean(b.searchInflightOccLatencies)
		s.P95InflightOccMergeLatency = percentile(b.searchInflightOccLatencies, 0.95)
		s.P99InflightOccMergeLatency = percentile(b.searchInflightOccLatencies, 0.99)
		if s.MeanSearchOpLatency > 0 {
			s.MeanInflightOccShare = s.MeanInflightOccMergeLatency / s.MeanSearchOpLatency
		}
		s.MeanInflightOccL2Dists = mean(b.searchInflightOccL2Dists)
		s.MeanInflightOccMergeComps = mean(b.searchInflightOccMergeComps)
		s.MeanInflightOccFilterTests = mean(b.searchInflightOccFilterTests)
		s.MeanInflightOccCandidates = mean(b.searchInflightOccCandidates)
		s.ResultBelowKCount = b.resultBelowKCount
	}

	if memStats.samplesCount > 0 {
		s.PeakMemoryMB = float64(memStats.peakUsage) / (1024 * 1024)
		s.AvgMemoryMB = memStats.avgUsage / (1024 * 1024)
	}
	if b.replicaRefresh != nil {
		s.ExtraStats = b.replicaRefresh.statsString()
	}
	if seen := b.m2VRouteDiagSeenQueries.Load(); seen > 0 {
		m2vRouteStats := fmt.Sprintf(
			"m2v_route_seen_queries:%d, m2v_route_no_route_queries:%d, m2v_route_candidate_regions:%d, m2v_route_opened_regions:%d, m2v_route_opened_postings:%d, m2v_member_bound_checks:%d, m2v_member_bound_skips:%d, m2v_hub_suppressed_regions:%d, m2v_wide_suppressed_regions:%d, m2v_weak_suppressed_regions:%d, m2v_strong_region_hits:%d, m2v_medium_region_hits:%d, m2v_weak_region_hits:%d",
			seen,
			b.m2VRouteDiagNoRouteQueries.Load(),
			b.m2VRouteDiagCandidateRegions.Load(),
			b.m2VRouteDiagOpenedRegions.Load(),
			b.m2VRouteDiagOpenedPostings.Load(),
			b.m2VRouteDiagMemberBoundChecks.Load(),
			b.m2VRouteDiagMemberBoundSkips.Load(),
			b.m2VRouteDiagHubSuppressed.Load(),
			b.m2VRouteDiagWideSuppressed.Load(),
			b.m2VRouteDiagWeakSuppressed.Load(),
			b.m2VRouteDiagStrongRegionHits.Load(),
			b.m2VRouteDiagMediumRegionHits.Load(),
			b.m2VRouteDiagWeakRegionHits.Load(),
		)
		if s.ExtraStats != "" {
			s.ExtraStats += ", "
		}
		s.ExtraStats += m2vRouteStats
	}
	if seen := b.m2VValidateSeen.Load(); seen > 0 {
		m2vStats := fmt.Sprintf(
			"m2v_validate_fn_seen_queries:%d, m2v_validate_fn_queries:%d, m2v_validate_fn_bruteforce_fresh:%d, m2v_validate_fn_fresh_winners:%d, m2v_validate_fn_missed_fresh_winners:%d, m2v_validate_fn_pending_winners:%d, m2v_validate_fn_active_region_winners:%d, m2v_validate_fn_opened_region_winners:%d, m2v_validate_fn_triangle_pruned_winners:%d",
			seen,
			b.m2VValidateRuns.Load(),
			b.m2VValidateFresh.Load(),
			b.m2VValidateWins.Load(),
			b.m2VValidateMiss.Load(),
			b.m2VValidatePendingWinner.Load(),
			b.m2VValidateActiveRegionWinner.Load(),
			b.m2VValidateOpenedRegionWinner.Load(),
			b.m2VValidateTrianglePrunedWinner.Load(),
		)
		if s.ExtraStats != "" {
			s.ExtraStats += ", "
		}
		s.ExtraStats += m2vStats
	}
	if seen := b.m2PathOneHopValidateSeen.Load(); seen > 0 {
		pathStats := fmt.Sprintf(
			"m2_path_onehop_validate_fn_seen_queries:%d, m2_path_onehop_validate_fn_queries:%d, m2_path_onehop_validate_fn_bruteforce_fresh:%d, m2_path_onehop_validate_fn_fresh_winners:%d, m2_path_onehop_validate_fn_missed_fresh_winners:%d, m2_path_onehop_validate_fn_pending_winners:%d, m2_path_onehop_validate_fn_selected_1hop_winners:%d, m2_path_onehop_validate_fn_all_path_1hop_winners:%d, m2_path_onehop_validate_fn_missed_all_path_1hop:%d, m2_path_onehop_validate_fn_missed_path_event_self:%d, m2_path_onehop_validate_fn_missed_selected_2hop:%d, m2_path_onehop_validate_fn_missed_all_path_2hop:%d, m2_path_onehop_validate_fn_missed_monotone_2hop:%d, m2_path_onehop_validate_fn_missed_radius_2hop:%d, m2_path_onehop_validate_fn_missed_supported_2hop:%d, m2_path_onehop_validate_fn_missed_recip_source_2hop:%d, m2_path_onehop_validate_fn_missed_recip_target_2hop:%d",
			seen,
			b.m2PathOneHopValidateRuns.Load(),
			b.m2PathOneHopValidateFresh.Load(),
			b.m2PathOneHopValidateWins.Load(),
			b.m2PathOneHopValidateMiss.Load(),
			b.m2PathOneHopValidatePendingWinner.Load(),
			b.m2PathOneHopValidateSelectedOneHop.Load(),
			b.m2PathOneHopValidateAllPathOneHop.Load(),
			b.m2PathOneHopValidateMissAllPathOneHop.Load(),
			b.m2PathOneHopValidateMissPathEventSelf.Load(),
			b.m2PathOneHopValidateMissSelectedTwoHop.Load(),
			b.m2PathOneHopValidateMissAllPathTwoHop.Load(),
			b.m2PathOneHopValidateMissMonotoneTwoHop.Load(),
			b.m2PathOneHopValidateMissRadiusTwoHop.Load(),
			b.m2PathOneHopValidateMissSupportedTwoHop.Load(),
			b.m2PathOneHopValidateMissRecipSourceTwoHop.Load(),
			b.m2PathOneHopValidateMissRecipTargetTwoHop.Load(),
		)
		if s.ExtraStats != "" {
			s.ExtraStats += ", "
		}
		s.ExtraStats += pathStats
	}

	// Print lock wait stats
	if len(b.searchLockWaits) > 0 {
		sort.Float64s(b.searchLockWaits)
		n := len(b.searchLockWaits)
		log.Printf("RWLock search wait: mean=%.2fms p95=%.2fms p99=%.2fms max=%.2fms count=%d (>1ms: %d)",
			mean(b.searchLockWaits),
			percentile(b.searchLockWaits, 0.95),
			percentile(b.searchLockWaits, 0.99),
			b.searchLockWaits[n-1],
			n,
			atomic.LoadInt64(&b.searchLockWaitCount))
	}

	return s
}

func monitorMemUsage(interval time.Duration, done <-chan struct{}) <-chan memResult {
	resultChan := make(chan memResult, 1)
	var samples []uint64

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				currentMem := GetCurrentMemoryUsage()
				samples = append(samples, currentMem)
			case <-done:
				if len(samples) == 0 {
					resultChan <- memResult{}
					return
				}

				var peak uint64
				var sum uint64
				for _, s := range samples {
					sum += s
					if s > peak {
						peak = s
					}
				}
				avg := float64(sum) / float64(len(samples))

				resultChan <- memResult{
					peakUsage:    peak,
					avgUsage:     avg,
					samplesCount: len(samples),
				}
				return
			}
		}
	}()

	return resultChan
}

func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	idx := int(float64(len(sorted)-1) * p)
	return sorted[idx]
}

// dumpRawLatencies writes per-sample latency arrays to CSV files for
// fluctuation / jitter analysis. Called when OCC_RAW_LATENCY_DIR env is set.
// Format: one value per line, header row identifying the column.
//
// When PerQueryLatency was enabled, the search arrays are per-query
// (amortized from per-batch by dividing by batch size); each row also carries
// the freshness and OCC merge telemetry captured at that sample.
func (b *Bench) dumpRawLatencies(dir string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("mkdir %s: %v", dir, err)
	}

	meta, err := os.Create(filepath.Join(dir, "meta.csv"))
	if err != nil {
		return err
	}
	metaW := bufio.NewWriter(meta)
	fmt.Fprintln(metaW, "key,value")
	fmt.Fprintf(metaW, "bench_start_unix_ns,%d\n", b.startTime.UnixNano())
	fmt.Fprintf(metaW, "bench_start_raw_ns,%d\n", b.startRawNs)
	fmt.Fprintf(metaW, "dump_unix_ns,%d\n", time.Now().UnixNano())
	metaW.Flush()
	meta.Close()

	writeFloatCol := func(path, header string, data []float64) error {
		f, err := os.Create(path)
		if err != nil {
			return err
		}
		defer f.Close()
		w := bufio.NewWriter(f)
		defer w.Flush()
		fmt.Fprintf(w, "%s\n", header)
		for _, v := range data {
			fmt.Fprintf(w, "%.6f\n", v)
		}
		return nil
	}

	// Insert-side raw columns.
	if len(b.insertOpLatencies) > 0 {
		if err := writeFloatCol(filepath.Join(dir, "insert_op_ms.csv"),
			"insert_op_ms", b.insertOpLatencies); err != nil {
			return err
		}
	}
	if len(b.insertE2ELatencies) > 0 {
		if err := writeFloatCol(filepath.Join(dir, "insert_e2e_ms.csv"),
			"insert_e2e_ms", b.insertE2ELatencies); err != nil {
			return err
		}
	}
	if len(b.insertOpLatencies) > 0 {
		f, err := os.Create(filepath.Join(dir, "insert_wide.csv"))
		if err != nil {
			return err
		}
		defer f.Close()
		w := bufio.NewWriter(f)
		defer w.Flush()
		insertWideHeaders := []string{
			"sample_idx",
			"finish_ms",
			"create_raw_ns",
			"start_raw_ns",
			"finish_raw_ns",
			"op_ms",
			"e2e_ms",
			"graph_insert_path_ms",
			"graph_insert_path_edges_scanned",
			"graph_insert_path_dist_comps",
			"graph_existing_neighbor_update_loop_ms",
			"graph_existing_neighbor_prune_ms",
			"graph_existing_neighbor_visits",
			"graph_existing_neighbor_loaded_edges",
			"graph_existing_neighbor_prune_calls",
			"graph_existing_neighbor_prune_candidates",
			"graph_existing_neighbor_prune_dist_comps",
			"graph_existing_neighbor_edges_written",
			"graph_existing_neighbor_edges_pruned",
			"graph_existing_neighbor_undo_edges_recorded",
			"graph_unique_lock_wait_ms",
			"graph_link_critical_ms",
			"graph_connect_ms",
			"graph_inserted_node_link_ms",
			"graph_select_new_neighbors_ms",
			"graph_existing_neighbor_load_scan_ms",
			"graph_existing_neighbor_append_ms",
			"graph_existing_neighbor_rewrite_ms",
			"graph_existing_neighbor_undo_record_ms",
			"graph_search_unique_lock_wait_ms",
			"graph_search_unique_lock_acqs",
			"graph_search_critical_ms",
			"batch_insert_wall_ms",
		}
		fmt.Fprintln(w, strings.Join(insertWideHeaders, ","))
		getf := func(arr []float64, i int) float64 {
			if i < len(arr) {
				return arr[i]
			}
			return 0
		}
		getu := func(arr []uint64, i int) uint64 {
			if i < len(arr) {
				return arr[i]
			}
			return 0
		}
		for i := 0; i < len(b.insertOpLatencies); i++ {
			values := []float64{
				getf(b.insertOpLatencies, i),
				getf(b.insertE2ELatencies, i),
				getf(b.insertGraphUpperSearchMs, i) + getf(b.insertGraphBaseSearchMs, i),
				getf(b.insertGraphUpperSearchEdgesScanned, i) + getf(b.insertGraphBaseSearchEdgesScanned, i),
				getf(b.insertGraphUpperSearchDistComps, i) + getf(b.insertGraphBaseSearchDistComps, i),
				getf(b.insertGraphExistingNeighborUpdateLoopMs, i),
				getf(b.insertGraphExistingNeighborPruneMs, i),
				getf(b.insertGraphExistingNeighborVisits, i),
				getf(b.insertGraphExistingNeighborLoadedEdges, i),
				getf(b.insertGraphExistingNeighborPrunes, i),
				getf(b.insertGraphExistingNeighborPruneCandidates, i),
				getf(b.insertGraphExistingNeighborPruneDistComps, i),
				getf(b.insertGraphExistingNeighborEdgesWritten, i),
				getf(b.insertGraphExistingNeighborEdgesPruned, i),
				getf(b.insertGraphExistingNeighborPrunedEdgesRecorded, i),
				getf(b.insertGraphUniqueLockWaitMs, i),
				getf(b.insertGraphLinkCriticalMs, i),
				getf(b.insertGraphConnectMs, i),
				getf(b.insertGraphInsertedNodeLinkMs, i),
				getf(b.insertGraphSelectNewNeighborsMs, i),
				getf(b.insertGraphExistingNeighborLoadScanMs, i),
				getf(b.insertGraphExistingNeighborAppendMs, i),
				getf(b.insertGraphExistingNeighborRewriteMs, i),
				getf(b.insertGraphExistingNeighborUndoRecordMs, i),
				getf(b.insertGraphSearchUniqueLockWaitMs, i),
				getf(b.insertGraphSearchUniqueLockAcqs, i),
				getf(b.insertGraphSearchCriticalMs, i),
				getf(b.insertBatchInsertWallMs, i),
			}
			fmt.Fprintf(w, "%d,%.6f,%d,%d,%d", i, getf(b.insertFinishMs, i),
				getu(b.insertCreateRawNs, i), getu(b.insertStartRawNs, i),
				getu(b.insertFinishRawNs, i))
			for _, v := range values {
				fmt.Fprintf(w, ",%.6f", v)
			}
			fmt.Fprintln(w)
		}
	}
	if len(b.commitEventMs) > 0 {
		f, err := os.Create(filepath.Join(dir, "commit_events.csv"))
		if err != nil {
			return err
		}
		defer f.Close()
		w := bufio.NewWriter(f)
		defer w.Flush()
		fmt.Fprintln(w, "event_idx,commit_ms,committed_offset")
		for i := 0; i < len(b.commitEventMs); i++ {
			offset := uint64(0)
			if i < len(b.commitEventOffset) {
				offset = b.commitEventOffset[i]
			}
			fmt.Fprintf(w, "%d,%.6f,%d\n", i, b.commitEventMs[i], offset)
		}
	}

	// Search-side raw columns (may be per-query or per-batch depending on
	// PerQueryLatency flag; see benchmark worker).
	if len(b.searchOpLatencies) > 0 {
		if err := writeFloatCol(filepath.Join(dir, "search_op_ms.csv"),
			"search_op_ms", b.searchOpLatencies); err != nil {
			return err
		}
	}
	if len(b.searchE2ELatencies) > 0 {
		if err := writeFloatCol(filepath.Join(dir, "search_e2e_ms.csv"),
			"search_e2e_ms", b.searchE2ELatencies); err != nil {
			return err
		}
	}

	n := len(b.searchOpLatencies)
	if n > 0 {
		f, err := os.Create(filepath.Join(dir, "search_compact.csv"))
		if err != nil {
			return err
		}
		defer f.Close()
		w := bufio.NewWriter(f)
		defer w.Flush()
		fmt.Fprintln(w, "sample_idx,measured,query_tag,finish_ms,create_raw_ns,start_raw_ns,finish_raw_ns,op_ms,e2e_ms")
		getf := func(arr []float64, i int) float64 {
			if i < len(arr) {
				return arr[i]
			}
			return 0
		}
		getu := func(arr []uint64, i int) uint64 {
			if i < len(arr) {
				return arr[i]
			}
			return 0
		}
		for i := 0; i < n; i++ {
			fmt.Fprintf(w, "%d,%.0f,%.0f,%.6f,%d,%d,%d,%.6f,%.6f\n",
				i, getf(b.searchMeasured, i), getf(b.searchQueryTags, i), getf(b.searchFinishMs, i),
				getu(b.searchCreateRawNs, i), getu(b.searchStartRawNs, i),
				getu(b.searchFinishRawNs, i), getf(b.searchOpLatencies, i),
				getf(b.searchE2ELatencies, i))
		}
	}
	if os.Getenv("RAW_LATENCY_SKIP_SEARCH_WIDE") == "1" {
		return nil
	}

	// Joined wide CSV for correlated analysis (search side only; one row per sample).
	if n > 0 {
		f, err := os.Create(filepath.Join(dir, "search_wide.csv"))
		if err != nil {
			return err
		}
		defer f.Close()
		w := bufio.NewWriter(f)
		defer w.Flush()
		searchWideHeaders := []string{
			"sample_idx",
			"measured",
			"query_tag",
			"finish_ms",
			"create_raw_ns",
			"start_raw_ns",
			"finish_raw_ns",
			"task_insert_offset",
			"committed_at_start",
			"view_offset",
			"physical_head",
			"op_ms",
			"e2e_ms",
			"search_index_ms",
			"search_post_index_ms",
			"search_results_lock_wait_ms",
			"search_results_lock_hold_ms",
			"search_record_lock_wait_ms",
			"freshness_total_pts",
			"freshness_contiguity_pts",
			"freshness_queue_pts",
			"freshness_active_front_pts",
			"freshness_missed_e2e_pts",
			"inflight_bf_pts",
			"occ_merge_ms",
			"occ_l2_dists",
			"occ_merge_comps",
			"occ_filter_tests",
			"occ_candidates",
			"diag_dist_comps",
			"diag_hops",
			"diag_future_skip_queries",
			"diag_future_skip_hops",
			"diag_recovery_triggers",
			"diag_recovered_edges",
			"diag_recovered_useful",
			"diag_modified_expansions",
			"diag_recovery_attempts",
			"diag_recovery_candidates",
			"diag_recovery_get_ms",
			"diag_recovery_loop_ms",
			"diag_rewrite_active_expansions",
			"diag_rewrite_active_queries",
			"diag_rewrite_recent_expansions",
			"diag_rewrite_recent_queries",
			"diag_rewrite_period_expansions",
			"diag_rewrite_period_queries",
			"diag_rewrite_period_active_sum",
			"diag_rewrite_period_active_max",
			"diag_batch_search_calls",
			"diag_batch_search_queries",
			"diag_batch_search_arena_ms",
			"diag_batch_search_knn_ms",
			"diag_batch_search_copy_ms",
			"diag_invisible_expansions",
			"diag_invisible_expansion_edges",
			"diag_invisible_candidate_dist_comps",
			"diag_invisible_candidate_enqueues",
			"diag_phase_overlap_samples",
			"diag_phase_existing_update_samples",
			"diag_phase_existing_update_queries",
			"diag_phase_link_critical_samples",
			"diag_phase_link_critical_queries",
			"diag_phase_load_scan_samples",
			"diag_phase_load_scan_queries",
			"diag_phase_append_samples",
			"diag_phase_append_queries",
			"diag_phase_prune_samples",
			"diag_phase_prune_queries",
			"diag_phase_undo_record_samples",
			"diag_phase_undo_record_queries",
			"diag_phase_rewrite_samples",
			"diag_phase_rewrite_queries",
			"work_searchknn_ms",
			"work_searchknn_thread_cpu_ms",
			"work_start_cpu",
			"work_end_cpu",
			"work_result_copy_ms",
			"work_entry_ms",
			"work_upper_search_ms",
			"work_base_search_ms",
			"work_result_materialize_ms",
			"work_snapshot_guard_ms",
			"work_visited_list_get_ms",
			"work_visited_list_release_ms",
			"work_upper_lock_wait_ms",
			"work_level0_lock_wait_ms",
			"work_distance_computations",
			"work_upper_distance_computations",
			"work_level0_distance_computations",
			"work_distance_compute_ms",
			"work_upper_distance_compute_ms",
			"work_level0_distance_compute_ms",
			"work_level0_queue_pop_ms",
			"work_level0_adj_fetch_ms",
			"work_level0_locality_capture_ms",
			"work_level0_candidate_loop_ms",
			"work_level0_visited_check_ms",
			"work_level0_visibility_check_ms",
			"work_level0_candidate_accept_ms",
			"work_upper_hops",
			"work_upper_edges_scanned",
			"work_level0_expansions",
			"work_level0_edges_scanned",
			"work_candidate_pops",
			"work_candidate_pushes",
			"work_visited_nodes",
			"work_result_pushes",
			"work_invisible_expansions",
			"work_invisible_edges",
			"work_invisible_candidate_dist_comps",
			"work_invisible_candidate_enqueues",
			"work_future_skip_hops",
			"work_rewrite_active_expansions",
			"work_rewrite_recent_expansions",
			"work_rewrite_period_expansions",
			"work_rewrite_period_active_sum",
			"work_rewrite_period_active_max",
			"work_expand_visible_count",
			"work_expand_recent_1k_hits",
			"work_expand_recent_4k_hits",
			"work_expand_recent_16k_hits",
			"work_expand_label_gap_sum",
			"work_expand_label_span",
			"work_expand_unique_label_4k_buckets",
			"work_expand_unique_data_4k_pages",
			"work_expand_unique_data_2m_pages",
			"work_expand_unique_adj_4k_pages",
			"work_expand_unique_adj_2m_pages",
			"work_expand_unique_overflow",
			"work_path_count",
			"work_path_first_label",
			"work_path_last_label",
			"work_path_min_label",
			"work_path_max_label",
			"work_path_label_span",
			"work_path_mean_abs_gap",
			"work_path_unique_4k_buckets",
			"work_path_min_visible_age",
			"work_path_mean_visible_age",
			"work_path_recent_1k_hits",
			"work_path_recent_4k_hits",
			"work_path_recent_16k_hits",
		}
		fmt.Fprintln(w, strings.Join(searchWideHeaders, ","))
		getf := func(arr []float64, i int) float64 {
			if i < len(arr) {
				return arr[i]
			}
			return 0
		}
		getu := func(arr []uint64, i int) uint64 {
			if i < len(arr) {
				return arr[i]
			}
			return 0
		}
		for i := 0; i < n; i++ {
			values := []float64{
				getf(b.searchTaskInsertOffset, i),
				getf(b.searchCommittedAtStart, i),
				getf(b.searchViewOffset, i),
				getf(b.searchPhysicalHead, i),
				getf(b.searchOpLatencies, i),
				getf(b.searchE2ELatencies, i),
				getf(b.searchIndexMs, i),
				getf(b.searchPostIndexMs, i),
				getf(b.searchResultsLockWaitMs, i),
				getf(b.searchResultsLockHoldMs, i),
				getf(b.searchRecordLockWaitMs, i),
				getf(b.searchFreshnessTotalPts, i),
				getf(b.searchFreshnessContiguityPts, i),
				getf(b.searchFreshnessQueuePts, i),
				getf(b.searchFreshnessActiveFrontPts, i),
				getf(b.searchFreshnessMissedE2EPts, i),
				getf(b.searchInflightBruteforcePts, i),
				getf(b.searchInflightOccLatencies, i),
				getf(b.searchInflightOccL2Dists, i),
				getf(b.searchInflightOccMergeComps, i),
				getf(b.searchInflightOccFilterTests, i),
				getf(b.searchInflightOccCandidates, i),
				getf(b.searchDiagDistComps, i),
				getf(b.searchDiagHops, i),
				getf(b.searchDiagFutureSkipQueries, i),
				getf(b.searchDiagFutureSkipHops, i),
				getf(b.searchDiagRecoveryTriggers, i),
				getf(b.searchDiagRecoveredEdges, i),
				getf(b.searchDiagUsefulRecovered, i),
				getf(b.searchDiagModifiedExpansions, i),
				getf(b.searchDiagRecoveryAttempts, i),
				getf(b.searchDiagRecoveryCandidates, i),
				getf(b.searchDiagRecoveryGetMs, i),
				getf(b.searchDiagRecoveryLoopMs, i),
				getf(b.searchDiagRewriteActiveExp, i),
				getf(b.searchDiagRewriteActiveQuery, i),
				getf(b.searchDiagRewriteRecentExp, i),
				getf(b.searchDiagRewriteRecentQuery, i),
				getf(b.searchDiagRewritePeriodExp, i),
				getf(b.searchDiagRewritePeriodQuery, i),
				getf(b.searchDiagRewritePeriodActiveSum, i),
				getf(b.searchDiagRewritePeriodActiveMax, i),
				getf(b.searchDiagBatchSearchCalls, i),
				getf(b.searchDiagBatchSearchQueries, i),
				getf(b.searchDiagBatchSearchArenaMs, i),
				getf(b.searchDiagBatchSearchKnnMs, i),
				getf(b.searchDiagBatchSearchCopyMs, i),
				getf(b.searchDiagInvisibleExpansions, i),
				getf(b.searchDiagInvisibleExpansionEdges, i),
				getf(b.searchDiagInvisibleCandidateDistComps, i),
				getf(b.searchDiagInvisibleCandidateEnqueues, i),
				getf(b.searchDiagPhaseOverlapSamples, i),
				getf(b.searchDiagPhaseExistingUpdateSamples, i),
				getf(b.searchDiagPhaseExistingUpdateQueries, i),
				getf(b.searchDiagPhaseLinkCriticalSamples, i),
				getf(b.searchDiagPhaseLinkCriticalQueries, i),
				getf(b.searchDiagPhaseLoadScanSamples, i),
				getf(b.searchDiagPhaseLoadScanQueries, i),
				getf(b.searchDiagPhaseAppendSamples, i),
				getf(b.searchDiagPhaseAppendQueries, i),
				getf(b.searchDiagPhasePruneSamples, i),
				getf(b.searchDiagPhasePruneQueries, i),
				getf(b.searchDiagPhaseUndoRecordSamples, i),
				getf(b.searchDiagPhaseUndoRecordQueries, i),
				getf(b.searchDiagPhaseRewriteSamples, i),
				getf(b.searchDiagPhaseRewriteQueries, i),
				getf(b.searchWorkSearchKnnMs, i),
				getf(b.searchWorkSearchKnnThreadCpuMs, i),
				getf(b.searchWorkStartCPU, i),
				getf(b.searchWorkEndCPU, i),
				getf(b.searchWorkResultCopyMs, i),
				getf(b.searchWorkEntryMs, i),
				getf(b.searchWorkUpperSearchMs, i),
				getf(b.searchWorkBaseSearchMs, i),
				getf(b.searchWorkResultMaterializeMs, i),
				getf(b.searchWorkSnapshotGuardMs, i),
				getf(b.searchWorkVisitedListGetMs, i),
				getf(b.searchWorkVisitedListReleaseMs, i),
				getf(b.searchWorkUpperLockWaitMs, i),
				getf(b.searchWorkLevel0LockWaitMs, i),
				getf(b.searchWorkDistanceComputations, i),
				getf(b.searchWorkUpperDistanceComputations, i),
				getf(b.searchWorkLevel0DistanceComputations, i),
				getf(b.searchWorkDistanceComputeMs, i),
				getf(b.searchWorkUpperDistanceComputeMs, i),
				getf(b.searchWorkLevel0DistanceComputeMs, i),
				getf(b.searchWorkLevel0QueuePopMs, i),
				getf(b.searchWorkLevel0AdjFetchMs, i),
				getf(b.searchWorkLevel0LocalityCaptureMs, i),
				getf(b.searchWorkLevel0CandidateLoopMs, i),
				getf(b.searchWorkLevel0VisitedCheckMs, i),
				getf(b.searchWorkLevel0VisibilityCheckMs, i),
				getf(b.searchWorkLevel0CandidateAcceptMs, i),
				getf(b.searchWorkUpperHops, i),
				getf(b.searchWorkUpperEdgesScanned, i),
				getf(b.searchWorkLevel0Expansions, i),
				getf(b.searchWorkLevel0EdgesScanned, i),
				getf(b.searchWorkCandidatePops, i),
				getf(b.searchWorkCandidatePushes, i),
				getf(b.searchWorkVisitedNodes, i),
				getf(b.searchWorkResultPushes, i),
				getf(b.searchWorkInvisibleExpansions, i),
				getf(b.searchWorkInvisibleEdges, i),
				getf(b.searchWorkInvisibleCandidateDistComps, i),
				getf(b.searchWorkInvisibleCandidateEnqueues, i),
				getf(b.searchWorkFutureSkipHops, i),
				getf(b.searchWorkRewriteActiveExpansions, i),
				getf(b.searchWorkRewriteRecentExpansions, i),
				getf(b.searchWorkRewritePeriodExpansions, i),
				getf(b.searchWorkRewritePeriodActiveSum, i),
				getf(b.searchWorkRewritePeriodActiveMax, i),
				getf(b.searchWorkExpandVisibleCount, i),
				getf(b.searchWorkExpandRecent1KHits, i),
				getf(b.searchWorkExpandRecent4KHits, i),
				getf(b.searchWorkExpandRecent16KHits, i),
				getf(b.searchWorkExpandLabelGapSum, i),
				getf(b.searchWorkExpandLabelSpan, i),
				getf(b.searchWorkExpandUniqueLabel4KBuckets, i),
				getf(b.searchWorkExpandUniqueData4KPages, i),
				getf(b.searchWorkExpandUniqueData2MPages, i),
				getf(b.searchWorkExpandUniqueAdj4KPages, i),
				getf(b.searchWorkExpandUniqueAdj2MPages, i),
				getf(b.searchWorkExpandUniqueOverflow, i),
				getf(b.searchWorkPathCount, i),
				getf(b.searchWorkPathFirstLabel, i),
				getf(b.searchWorkPathLastLabel, i),
				getf(b.searchWorkPathMinLabel, i),
				getf(b.searchWorkPathMaxLabel, i),
				getf(b.searchWorkPathLabelSpan, i),
				getf(b.searchWorkPathMeanAbsGap, i),
				getf(b.searchWorkPathUnique4KBuckets, i),
				getf(b.searchWorkPathMinVisibleAge, i),
				getf(b.searchWorkPathMeanVisibleAge, i),
				getf(b.searchWorkPathRecent1KHits, i),
				getf(b.searchWorkPathRecent4KHits, i),
				getf(b.searchWorkPathRecent16KHits, i),
			}
			fmt.Fprintf(w, "%d,%.0f,%.0f,%.6f,%d,%d,%d", i, getf(b.searchMeasured, i), getf(b.searchQueryTags, i), getf(b.searchFinishMs, i),
				getu(b.searchCreateRawNs, i), getu(b.searchStartRawNs, i),
				getu(b.searchFinishRawNs, i))
			for _, v := range values {
				fmt.Fprintf(w, ",%.6f", v)
			}
			fmt.Fprintln(w)
		}
	}

	return nil
}
