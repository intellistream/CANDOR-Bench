package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"ANN-CC-bench/bench/internal"
)

func toRows(flat []float32, dim int) [][]float32 {
	n := len(flat) / dim
	rows := make([][]float32, n)
	for i := 0; i < n; i++ {
		rows[i] = flat[i*dim : (i+1)*dim]
	}
	return rows
}

func loadGT(path string, wantK int) ([][]uint32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var nq, gtK uint32
	binary.Read(f, binary.LittleEndian, &nq)
	binary.Read(f, binary.LittleEndian, &gtK)
	fmt.Printf("GT file: nq=%d, k=%d\n", nq, gtK)

	allIDs := make([]int32, nq*gtK)
	binary.Read(f, binary.LittleEndian, &allIDs)

	gt := make([][]uint32, nq)
	for i := uint32(0); i < nq; i++ {
		limit := int(gtK)
		if limit > wantK {
			limit = wantK
		}
		row := make([]uint32, limit)
		for j := 0; j < limit; j++ {
			row[j] = uint32(allIDs[i*gtK+uint32(j)])
		}
		gt[i] = row
	}
	return gt, nil
}

func computeRecall(results [][]uint32, gt [][]uint32, k int) float64 {
	nq := len(results)
	if nq > len(gt) {
		nq = len(gt)
	}
	totalHits := 0
	totalExpected := 0
	for i := 0; i < nq; i++ {
		gtSet := make(map[uint32]bool)
		limit := k
		if limit > len(gt[i]) {
			limit = len(gt[i])
		}
		for _, id := range gt[i][:limit] {
			gtSet[id] = true
		}
		resLimit := k
		if resLimit > len(results[i]) {
			resLimit = len(results[i])
		}
		for _, id := range results[i][:resLimit] {
			if gtSet[id] {
				totalHits++
			}
		}
		totalExpected += limit
	}
	if totalExpected == 0 {
		return 0
	}
	return float64(totalHits) / float64(totalExpected)
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p/100.0*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// Chasing workload: insert and search run concurrently.
// Search threads continuously query, "chasing" the insertion frontier.
// After all inserts finish, measure final recall.
func runChasingExperiment(name string, indexType internal.IndexType,
	data []float32, dim int, npts int,
	queryRows [][]float32, gt [][]uint32,
	k int, efSearch uint, searchThreads int) {

	fmt.Printf("\n========== %s ==========\n", name)

	numCPU := runtime.NumCPU()
	buildSize := 500000
	insertSize := 500000
	if npts < buildSize+insertSize {
		buildSize = npts / 2
		insertSize = npts - buildSize
	}
	batchSize := 1000

	params := internal.IndexParams{
		Dim:            dim,
		MaxElements:    uint64(buildSize + insertSize + 1000),
		M:              16,
		EfConstruction: 200,
		Threads:        numCPU,
	}

	idx := internal.NewIndex(indexType, params)
	if idx == nil {
		fmt.Println("Failed to create index")
		return
	}
	defer idx.Close()

	// Configure per index type
	switch indexType {
	case internal.IndexTypeANNchorM1:
		idx.SetEnableMvcc(true)
	case internal.IndexTypeANNchorM3:
		// MVCC + Slipstream enabled internally; set generous TTL
		idx.SetSlipstreamTTL(100_000_000) // 100ms TTL
	case internal.IndexTypeHNSW:
		// pure HNSW, nothing to configure
	}

	// Build initial index
	buildRows := toRows(data[:buildSize*dim], dim)
	buildTags := make([]uint32, buildSize)
	for i := range buildTags {
		buildTags[i] = uint32(i)
	}

	t0 := time.Now()
	idx.Build(buildRows, buildTags)
	fmt.Printf("  Build: %d points in %v\n", buildSize, time.Since(t0))

	idx.SetQueryParams(internal.QueryParams{EfSearch: efSearch})

	// Prepare insert data
	insertFlat := data[buildSize*dim : (buildSize+insertSize)*dim]
	insertTags := make([]uint32, insertSize)
	for i := range insertTags {
		insertTags[i] = uint32(buildSize + i)
	}

	// --- Concurrent chasing workload ---
	var insertDone atomic.Bool
	var totalSearches atomic.Int64
	var searchLatencies []float64
	var mu sync.Mutex
	nq := len(queryRows)

	var wg sync.WaitGroup
	startTime := time.Now()

	// Search goroutines: chase the insertion frontier
	for t := 0; t < searchThreads; t++ {
		wg.Add(1)
		go func(tid int) {
			defer wg.Done()
			localLat := make([]float64, 0, 10000)
			qIdx := tid
			for !insertDone.Load() {
				qi := qIdx % nq
				q := queryRows[qi : qi+1]
				ts := time.Now()
				_, _, err := idx.BatchSearch(q, uint32(k), ^uint64(0))
				elapsed := time.Since(ts)
				if err == nil {
					localLat = append(localLat, float64(elapsed.Microseconds()))
					totalSearches.Add(1)
				}
				qIdx += searchThreads
			}
			mu.Lock()
			searchLatencies = append(searchLatencies, localLat...)
			mu.Unlock()
		}(t)
	}

	// Insert goroutine: batch insert
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < insertSize; i += batchSize {
			end := i + batchSize
			if end > insertSize {
				end = insertSize
			}
			batch := toRows(insertFlat[i*dim:end*dim], dim)
			tags := insertTags[i:end]
			idx.BatchInsert(batch, tags)
		}
		insertDone.Store(true)
	}()

	wg.Wait()
	totalTime := time.Since(startTime)

	sort.Float64s(searchLatencies)
	searchQPS := float64(totalSearches.Load()) / totalTime.Seconds()
	insertQPS := float64(insertSize) / totalTime.Seconds()

	fmt.Printf("  Duration: %v\n", totalTime)
	fmt.Printf("  Insert QPS: %.0f | Search QPS: %.0f | Total searches: %d\n",
		insertQPS, searchQPS, totalSearches.Load())

	if len(searchLatencies) > 0 {
		fmt.Printf("  Latency (μs): median=%.0f  p95=%.0f  p99=%.0f  max=%.0f\n",
			percentile(searchLatencies, 50),
			percentile(searchLatencies, 95),
			percentile(searchLatencies, 99),
			searchLatencies[len(searchLatencies)-1])
	}

	// Final recall on all queries after all inserts
	if gt != nil {
		finalNq := nq
		if finalNq > len(gt) {
			finalNq = len(gt)
		}
		results, _, err := idx.BatchSearch(queryRows[:finalNq], uint32(k), ^uint64(0))
		if err == nil {
			recall := computeRecall(results, gt[:finalNq], k)
			fmt.Printf("  Final recall@%d: %.4f\n", k, recall)
		}
	}

	stats := idx.DumpStats()
	fmt.Printf("  Stats: %s\n", stats)
}

func main() {
	numCPU := runtime.NumCPU()
	fmt.Printf("CPUs: %d\n", numCPU)

	data, npts, dimU, err := internal.LoadAlignedBin("../data/sift/sift_base.bin")
	if err != nil {
		panic(err)
	}
	dim := int(dimU)
	fmt.Printf("Loaded %d points, dim=%d\n", npts, dim)

	qFlat, nq, _, err := internal.LoadAlignedBin("../data/sift/sift_query.bin")
	if err != nil {
		panic(err)
	}
	queryRows := toRows(qFlat, dim)
	fmt.Printf("Loaded %d queries\n", nq)

	gt, err := loadGT("../data/sift/sift_base.gt20", 10)
	if err != nil {
		fmt.Printf("Warning: could not load GT: %v\n", err)
		gt = nil
	} else {
		fmt.Printf("Loaded %d ground truth entries\n", len(gt))
	}

	k := 10
	efSearch := uint(200)
	searchThreads := numCPU - 4
	if searchThreads < 4 {
		searchThreads = 4
	}
	if searchThreads > 56 {
		searchThreads = 56
	}

	fmt.Printf("\nConfig: k=%d, ef_search=%d, search_threads=%d, chasing workload\n",
		k, efSearch, searchThreads)

	// 1. Pure HNSW — no mechanisms at all
	runChasingExperiment("Pure HNSW (no mechanisms)",
		internal.IndexTypeHNSW, data, dim, int(npts), queryRows, gt, k, efSearch, searchThreads)

	// 2. M1 — MVCC only
	runChasingExperiment("ANNchor-M1 (MVCC only)",
		internal.IndexTypeANNchorM1, data, dim, int(npts), queryRows, gt, k, efSearch, searchThreads)

	// 3. M3 — MVCC + Slipstream (TTL=100ms)
	runChasingExperiment("ANNchor-M3 (MVCC + Slipstream, TTL=100ms)",
		internal.IndexTypeANNchorM3, data, dim, int(npts), queryRows, gt, k, efSearch, searchThreads)
}
