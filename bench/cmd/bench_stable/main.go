package main

import (
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

	type Config struct {
		Name      string
		IndexType internal.IndexType
		MVCC      bool
	}
	configs := []Config{
		{"HNSW-NoLock", internal.IndexTypeHNSW, false},
		{"ANNchor-Stable-MVCC", internal.IndexTypeANNchorM1, true},
	}

	buildSize := 500000
	insertSize := 500000
	if int(npts) < buildSize+insertSize {
		buildSize = int(npts) / 2
		insertSize = int(npts) - buildSize
	}

	searchThreads := numCPU - 4
	if searchThreads < 4 {
		searchThreads = 4
	}
	if searchThreads > 56 {
		searchThreads = 56
	}
	insertThreads := 4
	_ = insertThreads
	batchSize := 1000
	k := uint32(10)
	efSearch := uint(200)

	fmt.Printf("\nConfig: build=%d, insert=%d, search_threads=%d, batch=%d, ef=%d\n",
		buildSize, insertSize, searchThreads, batchSize, efSearch)

	for _, cfg := range configs {
		fmt.Printf("\n========== %s ==========\n", cfg.Name)

		params := internal.IndexParams{
			Dim:            int(dimU),
			MaxElements:    uint64(buildSize + insertSize + 1000),
			M:              16,
			EfConstruction: 200,
			Threads:        int(numCPU),
		}

		idx := internal.NewIndex(cfg.IndexType, params)
		if idx == nil {
			fmt.Printf("Failed to create index for %s\n", cfg.Name)
			continue
		}

		if cfg.MVCC {
			idx.SetEnableMvcc(true)
		}

		// Build
		buildRows := toRows(data[:buildSize*dim], dim)
		buildTags := make([]uint32, buildSize)
		for i := range buildTags {
			buildTags[i] = uint32(i)
		}

		t0 := time.Now()
		err := idx.Build(buildRows, buildTags)
		if err != nil {
			fmt.Printf("Build error: %v\n", err)
			idx.Close()
			continue
		}
		fmt.Printf("Build: %d points in %v (%.0f QPS)\n", buildSize, time.Since(t0),
			float64(buildSize)/time.Since(t0).Seconds())

		idx.SetQueryParams(internal.QueryParams{EfSearch: efSearch})

		// Prepare insert data
		insertFlat := data[buildSize*dim : (buildSize+insertSize)*dim]
		insertTags := make([]uint32, insertSize)
		for i := range insertTags {
			insertTags[i] = uint32(buildSize + i)
		}

		var insertDone atomic.Bool
		var totalSearches atomic.Int64
		var searchLatencies []float64
		var mu sync.Mutex

		var wg sync.WaitGroup
		startTime := time.Now()

		// Search goroutines
		for t := 0; t < searchThreads; t++ {
			wg.Add(1)
			go func(tid int) {
				defer wg.Done()
				localLatencies := make([]float64, 0, 10000)
				qIdx := tid
				for !insertDone.Load() {
					qi := qIdx % int(nq)
					q := queryRows[qi : qi+1]
					ts := time.Now()
					_, _, err := idx.BatchSearch(q, k, ^uint64(0))
					elapsed := time.Since(ts)
					if err == nil {
						localLatencies = append(localLatencies, float64(elapsed.Microseconds()))
						totalSearches.Add(1)
					}
					qIdx += searchThreads
				}
				mu.Lock()
				searchLatencies = append(searchLatencies, localLatencies...)
				mu.Unlock()
			}(t)
		}

		// Insert goroutine
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

		fmt.Printf("Duration: %v\n", totalTime)
		fmt.Printf("Search QPS: %.0f\n", searchQPS)
		fmt.Printf("Insert QPS: %.0f\n", insertQPS)
		fmt.Printf("Total searches: %d\n", totalSearches.Load())
		if len(searchLatencies) > 0 {
			fmt.Printf("Latency (μs): median=%.0f  p95=%.0f  p99=%.0f  p999=%.0f  max=%.0f\n",
				percentile(searchLatencies, 50),
				percentile(searchLatencies, 95),
				percentile(searchLatencies, 99),
				percentile(searchLatencies, 99.9),
				searchLatencies[len(searchLatencies)-1])
		}

		idx.Close()
	}

	os.Exit(0)
}
