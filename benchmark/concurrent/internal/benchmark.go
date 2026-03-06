package internal

import (
	"fmt"
	"log"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/time/rate"
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
	WaitForInsert bool
}

type Stat struct {
	InsertQPS            float64
	SearchQPS            float64
	MeanInsertOpLatency  float64
	MeanSearchOpLatency  float64
	P95InsertOpLatency   float64
	P99InsertOpLatency   float64
	P95SearchOpLatency   float64
	P99SearchOpLatency   float64
	MeanInsertE2ELatency float64
	P95InsertE2ELatency  float64
	P99InsertE2ELatency  float64
	MeanSearchE2ELatency float64
	P95SearchE2ELatency  float64
	P99SearchE2ELatency  float64
	ResultBelowKCount    int
	PeakMemoryMB         float64
	AvgMemoryMB          float64
}

type Bench struct {
	insertQueue        chan Task
	searchQueue        chan Task
	index              *Index
	rwMu               sync.RWMutex
	wg                 sync.WaitGroup
	insertOpLatencies  []float64
	searchOpLatencies  []float64
	insertE2ELatencies []float64
	searchE2ELatencies []float64
	insertLimiter      *rate.Limiter
	searchLimiter      *rate.Limiter
	searchResults      []*SearchResult
	resultsMu          sync.Mutex
	insertMu           sync.Mutex
	searchMu           sync.Mutex
	config             *Config
	insertPointCnt     int
	searchPointCnt     int
	globalInsertCnt    int64
	committedOffset    atomic.Uint64
	completionMu     sync.Mutex
	completedOffsets map[uint64]bool
	pendingOffsets   []uint64
	pendingIdx       int
	searchLagSum     int64
	searchLagCount   int64
	searchLagMax     int64
	startTime          time.Time
	resultBelowKCount  int
}

type memResult struct {
	peakUsage    uint64
	avgUsage     float64
	samplesCount int
}

func concurrentBench(index *Index, config Config) *Bench {
	batchSize := config.Workload.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}
	expectedBatches := (config.Data.MaxElements - config.Data.BeginNum) / batchSize
	if expectedBatches <= 0 {
		expectedBatches = 1000
	}

	b := &Bench{
		insertQueue:        make(chan Task, config.Workload.QueueSize),
		searchQueue:        make(chan Task, config.Workload.QueueSize),
		index:              index,
		insertOpLatencies:  make([]float64, 0, expectedBatches),
		searchOpLatencies:  make([]float64, 0, expectedBatches),
		insertE2ELatencies: make([]float64, 0, expectedBatches),
		searchE2ELatencies: make([]float64, 0, expectedBatches),
		insertLimiter:      buildLimiter(config.Workload.InsertEventRate, config.Workload.BatchSize),
		searchLimiter:      buildLimiter(config.Workload.SearchEventRate, config.Workload.BatchSize),
		config:             &config,
	}
	b.committedOffset.Store(uint64(config.Data.BeginNum))
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

func execBenchmark(index *Index, config *Config, data []float32, workloadQueries []float32, dataDim int, overallQueries []float32) ([]*SearchResult, Stat, error) {
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

	bench := concurrentBench(index, *config)
	expectedResults := config.Data.MaxElements * 2
	bench.searchResults = make([]*SearchResult, 0, expectedResults)

	log.Println("Start producing tasks and consuming tasks...")

	bench.startTime = time.Now()

	if config.Data.BeginNum >= config.Data.MaxElements {
		if doneChan != nil {
			close(doneChan)
		}
		<-memResultChan
		emptyStats := Stat{}
		finishBench(index, overallQueries, dataDim, config, emptyStats)
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

	_, err := StartProducers(
		config,
		data,
		dataDim,
		workloadQueries,
		config.Data.BeginNum,
		config.Data.MaxElements,
		len(workloadQueries)/dataDim,
		bench.insertQueue,
		bench.searchQueue,
		bench.insertLimiter,
		bench.searchLimiter,
		&bench.committedOffset,
		nil,
		nil,
	)
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

	finishBench(index, overallQueries, dataDim, config, stats)

	return bench.searchResults, stats, nil
}

func (b *Bench) startDispatchers() {
	b.index.SetQueryParams(BuildQueryParams(b.config))

	numThreads := b.config.Workload.NumThreads
	if numThreads <= 0 {
		numThreads = 1
	}

	insertWorkers := numThreads / 2
	if insertWorkers < 1 {
		insertWorkers = 1
	}
	searchWorkers := numThreads - insertWorkers
	if searchWorkers < 1 {
		searchWorkers = 1
	}

	log.Printf("Dispatchers: %d insert workers, %d search workers", insertWorkers, searchWorkers)

	for i := 0; i < insertWorkers; i++ {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			for task := range b.insertQueue {
				b.handleInsertTask(task, time.Now())
			}
		}()
	}

	for i := 0; i < searchWorkers; i++ {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			for task := range b.searchQueue {
				b.handleSearchTask(task, time.Now())
			}
		}()
	}
}

func (b *Bench) handleInsertTask(task Task, start time.Time) {
	if len(task.Data) == 0 {
		return
	}
	if b.config.Workload.WithExternalRWLock {
		b.rwMu.Lock()
	}
	err := b.index.BatchInsert(task.Data, task.Tags)
	if b.config.Workload.WithExternalRWLock {
		b.rwMu.Unlock()
	}
	if err != nil {
		log.Printf("Insert error: %v", err)
		return
	}
	insertOpLat := float64(time.Since(start).Microseconds()) / 1000.0
	insertE2ELat := float64(time.Since(task.CreateTime).Microseconds()) / 1000.0
	b.insertMu.Lock()
	b.insertOpLatencies = append(b.insertOpLatencies, insertOpLat)
	b.insertE2ELatencies = append(b.insertE2ELatencies, insertE2ELat)
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
		b.pendingIdx++
	}
	b.completionMu.Unlock()
}

func (b *Bench) handleSearchTask(task Task, start time.Time) {
	if len(task.Data) == 0 || task.RecallAt == 0 {
		return
	}

	viewOffset := b.committedOffset.Load()
	if viewOffset > task.InsertOffset {
		viewOffset = task.InsertOffset
	}

	currentInserted := atomic.LoadInt64(&b.globalInsertCnt)
	beginNum := int64(b.config.Data.BeginNum)
	lag := currentInserted + beginNum - int64(viewOffset)
	if lag < 0 {
		lag = 0
	}
	atomic.AddInt64(&b.searchLagSum, lag)
	atomic.AddInt64(&b.searchLagCount, 1)
	for {
		old := atomic.LoadInt64(&b.searchLagMax)
		if lag <= old {
			break
		}
		if atomic.CompareAndSwapInt64(&b.searchLagMax, old, lag) {
			break
		}
	}

	if b.config.Workload.WithExternalRWLock {
		b.rwMu.RLock()
	}
	results, _, err := b.index.BatchSearch(task.Data, uint32(task.RecallAt), uint64(viewOffset))
	if b.config.Workload.WithExternalRWLock {
		b.rwMu.RUnlock()
	}
	if err != nil {
		log.Printf("Search error: %v", err)
		return
	}

	b.resultsMu.Lock()
	for i, tags := range results {
		result := NewSearchResult(
			task.InsertOffset,
			uint64(task.Tags[i]),
			tags,
		)
		b.searchResults = append(b.searchResults, result)
	}
	b.resultsMu.Unlock()
	b.searchMu.Lock()
	if b.config.Workload.PerQueryLatency != nil && *b.config.Workload.PerQueryLatency && len(task.Data) > 0 {
		perQueryOp := float64(time.Since(start).Microseconds()) / 1000.0 / float64(len(task.Data))
		perQueryE2E := float64(time.Since(task.CreateTime).Microseconds()) / 1000.0 / float64(len(task.Data))
		for i := 0; i < len(task.Data); i++ {
			b.searchOpLatencies = append(b.searchOpLatencies, perQueryOp)
			b.searchE2ELatencies = append(b.searchE2ELatencies, perQueryE2E)
		}
	} else {
		b.searchOpLatencies = append(b.searchOpLatencies, float64(time.Since(start).Microseconds())/1000.0)
		b.searchE2ELatencies = append(b.searchE2ELatencies, float64(time.Since(task.CreateTime).Microseconds())/1000.0)
	}
	b.searchPointCnt += len(task.Data)
	b.searchMu.Unlock()
}

func (b *Bench) printProgress(totalInsert int) {
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
			var sLagAvg float64
			if sLagCount > 0 {
				sLagAvg = float64(atomic.LoadInt64(&b.searchLagSum)) / float64(sLagCount)
			}
			log.Printf("Progress: %d/%d (%d%%), Insert QPS: %.2f, Search QPS: %.2f, offset_lag: %d, search_snapshot_lag(avg/max): %.0f/%d pts",
				current, totalInsert, percent, insertQPS, searchQPS, offsetLag, sLagAvg, sLagMax)
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
		s.ResultBelowKCount = b.resultBelowKCount
	}

	if memStats.samplesCount > 0 {
		s.PeakMemoryMB = float64(memStats.peakUsage) / (1024 * 1024)
		s.AvgMemoryMB = memStats.avgUsage / (1024 * 1024)
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
