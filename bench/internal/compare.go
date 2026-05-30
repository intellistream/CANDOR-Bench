package internal

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func runRealtime(config *Config, data []float32, dataDim int, workloadQueries []float32) error {
	maxElements := config.Data.MaxElements
	begin := config.Data.BeginNum
	totalQueries := len(workloadQueries) / dataDim
	if totalQueries == 0 {
		return fmt.Errorf("realtime: workload query dataset is empty")
	}

	woLockCfg := *config
	woLockCfg.Workload.WithExternalRWLock = false

	wLockCfg := *config
	wLockCfg.Workload.WithExternalRWLock = true

	queryParams := BuildQueryParams(config)
	baselineSnapshot, err := captureBaselineSnapshot(config, data, dataDim)
	if err != nil {
		return fmt.Errorf("realtime: failed to capture baseline snapshot: %w", err)
	}
	if len(baselineSnapshot) > 0 {
		log.Printf("Realtime: using shared baseline snapshot for wolock/wlock comparison")
	} else {
		log.Printf("Realtime: snapshot unavailable for index type %s; each run will rebuild independently", config.Index.IndexType)
	}

	runMode := func(cfg *Config, replaySchedule []ScheduledTask, recordSchedule bool) (*Index, []*SearchResult, []ScheduledTask, int, error) {
		index, err := prepareRealtimeIndex(cfg, data, dataDim, baselineSnapshot)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		index.SetQueryParams(queryParams)
		results, schedule, stages, runErr := runRealtimePipeline(
			index,
			cfg,
			data,
			dataDim,
			workloadQueries,
			begin,
			maxElements,
			totalQueries,
			replaySchedule,
			recordSchedule,
		)
		if runErr != nil {
			index.Close()
			return nil, nil, nil, 0, runErr
		}
		return index, results, schedule, stages, nil
	}

	var capturedSchedule []ScheduledTask

	log.Printf("Realtime: running woLock (Record Phase)...")
	woLockIndex, woLockResults, woSchedule, woStages, err := runMode(&woLockCfg, nil, true)
	if err != nil {
		return fmt.Errorf("realtime: wolock pipeline failed: %w", err)
	}
	defer woLockIndex.Close()
	capturedSchedule = woSchedule
	log.Printf("Realtime: woLock finished, recorded %d search tasks", len(woSchedule))

	log.Printf("Realtime: running wLock (Replay Phase)...")
	wLockIndex, wLockResults, _, wStages, err := runMode(&wLockCfg, capturedSchedule, false)
	if err != nil {
		return fmt.Errorf("realtime: wlock pipeline failed: %w", err)
	}
	defer wLockIndex.Close()

	if len(woLockResults) == 0 && len(wLockResults) == 0 {
		log.Printf("Realtime: no search results collected")
	}

	woLockPaths, err := dumpOutputs(&woLockCfg, woLockIndex, woLockResults, fmt.Sprintf("%s_wolock", config.Index.IndexType))
	if err != nil {
		return err
	}
	wLockPaths, err := dumpOutputs(&wLockCfg, wLockIndex, wLockResults, fmt.Sprintf("%s_wlock", config.Index.IndexType))
	if err != nil {
		return err
	}
	log.Printf("Realtime: wrote wolock results to %s, wlock results to %s", woLockPaths["incr_res"], wLockPaths["incr_res"])

	woResPath := woLockPaths["incr_res"]
	wResPath := wLockPaths["incr_res"]
	if woResPath == "" || wResPath == "" {
		return fmt.Errorf("realtime: incremental result paths missing")
	}

	if config.Data.IncrGtPath != "" && config.Search.RecallAt > 0 {
		gtProvider, err := loadIncrementalGTProvider(config.Data.IncrGtPath, uint32(config.Search.RecallAt))
		if err != nil {
			return fmt.Errorf("realtime: failed to initialize incremental ground truth loader: %w", err)
		}
		diffSummary, err := computePartialDiffDistribution(wLockResults, woLockResults, uint32(config.Search.RecallAt), gtProvider)
		if err != nil {
			return fmt.Errorf("realtime: failed to compute diff distribution: %w", err)
		}
		if len(diffSummary) > 0 {
			baseDir := filepath.Dir(woResPath)
			diffFile := GenerateDiffFileName(woResPath, wResPath)
			diffPath := filepath.Join(baseDir, diffFile)
			if err := writePartialDiffDistribution(diffPath, diffSummary); err != nil {
				return fmt.Errorf("realtime: failed to write diff distribution: %w", err)
			}
			runDiffPlot(config, diffPath, "realtime")
		} else {
			log.Printf("Realtime: diff distribution empty; skipping plot script")
		}
	} else {
		log.Printf("Realtime: skipped diff plot (missing incremental ground truth or recall_at)")
	}

	totalStages := woStages
	if wStages > totalStages {
		totalStages = wStages
	}
	log.Printf("Realtime run completed after %d stages.", totalStages)
	return nil
}

func runRealtimePipeline(
	index *Index,
	cfg *Config,
	data []float32,
	dataDim int,
	workloadQueries []float32,
	begin, maxElements, totalQueries int,
	replaySchedule []ScheduledTask,
	recordSchedule bool,
) ([]*SearchResult, []ScheduledTask, int, error) {
	queueSize := cfg.Workload.QueueSize
	if queueSize < 0 {
		queueSize = cfg.Workload.NumThreads
		if queueSize <= 0 {
			queueSize = 1
		}
	}
	if cfg.Workload.WithExternalRWLock && queueSize > 1 {
		queueSize = 1
	}
	insertQueue := make(chan Task, queueSize)
	searchQueue := make(chan Task, queueSize)

	insertLimiter := buildLimiter(cfg.Workload.InsertEventRate, cfg.Workload.BatchSize)
	searchLimiter := buildLimiter(cfg.Workload.SearchEventRate, cfg.Workload.BatchSize)

	var committedOffset atomic.Uint64
	committedOffset.Store(uint64(begin))
	waitForCommitted := func(offset uint64) {
		for committedOffset.Load() < offset {
			time.Sleep(50 * time.Microsecond)
		}
	}

	results := make([]*SearchResult, 0)
	var resultsMu sync.Mutex
	startTime := time.Now()
	var searchQueries atomic.Uint64

	var rwMu sync.RWMutex
	var wg sync.WaitGroup

	var errOnce sync.Once
	var runErr error
	recordErr := func(err error) {
		if err == nil {
			return
		}
		errOnce.Do(func() {
			runErr = err
		})
	}

	var recorder *[]ScheduledTask
	if recordSchedule {
		localRec := make([]ScheduledTask, 0)
		recorder = &localRec
	}

	stageCountCh, err := StartProducers(
		cfg,
		data,
		dataDim,
		workloadQueries,
		begin,
		maxElements,
		totalQueries,
		insertQueue,
		searchQueue,
		insertLimiter,
		searchLimiter,
		&committedOffset,
		replaySchedule,
		recorder,
	)
	if err != nil {
		recordErr(err)
	}

	processInsert := func(task Task) {
		if len(task.Data) == 0 {
			return
		}
		if cfg.Workload.WithExternalRWLock {
			rwMu.Lock()
		}
		if task.InsertOffset%100000 == 0 {
			log.Printf("Processing batch: up to offset %d", task.InsertOffset)
		}
		err := index.BatchInsert(task.Data, task.Tags)
		if cfg.Workload.WithExternalRWLock {
			rwMu.Unlock()
		}
		if err != nil {
			log.Printf("Realtime: insert task failed: %v", err)
			recordErr(fmt.Errorf("insert task failed: %w", err))
			return
		}

		for {
			current := committedOffset.Load()
			if task.InsertOffset <= current {
				break
			}
			if committedOffset.CompareAndSwap(current, task.InsertOffset) {
				break
			}
		}
	}

	processSearch := func(task Task) {
		if len(task.Data) == 0 || task.RecallAt == 0 {
			return
		}
		if task.WaitForInsert {
			waitForCommitted(task.InsertOffset)
		}
		viewOffset := committedOffset.Load()
		if viewOffset > task.InsertOffset {
			viewOffset = task.InsertOffset
		}
		if cfg.Workload.WithExternalRWLock {
			rwMu.RLock()
		}
		raw, wm, err := index.BatchSearch(task.Data, task.RecallAt, uint64(viewOffset))
		if cfg.Workload.WithExternalRWLock {
			rwMu.RUnlock()
		}
		if err != nil {
			log.Printf("Realtime: search task failed: %v", err)
			recordErr(fmt.Errorf("search task failed: %w", err))
			return
		}

		if wm > 0 {
			viewOffset = wm
		}

		if len(raw) > 0 {
			resultsMu.Lock()
			results = appendSearchResults(results, raw, task.Tags, viewOffset)
			resultsMu.Unlock()
			searchQueries.Add(uint64(len(raw)))
		}
	}

	numThreads := cfg.Workload.NumThreads
	if numThreads <= 0 {
		numThreads = 1
	}

	insertRatio := 50
	if r := os.Getenv("INSERT_RATIO"); r != "" {
		if v, err := fmt.Sscanf(r, "%d", &insertRatio); v == 1 && err == nil && insertRatio >= 1 && insertRatio <= 99 {
		}
	}
	insertWorkers := numThreads * insertRatio / 100
	if insertWorkers < 1 {
		insertWorkers = 1
	}
	searchWorkers := numThreads - insertWorkers
	if searchWorkers < 1 {
		searchWorkers = 1
	}

	numaInsertCPUs := []int{}
	numaSearchCPUs := []int{}
	if os.Getenv("NUMA_SPLIT") == "1" {
		for i := 0; i < 24; i++ {
			numaInsertCPUs = append(numaInsertCPUs, i)
			numaInsertCPUs = append(numaInsertCPUs, i+48)
			numaSearchCPUs = append(numaSearchCPUs, i+24)
			numaSearchCPUs = append(numaSearchCPUs, i+72)
		}
		log.Printf("Realtime: NUMA split enabled: insert->node0, search->node1")
	}

	log.Printf("Realtime: dispatchers %d insert workers, %d search workers", insertWorkers, searchWorkers)

	for i := 0; i < insertWorkers; i++ {
		wg.Add(1)
		go func(cpus []int) {
			runtime.LockOSThread()
			if len(cpus) > 0 {
				pinThreadToCPUs(cpus)
			}
			defer wg.Done()
			for task := range insertQueue {
				processInsert(task)
			}
		}(numaInsertCPUs)
	}

	for i := 0; i < searchWorkers; i++ {
		wg.Add(1)
		go func(cpus []int) {
			runtime.LockOSThread()
			if len(cpus) > 0 {
				pinThreadToCPUs(cpus)
			}
			defer wg.Done()
			for task := range searchQueue {
				processSearch(task)
			}
		}(numaSearchCPUs)
	}

	wg.Wait()

	stageCount := 0
	if sc, ok := <-stageCountCh; ok {
		stageCount = sc
	}

	var finalSchedule []ScheduledTask
	if recorder != nil {
		finalSchedule = *recorder
	}

	modeLabel := "wlock"
	if !cfg.Workload.WithExternalRWLock {
		modeLabel = "wolock"
	}
	elapsed := time.Since(startTime).Seconds()
	inserted := committedOffset.Load() - uint64(begin)
	insQPS := float64(inserted) / elapsed
	searchQPS := float64(searchQueries.Load()) / elapsed
	log.Printf("Realtime %s: throughput insert=%.2f pts/s, search=%.2f pts/s (elapsed=%.2fs, inserted=%d, queries=%d)", modeLabel, insQPS, searchQPS, elapsed, inserted, searchQueries.Load())
	if strings.EqualFold(cfg.Index.IndexType, "annchor") || strings.EqualFold(cfg.Index.IndexType, "annchor-m1") || strings.EqualFold(cfg.Index.IndexType, "annchor-m2") || strings.EqualFold(cfg.Index.IndexType, "annchor-preempt") || strings.EqualFold(cfg.Index.IndexType, "annchor-trim") || strings.EqualFold(cfg.Index.IndexType, "hnsw-visible") {
		log.Printf("Realtime %s: annchor stats: %s", modeLabel, index.DumpStats())
	}

	return results, finalSchedule, stageCount, runErr
}

func dumpOutputs(cfg *Config, index *Index, results []*SearchResult, label string) (map[string]string, error) {
	paths := buildResultPaths(cfg, index)

	incrResPath := paths["incr_res"]
	if incrResPath == "" {
		return nil, fmt.Errorf("realtime: incremental result path missing")
	}

	if err := ensureDir(incrResPath); err != nil {
		return nil, fmt.Errorf("realtime: %w", err)
	}
	if err := DumpIncrResults(results, incrResPath); err != nil {
		return nil, fmt.Errorf("realtime: failed to dump incremental results: %w", err)
	}
	if strings.EqualFold(cfg.Index.IndexType, "annchor") || strings.EqualFold(cfg.Index.IndexType, "annchor-m1") || strings.EqualFold(cfg.Index.IndexType, "annchor-m2") || strings.EqualFold(cfg.Index.IndexType, "annchor-preempt") || strings.EqualFold(cfg.Index.IndexType, "annchor-trim") || strings.EqualFold(cfg.Index.IndexType, "hnsw-visible") {
		statsPath := strings.TrimSuffix(incrResPath, ".res") + "_stats.txt"
		statsPayload := fmt.Sprintf("label=%s\n%s\n", label, index.DumpStats())
		if err := os.WriteFile(statsPath, []byte(statsPayload), 0644); err != nil {
			return nil, fmt.Errorf("realtime: failed to write annchor stats: %w", err)
		}
		paths["stats"] = statsPath
	}

	if cfg.Data.IncrGtPath != "" {
		recallPath := paths["incr_recall"]
		if recallPath == "" {
			return nil, fmt.Errorf("realtime: recall output path missing")
		}
		if err := calcIncrRecall(
			defaultIncrRecallTool,
			incrResPath,
			cfg.Data.IncrGtPath,
			recallPath,
			label,
			cfg.StageQueryWindow(),
			cfg.Search.RecallAt,
		); err != nil {
			return nil, fmt.Errorf("realtime: incremental recall failed: %w", err)
		}
	} else {
		log.Printf("Test: incremental recall disabled (missing incr_gt_path)")
		paths["incr_recall"] = ""
	}

	return paths, nil
}

func prepareRealtimeIndex(cfg *Config, data []float32, dataDim int, snapshot []byte) (*Index, error) {
	if len(snapshot) > 0 {
		log.Printf("Realtime: restoring index from snapshot for %s mode...", cfg.Workload.QueryMode)
		index := createIndexInstance(cfg, dataDim)
		if !index.SupportsSnapshot() {
			index.Close()
			return nil, fmt.Errorf("index type %s does not support snapshot restore", cfg.Index.IndexType)
		}
		if err := index.Restore(snapshot); err != nil {
			index.Close()
			return nil, fmt.Errorf("restore from baseline snapshot failed: %w", err)
		}
		return index, nil
	}
	log.Printf("Realtime: building fresh index for %s mode...", cfg.Workload.QueryMode)
	return createAndBuildIndex(cfg, data, dataDim), nil
}

func captureBaselineSnapshot(config *Config, data []float32, dataDim int) ([]byte, error) {
	if !strings.EqualFold(config.Index.IndexType, "hnsw") && !strings.EqualFold(config.Index.IndexType, "hnsw-visible") && !strings.EqualFold(config.Index.IndexType, "annchor") && !strings.EqualFold(config.Index.IndexType, "annchor-m1") && !strings.EqualFold(config.Index.IndexType, "annchor-m2") && !strings.EqualFold(config.Index.IndexType, "annchor-preempt") && !strings.EqualFold(config.Index.IndexType, "annchor-trim") {
		return nil, nil
	}
	log.Printf("Realtime: capturing baseline snapshot...")
	baseCfg := *config
	baseCfg.Workload.WithExternalRWLock = false
	index := createAndBuildIndex(&baseCfg, data, dataDim)
	defer index.Close()
	if !index.SupportsSnapshot() {
		return nil, nil
	}
	snapshot, err := index.Snapshot()
	if err != nil {
		return nil, err
	}
	return snapshot, nil
}
