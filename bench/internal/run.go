package internal

import (
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"strings"
)

func Run(configPath string) error {
	configs, err := loadConfigVariants(configPath)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	for idx, config := range configs {
		if config.ThroughputSweep.Enabled {
			log.Printf("==== Running throughput sweep mode ====")
			if err := RunThroughputSweep(config, configPath); err != nil {
				return fmt.Errorf("throughput sweep failed: %w", err)
			}
			continue
		}

		baseLabel := config.VariantName
		if baseLabel == "" {
			baseLabel = fmt.Sprintf("variant_%d", idx+1)
		}

		repetitions := config.Repetitions
		if repetitions <= 0 {
			repetitions = 1
		}

		for rep := 1; rep <= repetitions; rep++ {
			runCfg := *config
			runCfg.VariantName = config.VariantName
			if repetitions > 1 {
				repToken := fmt.Sprintf("rep%d", rep)
				if runCfg.VariantName == "" {
					runCfg.VariantName = repToken
				} else {
					runCfg.VariantName = fmt.Sprintf("%s__%s", runCfg.VariantName, repToken)
				}
			}

			log.Printf("==== Running config variant: %s (rep %d/%d) ====", baseLabel, rep, repetitions)
			if err := runSingleConfig(&runCfg); err != nil {
				return fmt.Errorf("variant %s rep %d/%d failed: %w", baseLabel, rep, repetitions, err)
			}
		}
	}

	return nil
}

func runSingleConfig(config *Config) error {
	restoreEnv, err := applyMidInsertPreemptEnv(config)
	if err != nil {
		return fmt.Errorf("failed to apply mid-insert preempt env overrides: %w", err)
	}
	defer restoreEnv()

	data, _, dim, err := LoadAlignedBin(config.Data.DataPath)
	if err != nil {
		return fmt.Errorf("failed to load data: %w", err)
	}
	dataDim := int(dim)

	totalPoints := len(data) / dataDim
	if totalPoints == 0 {
		return fmt.Errorf("dataset %s contains no vectors", config.Data.DataPath)
	}

	clampDataBounds(config, totalPoints)

	workloadQueryPath := config.Data.IncrQueryPath
	if workloadQueryPath == "" {
		return fmt.Errorf("incr_query_path (or query_path/overall_query_path) must be provided")
	}

	workloadQueries, err := loadQueryDataset(workloadQueryPath, dataDim, "workload")
	if err != nil {
		return err
	}

	overallQueryPath := config.Data.OverallQueryPath
	var overallQueries []float32
	if overallQueryPath == "" || overallQueryPath == workloadQueryPath {
		overallQueryPath = workloadQueryPath
		overallQueries = workloadQueries
		log.Printf("Using workload queries as overall recall dataset (%s)", overallQueryPath)
	} else {
		overallQueries, err = loadQueryDataset(overallQueryPath, dataDim, "overall")
		if err != nil {
			return err
		}
	}

	go func() {
		log.Println(http.ListenAndServe("0.0.0.0:6060", nil))
	}()

	checkRecallConfig(config)

	if config.Compare.Enabled && !config.ShouldSimulate() {
		log.Printf("Analysis: running REALTIME mode (simulate.enabled=false)")
		if err := runRealtime(config, data, dataDim, workloadQueries); err != nil {
			return fmt.Errorf("analysis run failed: %w", err)
		}
		return nil
	}

	if config.ShouldSimulate() {
		log.Printf("Analysis: running SIMULATE mode (snapshot replay)")
		if err := runSimulate(config, data, dataDim, workloadQueries); err != nil {
			return fmt.Errorf("simulate run failed: %w", err)
		}
		return nil
	}

	if strings.EqualFold(config.Index.IndexType, "replica_snapshot") {
		writerConfig := *config
		writerConfig.Index.IndexType = "hnsw"

		writerIndex := createAndBuildIndex(&writerConfig, data, dataDim)
		defer writerIndex.Close()

		snapshot, err := writerIndex.Snapshot()
		if err != nil {
			return fmt.Errorf("failed to capture serving snapshot: %w", err)
		}

		servingIndex := createIndexInstance(&writerConfig, dataDim)
		defer servingIndex.Close()
		if err := servingIndex.Restore(snapshot); err != nil {
			return fmt.Errorf("failed to restore serving snapshot: %w", err)
		}

		incrResults, _, err := execBenchmarkWithServingIndex(writerIndex, servingIndex, config, data, workloadQueries, dataDim, overallQueries)
		if err != nil {
			return fmt.Errorf("benchmark pass failed: %w", err)
		}

		if config.Data.IncrGtPath != "" {
			log.Println("Calculating incremental recall ...")
			mainResultPaths := buildResultPaths(config, servingIndex)
			mainResPath := mainResultPaths["incr_res"]
			mainResDir := filepath.Dir(mainResPath)
			if err := os.MkdirAll(mainResDir, 0755); err != nil {
				return fmt.Errorf("failed to create main results output directory %s: %w", mainResDir, err)
			}
			if err := DumpIncrResults(incrResults, mainResPath); err != nil {
				return fmt.Errorf("failed to dump incremental results: %w", err)
			}
			recallPath := mainResultPaths["incr_recall"]
			if err := calcIncrRecall(
				defaultIncrRecallTool,
				mainResPath,
				config.Data.IncrGtPath,
				recallPath,
				config.Index.IndexType,
				config.StageQueryWindow(),
				config.Search.RecallAt,
			); err != nil {
				return err
			}
		}

		return nil
	}

	if isReplicaRefreshIndex(config.Index.IndexType) {
		writerConfig := *config
		writerConfig.Index.IndexType = "hnsw"

		writerIndex := createAndBuildIndex(&writerConfig, data, dataDim)
		defer writerIndex.Close()

		snapshot, err := writerIndex.Snapshot()
		if err != nil {
			return fmt.Errorf("failed to capture initial serving snapshot: %w", err)
		}

		servingIndex := createIndexInstance(&writerConfig, dataDim)
		if err := servingIndex.Restore(snapshot); err != nil {
			servingIndex.Close()
			return fmt.Errorf("failed to restore initial serving snapshot: %w", err)
		}
		servingIndex.SetQueryParams(BuildQueryParams(&writerConfig))

		replica := newReplicaRefreshManager(config, dataDim, servingIndex)
		defer replica.close()

		incrResults, _, err := execBenchmarkWithReplica(writerIndex, nil, replica, config, data, workloadQueries, dataDim, overallQueries)
		if err != nil {
			return fmt.Errorf("benchmark pass failed: %w", err)
		}

		if config.Data.IncrGtPath != "" {
			log.Println("Calculating incremental recall ...")
			mainResultPaths := buildResultPaths(config, replica.current())
			mainResPath := mainResultPaths["incr_res"]
			mainResDir := filepath.Dir(mainResPath)
			if err := os.MkdirAll(mainResDir, 0755); err != nil {
				return fmt.Errorf("failed to create main results output directory %s: %w", mainResDir, err)
			}
			if err := DumpIncrResults(incrResults, mainResPath); err != nil {
				return fmt.Errorf("failed to dump incremental results: %w", err)
			}
			recallPath := mainResultPaths["incr_recall"]
			if err := calcIncrRecall(
				defaultIncrRecallTool,
				mainResPath,
				config.Data.IncrGtPath,
				recallPath,
				config.Index.IndexType,
				config.StageQueryWindow(),
				config.Search.RecallAt,
			); err != nil {
				return err
			}
		}

		return nil
	}

	index := createAndBuildIndex(config, data, dataDim)
	defer index.Close()

	if err := maybePreinsertToMax(index, config, data, dataDim); err != nil {
		return fmt.Errorf("preinsert-to-max failed: %w", err)
	}

	incrResults, _, err := execBenchmark(index, config, data, workloadQueries, dataDim, overallQueries)
	if err != nil {
		return fmt.Errorf("benchmark pass failed: %w", err)
	}

	needCalcIncrRecall := config.Data.IncrGtPath != ""

	mainResultPaths := buildResultPaths(config, index)

	if needCalcIncrRecall {
		log.Println("Calculating incremental recall ...")
		mainResPath := mainResultPaths["incr_res"]
		mainResDir := filepath.Dir(mainResPath)
		if err := os.MkdirAll(mainResDir, 0755); err != nil {
			return fmt.Errorf("failed to create main results output directory %s: %w", mainResDir, err)
		}

		if err := DumpIncrResults(incrResults, mainResPath); err != nil {
			return fmt.Errorf("failed to dump incremental results: %w", err)
		}

		recallPath := mainResultPaths["incr_recall"]
		if err := calcIncrRecall(
			defaultIncrRecallTool,
			mainResPath,
			config.Data.IncrGtPath,
			recallPath,
			config.Index.IndexType,
			config.StageQueryWindow(),
			config.Search.RecallAt,
		); err != nil {
			return err
		}
	}

	return nil
}

func maybePreinsertToMax(index *Index, config *Config, data []float32, dataDim int) error {
	raw := strings.TrimSpace(os.Getenv("ANNCHOR_PREINSERT_TO_MAX"))
	if raw == "" || raw == "0" || strings.EqualFold(raw, "false") || strings.EqualFold(raw, "no") {
		return nil
	}
	begin := config.Data.BeginNum
	maxElements := config.Data.MaxElements
	if begin >= maxElements {
		log.Printf("Preinsert-to-max requested, but begin_num=%d max_elements=%d; nothing to insert", begin, maxElements)
		return nil
	}
	batchSize := config.Workload.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}
	log.Printf("Preinsert-to-max: inserting [%d, %d) before benchmark, batch_size=%d", begin, maxElements, batchSize)
	for start := begin; start < maxElements; start += batchSize {
		end := start + batchSize
		if end > maxElements {
			end = maxElements
		}
		if err := index.BatchInsert(sliceVectorsRange(data, dataDim, start, end), makeSequentialTags(start, end)); err != nil {
			return fmt.Errorf("batch [%d, %d): %w", start, end, err)
		}
	}
	config.Data.BeginNum = maxElements
	log.Printf("Preinsert-to-max complete; benchmark visible begin_num set to %d", config.Data.BeginNum)
	return nil
}

func clampDataBounds(config *Config, totalPoints int) {
	if config.Data.MaxElements <= 0 || config.Data.MaxElements > totalPoints {
		if config.Data.MaxElements > 0 && config.Data.MaxElements != totalPoints {
			log.Printf("Configured max_elements=%d exceeds available dataset size (%d); clamping", config.Data.MaxElements, totalPoints)
		} else {
			log.Printf("Configured max_elements=%d; defaulting to dataset size (%d)", config.Data.MaxElements, totalPoints)
		}
		config.Data.MaxElements = totalPoints
	}

	if config.Data.BeginNum < 0 {
		log.Printf("Configured begin_num=%d; defaulting to max_elements (%d)", config.Data.BeginNum, config.Data.MaxElements)
		config.Data.BeginNum = config.Data.MaxElements
	} else if config.Data.BeginNum > config.Data.MaxElements {
		log.Printf("Configured begin_num=%d exceeds available dataset size (%d); clamping", config.Data.BeginNum, config.Data.MaxElements)
		config.Data.BeginNum = config.Data.MaxElements
	}
}

func loadQueryDataset(path string, expectedDim int, label string) ([]float32, error) {
	queries, _, dim, err := LoadAlignedBin(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load %s queries: %w", label, err)
	}
	if int(dim) != expectedDim {
		return nil, fmt.Errorf("%s query dimension mismatch: got %d, expected %d", label, dim, expectedDim)
	}
	log.Printf("Loaded %d %s queries from %s", len(queries)/expectedDim, label, path)
	return queries, nil
}

type envSnapshot struct {
	key     string
	value   string
	existed bool
}

func applyMidInsertPreemptEnv(config *Config) (func(), error) {
	type envEntry struct {
		key   string
		value string
	}

	entries := make([]envEntry, 0, 16)
	addBool := func(key string, value *bool) {
		if value == nil {
			return
		}
		if *value {
			entries = append(entries, envEntry{key: key, value: "1"})
			return
		}
		entries = append(entries, envEntry{key: key, value: "0"})
	}
	addInt := func(key string, value *int) {
		if value == nil {
			return
		}
		entries = append(entries, envEntry{key: key, value: fmt.Sprintf("%d", *value)})
	}

	addBool("ANN_MID_INSERT_PREEMPT_ENABLE", config.Search.MidInsertPreemptEnable)
	addInt("ANN_MID_INSERT_PREEMPT_K", config.Search.MidInsertPreemptK)
	addBool("ANN_MID_INSERT_PREEMPT_REVALIDATE", config.Search.MidInsertPreemptRevalidate)
	addBool("ANN_MID_INSERT_PREEMPT_SHADOW_REPLAN", config.Search.MidInsertShadowReplan)
	addBool("ANN_MID_INSERT_PREEMPT_HARM_GUARD", config.Search.MidInsertHarmGuard)
	addBool("ANN_MID_INSERT_PREEMPT_HARM_MICRO_REPLAN", config.Search.MidInsertHarmMicroReplan)
	addInt("ANN_MID_INSERT_PREEMPT_HARM_MICRO_PROBE_EF", config.Search.MidInsertHarmMicroProbeEf)
	addInt("ANN_MID_INSERT_PREEMPT_MAX_WAIT_US", config.Search.MidInsertPreemptMaxWaitUs)
	addInt("ANN_MID_INSERT_PREEMPT_ACTIVATE_AFTER_COMMITS", config.Search.MidInsertActivateAfterCommits)
	addInt("ANN_MID_INSERT_PREEMPT_EVERY_N", config.Search.MidInsertPreemptEveryN)
	addInt("ANN_MID_INSERT_PREEMPT_MAX_INFLIGHT", config.Search.MidInsertPreemptMaxInflight)
	addInt("ANN_MID_INSERT_PREEMPT_HARM_MICRO_POOL_CAP", config.Search.MidInsertHarmMicroPoolCap)
	addBool("ANN_MID_INSERT_PREEMPT_HARM_ONLINE_POLICY", config.Search.MidInsertHarmOnlinePolicy)
	addInt("ANN_MID_INSERT_PREEMPT_HARM_BUSY_WAIT_COMMITS", config.Search.MidInsertHarmBusyWaitCommits)
	addInt("ANN_MID_INSERT_PREEMPT_HARM_SEARCH_BACKLOG_THRESHOLD", config.Search.MidInsertHarmSearchBacklogThreshold)
	addInt("ANN_MID_INSERT_PREEMPT_HARM_PRIORITY_SEARCH_THRESHOLD", config.Search.MidInsertHarmPrioritySearchThreshold)
	addInt("ANN_MID_INSERT_PREEMPT_HARM_FULL_FOREIGN_OUTRANK", config.Search.MidInsertHarmFullForeignOutrank)
	addInt("ANN_MID_INSERT_PREEMPT_HARM_FULL_SELECTED_FRONTIER_TOUCHED", config.Search.MidInsertHarmFullSelectedFrontierTouched)
	addBool("ANN_MID_INSERT_PREEMPT_HARM_DEFER_ENABLED", config.Search.MidInsertHarmDeferEnabled)
	addInt("ANN_MID_INSERT_PREEMPT_HARM_DEFER_QUEUE_CAP", config.Search.MidInsertHarmDeferQueueCap)
	addInt("ANN_MID_INSERT_PREEMPT_HARM_DEFER_DRAIN_BUDGET", config.Search.MidInsertHarmDeferDrainBudget)
	addInt("ANN_MID_INSERT_PREEMPT_HARM_DEFER_HIGH_WATERMARK_PCT", config.Search.MidInsertHarmDeferHighWatermarkPct)

	if len(entries) == 0 {
		return func() {}, nil
	}

	snapshots := make([]envSnapshot, 0, len(entries))
	labels := make([]string, 0, len(entries))
	restore := func() {
		for i := len(snapshots) - 1; i >= 0; i-- {
			snap := snapshots[i]
			if snap.existed {
				_ = os.Setenv(snap.key, snap.value)
			} else {
				_ = os.Unsetenv(snap.key)
			}
		}
	}

	for _, entry := range entries {
		oldValue, existed := os.LookupEnv(entry.key)
		snapshots = append(snapshots, envSnapshot{
			key:     entry.key,
			value:   oldValue,
			existed: existed,
		})
		if err := os.Setenv(entry.key, entry.value); err != nil {
			restore()
			return nil, err
		}
		labels = append(labels, fmt.Sprintf("%s=%s", entry.key, entry.value))
	}

	log.Printf("Applied mid-insert preempt env overrides: %s", strings.Join(labels, ", "))
	return restore, nil
}
