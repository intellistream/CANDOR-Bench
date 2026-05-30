package internal

import (
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

var statsFile = "benchmark_results.csv"

func filteredVariantSuffix(config *Config) string {
	if config.VariantName == "" {
		return ""
	}
	parts := strings.Split(config.VariantName, "__")
	filtered := make([]string, 0, len(parts))
	seen := make(map[string]bool, len(parts))
	for _, part := range parts {
		if part == "" || seen[part] {
			continue
		}
		if strings.HasPrefix(part, "m") && len(part) > 1 && isDigit(part[1]) {
			continue
		}
		if strings.HasPrefix(part, "ef_construction") {
			continue
		}
		if strings.HasPrefix(part, "ef_search") {
			continue
		}
		if strings.HasPrefix(part, "num_threads") {
			continue
		}
		if strings.HasPrefix(part, "batch_size") {
			continue
		}
		if strings.HasPrefix(part, "use_node_lock") {
			continue
		}

		short := compactRateToken(part)
		seen[short] = true
		filtered = append(filtered, short)
	}
	if len(filtered) == 0 {
		return ""
	}
	return "_" + strings.Join(filtered, "_")
}

func EffectiveUseNodeLock(config *Config) (bool, bool) {
	var useNodeLock bool
	var applicable bool

	switch strings.ToLower(config.Index.IndexType) {
	case "hnsw", "hnsw-visible", "replica_snapshot", "replica_refresh", "segmented", "annchor", "annchor-m1", "annchor-m2", "annchor-preempt", "annchor-trim", "annchor-m3", "annchor-m3-off", "annchor-m3-prune-assist", "annchor-m3-append-only", "annchor-m3-ablation":
		useNodeLock = true
		if config.Search.UseNodeLock != nil {
			useNodeLock = *config.Search.UseNodeLock
		} else if config.Workload.UseNodeLock != nil {
			useNodeLock = *config.Workload.UseNodeLock
		}
		applicable = true
	default:
		return false, false
	}

	return useNodeLock, applicable
}

func compactRateToken(token string) string {
	if strings.HasPrefix(token, "w") && strings.Contains(token, "-r") {
		parts := strings.Split(token, "-")
		if len(parts) == 2 {
			wPart := compactPrefixedNumber(parts[0], "w")
			rPart := compactPrefixedNumber(parts[1], "r")
			return fmt.Sprintf("%s-%s", wPart, rPart)
		}
	}
	if !strings.HasPrefix(token, "r") {
		return token
	}
	body := strings.TrimPrefix(token, "r")
	parts := strings.Split(body, "-")
	if len(parts) != 2 {
		return token
	}
	a := compactNumber(parts[0])
	b := compactNumber(parts[1])
	if a == "" || b == "" {
		return token
	}
	return fmt.Sprintf("r%s-%s", a, b)
}

func compactPrefixedNumber(s, prefix string) string {
	if !strings.HasPrefix(s, prefix) {
		return s
	}
	val := strings.TrimPrefix(s, prefix)
	c := compactNumber(val)
	if c == "" {
		return s
	}
	return prefix + c
}

func compactNumber(raw string) string {
	n, err := strconv.Atoi(raw)
	if err != nil {
		return ""
	}
	switch {
	case n%1_000_000 == 0:
		return fmt.Sprintf("%dm", n/1_000_000)
	case n%1_000 == 0:
		return fmt.Sprintf("%dk", n/1_000)
	default:
		return raw
	}
}

func isDigit(b byte) bool {
	return b >= '0' && b <= '9'
}

func buildResultPaths(config *Config, index *Index) map[string]string {
	datasetName := config.Data.DatasetName

	querySource := config.Data.IncrQueryPath
	if querySource == "" {
		querySource = config.Data.QueryPath
	}
	if querySource == "" {
		querySource = config.Data.OverallQueryPath
	}
	queryPath := filepath.Base(querySource)
	queryName := strings.TrimSuffix(queryPath, ".bin")

	basePath := filepath.Join(config.Result.OutputDir, config.Index.IndexType, datasetName, queryName)
	filesPath := filepath.Join(basePath, "files")

	indexConfig := IndexConfig{
		IndexType:      config.Index.IndexType,
		M:              config.Index.M,
		EfConstruction: config.Index.EfConstruction,
		EfSearch:       config.Search.EfSearch,
		SealedType:     config.Index.SealedType,
		SealThreshold:  config.Index.SealThreshold,
	}
	algorithmSuffix := BuildAlgorithmSuffix(indexConfig)
	nodeLockSuffix := ""
	if useNodeLock, ok := EffectiveUseNodeLock(config); ok {
		if useNodeLock {
			nodeLockSuffix = "_nl1"
		} else {
			nodeLockSuffix = "_nl0"
		}
	}
	indexSuffix := fmt.Sprintf("%s_t%d%s", algorithmSuffix, config.Workload.NumThreads, nodeLockSuffix)

	variantSuffix := filteredVariantSuffix(config)
	workloadSuffix := fmt.Sprintf("_b%d", config.Workload.BatchSize)
	if variantSuffix != "" {
		workloadSuffix += variantSuffix
	}
	if config.Workload.QueueSize > 0 {
		workloadSuffix += fmt.Sprintf("_qs%d", config.Workload.QueueSize)
	}
	workloadSuffix += config.QueryModeSuffix()
	if config.Workload.WithExternalRWLock {
		workloadSuffix += "_wl"
	} else {
		workloadSuffix += "_wol"
	}

	fileName := fmt.Sprintf("%s%s", indexSuffix, workloadSuffix)

	paths := make(map[string]string)
	paths["base"] = basePath
	paths["overall_res"] = filepath.Join(filesPath, fileName+".res")
	incrementSuffix := fmt.Sprintf("_i%d_k%d", config.Workload.BatchSize, config.Search.RecallAt)
	paths["incr_res"] = filepath.Join(filesPath, fileName+incrementSuffix+".res")
	paths["incr_recall"] = filepath.Join(filesPath, fileName+incrementSuffix+".rc")
	paths["lagging_incr_res"] = filepath.Join(filesPath, fileName+incrementSuffix+"_lag.res")
	paths["lagging_incr_recall"] = filepath.Join(filesPath, fileName+incrementSuffix+"_lag.rc")
	partialMode := normalizePartialMode(partialModeLagInsert)
	partialSeed := int64(0)
	if simCfg := config.Compare.Simulate; simCfg != nil {
		partialMode = normalizePartialMode(simCfg.Partial.Mode)
		partialSeed = simCfg.Partial.Seed
	}
	partialSuffix := fmt.Sprintf("_partial_%s", partialModeName(partialMode))
	if partialSeed != 0 {
		partialSuffix = fmt.Sprintf("%s_s%d", partialSuffix, partialSeed)
	}
	paths["partial_incr_res"] = filepath.Join(filesPath, fileName+incrementSuffix+partialSuffix+".res")
	paths["partial_incr_recall"] = filepath.Join(filesPath, fileName+incrementSuffix+partialSuffix+".rc")
	paths["partial_diff_path"] = filepath.Join(filesPath, fileName+incrementSuffix+partialSuffix+"_diff_distribution.csv")

	if baselineAlgo, exists := baselineAlgoMap[config.Index.IndexType]; exists {
		baselineBasePath := filepath.Join(config.Result.OutputDir, baselineAlgo, datasetName, queryName)
		paths["baseline_incr_res"] = filepath.Join(filepath.Join(baselineBasePath, "files"), fileName+incrementSuffix+".res")
		paths["baseline_incr_recall"] = filepath.Join(filepath.Join(baselineBasePath, "files"), fileName+incrementSuffix+".rc")
	}

	return paths
}

func writeResultsToCSV(index *Index, config *Config, stats Stat, recall float64) error {
	outputDir := config.Result.OutputDir
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %v", err)
	}

	resultPath := filepath.Join(outputDir, statsFile)
	fileExists := false
	if _, err := os.Stat(resultPath); err == nil {
		fileExists = true
	}

	file, err := os.OpenFile(resultPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open result file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{
		"algorithm", "threads", "write_batch_size", "read_batch_size", "dataset_name", "variant_name", "query_mode", "insert_thread_ratio_env", "with_external_rw_lock", "use_node_lock", "insert_input_rate", "search_input_rate", "index_params", "query_params",
		"insert_qps (per-point)", "insert_mean_op_latency_ms (per-batch)", "insert_p95_op_latency_ms (per-batch)", "insert_p99_op_latency_ms (per-batch)",
		"insert_mean_e2e_latency_ms (per-batch)", "insert_p95_e2e_latency_ms (per-batch)", "insert_p99_e2e_latency_ms (per-batch)",
		"search_qps (per-point)", "search_mean_op_latency_ms (per-batch)", "search_p95_op_latency_ms (per-batch)", "search_p99_op_latency_ms (per-batch)",
		"search_mean_e2e_latency_ms (per-batch)", "search_p95_e2e_latency_ms (per-batch)", "search_p99_e2e_latency_ms (per-batch)",
		"freshness_total_pts_mean", "freshness_total_pts_p95", "freshness_total_pts_p99",
		"freshness_contiguity_pts_mean", "freshness_contiguity_pts_p95", "freshness_contiguity_pts_p99",
		"freshness_queue_pts_mean", "freshness_queue_pts_p95", "freshness_queue_pts_p99",
		"freshness_active_front_pts_mean", "freshness_active_front_pts_p95", "freshness_active_front_pts_p99",
		"freshness_missed_e2e_pts_mean", "freshness_missed_e2e_pts_p95", "freshness_missed_e2e_pts_p99",
		"inflight_bf_enabled", "inflight_join_enabled", "inflight_bf_pts_mean", "inflight_bf_pts_p95", "inflight_bf_pts_p99",
		"inflight_occ_merge_mean_ms", "inflight_occ_merge_p95_ms", "inflight_occ_merge_p99_ms", "inflight_occ_share_search_op", "inflight_occ_pct_search_op",
		"inflight_occ_l2_dists_mean", "inflight_occ_merge_comps_mean", "inflight_occ_filter_tests_mean", "inflight_occ_candidates_mean",
		"result_below_k_count", "recall", "peak_memory_mb", "avg_memory_mb", "stats",
	}

	if !fileExists {
		if err := writer.Write(header); err != nil {
			return fmt.Errorf("failed to write header: %v", err)
		}
	}

	indexParams := index.DumpIndexParams()
	queryParams := index.DumpQueryParams()
	statsStr := index.DumpStats()
	if stats.ExtraStats != "" {
		if statsStr != "" {
			statsStr += ", "
		}
		statsStr += stats.ExtraStats
	}

	useNodeLockStr := "n/a"
	if useNodeLock, ok := EffectiveUseNodeLock(config); ok {
		useNodeLockStr = fmt.Sprintf("%t", useNodeLock)
	}
	insertThreadRatio := os.Getenv("INSERT_RATIO")
	if insertThreadRatio == "" {
		insertThreadRatio = "50"
	}

	row := []string{
		config.Index.IndexType,
		fmt.Sprintf("%d", config.Workload.NumThreads),
		fmt.Sprintf("%d", config.Workload.BatchSize),
		fmt.Sprintf("%d", config.Workload.BatchSize),
		config.Data.DatasetName,
		config.VariantName,
		config.Workload.QueryMode,
		insertThreadRatio,
		fmt.Sprintf("%t", config.Workload.WithExternalRWLock),
		useNodeLockStr,
		fmt.Sprintf("%.2f", config.InsertRate()),
		fmt.Sprintf("%.2f", config.SearchRate()),
		indexParams,
		queryParams,
		fmt.Sprintf("%.4f", stats.InsertQPS),
		fmt.Sprintf("%.4f", stats.MeanInsertOpLatency),
		fmt.Sprintf("%.4f", stats.P95InsertOpLatency),
		fmt.Sprintf("%.4f", stats.P99InsertOpLatency),
		fmt.Sprintf("%.4f", stats.MeanInsertE2ELatency),
		fmt.Sprintf("%.4f", stats.P95InsertE2ELatency),
		fmt.Sprintf("%.4f", stats.P99InsertE2ELatency),
		fmt.Sprintf("%.4f", stats.SearchQPS),
		fmt.Sprintf("%.4f", stats.MeanSearchOpLatency),
		fmt.Sprintf("%.4f", stats.P95SearchOpLatency),
		fmt.Sprintf("%.4f", stats.P99SearchOpLatency),
		fmt.Sprintf("%.4f", stats.MeanSearchE2ELatency),
		fmt.Sprintf("%.4f", stats.P95SearchE2ELatency),
		fmt.Sprintf("%.4f", stats.P99SearchE2ELatency),
		fmt.Sprintf("%.4f", stats.MeanFreshnessTotalPts),
		fmt.Sprintf("%.4f", stats.P95FreshnessTotalPts),
		fmt.Sprintf("%.4f", stats.P99FreshnessTotalPts),
		fmt.Sprintf("%.4f", stats.MeanFreshnessContiguityPts),
		fmt.Sprintf("%.4f", stats.P95FreshnessContiguityPts),
		fmt.Sprintf("%.4f", stats.P99FreshnessContiguityPts),
		fmt.Sprintf("%.4f", stats.MeanFreshnessQueuePts),
		fmt.Sprintf("%.4f", stats.P95FreshnessQueuePts),
		fmt.Sprintf("%.4f", stats.P99FreshnessQueuePts),
		fmt.Sprintf("%.4f", stats.MeanFreshnessActiveFrontPts),
		fmt.Sprintf("%.4f", stats.P95FreshnessActiveFrontPts),
		fmt.Sprintf("%.4f", stats.P99FreshnessActiveFrontPts),
		fmt.Sprintf("%.4f", stats.MeanFreshnessMissedE2EPts),
		fmt.Sprintf("%.4f", stats.P95FreshnessMissedE2EPts),
		fmt.Sprintf("%.4f", stats.P99FreshnessMissedE2EPts),
		fmt.Sprintf("%t", stats.InflightBruteforceEnabled),
		fmt.Sprintf("%t", stats.InflightJoinEnabled),
		fmt.Sprintf("%.4f", stats.MeanInflightBruteforcePts),
		fmt.Sprintf("%.4f", stats.P95InflightBruteforcePts),
		fmt.Sprintf("%.4f", stats.P99InflightBruteforcePts),
		fmt.Sprintf("%.4f", stats.MeanInflightOccMergeLatency),
		fmt.Sprintf("%.4f", stats.P95InflightOccMergeLatency),
		fmt.Sprintf("%.4f", stats.P99InflightOccMergeLatency),
		fmt.Sprintf("%.6f", stats.MeanInflightOccShare),
		fmt.Sprintf("%.2f", stats.MeanInflightOccShare*100.0),
		fmt.Sprintf("%.4f", stats.MeanInflightOccL2Dists),
		fmt.Sprintf("%.4f", stats.MeanInflightOccMergeComps),
		fmt.Sprintf("%.4f", stats.MeanInflightOccFilterTests),
		fmt.Sprintf("%.4f", stats.MeanInflightOccCandidates),
		fmt.Sprintf("%d", stats.ResultBelowKCount),
		fmt.Sprintf("%.3f", recall),
		fmt.Sprintf("%.4f", stats.PeakMemoryMB),
		fmt.Sprintf("%.4f", stats.AvgMemoryMB),
		statsStr,
	}

	if err := writer.Write(row); err != nil {
		return fmt.Errorf("failed to write data row: %v", err)
	}

	log.Printf("Results written to: %s", resultPath)
	return nil
}

func calcOverallRecallForIndex(index *Index, config *Config, queries []float32, dataDim int) (float64, error) {
	if config.Data.OverallGtPath == "" {
		log.Println("No ground truth or recall tool path provided, skipping overall recall check")
		return 0, nil
	}

	if len(queries) == 0 {
		return 0, fmt.Errorf("no queries available for overall recall calculation")
	}

	log.Println("Calculating recall against ground truth ...")

	totalQueries := len(queries) / dataDim
	if totalQueries == 0 {
		return 0, fmt.Errorf("overall recall requested but no queries available")
	}

	targetQueries, err := determineOverallQueryCount(config.Data.OverallGtPath, totalQueries)
	if err != nil {
		return 0, err
	}
	if targetQueries < totalQueries {
		log.Printf("Overall recall: limiting queries from %d to %d to match ground truth", totalQueries, targetQueries)
	}

	index.SetQueryParams(BuildQueryParams(config))

	recallAt := config.Search.RecallAt

	resultPaths := buildResultPaths(config, index)
	outPath := resultPaths["overall_res"]

	if err := ensureDir(outPath); err != nil {
		return 0, err
	}
	file, err := os.Create(outPath)
	if err != nil {
		return 0, fmt.Errorf("failed to create result file: %v", err)
	}
	defer file.Close()

	batchedQueries := make([][]float32, targetQueries)
	for i := 0; i < targetQueries; i++ {
		batchedQueries[i] = queries[i*dataDim : (i+1)*dataDim]
	}
	const noVisibilityCap = ^uint64(0)
	tags, _, err := index.BatchSearch(batchedQueries, uint32(recallAt), noVisibilityCap)
	if err != nil {
		return 0, fmt.Errorf("batch search error: %v", err)
	}

	n := int32(len(tags))
	k := int32(len(tags[0]))
	if err := binary.Write(file, binary.LittleEndian, n); err != nil {
		return 0, fmt.Errorf("failed to write n: %v", err)
	}
	if err := binary.Write(file, binary.LittleEndian, k); err != nil {
		return 0, fmt.Errorf("failed to write k: %v", err)
	}
	for _, result := range tags {
		for _, tag := range result {
			if err := binary.Write(file, binary.LittleEndian, tag); err != nil {
				return 0, fmt.Errorf("failed to write tag: %v", err)
			}
		}
	}

	if err := file.Sync(); err != nil {
		return 0, fmt.Errorf("failed to sync result file: %v", err)
	}

	gtPath := config.Data.OverallGtPath
	recallToolPath := defaultOverallRecallTool
	log.Printf("Calculating overall recall: tool=%s gt=%s res=%s k=%d", recallToolPath, gtPath, outPath, recallAt)
	recall, err := CalculateRecall(recallToolPath, gtPath, outPath, int(recallAt))
	if err != nil {
		return 0, fmt.Errorf("failed to calculate recall: %v", err)
	}
	log.Printf("Overall recall@%d = %.4f", recallAt, recall)

	return recall, nil
}

func determineOverallQueryCount(gtPath string, available int) (int, error) {
	file, err := os.Open(gtPath)
	if err != nil {
		return 0, fmt.Errorf("failed to open ground truth file: %v", err)
	}
	defer file.Close()

	var n int32
	if err := binary.Read(file, binary.LittleEndian, &n); err != nil {
		return 0, fmt.Errorf("failed to read ground truth query count: %v", err)
	}
	if n <= 0 {
		return 0, fmt.Errorf("ground truth file reports non-positive query count: %d", n)
	}

	var k int32
	if err := binary.Read(file, binary.LittleEndian, &k); err != nil {
		return 0, fmt.Errorf("failed to read ground truth top-k value: %v", err)
	}
	if k <= 0 {
		return 0, fmt.Errorf("ground truth file reports non-positive top-k value: %d", k)
	}

	if available < int(n) {
		return 0, fmt.Errorf("overall recall: only %d queries available but ground truth expects %d", available, n)
	}
	return min(available, int(n)), nil
}

func finishBench(index *Index, queries []float32, dataDim int, config *Config, stats Stat) {
	var recall float64
	var err error
	if config.Data.OverallGtPath != "" {
		recall, err = calcOverallRecallForIndex(index, config, queries, dataDim)
		if err != nil {
			log.Printf("WARNING: failed to check recall: %v (continuing anyway)", err)
		}
	}

	if err := writeResultsToCSV(index, config, stats, recall); err != nil {
		log.Fatalf("failed to write results to CSV: %v", err)
	}
}

func calcIncrRecall(toolPath, resPath, gtPath, recallPath, algoName string, windowSize int, recallAt uint32) error {
	cmd := exec.Command(toolPath,
		"--gt_path", gtPath,
		"--res_path", resPath,
		"--recall_path", recallPath,
		"--k", fmt.Sprintf("%d", recallAt),
		"--inc", fmt.Sprintf("%d", windowSize),
	)
	log.Printf("Calculating incremental recall for %s, cmd: %s", algoName, strings.Join(cmd.Args, " "))

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to calculate incremental recall for %s cmd: %v\nOutput:\n%s", algoName, err, string(output))
	}
	log.Printf("Incremental recall for %s calculated", algoName)

	return nil
}

func GenerateDiffFileName(woPath, wPath string) string {
	woFile := strings.TrimSuffix(filepath.Base(woPath), filepath.Ext(woPath))
	wFile := strings.TrimSuffix(filepath.Base(wPath), filepath.Ext(wPath))
	diffFile := fmt.Sprintf("%s_vs_%s_diff.csv", woFile, wFile)

	if strings.Contains(woFile, "_wol") && strings.Contains(wFile, "_wl") {
		prefixWo := strings.Split(woFile, "_wol")[0]
		prefixW := strings.Split(wFile, "_wl")[0]
		if prefixWo == prefixW {
			suffixWo := strings.TrimPrefix(woFile, prefixWo+"_wol")
			suffixW := strings.TrimPrefix(wFile, prefixW+"_wl")
			if suffixWo == suffixW {
				lastUnderscore := strings.LastIndex(prefixWo, "_")
				if lastUnderscore != -1 {
					commonPart := prefixWo[:lastUnderscore]
					ratePart := prefixWo[lastUnderscore+1:]

					if strings.HasPrefix(ratePart, "r") || strings.HasPrefix(ratePart, "w") {
						return fmt.Sprintf("%s%s_%s-wol_vs_wl_diff.csv", commonPart, suffixWo, ratePart)
					}
				}
				diffFile = fmt.Sprintf("%s-wol_vs_wl%s_diff.csv", prefixWo, suffixWo)
			}
		}
	}
	return diffFile
}
