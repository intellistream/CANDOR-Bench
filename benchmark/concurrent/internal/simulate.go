package internal

import (
	"fmt"
	"log"
	"math/rand"
)

const (
	partialModeLagInsert   = 1
	partialModeLeadInsert  = 2
	partialModeCarryInsert = 3
)

func normalizePartialMode(mode int) int {
	switch mode {
	case partialModeLagInsert, partialModeLeadInsert, partialModeCarryInsert:
		return mode
	default:
		return partialModeLagInsert
	}
}

func partialModeName(mode int) string {
	switch mode {
	case partialModeLeadInsert:
		return "lead_insert"
	case partialModeCarryInsert:
		return "carry_insert"
	case partialModeLagInsert:
		fallthrough
	default:
		return "lag_insert"
	}
}

func runSimulate(config *Config, data []float32, dataDim int, workloadQueries []float32) error {
	maxElements := config.Data.MaxElements
	begin := config.Data.BeginNum
	totalQueries := len(workloadQueries) / dataDim
	if totalQueries == 0 {
		return fmt.Errorf("simulate: no workload queries available")
	}

	simulateCfg := config.Compare.Simulate
	if simulateCfg == nil {
		return fmt.Errorf("simulate: simulate configuration missing")
	}

	partialCfg := &simulateCfg.Partial
	partialEnabled := simulateCfg.Enabled && partialCfg.Enabled
	if partialEnabled && partialCfg.Seed == 0 {
		partialCfg.Seed = 1
	}
	partialMode := normalizePartialMode(partialCfg.Mode)

	var gtProvider *incrementalGTProvider
	if partialEnabled {
		if config.Data.IncrGtPath != "" && config.Search.RecallAt > 0 {
			var err error
			gtProvider, err = loadIncrementalGTProvider(config.Data.IncrGtPath, uint32(config.Search.RecallAt))
			if err != nil {
				return fmt.Errorf("simulate: failed to initialize incremental ground truth loader: %w", err)
			}
		} else {
			log.Printf("Simulate: incremental ground truth unavailable; recall-based partial diff summaries will be skipped")
		}
	}

	if config.Workload.QueryMode == queryModeChasing || config.Workload.QueryMode == queryModePeeking {
		log.Printf("Simulate: skipping timestamp reassignment for Chasing/Peeking mode (preserving original relative order)")
		return nil
	}

	log.Printf("Lag timeline: initializing lead index (current state)")
	leadIndex := createAndBuildIndex(config, data, dataDim)
	defer leadIndex.Close()

	lagConfig := *config
	log.Printf("Lag timeline: initializing lag index (will replay snapshots)")
	lagIndex := createAndBuildIndex(&lagConfig, data, dataDim)
	defer lagIndex.Close()

	var partialResults []*SearchResult
	var carrySnapshot []byte
	if partialEnabled {
		leadIndex.EnableInsertTelemetry(true)
		partialResults = make([]*SearchResult, 0)
	}

	if !leadIndex.SupportsSnapshot() {
		return fmt.Errorf("simulate: index type %s does not support snapshots", config.Index.IndexType)
	}

	queryParams := BuildQueryParams(config)
	leadIndex.SetQueryParams(queryParams)
	lagIndex.SetQueryParams(queryParams)

	prevSnapshot, err := leadIndex.Snapshot()
	if err != nil {
		return fmt.Errorf("simulate: failed to capture initial snapshot: %w", err)
	}

	tags := make([]uint32, maxElements)
	for i := 0; i < maxElements; i++ {
		tags[i] = uint32(i)
	}

	consistentResults := make([]*SearchResult, 0)
	laggingResults := make([]*SearchResult, 0)

	offset := begin
	batchSize := config.Workload.BatchSize
	stage := 0
	cursor := 0

	for offset < maxElements {
		next := min(offset+batchSize, maxElements)
		if next <= offset {
			break
		}

		log.Printf("Trace stage %d: applying inserts %d -> %d", stage, offset, next)

		stageQueries, stageQueryTags, err := buildQueriesForMode(
			config.Workload.QueryMode,
			workloadQueries,
			data,
			dataDim,
			batchSize,
			config.StageQueryWindow(),
			totalQueries,
			offset,
			next,
			&cursor,
			config.Workload.ZipfianSkew,
		)
		if err != nil {
			return fmt.Errorf("simulate: stage %d query build failed: %w", stage, err)
		}
		if len(stageQueries) == 0 {
			log.Printf("Simulate: no queries issued at offset %d (stage %d)", next, stage)
			offset = next
			stage++
			continue
		}

		batchData := sliceVectorsRange(data, dataDim, offset, next)
		batchTags := tags[offset:next]

		if err := lagIndex.Restore(prevSnapshot); err != nil {
			return fmt.Errorf("simulate: restore stage %d failed: %w", stage, err)
		}
		lagIndex.SetQueryParams(queryParams)

		var stageStats []uint64
		var stageMax uint64
		if partialEnabled {
			stageStats = make([]uint64, len(batchData))
			if err := leadIndex.BatchPartialInsert(batchData, batchTags, nil, stageStats); err != nil {
				return fmt.Errorf("simulate: insert stage %d (offset %d->%d) failed: %w", stage, offset, next, err)
			}
			for _, v := range stageStats {
				if v > stageMax {
					stageMax = v
				}
			}
		} else {
			if err := leadIndex.BatchInsert(batchData, batchTags); err != nil {
				return fmt.Errorf("simulate: insert stage %d (offset %d->%d) failed: %w", stage, offset, next, err)
			}
		}

		consistentRaw, _, err := leadIndex.BatchSearch(stageQueries, uint32(config.Search.RecallAt), 0)
		if err != nil {
			return fmt.Errorf("simulate: search (current) stage %d failed: %w", stage, err)
		}

		consistentResults = appendSearchResults(consistentResults, consistentRaw, stageQueryTags, uint64(next))

		laggingRaw, _, err := lagIndex.BatchSearch(stageQueries, uint32(config.Search.RecallAt), 0)
		if err != nil {
			return fmt.Errorf("simulate: search (lagging) stage %d failed: %w", stage, err)
		}

		laggingResults = appendSearchResults(laggingResults, laggingRaw, stageQueryTags, uint64(next))

		fullSnapshot, err := leadIndex.Snapshot()
		if err != nil {
			return fmt.Errorf("simulate: snapshot stage %d failed: %w", stage, err)
		}

		var (
			mode2Data  [][]float32
			mode2Tags  []uint32
			mode2Stats []uint64
			mode2Max   uint64
		)

		if partialEnabled && partialMode == partialModeLeadInsert {
			futureEnd := next + batchSize
			if futureEnd > maxElements {
				futureEnd = maxElements
			}
			if futureEnd > next {
				mode2Data = sliceVectorsRange(data, dataDim, next, futureEnd)
				mode2Tags = tags[next:futureEnd]
				mode2Stats = make([]uint64, len(mode2Data))

				if err := lagIndex.Restore(fullSnapshot); err != nil {
					return fmt.Errorf("simulate: partial mode %s restore (stats) stage %d failed: %w", partialModeName(partialMode), stage, err)
				}
				lagIndex.SetQueryParams(queryParams)
				if err := lagIndex.BatchPartialInsert(mode2Data, mode2Tags, nil, mode2Stats); err != nil {
					return fmt.Errorf("simulate: partial mode %s stats stage %d failed: %w", partialModeName(partialMode), stage, err)
				}
				for _, v := range mode2Stats {
					if v > mode2Max {
						mode2Max = v
					}
				}
			} else {
				log.Printf("Simulate: partial mode %s has no future batch at stage %d", partialModeName(partialMode), stage)
			}
		}

		if partialEnabled {
			var (
				templateSnapshot []byte
				trialData        [][]float32
				trialTags        []uint32
				trialStats       []uint64
				trialMax         uint64
				trialOffset      uint64
				ready            = true
			)

			switch partialMode {
			case partialModeLeadInsert:
				if len(mode2Data) == 0 {
					log.Printf("Simulate: partial mode %s skipped at stage %d (no data)", partialModeName(partialMode), stage)
					ready = false
					break
				}
				templateSnapshot = fullSnapshot
				trialData = mode2Data
				trialTags = mode2Tags
				trialStats = mode2Stats
				trialMax = mode2Max
				trialOffset = uint64(next)
			case partialModeCarryInsert:
				templateSnapshot = prevSnapshot
				trialData = batchData
				trialTags = batchTags
				trialStats = stageStats
				trialMax = stageMax
				trialOffset = uint64(next)
			case partialModeLagInsert:
				fallthrough
			default:
				templateSnapshot = prevSnapshot
				trialData = batchData
				trialTags = batchTags
				trialStats = stageStats
				trialMax = stageMax
				trialOffset = uint64(offset + len(batchData))
			}

			if !ready {
			} else if len(trialData) == 0 {
				log.Printf("Simulate: partial mode %s has empty data at stage %d", partialModeName(partialMode), stage)
			} else {
				rng := rand.New(rand.NewSource(partialCfg.Seed + int64(stage)*1000))

				var baseSnapshot []byte
				if partialMode == partialModeCarryInsert {
					if carrySnapshot != nil {
						baseSnapshot = carrySnapshot
					} else {
						baseSnapshot = prevSnapshot
					}
				} else {
					baseSnapshot = templateSnapshot
				}
				if baseSnapshot == nil {
					baseSnapshot = prevSnapshot
				}

				if err := lagIndex.Restore(baseSnapshot); err != nil {
					return fmt.Errorf("simulate: partial restore stage %d failed: %w", stage, err)
				}
				lagIndex.SetQueryParams(queryParams)

				limits := make([]uint64, len(trialData))
				for i := 0; i < len(trialData); i++ {
					maxUpdates := uint64(1)
					if i < len(trialStats) && trialStats[i] > 0 {
						maxUpdates = trialStats[i]
					} else if trialMax > 0 {
						maxUpdates = trialMax
					}
					if maxUpdates == 0 {
						limits[i] = 0
						continue
					}
					limits[i] = uint64(rng.Int63n(int64(maxUpdates) + 1))
				}

				if err := lagIndex.BatchPartialInsert(trialData, trialTags, limits, nil); err != nil {
					return fmt.Errorf("simulate: partial insert stage %d failed: %w", stage, err)
				}

				partialRaw, _, err := lagIndex.BatchSearch(stageQueries, uint32(config.Search.RecallAt), 0)
				if err != nil {
					return fmt.Errorf("simulate: partial search stage %d failed: %w", stage, err)
				}
				partialResults = appendSearchResults(partialResults, partialRaw, stageQueryTags, trialOffset)

				if partialMode == partialModeCarryInsert {
					snap, err := lagIndex.Snapshot()
					if err != nil {
						return fmt.Errorf("simulate: partial snapshot stage %d failed: %w", stage, err)
					}
					carrySnapshot = snap
				}
			}
		}

		prevSnapshot = fullSnapshot

		offset = next
		stage++
	}
	prevSnapshot = nil

	resultPaths := buildResultPaths(config, leadIndex)
	log.Printf("Trace completed after %d stages; results base directory: %s", stage, resultPaths["base"])
	mainResPath := resultPaths["incr_res"]
	if err := ensureDir(mainResPath); err != nil {
		return fmt.Errorf("simulate: %w", err)
	}
	if err := DumpIncrResults(consistentResults, mainResPath); err != nil {
		return fmt.Errorf("simulate: failed to dump consistent results: %w", err)
	}

	lagResPath := resultPaths["lagging_incr_res"]
	if err := ensureDir(lagResPath); err != nil {
		return fmt.Errorf("simulate: %w", err)
	}
	if err := DumpIncrResults(laggingResults, lagResPath); err != nil {
		return fmt.Errorf("simulate: failed to dump lagging results: %w", err)
	}

	if partialEnabled && len(partialResults) > 0 {
		partialResPath := resultPaths["partial_incr_res"]
		if err := ensureDir(partialResPath); err != nil {
			return fmt.Errorf("simulate: %w", err)
		}
		if err := DumpIncrResults(partialResults, partialResPath); err != nil {
			return fmt.Errorf("simulate: failed to dump partial results: %w", err)
		}
		if config.Data.IncrGtPath != "" {
			partialRecallPath := resultPaths["partial_incr_recall"]
			if err := calcIncrRecall(
				defaultIncrRecallTool,
				partialResPath,
				config.Data.IncrGtPath,
				partialRecallPath,
				fmt.Sprintf("%s_partial", config.Index.IndexType),
				config.StageQueryWindow(),
				config.Search.RecallAt,
			); err != nil {
				return fmt.Errorf("simulate: incremental recall (partial) failed: %w", err)
			}
		}
	}

	if partialEnabled && gtProvider != nil {
		diffSummary, err := computePartialDiffDistribution(consistentResults, partialResults, uint32(config.Search.RecallAt), gtProvider)
		if err != nil {
			return fmt.Errorf("simulate: failed to compute partial diff summaries: %w", err)
		}
		if len(diffSummary) > 0 {
			diffPath := resultPaths["partial_diff_path"]
			if err := writePartialDiffDistribution(diffPath, diffSummary); err != nil {
				return fmt.Errorf("simulate: failed to write partial diff distribution: %w", err)
			}
			runDiffPlot(config, diffPath, "simulate")
		}
	} else if partialEnabled {
		log.Printf("Simulate: skipping partial diff CSVs because incremental ground truth was not loaded")
	}

	if config.Data.IncrGtPath != "" {
		mainRecallPath := resultPaths["incr_recall"]
		if err := calcIncrRecall(
			defaultIncrRecallTool,
			mainResPath,
			config.Data.IncrGtPath,
			mainRecallPath,
			config.Index.IndexType,
			config.StageQueryWindow(),
			config.Search.RecallAt,
		); err != nil {
			return fmt.Errorf("simulate: incremental recall (current) failed: %w", err)
		}

		lagRecallPath := resultPaths["lagging_incr_recall"]
		if err := calcIncrRecall(
			defaultIncrRecallTool,
			lagResPath,
			config.Data.IncrGtPath,
			lagRecallPath,
			fmt.Sprintf("%s_lagging", config.Index.IndexType),
			config.StageQueryWindow(),
			config.Search.RecallAt,
		); err != nil {
			return fmt.Errorf("simulate: incremental recall (lagging) failed: %w", err)
		}
	} else {
		log.Printf("Simulate: incremental recall disabled (missing incr_gt_path)")
	}

	log.Printf("Simulate completed after %d stages.", stage)
	return nil
}

func sliceVectorsRange(flat []float32, dim int, begin, end int) [][]float32 {
	length := end - begin
	if length <= 0 {
		return nil
	}

	result := make([][]float32, 0, length)
	for idx := begin; idx < end; idx++ {
		start := idx * dim
		result = append(result, flat[start:start+dim])
	}
	return result
}

func makeSequentialTags(begin, end int) []uint32 {
	length := end - begin
	if length <= 0 {
		return nil
	}

	tags := make([]uint32, length)
	for i := 0; i < length; i++ {
		tags[i] = uint32(begin + i)
	}
	return tags
}

func appendSearchResults(dst []*SearchResult, batch [][]uint32, queryTags []uint32, insertOffset uint64) []*SearchResult {
	for i, tags := range batch {
		qtag := uint64(0)
		if i < len(queryTags) {
			qtag = uint64(queryTags[i])
		}
		dst = append(dst, NewSearchResult(insertOffset, qtag, tags))
	}
	return dst
}
