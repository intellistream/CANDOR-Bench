package internal

import (
	"context"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/time/rate"
)

type scheduledInsertTask struct {
	Begin        int
	End          int
	ReleaseAfter time.Duration
	BurstIndex   int
	BurstStart   time.Duration
}

type scheduledSearchTask struct {
	InsertOffset uint64
	QueryTags    []uint32
	ReleaseAfter time.Duration
	Measured     bool
}

type producerSchedule struct {
	InsertTasks   []scheduledInsertTask
	SearchTasks   []scheduledSearchTask
	InsertHorizon time.Duration
}

func measuredSearchEvery() int {
	value, err := strconv.Atoi(os.Getenv("MEASURED_SEARCH_EVERY_N"))
	if err != nil || value <= 1 {
		return 1
	}
	return value
}

func isMeasuredSearch(seq int, every int) bool {
	return every <= 1 || seq%every == 0
}

func searchRateFromEnv(name string) (float64, bool, error) {
	raw := os.Getenv(name)
	if raw == "" {
		return 0, false, nil
	}
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, true, fmt.Errorf("%s must be numeric: %w", name, err)
	}
	return value, true, nil
}

func searchRateListFromEnv(name string) ([]float64, error) {
	raw := os.Getenv(name)
	if raw == "" {
		return nil, nil
	}
	parts := strings.Split(raw, ",")
	values := make([]float64, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		value, err := strconv.ParseFloat(part, 64)
		if err != nil {
			return nil, fmt.Errorf("%s must be a comma-separated numeric list: %w", name, err)
		}
		if value < 0 {
			return nil, fmt.Errorf("%s cannot contain negative rates", name)
		}
		values = append(values, value)
	}
	return values, nil
}

func periodRate(values []float64, fallback float64, idx int) float64 {
	if len(values) == 0 {
		return fallback
	}
	if idx < len(values) {
		return values[idx]
	}
	return values[len(values)-1]
}

func setLimiterEventRate(limiter *rate.Limiter, rateValue float64, batchSize int) {
	if limiter == nil || rateValue <= 0 {
		return
	}
	if batchSize <= 0 {
		batchSize = 1
	}
	limiter.SetLimit(rate.Limit(rateValue / float64(batchSize)))
}

func commitGatedBurstsEnabled() bool {
	raw := os.Getenv("COMMIT_GATED_INSERT_BURSTS")
	return raw == "1" || raw == "true" || raw == "TRUE"
}

func commitGatedCooldown() time.Duration {
	raw := os.Getenv("COMMIT_GATED_COOLDOWN_MS")
	if raw == "" {
		return 500 * time.Millisecond
	}
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil || value < 0 {
		return 500 * time.Millisecond
	}
	return time.Duration(value * float64(time.Millisecond))
}

func pauseBackgroundSearchDuringInsert() bool {
	raw := os.Getenv("BACKGROUND_SEARCH_PAUSE_DURING_INSERT")
	if raw == "" {
		return true
	}
	return !(raw == "0" || raw == "false" || raw == "FALSE")
}

func buildProducerSchedule(cfg *Config, totalQueries int) (*producerSchedule, error) {
	insertTasks, insertHorizon, err := buildScheduledInsertTasks(cfg)
	if err != nil {
		return nil, err
	}
	if cfg.Workload.ScheduleHorizonMs > 0 {
		explicitHorizon := time.Duration(cfg.Workload.ScheduleHorizonMs * float64(time.Millisecond))
		if explicitHorizon > insertHorizon {
			insertHorizon = explicitHorizon
		}
	}

	schedule := &producerSchedule{
		InsertTasks:   insertTasks,
		InsertHorizon: insertHorizon,
	}

	if cfg.Search.RecallAt == 0 || cfg.Workload.SearchEventRate <= 0 {
		return schedule, nil
	}

	searchTasks, err := buildScheduledSearchTasks(cfg, totalQueries, insertTasks, insertHorizon)
	if err != nil {
		return nil, err
	}
	schedule.SearchTasks = searchTasks
	return schedule, nil
}

func buildScheduledInsertTasks(cfg *Config) ([]scheduledInsertTask, time.Duration, error) {
	if len(cfg.Workload.InsertBurstSchedule) > 0 {
		return buildBurstScheduledInsertTasks(cfg)
	}
	return buildUniformScheduledInsertTasks(
		cfg.Data.BeginNum,
		cfg.Data.MaxElements,
		cfg.Workload.BatchSize,
		cfg.Workload.InsertEventRate,
	)
}

func buildUniformScheduledInsertTasks(begin, maxElements, batchSize int, insertRate float64) ([]scheduledInsertTask, time.Duration, error) {
	if batchSize <= 0 {
		return nil, 0, fmt.Errorf("scheduled producer requires batch_size > 0")
	}
	if maxElements <= begin {
		return nil, 0, nil
	}

	limiter := buildLimiter(insertRate, batchSize)
	current := time.Unix(0, 0)
	zero := current
	tasks := make([]scheduledInsertTask, 0, (maxElements-begin+batchSize-1)/batchSize)

	for offset := begin; offset < maxElements; {
		next := min(offset+batchSize, maxElements)
		releaseAfter, err := nextScheduledRelease(limiter, &current, zero)
		if err != nil {
			return nil, 0, err
		}
		tasks = append(tasks, scheduledInsertTask{
			Begin:        offset,
			End:          next,
			ReleaseAfter: releaseAfter,
			BurstIndex:   0,
			BurstStart:   0,
		})
		offset = next
	}

	if len(tasks) == 0 {
		return tasks, 0, nil
	}
	return tasks, tasks[len(tasks)-1].ReleaseAfter, nil
}

func buildBurstScheduledInsertTasks(cfg *Config) ([]scheduledInsertTask, time.Duration, error) {
	batchSize := cfg.Workload.BatchSize
	if batchSize <= 0 {
		return nil, 0, fmt.Errorf("scheduled producer requires batch_size > 0")
	}
	if cfg.Data.MaxElements <= cfg.Data.BeginNum {
		return nil, 0, nil
	}

	offset := cfg.Data.BeginNum
	tasks := make([]scheduledInsertTask, 0, (cfg.Data.MaxElements-cfg.Data.BeginNum+batchSize-1)/batchSize)
	horizon := time.Duration(0)
	for i, window := range cfg.Workload.InsertBurstSchedule {
		if window.StartMs < 0 {
			return nil, 0, fmt.Errorf("insert_burst_schedule[%d].start_ms cannot be negative", i)
		}
		if window.DurationMs <= 0 {
			return nil, 0, fmt.Errorf("insert_burst_schedule[%d].duration_ms must be positive", i)
		}
		rate := window.InsertEventRate
		if rate <= 0 {
			rate = cfg.Workload.InsertEventRate
		}
		if rate <= 0 {
			return nil, 0, fmt.Errorf("insert_burst_schedule[%d] needs insert_event_rate > 0", i)
		}

		start := time.Duration(window.StartMs * float64(time.Millisecond))
		end := start + time.Duration(window.DurationMs*float64(time.Millisecond))
		if end > horizon {
			horizon = end
		}
		interval := time.Duration(float64(batchSize) / rate * float64(time.Second))
		if interval <= 0 {
			return nil, 0, fmt.Errorf("insert_burst_schedule[%d] rate %.6g is too high for batch_size=%d", i, rate, batchSize)
		}

		for releaseAfter := start; releaseAfter < end && offset < cfg.Data.MaxElements; releaseAfter += interval {
			next := min(offset+batchSize, cfg.Data.MaxElements)
			tasks = append(tasks, scheduledInsertTask{
				Begin:        offset,
				End:          next,
				ReleaseAfter: releaseAfter,
				BurstIndex:   i,
				BurstStart:   start,
			})
			offset = next
		}
	}
	if offset < cfg.Data.MaxElements {
		return nil, 0, fmt.Errorf(
			"insert_burst_schedule capacity ended at offset %d, need max_elements=%d",
			offset,
			cfg.Data.MaxElements,
		)
	}
	if len(tasks) > 0 && tasks[len(tasks)-1].ReleaseAfter > horizon {
		horizon = tasks[len(tasks)-1].ReleaseAfter
	}
	return tasks, horizon, nil
}

func buildScheduledSearchTasks(cfg *Config, totalQueries int, insertTasks []scheduledInsertTask, insertHorizon time.Duration) ([]scheduledSearchTask, error) {
	measuredRate, measuredSet, err := searchRateFromEnv("MEASURED_SEARCH_EVENT_RATE")
	if err != nil {
		return nil, err
	}
	backgroundRate, _, err := searchRateFromEnv("BACKGROUND_SEARCH_EVENT_RATE")
	if err != nil {
		return nil, err
	}
	backgroundRates, err := searchRateListFromEnv("BACKGROUND_SEARCH_EVENT_RATES")
	if err != nil {
		return nil, err
	}
	if len(backgroundRates) > 0 {
		backgroundRate = backgroundRates[0]
	}
	if !measuredSet {
		measuredRate = cfg.Workload.SearchEventRate
	}
	if measuredRate <= 0 && backgroundRate <= 0 {
		return nil, nil
	}
	if !measuredSet && backgroundRate <= 0 {
		return buildScheduledSearchLane(cfg, totalQueries, insertTasks, insertHorizon, cfg.Workload.SearchEventRate, false, false, true)
	}

	var searchTasks []scheduledSearchTask
	measuredTasks, err := buildScheduledSearchLane(cfg, totalQueries, insertTasks, insertHorizon, measuredRate, true, false, false)
	if err != nil {
		return nil, err
	}
	searchTasks = append(searchTasks, measuredTasks...)
	backgroundTasks, err := buildScheduledSearchLane(cfg, totalQueries, insertTasks, insertHorizon, backgroundRate, false, true, false)
	if err != nil {
		return nil, err
	}
	searchTasks = append(searchTasks, backgroundTasks...)
	sort.SliceStable(searchTasks, func(i, j int) bool {
		return searchTasks[i].ReleaseAfter < searchTasks[j].ReleaseAfter
	})
	return searchTasks, nil
}

func buildScheduledSearchLane(
	cfg *Config,
	totalQueries int,
	insertTasks []scheduledInsertTask,
	insertHorizon time.Duration,
	searchRate float64,
	measured bool,
	skipInsertWindows bool,
	useEveryNthMeasurement bool,
) ([]scheduledSearchTask, error) {
	limiter := buildLimiter(searchRate, cfg.Workload.BatchSize)
	if limiter == nil {
		return nil, nil
	}

	current := time.Unix(0, 0)
	zero := current
	plannedOffset := cfg.Data.BeginNum
	insertIdx := 0
	cursor := 0
	searchTasks := make([]scheduledSearchTask, 0)
	measuredEvery := measuredSearchEvery()
	searchSeq := 0

	for {
		releaseAfter, err := nextScheduledRelease(limiter, &current, zero)
		if err != nil {
			return nil, err
		}
		if releaseAfter > insertHorizon {
			break
		}
		if skipInsertWindows && inInsertWindow(cfg, releaseAfter) {
			continue
		}

		for insertIdx < len(insertTasks) && insertTasks[insertIdx].ReleaseAfter <= releaseAfter {
			plannedOffset = insertTasks[insertIdx].End
			insertIdx++
		}

		tags, err := buildQueryTagsForMode(
			cfg.Workload.QueryMode,
			cfg.Workload.BatchSize,
			cfg.StageQueryWindow(),
			totalQueries,
			plannedOffset,
			&cursor,
			cfg.Workload.ZipfianSkew,
		)
		if err != nil {
			return nil, err
		}
		if len(tags) == 0 {
			continue
		}

		searchTasks = append(searchTasks, scheduledSearchTask{
			InsertOffset: uint64(plannedOffset),
			QueryTags:    append([]uint32(nil), tags...),
			ReleaseAfter: releaseAfter,
			Measured:     measured || (useEveryNthMeasurement && isMeasuredSearch(searchSeq, measuredEvery)),
		})
		searchSeq++
	}

	return searchTasks, nil
}

func inInsertWindow(cfg *Config, releaseAfter time.Duration) bool {
	for _, window := range cfg.Workload.InsertBurstSchedule {
		start := time.Duration(window.StartMs * float64(time.Millisecond))
		end := start + time.Duration(window.DurationMs*float64(time.Millisecond))
		if releaseAfter >= start && releaseAfter < end {
			return true
		}
	}
	return false
}

func nextScheduledRelease(limiter *rate.Limiter, current *time.Time, zero time.Time) (time.Duration, error) {
	if limiter == nil {
		return current.Sub(zero), nil
	}
	reservation := limiter.ReserveN(*current, 1)
	if !reservation.OK() {
		return 0, fmt.Errorf("failed to reserve scheduled producer event")
	}
	delay := reservation.DelayFrom(*current)
	*current = current.Add(delay)
	return current.Sub(zero), nil
}

func StartScheduledProducers(
	cfg *Config,
	schedule *producerSchedule,
	data []float32,
	dataDim int,
	workloadQueries []float32,
	insertQueue chan<- Task,
	searchQueue chan<- Task,
	measuredSearchQueue chan<- Task,
	committedOffset *atomic.Uint64,
) (<-chan int, error) {
	if schedule == nil {
		return nil, fmt.Errorf("scheduled producer requires a non-nil schedule")
	}
	if commitGatedBurstsEnabled() && len(cfg.Workload.InsertBurstSchedule) > 0 {
		if committedOffset == nil {
			return nil, fmt.Errorf("commit-gated scheduled producer requires committed offset")
		}
		return StartCommitGatedScheduledProducers(
			cfg,
			schedule,
			data,
			dataDim,
			workloadQueries,
			insertQueue,
			searchQueue,
			measuredSearchQueue,
			committedOffset,
		)
	}

	stageCountCh := make(chan int, 1)
	go func() {
		defer close(stageCountCh)

		base := time.Now()
		var wg sync.WaitGroup
		wg.Add(2)

		go func() {
			defer wg.Done()
			defer close(insertQueue)
			for _, task := range schedule.InsertTasks {
				scheduledAt := base.Add(task.ReleaseAfter)
				sleepUntil(scheduledAt)
				releaseRawNs := MonotonicRawNs()
				insertQueue <- Task{
					Type:         InsertTask,
					Data:         sliceVectorsRange(data, dataDim, task.Begin, task.End),
					Tags:         makeSequentialTags(task.Begin, task.End),
					InsertOffset: uint64(task.End),
					CreateTime:   scheduledAt,
					CreateRawNs:  releaseRawNs,
				}
			}
		}()

		go func() {
			defer wg.Done()
			defer close(searchQueue)
			if measuredSearchQueue != nil {
				defer close(measuredSearchQueue)
			}
			if cfg.Search.RecallAt == 0 {
				return
			}
			for _, task := range schedule.SearchTasks {
				scheduledAt := base.Add(task.ReleaseAfter)
				sleepUntil(scheduledAt)
				queries := buildQueriesFromTags(workloadQueries, dataDim, task.QueryTags)
				if len(queries) == 0 {
					continue
				}
				releaseRawNs := MonotonicRawNs()
				out := searchQueue
				if task.Measured && measuredSearchQueue != nil && measuredLanes() > 0 {
					out = measuredSearchQueue
				}
				out <- Task{
					Type:          SearchTask,
					Data:          queries,
					Tags:          task.QueryTags,
					RecallAt:      cfg.Search.RecallAt,
					InsertOffset:  task.InsertOffset,
					WaitForInsert: false,
					CreateTime:    scheduledAt,
					CreateRawNs:   releaseRawNs,
					Measured:      task.Measured,
				}
			}
		}()

		wg.Wait()
		stageCountCh <- len(schedule.InsertTasks)
	}()

	return stageCountCh, nil
}

func sleepUntil(target time.Time) {
	if delay := time.Until(target); delay > 0 {
		time.Sleep(delay)
	}
}

func groupedInsertBursts(tasks []scheduledInsertTask) [][]scheduledInsertTask {
	if len(tasks) == 0 {
		return nil
	}
	groups := make([][]scheduledInsertTask, 0)
	for _, task := range tasks {
		if len(groups) == 0 || groups[len(groups)-1][0].BurstIndex != task.BurstIndex {
			groups = append(groups, []scheduledInsertTask{task})
			continue
		}
		groups[len(groups)-1] = append(groups[len(groups)-1], task)
	}
	return groups
}

func StartCommitGatedScheduledProducers(
	cfg *Config,
	schedule *producerSchedule,
	data []float32,
	dataDim int,
	workloadQueries []float32,
	insertQueue chan<- Task,
	searchQueue chan<- Task,
	measuredSearchQueue chan<- Task,
	committedOffset *atomic.Uint64,
) (<-chan int, error) {
	stageCountCh := make(chan int, 1)
	measuredRate, measuredSet, err := searchRateFromEnv("MEASURED_SEARCH_EVENT_RATE")
	if err != nil {
		return nil, err
	}
	backgroundRate, _, err := searchRateFromEnv("BACKGROUND_SEARCH_EVENT_RATE")
	if err != nil {
		return nil, err
	}
	backgroundRates, err := searchRateListFromEnv("BACKGROUND_SEARCH_EVENT_RATES")
	if err != nil {
		return nil, err
	}
	if len(backgroundRates) > 0 {
		backgroundRate = backgroundRates[0]
	}
	if !measuredSet {
		measuredRate = cfg.Workload.SearchEventRate
	}

	go func() {
		defer close(stageCountCh)

		base := time.Now()
		totalQueries := len(workloadQueries) / dataDim
		stopSearch := make(chan struct{})
		var insertActive atomic.Bool
		var searchWG sync.WaitGroup

		startSearchLane := func(rateValue float64, measured bool, pauseDuringInsert bool) *rate.Limiter {
			if rateValue <= 0 || cfg.Search.RecallAt == 0 {
				return nil
			}
			limiter := buildLimiter(rateValue, cfg.Workload.BatchSize)
			if limiter == nil {
				return nil
			}
			searchWG.Add(1)
			go func() {
				defer searchWG.Done()
				cursor := 0
				for {
					select {
					case <-stopSearch:
						return
					default:
					}
					if pauseDuringInsert && insertActive.Load() {
						time.Sleep(time.Millisecond)
						continue
					}
					if err := limiter.Wait(context.Background()); err != nil {
						log.Printf("Scheduled producer: search limiter wait failed: %v", err)
						return
					}
					select {
					case <-stopSearch:
						return
					default:
					}
					if pauseDuringInsert && insertActive.Load() {
						continue
					}
					view := int(committedOffset.Load())
					tags, err := buildQueryTagsForMode(
						cfg.Workload.QueryMode,
						cfg.Workload.BatchSize,
						cfg.StageQueryWindow(),
						totalQueries,
						view,
						&cursor,
						cfg.Workload.ZipfianSkew,
					)
					if err != nil || len(tags) == 0 {
						if err != nil {
							log.Printf("Scheduled producer: search tags failed: %v", err)
						}
						continue
					}
					releaseRawNs := MonotonicRawNs()
					out := searchQueue
					if measured && measuredSearchQueue != nil && measuredLanes() > 0 {
						out = measuredSearchQueue
					}
					out <- Task{
						Type:          SearchTask,
						Data:          buildQueriesFromTags(workloadQueries, dataDim, tags),
						Tags:          tags,
						RecallAt:      cfg.Search.RecallAt,
						InsertOffset:  uint64(view),
						WaitForInsert: false,
						CreateTime:    time.Now(),
						CreateRawNs:   releaseRawNs,
						Measured:      measured,
					}
				}
			}()
			return limiter
		}

		startSearchLane(measuredRate, true, false)
		backgroundLimiter := startSearchLane(backgroundRate, false, pauseBackgroundSearchDuringInsert())

		bursts := groupedInsertBursts(schedule.InsertTasks)
		cooldown := commitGatedCooldown()
		stageCount := 0
		for burstIdx, burst := range bursts {
			if len(burst) == 0 {
				continue
			}
			var burstBase time.Time
			if burstIdx == 0 {
				burstBase = base.Add(burst[0].BurstStart)
				sleepUntil(burstBase)
			} else {
				burstBase = time.Now()
			}
			insertActive.Store(true)
			for _, task := range burst {
				scheduledAt := burstBase.Add(task.ReleaseAfter - task.BurstStart)
				sleepUntil(scheduledAt)
				releaseRawNs := MonotonicRawNs()
				insertQueue <- Task{
					Type:         InsertTask,
					Data:         sliceVectorsRange(data, dataDim, task.Begin, task.End),
					Tags:         makeSequentialTags(task.Begin, task.End),
					InsertOffset: uint64(task.End),
					CreateTime:   scheduledAt,
					CreateRawNs:  releaseRawNs,
				}
				stageCount++
			}
			target := uint64(burst[len(burst)-1].End)
			for committedOffset.Load() < target {
				time.Sleep(time.Millisecond)
			}
			if burstIdx+1 < len(bursts) {
				setLimiterEventRate(backgroundLimiter, periodRate(backgroundRates, backgroundRate, burstIdx+1), cfg.Workload.BatchSize)
			}
			insertActive.Store(false)
			if cooldown > 0 {
				time.Sleep(cooldown)
			}
		}

		close(insertQueue)
		close(stopSearch)
		searchWG.Wait()
		close(searchQueue)
		if measuredSearchQueue != nil {
			close(measuredSearchQueue)
		}
		stageCountCh <- stageCount
	}()

	return stageCountCh, nil
}
