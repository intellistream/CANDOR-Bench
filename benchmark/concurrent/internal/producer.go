package internal

import (
	"context"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/time/rate"
)

type ScheduledTask struct {
	InsertOffset uint64
	QueryTags    []uint32
}

func StartProducers(
	cfg *Config,
	data []float32,
	dataDim int,
	workloadQueries []float32,
	begin, maxElements, totalQueries int,
	insertQueue chan<- Task,
	searchQueue chan<- Task,
	insertLimiter *rate.Limiter,
	searchLimiter *rate.Limiter,
	committedOffset *atomic.Uint64,
	replaySchedule []ScheduledTask,
	scheduleRecorder *[]ScheduledTask,
) (<-chan int, error) {
	stageCountCh := make(chan int, 1)

	go func() {
		defer close(stageCountCh)

		offset := begin
		stage := 0
		batchSize := cfg.Workload.BatchSize
		cursor := 0

		var wg sync.WaitGroup
		wg.Add(2)

		var insertDone atomic.Bool

		var replayMu sync.Mutex
		replayCond := sync.NewCond(&replayMu)
		var maxAllowedOffset uint64 = ^uint64(0)

		if len(replaySchedule) > 0 {
			maxAllowedOffset = replaySchedule[0].InsertOffset
		}

		go func() {
			defer wg.Done()
			defer close(insertQueue)

			for offset < maxElements {
				if len(replaySchedule) > 0 {
					replayMu.Lock()
					for uint64(offset) >= maxAllowedOffset {
						replayCond.Wait()
					}
					replayMu.Unlock()
				}

				next := min(offset+batchSize, maxElements)
				if next <= offset {
					break
				}

				if len(replaySchedule) > 0 {
					replayMu.Lock()
					if uint64(next) > maxAllowedOffset {
						next = int(maxAllowedOffset)
					}
					replayMu.Unlock()
				}

				if next <= offset {
					continue
				}

				batchData := sliceVectorsRange(data, dataDim, offset, next)
				batchTags := makeSequentialTags(offset, next)
				insertTask := Task{
					Type:         InsertTask,
					Data:         batchData,
					Tags:         batchTags,
					InsertOffset: uint64(next),
					CreateTime:   time.Now(),
				}

				if insertLimiter != nil {
					if err := insertLimiter.Wait(context.Background()); err != nil {
						log.Printf("Producer: insert limiter wait failed: %v", err)
						return
					}
				}
				insertQueue <- insertTask

				offset = next
				stage++
			}
			insertDone.Store(true)
		}()

		go func() {
			defer wg.Done()
			defer close(searchQueue)

			if cfg.Search.RecallAt == 0 || (len(replaySchedule) == 0 && cfg.Workload.SearchEventRate <= 0) {
				if len(replaySchedule) > 0 {
					replayMu.Lock()
					maxAllowedOffset = ^uint64(0)
					replayCond.Broadcast()
					replayMu.Unlock()
				}
				return
			}

			if len(replaySchedule) > 0 {
				log.Printf("Producer: REPLAY mode active with %d scheduled tasks", len(replaySchedule))
				log.Printf("Producer: schedule granularity is per-batch (batch_size=%d, queries per task ~= batch_size)", cfg.Workload.BatchSize)
				for i, task := range replaySchedule {
					for {
						current := committedOffset.Load()
						if current >= task.InsertOffset {
							break
						}
						if insertDone.Load() && current < task.InsertOffset {
							log.Printf("Producer: Replay warning - insert finished at %d before reaching target %d", current, task.InsertOffset)
							return
						}
						time.Sleep(100 * time.Microsecond)
					}

					queries := buildQueriesFromTags(workloadQueries, dataDim, task.QueryTags)
					if len(queries) > 0 {
						searchQueue <- Task{
							Type:          SearchTask,
							Data:          queries,
							Tags:          task.QueryTags,
							RecallAt:      cfg.Search.RecallAt,
							InsertOffset:  task.InsertOffset,
							WaitForInsert: false,
							CreateTime:    time.Now(),
						}
					}

					replayMu.Lock()
					if i+1 < len(replaySchedule) {
						maxAllowedOffset = replaySchedule[i+1].InsertOffset
					} else {
						maxAllowedOffset = ^uint64(0)
					}
					replayCond.Broadcast()
					replayMu.Unlock()
				}
				return
			}

			for !insertDone.Load() {
				if searchLimiter != nil {
					if err := searchLimiter.Wait(context.Background()); err != nil {
						log.Printf("Producer: search limiter wait failed: %v", err)
						return
					}
				}

				currentMax := int(committedOffset.Load())

				var stageQueries [][]float32
				var stageQueryTags []uint32
				var err error

				stageQueries, stageQueryTags, err = buildQueriesForMode(
					cfg.Workload.QueryMode,
					workloadQueries,
					data,
					dataDim,
					cfg.Workload.BatchSize,
					cfg.StageQueryWindow(),
					totalQueries,
					0,
					currentMax,
					&cursor,
					cfg.Workload.ZipfianSkew,
				)
				if err != nil {
					log.Printf("Producer: query generation failed: %v", err)
					return
				}

				if len(stageQueries) > 0 {
					if scheduleRecorder != nil {
						lastIdx := len(*scheduleRecorder) - 1
						if lastIdx < 0 || (*scheduleRecorder)[lastIdx].InsertOffset != uint64(currentMax) {
							*scheduleRecorder = append(*scheduleRecorder, ScheduledTask{
								InsertOffset: uint64(currentMax),
								QueryTags:    append([]uint32(nil), stageQueryTags...),
							})
						}
					}

					searchQueue <- Task{
						Type:          SearchTask,
						Data:          stageQueries,
						Tags:          stageQueryTags,
						RecallAt:      cfg.Search.RecallAt,
						InsertOffset:  uint64(currentMax),
						WaitForInsert: false,
						CreateTime:    time.Now(),
					}
				}
			}
		}()

		wg.Wait()
		stageCountCh <- stage
	}()

	return stageCountCh, nil
}

func buildQueriesFromTags(source []float32, dim int, tags []uint32) [][]float32 {
	if len(tags) == 0 || dim <= 0 {
		return nil
	}
	queries := make([][]float32, 0, len(tags))
	for _, tag := range tags {
		start := int(tag) * dim
		end := start + dim
		if start >= 0 && end <= len(source) {
			queries = append(queries, source[start:end])
		} else {
			log.Printf("buildQueriesFromTags: tag %d out of range (source len=%d, dim=%d)", tag, len(source), dim)
		}
	}
	return queries
}
