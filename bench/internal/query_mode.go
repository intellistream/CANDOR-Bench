package internal

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
)

func buildQueriesForMode(
	mode string,
	workload []float32,
	data []float32,
	dim int,
	batchSize int,
	stageWindow int,
	totalQueries int,
	startInsertOffset int,
	endInsertOffset int,
	cursor *int,
	skew ...float64,
) ([][]float32, []uint32, error) {
	_ = data
	_ = startInsertOffset

	if batchSize <= 0 || dim <= 0 {
		return nil, nil, nil
	}

	tags, err := buildQueryTagsForMode(mode, batchSize, stageWindow, totalQueries, endInsertOffset, cursor, skew...)
	if err != nil || len(tags) == 0 {
		return nil, tags, err
	}

	queries := buildQueriesFromTags(workload, dim, tags)
	if len(queries) == 0 {
		return nil, nil, nil
	}
	return queries, tags, nil
}

func buildQueryTagsForMode(
	mode string,
	batchSize int,
	stageWindow int,
	totalQueries int,
	endInsertOffset int,
	cursor *int,
	skew ...float64,
) ([]uint32, error) {
	if batchSize <= 0 {
		return nil, nil
	}
	if cursor == nil {
		localCursor := 0
		cursor = &localCursor
	}

	switch mode {
	case queryModeRoundRobin:
		if totalQueries == 0 {
			return nil, fmt.Errorf("query mode round robin requires a workload query dataset")
		}

		tags := make([]uint32, 0, batchSize)
		for i := 0; i < batchSize; i++ {
			idx := *cursor % totalQueries
			tags = append(tags, uint32(idx))
			*cursor = *cursor + 1
		}
		return tags, nil

	case queryModeChasing:
		if totalQueries == 0 {
			return nil, fmt.Errorf("query mode chasing requires a workload query dataset")
		}
		if endInsertOffset > totalQueries {
			return nil, fmt.Errorf("query mode chasing window exceeds workload queries (offset=%d, queries=%d)", endInsertOffset, totalQueries)
		}

		windowEnd := endInsertOffset
		windowStart := windowEnd - stageWindow
		if windowStart < 0 {
			windowStart = 0
		}
		windowSize := windowEnd - windowStart
		if windowSize <= 0 {
			log.Printf("query mode chasing: empty query window at offset %d", endInsertOffset)
			return nil, nil
		}

		tags := make([]uint32, 0, batchSize)
		for i := 0; i < batchSize; i++ {
			idx := windowStart + (i % windowSize)
			tags = append(tags, uint32(idx))
		}
		return tags, nil

	case queryModePeeking:
		if totalQueries == 0 {
			return nil, fmt.Errorf("query mode peeking requires a workload query dataset")
		}
		if endInsertOffset > totalQueries {
			return nil, fmt.Errorf("query mode peeking window exceeds workload queries (offset=%d, queries=%d)", endInsertOffset, totalQueries)
		}

		windowStart := endInsertOffset
		windowEnd := windowStart + stageWindow
		if windowStart >= totalQueries {
			return nil, nil
		}
		if windowEnd > totalQueries {
			windowEnd = totalQueries
		}
		windowSize := windowEnd - windowStart
		if windowSize <= 0 {
			return nil, nil
		}

		tags := make([]uint32, 0, batchSize)
		for i := 0; i < batchSize; i++ {
			offset := i % windowSize
			idx := windowStart + offset
			tags = append(tags, uint32(idx))
		}
		return tags, nil

	case queryModeZipfian:
		if totalQueries == 0 {
			return nil, fmt.Errorf("query mode zipfian requires a workload query dataset")
		}

		s := 0.99
		if len(skew) > 0 && skew[0] > 0 {
			s = skew[0]
		}
		zipf := getOrCreateZipfianSampler(totalQueries, s)
		tags := make([]uint32, 0, batchSize)
		for i := 0; i < batchSize; i++ {
			tags = append(tags, uint32(zipf.Uint64()))
		}
		return tags, nil

	default:
		return nil, fmt.Errorf("unsupported query_mode %s", mode)
	}
}

type zipfianSampler struct {
	cdf []float64
	n   int
	mu  sync.Mutex
	rng *rand.Rand
}

var (
	cachedZipfMu      sync.Mutex
	cachedZipfSampler *zipfianSampler
	cachedZipfN       int
	cachedZipfSkew    float64
)

func getOrCreateZipfianSampler(n int, skew float64) *zipfianSampler {
	cachedZipfMu.Lock()
	defer cachedZipfMu.Unlock()
	if cachedZipfSampler != nil && cachedZipfN == n && cachedZipfSkew == skew {
		return cachedZipfSampler
	}
	s := newZipfianSampler(n, skew)
	cachedZipfSampler = s
	cachedZipfN = n
	cachedZipfSkew = skew
	return s
}

func newZipfianSampler(n int, skew float64) *zipfianSampler {
	if n <= 0 {
		return nil
	}
	rng := rand.New(rand.NewSource(rand.Int63()))

	weights := make([]float64, n)
	sum := 0.0
	for k := 0; k < n; k++ {
		w := 1.0 / math.Pow(float64(k+1), skew)
		weights[k] = w
		sum += w
	}
	cdf := make([]float64, n)
	cumul := 0.0
	for k := 0; k < n; k++ {
		cumul += weights[k] / sum
		cdf[k] = cumul
	}
	cdf[n-1] = 1.0
	return &zipfianSampler{cdf: cdf, n: n, rng: rng}
}

func (z *zipfianSampler) Uint64() uint64 {
	z.mu.Lock()
	u := z.rng.Float64()
	z.mu.Unlock()
	lo, hi := 0, len(z.cdf)-1
	for lo < hi {
		mid := (lo + hi) / 2
		if z.cdf[mid] < u {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return uint64(lo)
}
