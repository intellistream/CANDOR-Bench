package internal

import (
	"fmt"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const defaultReplicaRefreshInterval = 50_000

func isReplicaRefreshIndex(indexType string) bool {
	return strings.EqualFold(indexType, "replica_refresh")
}

type replicaRefreshManager struct {
	writerConfig    Config
	dataDim         int
	refreshInterval uint64

	writerMu sync.Mutex
	orderMu  sync.Mutex
	orderCV  *sync.Cond
	activeMu sync.RWMutex
	active   *Index

	nextInsertStart   uint64
	nextRefreshOffset uint64
	visibleOffset     atomic.Uint64

	statsMu         sync.Mutex
	refreshCount    int
	refreshTotalMs  float64
	refreshMaxMs    float64
	snapshotTotalMs float64
	restoreTotalMs  float64
}

func newReplicaRefreshManager(config *Config, dataDim int, initialServing *Index) *replicaRefreshManager {
	interval := config.Replica.RefreshInterval
	if interval <= 0 {
		interval = defaultReplicaRefreshInterval
	}
	next := config.Data.BeginNum + interval
	if next <= config.Data.BeginNum {
		next = config.Data.BeginNum + defaultReplicaRefreshInterval
	}
	m := &replicaRefreshManager{
		writerConfig:      *config,
		dataDim:           dataDim,
		refreshInterval:   uint64(interval),
		active:            initialServing,
		nextInsertStart:   uint64(config.Data.BeginNum),
		nextRefreshOffset: uint64(next),
	}
	m.orderCV = sync.NewCond(&m.orderMu)
	m.writerConfig.Index.IndexType = "hnsw"
	m.visibleOffset.Store(uint64(config.Data.BeginNum))
	log.Printf("Replica refresh: interval=%d points, initial_visible=%d", interval, config.Data.BeginNum)
	return m
}

func (m *replicaRefreshManager) beginInsert(startOffset uint64) {
	if m == nil {
		return
	}
	m.orderMu.Lock()
	for startOffset != m.nextInsertStart {
		m.orderCV.Wait()
	}
	m.orderMu.Unlock()
	m.writerMu.Lock()
}

func (m *replicaRefreshManager) finishInsert(endOffset uint64) {
	if m == nil {
		return
	}
	m.writerMu.Unlock()
	m.orderMu.Lock()
	if endOffset > m.nextInsertStart {
		m.nextInsertStart = endOffset
	}
	m.orderCV.Broadcast()
	m.orderMu.Unlock()
}

func (m *replicaRefreshManager) current() *Index {
	if m == nil {
		return nil
	}
	m.activeMu.RLock()
	idx := m.active
	m.activeMu.RUnlock()
	return idx
}

func (m *replicaRefreshManager) acquire() (*Index, func()) {
	if m == nil {
		return nil, func() {}
	}
	m.activeMu.RLock()
	return m.active, m.activeMu.RUnlock
}

func (m *replicaRefreshManager) visible() uint64 {
	if m == nil {
		return 0
	}
	return m.visibleOffset.Load()
}

func (m *replicaRefreshManager) maybeRefresh(writer *Index, offset uint64) error {
	if m == nil || writer == nil || m.refreshInterval == 0 {
		return nil
	}
	if offset < m.nextRefreshOffset {
		return nil
	}

	refreshStart := time.Now()
	snapshotStart := time.Now()
	snapshot, err := writer.Snapshot()
	snapshotMs := float64(time.Since(snapshotStart).Microseconds()) / 1000.0
	if err != nil {
		return fmt.Errorf("replica refresh snapshot failed at offset %d: %w", offset, err)
	}

	newServing := createIndexInstance(&m.writerConfig, m.dataDim)
	restoreStart := time.Now()
	if err := newServing.Restore(snapshot); err != nil {
		newServing.Close()
		return fmt.Errorf("replica refresh restore failed at offset %d: %w", offset, err)
	}
	newServing.SetQueryParams(BuildQueryParams(&m.writerConfig))
	restoreMs := float64(time.Since(restoreStart).Microseconds()) / 1000.0

	m.activeMu.Lock()
	old := m.active
	m.active = newServing
	for offset >= m.nextRefreshOffset {
		m.nextRefreshOffset += m.refreshInterval
	}
	m.visibleOffset.Store(offset)
	m.activeMu.Unlock()
	if old != nil {
		old.Close()
	}

	refreshMs := float64(time.Since(refreshStart).Microseconds()) / 1000.0
	m.statsMu.Lock()
	m.refreshCount++
	m.refreshTotalMs += refreshMs
	m.snapshotTotalMs += snapshotMs
	m.restoreTotalMs += restoreMs
	if refreshMs > m.refreshMaxMs {
		m.refreshMaxMs = refreshMs
	}
	count := m.refreshCount
	m.statsMu.Unlock()

	log.Printf("Replica refresh: offset=%d count=%d total=%.2fms snapshot=%.2fms restore=%.2fms next=%d",
		offset, count, refreshMs, snapshotMs, restoreMs, m.nextRefreshOffset)
	return nil
}

func (m *replicaRefreshManager) statsString() string {
	if m == nil {
		return ""
	}
	m.statsMu.Lock()
	defer m.statsMu.Unlock()
	mean := 0.0
	snapMean := 0.0
	restoreMean := 0.0
	if m.refreshCount > 0 {
		mean = m.refreshTotalMs / float64(m.refreshCount)
		snapMean = m.snapshotTotalMs / float64(m.refreshCount)
		restoreMean = m.restoreTotalMs / float64(m.refreshCount)
	}
	return fmt.Sprintf(
		"replica_refresh_interval:%d, replica_refresh_count:%d, replica_visible_offset:%d, replica_refresh_mean_ms:%.4f, replica_refresh_max_ms:%.4f, replica_snapshot_mean_ms:%.4f, replica_restore_mean_ms:%.4f",
		m.refreshInterval,
		m.refreshCount,
		m.visibleOffset.Load(),
		mean,
		m.refreshMaxMs,
		snapMean,
		restoreMean,
	)
}

func (m *replicaRefreshManager) close() {
	if m == nil {
		return
	}
	m.activeMu.Lock()
	active := m.active
	m.active = nil
	m.activeMu.Unlock()

	if active != nil {
		active.Close()
	}
}
