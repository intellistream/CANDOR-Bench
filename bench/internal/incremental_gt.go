package internal

import (
	"bufio"
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type splitFileInfo struct {
	filename    string
	startOffset uint64
	endOffset   uint64
}

type incrementalGTProvider struct {
	baseDir    string
	recallAt   uint32
	splits     []splitFileInfo
	cachedFile string
	cache      map[uint64]map[uint64][]uint32
}

type partialDiffSummary struct {
	StageOffset      uint64
	Diff             float64
	Count            int
	Percentage       float64
	SumRecallBase    float64
	SumRecallPartial float64
	SumFutureTags    float64
}

func loadIncrementalGTProvider(indexPath string, recallAt uint32) (*incrementalGTProvider, error) {
	splits, err := parseIncrementalGTIndex(indexPath)
	if err != nil {
		return nil, err
	}
	if len(splits) == 0 {
		return nil, fmt.Errorf("incremental ground truth index %s contained no entries", indexPath)
	}

	return &incrementalGTProvider{
		baseDir:  filepath.Dir(indexPath),
		recallAt: recallAt,
		splits:   splits,
		cache:    make(map[uint64]map[uint64][]uint32),
	}, nil
}

func parseIncrementalGTIndex(indexPath string) ([]splitFileInfo, error) {
	file, err := os.Open(indexPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open incremental ground truth index %s: %w", indexPath, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var splits []splitFileInfo
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}

		startVal, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse start offset in %s: %w", indexPath, err)
		}
		endVal, err := strconv.ParseInt(fields[2], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse end offset in %s: %w", indexPath, err)
		}
		if startVal < 0 {
			startVal = 0
		}
		if endVal < 0 {
			endVal = 0
		}

		splits = append(splits, splitFileInfo{
			filename:    fields[0],
			startOffset: uint64(startVal),
			endOffset:   uint64(endVal),
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to read incremental ground truth index %s: %w", indexPath, err)
	}

	return splits, nil
}

func (p *incrementalGTProvider) topK(offset uint64, queryTag uint64) ([]uint32, error) {
	if p == nil {
		return nil, fmt.Errorf("incremental ground truth provider is nil")
	}
	split := p.findSplit(offset)
	if split == nil {
		return nil, nil
	}
	if err := p.ensureFileLoaded(split.filename); err != nil {
		return nil, err
	}
	stageMap := p.cache[offset]
	if stageMap == nil {
		return nil, nil
	}
	return stageMap[queryTag], nil
}

func (p *incrementalGTProvider) findSplit(offset uint64) *splitFileInfo {
	for i := range p.splits {
		split := &p.splits[i]
		if offset >= split.startOffset && offset <= split.endOffset {
			return split
		}
	}
	return nil
}

func (p *incrementalGTProvider) ensureFileLoaded(filename string) error {
	if p.cachedFile == filename {
		return nil
	}
	if err := p.loadSplitFile(filename); err != nil {
		return err
	}
	p.cachedFile = filename
	return nil
}

func (p *incrementalGTProvider) loadSplitFile(filename string) error {
	path := filepath.Join(p.baseDir, filename)

	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open incremental ground truth split %s: %w", path, err)
	}
	defer file.Close()

	var (
		nPerBatch  int32
		kFile      int32
		numBatches int32
	)
	if err := binary.Read(file, binary.LittleEndian, &nPerBatch); err != nil {
		return fmt.Errorf("failed to read n_per_batch from %s: %w", path, err)
	}
	if err := binary.Read(file, binary.LittleEndian, &kFile); err != nil {
		return fmt.Errorf("failed to read k from %s: %w", path, err)
	}
	if err := binary.Read(file, binary.LittleEndian, &numBatches); err != nil {
		return fmt.Errorf("failed to read batch count from %s: %w", path, err)
	}
	if nPerBatch <= 0 || kFile <= 0 || numBatches <= 0 {
		return fmt.Errorf("incremental ground truth split %s has invalid header values", path)
	}

	payloadStart, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return fmt.Errorf("failed to determine payload start for %s: %w", path, err)
	}
	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("failed to stat incremental ground truth split %s: %w", path, err)
	}
	payloadSize := info.Size() - payloadStart

	perBatchBase := int64(8) + int64(nPerBatch)*int64(kFile)*(4+4)
	expectedPayload := perBatchBase * int64(numBatches)
	queryTagBytes := int64(nPerBatch) * 4 * int64(numBatches)

	hasQueryTags := false
	if payloadSize == expectedPayload+queryTagBytes {
		hasQueryTags = true
	} else if payloadSize != expectedPayload {
		return fmt.Errorf("incremental ground truth split %s has unexpected size: got %d bytes, expected %d or %d (with query tags)", path, payloadSize, expectedPayload, expectedPayload+queryTagBytes)
	}

	p.cache = make(map[uint64]map[uint64][]uint32, numBatches)

	distCount := int(nPerBatch) * int(kFile)
	distancesBuf := make([]float32, distCount)
	indicesBuf := make([]uint32, distCount)
	queryTagsBuf := make([]uint32, nPerBatch)

	limit := int(p.recallAt)
	if limit <= 0 || limit > int(kFile) {
		limit = int(kFile)
	}
	if limit <= 0 {
		return nil
	}

	for batch := 0; batch < int(numBatches); batch++ {
		var offset uint64
		if err := binary.Read(file, binary.LittleEndian, &offset); err != nil {
			return fmt.Errorf("failed to read batch %d offset from %s: %w", batch, path, err)
		}

		queryTags := make([]uint64, nPerBatch)
		if hasQueryTags {
			if err := binary.Read(file, binary.LittleEndian, queryTagsBuf); err != nil {
				return fmt.Errorf("failed to read batch %d query tags from %s: %w", batch, path, err)
			}
			for i := 0; i < int(nPerBatch); i++ {
				queryTags[i] = uint64(queryTagsBuf[i])
			}
		} else {
			for i := 0; i < int(nPerBatch); i++ {
				queryTags[i] = uint64(i)
			}
		}

		if err := binary.Read(file, binary.LittleEndian, distancesBuf); err != nil {
			return fmt.Errorf("failed to read batch %d distances from %s: %w", batch, path, err)
		}
		if err := binary.Read(file, binary.LittleEndian, indicesBuf); err != nil {
			return fmt.Errorf("failed to read batch %d indices from %s: %w", batch, path, err)
		}

		stageMap := p.cache[offset]
		if stageMap == nil {
			stageMap = make(map[uint64][]uint32, nPerBatch)
			p.cache[offset] = stageMap
		}

		for i := 0; i < int(nPerBatch); i++ {
			start := i * int(kFile)
			end := start + limit
			if end > start+int(kFile) || end > len(indicesBuf) {
				end = start + int(kFile)
				if end > len(indicesBuf) {
					end = len(indicesBuf)
				}
			}
			if end <= start {
				continue
			}
			tags := make([]uint32, end-start)
			copy(tags, indicesBuf[start:end])
			stageMap[queryTags[i]] = tags
		}
	}

	return nil
}

func computePartialDiffDistribution(consistent []*SearchResult, partial []*SearchResult, recallAt uint32, gt *incrementalGTProvider) (map[uint64][]partialDiffSummary, error) {
	if gt == nil {
		return nil, fmt.Errorf("incremental ground truth provider is nil")
	}
	if recallAt == 0 {
		return nil, fmt.Errorf("recallAt must be greater than zero")
	}

	stageTotals := make(map[uint64]int)
	stageBuckets := make(map[uint64]map[string]*partialDiffSummary)

	baseline := make(map[uint64]map[uint64]*SearchResult, len(consistent))
	baselineRecall := make(map[uint64]map[uint64]float64, len(consistent))
	gtCache := make(map[uint64]map[uint64][]uint32)

	for _, res := range consistent {
		stageMap := baseline[res.InsertOffset]
		if stageMap == nil {
			stageMap = make(map[uint64]*SearchResult)
			baseline[res.InsertOffset] = stageMap
		}
		stageMap[res.QueryTag] = res

		gtTags, err := gt.topK(res.InsertOffset, res.QueryTag)
		if err != nil {
			return nil, err
		}
		if len(gtTags) == 0 {
			continue
		}

		stageGT := gtCache[res.InsertOffset]
		if stageGT == nil {
			stageGT = make(map[uint64][]uint32)
			gtCache[res.InsertOffset] = stageGT
		}
		stageGT[res.QueryTag] = gtTags

		stageRecall := baselineRecall[res.InsertOffset]
		if stageRecall == nil {
			stageRecall = make(map[uint64]float64)
			baselineRecall[res.InsertOffset] = stageRecall
		}
		stageRecall[res.QueryTag] = computeRecallAgainstGT(gtTags, res.Tags, recallAt)
	}

	for _, res := range partial {
		stageMap := baseline[res.InsertOffset]
		if stageMap == nil {
			continue
		}
		baseRes := stageMap[res.QueryTag]
		if baseRes == nil {
			continue
		}

		stageRecall := baselineRecall[res.InsertOffset]
		if stageRecall == nil {
			continue
		}
		baseRecall, ok := stageRecall[res.QueryTag]
		if !ok {
			gtTags, err := gt.topK(res.InsertOffset, res.QueryTag)
			if err != nil {
				return nil, err
			}
			if len(gtTags) == 0 {
				continue
			}
			baseRecall = computeRecallAgainstGT(gtTags, baseRes.Tags, recallAt)
			stageRecall[res.QueryTag] = baseRecall

			stageGT := gtCache[res.InsertOffset]
			if stageGT == nil {
				stageGT = make(map[uint64][]uint32)
				gtCache[res.InsertOffset] = stageGT
			}
			stageGT[res.QueryTag] = gtTags
		}

		stageGT := gtCache[res.InsertOffset]
		var gtTags []uint32
		if stageGT != nil {
			gtTags = stageGT[res.QueryTag]
		}
		if len(gtTags) == 0 {
			var err error
			gtTags, err = gt.topK(res.InsertOffset, res.QueryTag)
			if err != nil {
				return nil, err
			}
			if len(gtTags) == 0 {
				continue
			}
			if stageGT == nil {
				stageGT = make(map[uint64][]uint32)
				gtCache[res.InsertOffset] = stageGT
			}
			stageGT[res.QueryTag] = gtTags
		}

		partialRecall := computeRecallAgainstGT(gtTags, res.Tags, recallAt)
		diff := baseRecall - partialRecall

		stageTotals[res.InsertOffset]++

		roundedDiff := math.Round(diff*1e6) / 1e6
		key := fmt.Sprintf("%.6f", roundedDiff)
		bucket := stageBuckets[res.InsertOffset]
		if bucket == nil {
			bucket = make(map[string]*partialDiffSummary)
			stageBuckets[res.InsertOffset] = bucket
		}
		summary := bucket[key]
		if summary == nil {
			summary = &partialDiffSummary{
				StageOffset: res.InsertOffset,
				Diff:        roundedDiff,
			}
			bucket[key] = summary
		}
		summary.Count++
		summary.SumRecallBase += baseRecall
		summary.SumRecallPartial += partialRecall

		futureTags := 0
		for _, tag := range res.Tags {
			if uint64(tag) >= res.InsertOffset {
				futureTags++
			}
		}
		summary.SumFutureTags += float64(futureTags)
	}

	result := make(map[uint64][]partialDiffSummary, len(stageBuckets))
	for stage, bucket := range stageBuckets {
		total := stageTotals[stage]
		if total == 0 {
			continue
		}
		summaries := make([]partialDiffSummary, 0, len(bucket))
		for _, summary := range bucket {
			summary.Percentage = (float64(summary.Count) / float64(total)) * 100.0
			summaries = append(summaries, *summary)
		}
		sort.Slice(summaries, func(i, j int) bool {
			if summaries[i].Diff == summaries[j].Diff {
				return summaries[i].Count > summaries[j].Count
			}
			return summaries[i].Diff > summaries[j].Diff
		})
		result[stage] = summaries
	}

	return result, nil
}

func computeRecallAgainstGT(gtTags []uint32, resultTags []uint32, recallAt uint32) float64 {
	limit := int(recallAt)
	if limit <= 0 {
		return 0
	}
	if len(gtTags) < limit {
		limit = len(gtTags)
	}
	if limit == 0 {
		return 0
	}

	refSet := make(map[uint32]struct{}, limit)
	for i := 0; i < limit; i++ {
		refSet[gtTags[i]] = struct{}{}
	}

	matches := 0
	maxCompare := len(resultTags)
	if maxCompare > limit {
		maxCompare = limit
	}
	for i := 0; i < maxCompare; i++ {
		if _, ok := refSet[resultTags[i]]; ok {
			matches++
		}
	}
	return float64(matches) / float64(limit)
}

func writePartialDiffDistribution(path string, diffMap map[uint64][]partialDiffSummary) error {
	if path == "" || len(diffMap) == 0 {
		return nil
	}

	stages := make([]uint64, 0, len(diffMap))
	for stage := range diffMap {
		stages = append(stages, stage)
	}
	sort.Slice(stages, func(i, j int) bool { return stages[i] < stages[j] })

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}

	writer := csv.NewWriter(file)
	header := []string{"stage_offset", "diff (baseline - partial)", "count", "percentage", "avg_recall_base", "avg_recall_partial", "avg_future_tags"}
	if err := writer.Write(header); err != nil {
		file.Close()
		return err
	}

	for idx, stage := range stages {
		records := diffMap[stage]
		if len(records) == 0 {
			continue
		}
		sort.Slice(records, func(i, j int) bool {
			if records[i].Diff == records[j].Diff {
				return records[i].Count > records[j].Count
			}
			return records[i].Diff > records[j].Diff
		})
		for _, rec := range records {
			row := []string{
				strconv.FormatUint(rec.StageOffset, 10),
				fmt.Sprintf("%.6f", rec.Diff),
				strconv.Itoa(rec.Count),
				fmt.Sprintf("%.4f", rec.Percentage),
				fmt.Sprintf("%.6f", rec.SumRecallBase/float64(rec.Count)),
				fmt.Sprintf("%.6f", rec.SumRecallPartial/float64(rec.Count)),
				fmt.Sprintf("%.6f", rec.SumFutureTags/float64(rec.Count)),
			}
			if err := writer.Write(row); err != nil {
				writer.Flush()
				file.Close()
				return err
			}
		}
		if idx < len(stages)-1 {
			nextHasData := false
			for j := idx + 1; j < len(stages); j++ {
				if len(diffMap[stages[j]]) > 0 {
					nextHasData = true
					break
				}
			}
			if nextHasData {
				if err := writer.Write([]string{"", "", "", "", "", "", ""}); err != nil {
					writer.Flush()
					file.Close()
					return err
				}
			}
		}
	}

	writer.Flush()
	if err := writer.Error(); err != nil {
		file.Close()
		return err
	}
	file.Close()
	log.Printf("Recall diff summary: wrote %s (%d stages)", path, len(stages))

	return nil
}
