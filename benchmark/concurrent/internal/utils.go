package internal

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/time/rate"
)

/*
#cgo LDFLAGS: -ljemalloc
#include <jemalloc/jemalloc.h>
static unsigned long long je_stats_allocated_bytes(void) {
    size_t epoch = 1, sz = sizeof(epoch);
    mallctl("epoch", &epoch, &sz, &epoch, sizeof(epoch));
    unsigned long long allocated = 0; sz = sizeof(allocated);
    if (mallctl("stats.allocated", &allocated, &sz, NULL, 0) != 0) return 0;
    return allocated;
}
*/
import "C"

type SearchResult struct {
	InsertOffset uint64
	QueryTag     uint64
	Tags         []uint32
}

func NewSearchResult(offset, qtag uint64, tags []uint32) *SearchResult {
	return &SearchResult{
		InsertOffset: offset,
		QueryTag:     qtag,
		Tags:         tags,
	}
}

func ensureDir(path string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}
	return nil
}

func LoadAlignedBin(binFile string) (data []float32, npts, dim uint32, err error) {
	log.Printf("Reading bin file %s ...", binFile)

	file, err := os.Open(binFile)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to get file info: %v", err)
	}
	actualFileSize := fileInfo.Size()

	var nptsI32, dimI32 int32
	if err := binary.Read(file, binary.LittleEndian, &nptsI32); err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read npts: %v", err)
	}
	if err := binary.Read(file, binary.LittleEndian, &dimI32); err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read dim: %v", err)
	}

	npts = uint32(nptsI32)
	dim = uint32(dimI32)

	expectedFileSize := int64(npts)*int64(dim)*4 + 8
	if actualFileSize != expectedFileSize {
		return nil, 0, 0, fmt.Errorf("file size mismatch: actual=%d, expected=%d (npts=%d, dim=%d)",
			actualFileSize, expectedFileSize, npts, dim)
	}

	log.Printf("Metadata: pts = %d, dims = %d", npts, dim)

	allocSize := uint64(npts) * uint64(dim) * 4
	log.Printf("Allocating memory of %d bytes... ", allocSize)
	data = make([]float32, npts*dim)

	if err := binary.Read(file, binary.LittleEndian, data); err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read data: %v", err)
	}
	log.Println("Load bin file done.")

	return data, npts, dim, nil
}

func DumpIncrResults(results []*SearchResult, outputPath string) error {
	sort.Slice(results, func(i, j int) bool {
		if results[i].InsertOffset != results[j].InsertOffset {
			return results[i].InsertOffset < results[j].InsertOffset
		}
		return results[i].QueryTag < results[j].QueryTag
	})

	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %v", outputPath, err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	numQueries := uint64(len(results))
	if err := binary.Write(writer, binary.LittleEndian, &numQueries); err != nil {
		return fmt.Errorf("failed to write number of entries: %v", err)
	}

	for i, res := range results {
		if err := binary.Write(writer, binary.LittleEndian, &res.InsertOffset); err != nil {
			return fmt.Errorf("failed to write InsertOffset for entry %d: %v", i, err)
		}
		if err := binary.Write(writer, binary.LittleEndian, &res.QueryTag); err != nil {
			return fmt.Errorf("failed to write QueryTag for entry %d: %v", i, err)
		}

		numTags := uint64(len(res.Tags))
		if err := binary.Write(writer, binary.LittleEndian, &numTags); err != nil {
			return fmt.Errorf("failed to write numTags for entry %d: %v", i, err)
		}
		if err := binary.Write(writer, binary.LittleEndian, res.Tags); err != nil {
			return fmt.Errorf("failed to write Tags for entry %d: %v", i, err)
		}

		var numDists uint64
		if err := binary.Write(writer, binary.LittleEndian, &numDists); err != nil {
			return fmt.Errorf("failed to write numDists for entry %d: %v", i, err)
		}
	}

	log.Printf("Successfully dumped %d entries to %s\n", numQueries, outputPath)
	return nil
}

func buildLimiter(rateValue float64, batchSize int) *rate.Limiter {
	if rateValue <= 0 {
		return nil
	}
	if batchSize <= 0 {
		batchSize = 1
	}
	perBatch := rateValue / float64(batchSize)
	if perBatch <= 0 {
		return nil
	}
	return rate.NewLimiter(rate.Limit(perBatch), batchSize)
}

func GetCurrentMemoryUsage() uint64 {
	return uint64(C.je_stats_allocated_bytes())
}

func CalculateRecall(toolPath, gtPath, resPath string, k int) (float64, error) {
	if toolPath == "" {
		return 0, fmt.Errorf("recall tool path is empty")
	}
	if gtPath == "" {
		return 0, fmt.Errorf("ground truth path is empty")
	}
	if resPath == "" {
		return 0, fmt.Errorf("result path is empty")
	}

	cmd := exec.Command(
		toolPath,
		"--gt_path", gtPath,
		"--res_path", resPath,
		"--k", fmt.Sprintf("%d", k),
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("overall recall tool failed: %v\nOutput:\n%s", err, string(output))
	}

	lines := strings.Split(string(output), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}

		candidates := []string{
			line,
			strings.TrimPrefix(strings.ToLower(line), "recall"),
			strings.TrimPrefix(strings.ToLower(line), "recall:"),
		}

		for _, candidate := range candidates {
			fields := strings.Fields(strings.ReplaceAll(candidate, "=", " "))
			for j := len(fields) - 1; j >= 0; j-- {
				token := strings.TrimSpace(fields[j])
				if token == "" {
					continue
				}
				isPercent := strings.HasSuffix(token, "%")
				token = strings.TrimSuffix(token, "%")
				value, err := strconv.ParseFloat(token, 64)
				if err != nil {
					continue
				}
				if isPercent {
					return value / 100.0, nil
				}
				return value, nil
			}
		}
	}

	return 0, fmt.Errorf("failed to parse recall from output:\n%s", string(output))
}
