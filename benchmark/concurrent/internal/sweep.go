package internal

import (
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
)

func RunThroughputSweep(config *Config, configPath string) error {
	sweep := config.ThroughputSweep
	if sweep.Steps <= 0 {
		sweep.Steps = 10
	}
	if sweep.StartRate <= 0 || sweep.EndRate <= 0 {
		return fmt.Errorf("throughput_sweep: start_rate and end_rate must be positive")
	}

	outputDir := config.Result.OutputDir
	if outputDir == "" {
		outputDir = "."
	}
	csvPath := filepath.Join(outputDir, "throughput_latency_curve.csv")

	f, err := os.Create(csvPath)
	if err != nil {
		return fmt.Errorf("failed to create %s: %w", csvPath, err)
	}
	defer f.Close()

	fmt.Fprintf(f, "target_search_rate,actual_search_qps,mean_search_op_latency_ms,p95_search_op_latency_ms,p99_search_op_latency_ms,mean_search_e2e_latency_ms,p95_search_e2e_latency_ms,p99_search_e2e_latency_ms\n")

	logStep := (math.Log(sweep.EndRate) - math.Log(sweep.StartRate)) / float64(sweep.Steps-1)

	for i := 0; i < sweep.Steps; i++ {
		targetRate := math.Exp(math.Log(sweep.StartRate) + float64(i)*logStep)

		log.Printf("==== Throughput Sweep step %d/%d: target_search_rate=%.0f ====", i+1, sweep.Steps, targetRate)

		variant := *config
		variant.Workload.SearchEventRate = targetRate
		variant.ThroughputSweep.Enabled = false

		data, _, dim, err := LoadAlignedBin(variant.Data.DataPath)
		if err != nil {
			return fmt.Errorf("sweep step %d: failed to load data: %w", i+1, err)
		}
		dataDim := int(dim)
		totalPoints := len(data) / dataDim
		clampDataBounds(&variant, totalPoints)

		workloadQueries, err := loadQueryDataset(variant.Data.IncrQueryPath, dataDim, "workload")
		if err != nil {
			return fmt.Errorf("sweep step %d: %w", i+1, err)
		}

		overallQueries := workloadQueries
		if variant.Data.OverallQueryPath != "" && variant.Data.OverallQueryPath != variant.Data.IncrQueryPath {
			overallQueries, err = loadQueryDataset(variant.Data.OverallQueryPath, dataDim, "overall")
			if err != nil {
				return fmt.Errorf("sweep step %d: %w", i+1, err)
			}
		}

		index := createAndBuildIndex(&variant, data, dataDim)
		_, stats, err := execBenchmark(index, &variant, data, workloadQueries, dataDim, overallQueries)
		index.Close()
		if err != nil {
			log.Printf("Sweep step %d failed: %v", i+1, err)
			continue
		}

		fmt.Fprintf(f, "%.0f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
			targetRate,
			stats.SearchQPS,
			stats.MeanSearchOpLatency,
			stats.P95SearchOpLatency,
			stats.P99SearchOpLatency,
			stats.MeanSearchE2ELatency,
			stats.P95SearchE2ELatency,
			stats.P99SearchE2ELatency,
		)
		f.Sync()
	}

	log.Printf("Throughput-latency curve written to %s", csvPath)
	return nil
}
