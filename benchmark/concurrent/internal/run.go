package internal

import (
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"
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
		label := config.VariantName
		if label == "" {
			label = fmt.Sprintf("variant_%d", idx+1)
		}
		log.Printf("==== Running config variant: %s ====", label)
		if err := runSingleConfig(config); err != nil {
			return fmt.Errorf("variant %s failed: %w", label, err)
		}
	}

	return nil
}

func runSingleConfig(config *Config) error {
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

	index := createAndBuildIndex(config, data, dataDim)
	defer index.Close()

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
