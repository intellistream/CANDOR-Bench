package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"ANN-CC-bench/bench/internal"
)

func main() {
	configPath := flag.String("config", "", "config file path")
	flag.Parse()

	if *configPath == "" {
		log.Fatal("Usage: recall_sweep -config <config.yaml>")
	}

	configs, err := internal.LoadConfigVariantsPublic(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}
	if len(configs) == 0 {
		log.Fatal("No config variants found")
	}

	// Use first config to load data and build index
	baseConfig := configs[0]

	data, _, dim, err := internal.LoadAlignedBin(baseConfig.Data.DataPath)
	if err != nil {
		log.Fatalf("Failed to load data: %v", err)
	}
	dataDim := int(dim)

	totalPoints := len(data) / dataDim
	internal.ClampDataBoundsPublic(baseConfig, totalPoints)

	// Load queries
	overallQueries, err := internal.LoadQueryDatasetPublic(baseConfig.Data.OverallQueryPath, dataDim, "overall")
	if err != nil {
		log.Fatalf("Failed to load queries: %v", err)
	}

	// Build index ONCE
	log.Printf("Building index with %d points, %d threads...", baseConfig.Data.BeginNum, baseConfig.Workload.NumThreads)
	startBuild := time.Now()
	index := internal.CreateAndBuildIndexPublic(baseConfig, data, dataDim)
	log.Printf("Index built in %v", time.Since(startBuild))

	// Prepare CSV output
	outputDir := baseConfig.Result.OutputDir
	os.MkdirAll(outputDir, 0755)
	csvPath := fmt.Sprintf("%s/recall_sweep.csv", outputDir)
	f, err := os.Create(csvPath)
	if err != nil {
		log.Fatalf("Failed to create CSV: %v", err)
	}
	defer f.Close()
	fmt.Fprintln(f, "ef_search,recall,search_qps")

	// Sweep ef_search values
	for _, config := range configs {
		internal.ClampDataBoundsPublic(config, totalPoints)
		qp := internal.BuildQueryParams(config)
		index.SetQueryParams(qp)

		efLabel := fmt.Sprintf("ef_s:%d", config.Search.EfSearch)
		log.Printf("Testing %s ...", efLabel)

		recall, err := internal.CalcOverallRecallForIndexPublic(index, config, overallQueries, dataDim)
		if err != nil {
			log.Printf("  %s: recall error: %v", efLabel, err)
			continue
		}

		log.Printf("  %s: recall=%.4f", efLabel, recall)
		fmt.Fprintf(f, "%d,%.4f,0\n", config.Search.EfSearch, recall)
		f.Sync()
	}

	index.Close()
	log.Printf("Results written to %s", csvPath)
}
