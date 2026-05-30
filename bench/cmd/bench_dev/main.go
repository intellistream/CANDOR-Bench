package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"

	"ANN-CC-bench/bench/internal"
)

func main() {
	configPath := flag.String("config", "", "path to YAML config file")
	flag.Parse()

	if *configPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s -config <config.yaml>\n", os.Args[0])
		os.Exit(1)
	}

	log.Printf("CPUs: %d", runtime.NumCPU())
	if err := internal.Run(*configPath); err != nil {
		log.Fatalf("Benchmark failed: %v", err)
	}
}
