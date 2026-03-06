package main

import (
	"flag"
	"log"

	"CANDOR-Bench/benchmark/concurrent/internal"
)

func main() {
	configPath := flag.String("config", "config/config.yaml", "config file path")
	flag.Parse()

	if err := internal.Run(*configPath); err != nil {
		log.Fatalf("benchmark failed: %v", err)
	}
}
