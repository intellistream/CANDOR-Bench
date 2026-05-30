package main

import (
	"fmt"
	"time"

	"ANN-CC-bench/bench/internal"
)

func main() {
	params := internal.IndexParams{
		Dim:            128,
		MaxElements:    1000000,
		M:              16,
		EfConstruction: 200,
		Threads:        32,
		UseNodeLock:    true,
	}

	idx := internal.NewIndex(internal.IndexTypeANNchor, params)
	defer idx.Close()

	idx.SetEnableMvcc(true)
	idx.ActivateEUL()

	// Load SIFT data
	data, npts, dim, err := internal.LoadAlignedBin("../data/sift/sift_base.bin")
	if err != nil {
		panic(err)
	}
	fmt.Printf("Loaded %d vectors, dim=%d\n", npts, dim)

	// Build initial 500K
	initData := make([][]float32, 500000)
	for i := 0; i < 500000; i++ {
		initData[i] = data[i*int(dim) : (i+1)*int(dim)]
	}
	initTags := make([]uint32, 500000)
	for i := range initTags {
		initTags[i] = uint32(i)
	}
	fmt.Println("Building initial 500K...")
	start := time.Now()
	if err := idx.Build(initData, initTags); err != nil {
		panic(err)
	}
	fmt.Printf("Build took %v\n", time.Since(start))

	// Insert remaining 500K in batches of 20
	batchSize := 20
	fmt.Printf("Inserting 500K points in batches of %d...\n", batchSize)
	start = time.Now()
	for i := 500000; i < int(npts); i += batchSize {
		end := i + batchSize
		if end > int(npts) {
			end = int(npts)
		}
		n := end - i
		batch := make([][]float32, n)
		tags := make([]uint32, n)
		for j := 0; j < n; j++ {
			vi := i + j
			batch[j] = data[vi*int(dim) : (vi+1)*int(dim)]
			tags[j] = uint32(vi)
		}
		idx.BatchInsert(batch, tags)
	}
	fmt.Printf("Insert took %v\n", time.Since(start))

	// Compact
	fmt.Println("\nRunning compact...")
	start = time.Now()
	compacted := idx.Compact()
	elapsed := time.Since(start)
	fmt.Printf("Compact: %d nodes compacted in %v (%.3f ms)\n", compacted, elapsed, float64(elapsed.Microseconds())/1000.0)

	// Second compact (should be instant)
	fmt.Println("Running compact again (should be instant)...")
	start = time.Now()
	compacted2 := idx.Compact()
	elapsed2 := time.Since(start)
	fmt.Printf("Compact (2nd): %d nodes in %v (%.3f ms)\n", compacted2, elapsed2, float64(elapsed2.Microseconds())/1000.0)
}
