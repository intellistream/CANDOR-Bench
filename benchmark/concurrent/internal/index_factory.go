package internal

import (
	"log"
)

var baselineAlgoMap = map[string]string{}

func createAndBuildIndex(config *Config, data []float32, dataDim int) *Index {
	index := createIndexInstance(config, dataDim)
	beginNum := config.Data.BeginNum
	if beginNum <= 0 {
		return index
	}

	totalPoints := len(data) / dataDim
	if beginNum > totalPoints {
		beginNum = totalPoints
	}
	if beginNum <= 0 {
		return index
	}

	preData := sliceVectorsRange(data, dataDim, 0, beginNum)
	preTags := makeSequentialTags(0, beginNum)
	log.Printf("Building index with %d initial elements...", len(preData))
	if err := index.Build(preData, preTags); err != nil {
		log.Fatalf("failed to build index: %v", err)
	}

	return index
}

func createIndexInstance(config *Config, dataDim int) *Index {
	var index *Index
	switch config.Index.IndexType {
	case "hnsw":
		useNodeLock, _ := EffectiveUseNodeLock(config)
		params := IndexParams{
			Dim:            dataDim,
			MaxElements:    uint64(config.Data.MaxElements),
			M:              config.Index.M,
			EfConstruction: config.Index.EfConstruction,
			Threads:        config.Workload.NumThreads,
			DataType:       DataTypeFloat,
			UseNodeLock:    useNodeLock,
			Metric:         getMetric(config.Data.DatasetName),

			VisitLimit: config.Index.VisitLimit,
		}
		index = NewIndex(IndexTypeHNSW, params)
	case "parlayhnsw":
		params := IndexParams{
			Dim:            dataDim,
			MaxElements:    uint64(config.Data.MaxElements),
			M:              config.Index.M,
			EfConstruction: config.Index.EfConstruction,
			LevelM:         config.Index.LevelM,
			Alpha:          config.Index.Alpha,
			Threads:        config.Workload.NumThreads,
			DataType:       DataTypeFloat,
			VisitLimit:     config.Index.VisitLimit,
		}
		index = NewIndex(IndexTypeParlayHNSW, params)
	case "parlayvamana":
		params := IndexParams{
			Dim:            dataDim,
			MaxElements:    uint64(config.Data.MaxElements),
			M:              config.Index.M,
			EfConstruction: config.Index.EfConstruction,
			Alpha:          config.Index.Alpha,
			Threads:        config.Workload.NumThreads,
			DataType:       DataTypeFloat,
			VisitLimit:     config.Index.VisitLimit,
		}
		index = NewIndex(IndexTypeParlayVamana, params)
	case "vamana":
		params := IndexParams{
			Dim:            dataDim,
			MaxElements:    uint64(config.Data.MaxElements),
			M:              config.Index.M,
			EfConstruction: config.Index.EfConstruction,
			Alpha:          config.Index.Alpha,
			Threads:        config.Workload.NumThreads,
			DataType:       DataTypeFloat,
			VisitLimit:     config.Index.VisitLimit,
		}
		index = NewIndex(IndexTypeVamana, params)
	case "segmented":
		baseType := config.Index.SealedType
		if baseType == "" {
			baseType = "hnsw"
		}
		baseIndexType := map[string]IndexType{
			"hnsw":         IndexTypeHNSW,
			"parlayhnsw":   IndexTypeParlayHNSW,
			"parlayvamana": IndexTypeParlayVamana,
			"vamana":       IndexTypeVamana,
		}[baseType]
		if baseIndexType == 0 && baseType != "hnsw" {
			log.Fatalf("unsupported sealed index type for segmented index: %s\n", baseType)
		}
		threshold := config.Index.SealThreshold
		if threshold <= 0 {
			threshold = config.Index.VisitLimit
		}
		if threshold <= 0 {
			threshold = config.Workload.BatchSize
		}
		if threshold <= 0 {
			threshold = 50000
		}

		useNodeLock, _ := EffectiveUseNodeLock(config)
		params := IndexParams{
			Dim:            dataDim,
			MaxElements:    uint64(config.Data.MaxElements),
			M:              config.Index.M,
			EfConstruction: config.Index.EfConstruction,
			LevelM:         config.Index.LevelM,
			Alpha:          config.Index.Alpha,
			Threads:        config.Workload.NumThreads,
			DataType:       DataTypeFloat,
			UseNodeLock:    useNodeLock,
			VisitLimit:     config.Index.VisitLimit,
			SealThreshold:  threshold,
			SealedType:     baseIndexType,
		}
		index = NewIndex(IndexTypeSegment, params)
	default:
		log.Fatalf("unsupported index type: %s\n", config.Index.IndexType)
	}

	return index
}

func getMetric(datasetName string) Metric {
	if datasetName == "glove1.2m" || datasetName == "glove" || datasetName == "glove-100-angular" {
		return MetricCosine
	}
	return MetricL2
}
