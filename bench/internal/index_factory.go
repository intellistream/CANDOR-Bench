package internal

import (
	"log"
)

var baselineAlgoMap = map[string]string{
	"annchor":         "hnsw",
	"annchor-m1":      "hnsw",
	"annchor-m2":      "hnsw",
	"annchor-m3":      "hnsw",
	"annchor-preempt": "hnsw",
	"annchor-trim":    "hnsw",
	"hnsw-visible":    "hnsw",
}

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
		if config.Search.EnableS3 != nil {
			index.SetEnableS3(*config.Search.EnableS3)
			log.Printf("HNSW S3: %v", *config.Search.EnableS3)
		}
		if config.Search.S3ProximityThreshold != nil {
			index.SetS3ProximityThreshold(*config.Search.S3ProximityThreshold)
			log.Printf("HNSW S3 threshold: %v", *config.Search.S3ProximityThreshold)
		}
		if config.Search.EnablePathSkip != nil {
			index.SetEnablePathSkip(*config.Search.EnablePathSkip)
			log.Printf("HNSW PathSkip: %v", *config.Search.EnablePathSkip)
		}
		if config.Search.EnableCandidateInjection != nil {
			index.SetEnableCandidateInjection(*config.Search.EnableCandidateInjection)
			log.Printf("HNSW CandidateInjection: %v", *config.Search.EnableCandidateInjection)
		}
		if config.Search.EnableSearchSharing != nil {
			index.SetEnableSearchSharing(*config.Search.EnableSearchSharing)
			log.Printf("HNSW SearchSharing: %v", *config.Search.EnableSearchSharing)
		}
		if config.Search.SearchSharingCheckInterval != nil {
			index.SetSearchSharingCheckInterval(*config.Search.SearchSharingCheckInterval)
			log.Printf("HNSW SearchSharingCheckInterval: %d", *config.Search.SearchSharingCheckInterval)
		}
		if config.Search.EnableWarmStart != nil {
			index.SetEnableWarmStart(*config.Search.EnableWarmStart)
			log.Printf("HNSW WarmStart: %v", *config.Search.EnableWarmStart)
		}
	case "hnsw-visible":
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
		index = NewIndex(IndexTypeHNSWVisible, params)
		mode := visibilityModeValue(config.Search.VisibilityMode)
		index.SetVisibilityMode(mode)
		log.Printf("HNSWVisible visibility mode: %s (%d)", visibilityModeName(mode), mode)
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
	case "annchor":
		useNodeLock, _ := EffectiveUseNodeLock(config)
		params := IndexParams{
			Dim:            dataDim,
			MaxElements:    uint64(config.Data.MaxElements),
			M:              config.Index.M,
			EfConstruction: config.Index.EfConstruction,
			Alpha:          config.Index.Alpha,
			Threads:        config.Workload.NumThreads,
			DataType:       DataTypeFloat,
			VisitLimit:     config.Index.VisitLimit,
			UseNodeLock:    useNodeLock,
			Metric:         getMetric(config.Data.DatasetName),
		}
		index = NewIndex(IndexTypeANNchor, params)
		if config.Search.EnableMvcc != nil {
			index.SetEnableMvcc(*config.Search.EnableMvcc)
			log.Printf("ANNchor MVCC: %v", *config.Search.EnableMvcc)
		}
		if config.Search.EnableUndoRecovery != nil {
			index.SetEnableUndoRecovery(*config.Search.EnableUndoRecovery)
			log.Printf("ANNchor UndoRecovery: %v", *config.Search.EnableUndoRecovery)
		}

		if config.Search.EnableS3 != nil {
			index.SetEnableS3(*config.Search.EnableS3)
			log.Printf("ANNchor S3: %v", *config.Search.EnableS3)
		}
		if config.Search.S3ProximityThreshold != nil {
			index.SetS3ProximityThreshold(*config.Search.S3ProximityThreshold)
			log.Printf("ANNchor S3 threshold: %v", *config.Search.S3ProximityThreshold)
		}
	case "annchor-dev":
		useNodeLock, _ := EffectiveUseNodeLock(config)
		params := IndexParams{
			Dim:            dataDim,
			MaxElements:    uint64(config.Data.MaxElements),
			M:              config.Index.M,
			EfConstruction: config.Index.EfConstruction,
			Alpha:          config.Index.Alpha,
			Threads:        config.Workload.NumThreads,
			DataType:       DataTypeFloat,
			VisitLimit:     config.Index.VisitLimit,
			UseNodeLock:    useNodeLock,
			Metric:         getMetric(config.Data.DatasetName),
		}
		index = NewIndex(IndexTypeANNchorDev, params)
		if config.Search.EnableMvcc != nil {
			index.SetEnableMvcc(*config.Search.EnableMvcc)
			log.Printf("ANNchorDev MVCC: %v", *config.Search.EnableMvcc)
		}
		if config.Search.EnableUndoRecovery != nil {
			index.SetEnableUndoRecovery(*config.Search.EnableUndoRecovery)
			log.Printf("ANNchorDev UndoRecovery: %v", *config.Search.EnableUndoRecovery)
		}
		if config.Search.EnableCnr != nil {
			index.SetEnableCnr(*config.Search.EnableCnr)
			log.Printf("ANNchorDev EnableCnr: %v", *config.Search.EnableCnr)
		}
		if config.Search.CnrDegreeThreshold != nil {
			index.SetCnrDegreeThreshold(float32(*config.Search.CnrDegreeThreshold))
			log.Printf("ANNchorDev CnrDegreeThreshold: %v", *config.Search.CnrDegreeThreshold)
		}
		if config.Search.CnrMaxRecover != nil {
			index.SetCnrMaxRecover(*config.Search.CnrMaxRecover)
			log.Printf("ANNchorDev CnrMaxRecover: %v", *config.Search.CnrMaxRecover)
		}
		if config.Search.CnrStagnationHops != nil {
			index.SetCnrStagnationHops(*config.Search.CnrStagnationHops)
			log.Printf("ANNchorDev CnrStagnationHops: %v", *config.Search.CnrStagnationHops)
		}
		if config.Search.EnableM2DualPath != nil {
			index.SetEnableM2DualPath(*config.Search.EnableM2DualPath)
			log.Printf("ANNchorDev EnableM2DualPath: %v", *config.Search.EnableM2DualPath)
		}
		if config.Search.M2RiskHops != nil {
			index.SetM2RiskHops(*config.Search.M2RiskHops)
			log.Printf("ANNchorDev M2RiskHops: %v", *config.Search.M2RiskHops)
		}
		if config.Search.M2AssistBudget != nil {
			index.SetM2AssistBudget(*config.Search.M2AssistBudget)
			log.Printf("ANNchorDev M2AssistBudget: %v", *config.Search.M2AssistBudget)
		}
	case "annchor-m1":
		useNodeLock, _ := EffectiveUseNodeLock(config)
		params := IndexParams{
			Dim:            dataDim,
			MaxElements:    uint64(config.Data.MaxElements),
			M:              config.Index.M,
			EfConstruction: config.Index.EfConstruction,
			Alpha:          config.Index.Alpha,
			Threads:        config.Workload.NumThreads,
			DataType:       DataTypeFloat,
			VisitLimit:     config.Index.VisitLimit,
			UseNodeLock:    useNodeLock,
			Metric:         getMetric(config.Data.DatasetName),
		}
		index = NewIndex(IndexTypeANNchorM1, params)
		if config.Search.EnableMvcc != nil {
			index.SetEnableMvcc(*config.Search.EnableMvcc)
			log.Printf("ANNchorM1 MVCC: %v", *config.Search.EnableMvcc)
		}
		if config.Search.EnableUndoRecovery != nil {
			index.SetEnableUndoRecovery(*config.Search.EnableUndoRecovery)
			log.Printf("ANNchorM1 UndoRecovery: %v", *config.Search.EnableUndoRecovery)
		}
	case "annchor-m2":
		useNodeLock, _ := EffectiveUseNodeLock(config)
		params := IndexParams{
			Dim:            dataDim,
			MaxElements:    uint64(config.Data.MaxElements),
			M:              config.Index.M,
			EfConstruction: config.Index.EfConstruction,
			Alpha:          config.Index.Alpha,
			Threads:        config.Workload.NumThreads,
			DataType:       DataTypeFloat,
			VisitLimit:     config.Index.VisitLimit,
			UseNodeLock:    useNodeLock,
			Metric:         getMetric(config.Data.DatasetName),
		}
		index = NewIndex(IndexTypeANNchorM2, params)
		if config.Search.EnableMvcc != nil {
			index.SetEnableMvcc(*config.Search.EnableMvcc)
			log.Printf("ANNchorM2 MVCC: %v", *config.Search.EnableMvcc)
		}
		if config.Search.EnableUndoRecovery != nil {
			index.SetEnableUndoRecovery(*config.Search.EnableUndoRecovery)
			log.Printf("ANNchorM2 UndoRecovery: %v", *config.Search.EnableUndoRecovery)
		}
	case "annchor-trim":
		useNodeLock, _ := EffectiveUseNodeLock(config)
		params := IndexParams{
			Dim:            dataDim,
			MaxElements:    uint64(config.Data.MaxElements),
			M:              config.Index.M,
			EfConstruction: config.Index.EfConstruction,
			Alpha:          config.Index.Alpha,
			Threads:        config.Workload.NumThreads,
			DataType:       DataTypeFloat,
			VisitLimit:     config.Index.VisitLimit,
			UseNodeLock:    useNodeLock,
			Metric:         getMetric(config.Data.DatasetName),
		}
		index = NewIndex(IndexTypeANNchorTrim, params)
		if config.Search.EnableMvcc != nil {
			index.SetEnableMvcc(*config.Search.EnableMvcc)
			log.Printf("ANNchorTrim MVCC: %v", *config.Search.EnableMvcc)
		}
		if config.Search.EnableUndoRecovery != nil {
			index.SetEnableUndoRecovery(*config.Search.EnableUndoRecovery)
			log.Printf("ANNchorTrim UndoRecovery: %v", *config.Search.EnableUndoRecovery)
		}
		if config.Search.EnableTrimRecoveryFilter != nil {
			index.SetEnableTrimRecoveryFilter(*config.Search.EnableTrimRecoveryFilter)
			log.Printf("ANNchorTrim EnableTrimRecoveryFilter: %v", *config.Search.EnableTrimRecoveryFilter)
		}
		if config.Search.TrimRecoveryRelaxFactor != nil {
			index.SetTrimRecoveryRelaxFactor(*config.Search.TrimRecoveryRelaxFactor)
			log.Printf("ANNchorTrim TrimRecoveryRelaxFactor: %.3f", *config.Search.TrimRecoveryRelaxFactor)
		}
		if config.Search.TrimRecoveryMarginRatio != nil {
			index.SetTrimRecoveryMarginRatio(*config.Search.TrimRecoveryMarginRatio)
			log.Printf("ANNchorTrim TrimRecoveryMarginRatio: %.3f", *config.Search.TrimRecoveryMarginRatio)
		}
	case "annchor-preempt":
		useNodeLock, _ := EffectiveUseNodeLock(config)
		params := IndexParams{
			Dim:            dataDim,
			MaxElements:    uint64(config.Data.MaxElements),
			M:              config.Index.M,
			EfConstruction: config.Index.EfConstruction,
			Alpha:          config.Index.Alpha,
			Threads:        config.Workload.NumThreads,
			DataType:       DataTypeFloat,
			VisitLimit:     config.Index.VisitLimit,
			UseNodeLock:    useNodeLock,
			Metric:         getMetric(config.Data.DatasetName),
		}
		index = NewIndex(IndexTypeANNchorPreempt, params)
		if config.Search.EnableMvcc != nil {
			index.SetEnableMvcc(*config.Search.EnableMvcc)
			log.Printf("ANNchorPreempt MVCC: %v", *config.Search.EnableMvcc)
		}
		if config.Search.EnableUndoRecovery != nil {
			index.SetEnableUndoRecovery(*config.Search.EnableUndoRecovery)
			log.Printf("ANNchorPreempt UndoRecovery: %v", *config.Search.EnableUndoRecovery)
		}
		if config.Search.EnablePreemptM2 != nil {
			index.SetPreemptEnableM2(*config.Search.EnablePreemptM2)
			log.Printf("ANNchorPreempt EnablePreemptM2: %v", *config.Search.EnablePreemptM2)
		}
		if config.Search.PreemptQuantumPoints != nil {
			index.SetPreemptQuantumPoints(*config.Search.PreemptQuantumPoints)
			log.Printf("ANNchorPreempt PreemptQuantumPoints: %v", *config.Search.PreemptQuantumPoints)
		}
		if config.Search.PreemptSearchBacklogThresh != nil {
			index.SetPreemptSearchBacklogThreshold(*config.Search.PreemptSearchBacklogThresh)
			log.Printf("ANNchorPreempt PreemptSearchBacklogThreshold: %v", *config.Search.PreemptSearchBacklogThresh)
		}
		if config.Search.PreemptMaxYieldsPerBatch != nil {
			index.SetPreemptMaxYieldsPerBatch(*config.Search.PreemptMaxYieldsPerBatch)
			log.Printf("ANNchorPreempt PreemptMaxYieldsPerBatch: %v", *config.Search.PreemptMaxYieldsPerBatch)
		}
		if config.Search.PreemptBudgetWindowUs != nil {
			index.SetPreemptBudgetWindowUs(*config.Search.PreemptBudgetWindowUs)
			log.Printf("ANNchorPreempt PreemptBudgetWindowUs: %v", *config.Search.PreemptBudgetWindowUs)
		}
		if config.Search.PreemptBudgetPct != nil {
			index.SetPreemptBudgetPct(*config.Search.PreemptBudgetPct)
			log.Printf("ANNchorPreempt PreemptBudgetPct: %.2f", *config.Search.PreemptBudgetPct)
		}
		if config.Search.PreemptPriorityCap != nil {
			index.SetPreemptPriorityCap(*config.Search.PreemptPriorityCap)
			log.Printf("ANNchorPreempt PreemptPriorityCap: %v", *config.Search.PreemptPriorityCap)
		}
	case "annchor-m3":
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
		}
		index = NewIndex(IndexTypeANNchorM3, params)
		if config.Search.EnableUndoRecovery != nil {
			index.SetEnableUndoRecovery(*config.Search.EnableUndoRecovery)
			log.Printf("ANNchorM3 UndoRecovery: %v", *config.Search.EnableUndoRecovery)
		}
		log.Printf("ANNchorM3: clean ANNchorM2 fork")
	case "annchor-m3-off":
		// Retired alias retained for old configs; maps to clean ANNchorM3.
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
		}
		index = NewIndex(IndexTypeANNchorM3, params)
		if config.Search.EnableUndoRecovery != nil {
			index.SetEnableUndoRecovery(*config.Search.EnableUndoRecovery)
			log.Printf("ANNchorM3-OFF UndoRecovery: %v", *config.Search.EnableUndoRecovery)
		}
		log.Printf("ANNchorM3-OFF: retired alias mapped to clean ANNchorM3")
	case "annchor-m3-prune-assist":
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
		}
		index = NewIndex(IndexTypeANNchorM3, params)
		if config.Search.EnableUndoRecovery != nil {
			index.SetEnableUndoRecovery(*config.Search.EnableUndoRecovery)
			log.Printf("ANNchorM3-PruneAssist UndoRecovery: %v", *config.Search.EnableUndoRecovery)
		}
		log.Printf("ANNchorM3-PruneAssist: retired alias mapped to clean ANNchorM3")
	case "annchor-m3-append-only":
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
		}
		index = NewIndex(IndexTypeANNchorM3, params)
		if config.Search.EnableUndoRecovery != nil {
			index.SetEnableUndoRecovery(*config.Search.EnableUndoRecovery)
			log.Printf("ANNchorM3-AppendOnly UndoRecovery: %v", *config.Search.EnableUndoRecovery)
		}
		log.Printf("ANNchorM3-AppendOnly: retired alias mapped to clean ANNchorM3")
	case "annchor-m3-ablation":
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
		}
		index = NewIndex(IndexTypeANNchorM3, params)
		if config.Search.EnableUndoRecovery != nil {
			index.SetEnableUndoRecovery(*config.Search.EnableUndoRecovery)
			log.Printf("ANNchorM3-Ablation UndoRecovery: %v", *config.Search.EnableUndoRecovery)
		}
		log.Printf("ANNchorM3-Ablation: retired alias mapped to clean ANNchorM3")
	case "segmented":
		baseType := config.Index.SealedType
		if baseType == "" {
			baseType = "hnsw"
		}
		baseIndexType := map[string]IndexType{
			"hnsw":                    IndexTypeHNSW,
			"parlayhnsw":              IndexTypeParlayHNSW,
			"parlayvamana":            IndexTypeParlayVamana,
			"vamana":                  IndexTypeVamana,
			"annchor":                 IndexTypeANNchor,
			"annchor-m1":              IndexTypeANNchorM1,
			"annchor-m2":              IndexTypeANNchorM2,
			"annchor-preempt":         IndexTypeANNchorPreempt,
			"annchor-trim":            IndexTypeANNchorTrim,
			"annchor-m3":              IndexTypeANNchorM3,
			"annchor-m3-off":          IndexTypeANNchorM3,
			"annchor-m3-prune-assist": IndexTypeANNchorM3,
			"annchor-m3-append-only":  IndexTypeANNchorM3,
			"annchor-m3-ablation":     IndexTypeANNchorM3,
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

func visibilityModeValue(mode string) int {
	switch mode {
	case "post_filter_refill", "post-filter-refill", "post_filter", "post-filter":
		return 1
	case "traversal_filter", "traversal-filter", "in_search_filter", "in-search-filter":
		return 2
	default:
		return 0
	}
}

func visibilityModeName(mode int) string {
	switch mode {
	case 1:
		return "post_filter_refill"
	case 2:
		return "traversal_filter"
	default:
		return "result_filter"
	}
}

func getMetric(datasetName string) Metric {
	if datasetName == "glove1.2m" || datasetName == "glove" || datasetName == "glove-100-angular" {
		return MetricCosine
	}
	return MetricL2
}
