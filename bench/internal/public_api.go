package internal

// Public wrappers for recall_sweep cmd

func LoadConfigVariantsPublic(filename string) ([]*Config, error) {
	return loadConfigVariants(filename)
}

func ClampDataBoundsPublic(config *Config, totalPoints int) {
	clampDataBounds(config, totalPoints)
}

func CreateAndBuildIndexPublic(config *Config, data []float32, dataDim int) *Index {
	return createAndBuildIndex(config, data, dataDim)
}

func LoadQueryDatasetPublic(path string, expectedDim int, label string) ([]float32, error) {
	return loadQueryDataset(path, expectedDim, label)
}

func CalcOverallRecallForIndexPublic(index *Index, config *Config, queries []float32, dataDim int) (float64, error) {
	return calcOverallRecallForIndex(index, config, queries, dataDim)
}
