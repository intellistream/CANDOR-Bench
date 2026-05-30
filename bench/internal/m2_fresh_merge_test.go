package internal

import "testing"

func TestANNchorM2FreshMergeGraphRegion(t *testing.T) {
	t.Setenv("ANNCHOR_M2_REGION_CAP", "1")
	t.Setenv("ANNCHOR_M2_REGION_OVERLAP", "0")
	t.Setenv("ANNCHOR_M2_EDGE_ANCHOR_K", "2")
	t.Setenv("ANNCHOR_M2_REGION_MAX", "8")

	idx := NewIndex(IndexTypeANNchorM2, IndexParams{
		Dim:            2,
		MaxElements:    16,
		M:              4,
		EfConstruction: 20,
		Threads:        1,
		DataType:       DataTypeFloat,
		UseNodeLock:    true,
		Metric:         MetricL2,
	})
	defer idx.Close()

	base := [][]float32{
		{0, 0},
		{10, 0},
		{0, 10},
		{10, 10},
	}
	if err := idx.Build(base, []uint32{0, 1, 2, 3}); err != nil {
		t.Fatalf("build ANNchorM2: %v", err)
	}

	queries := [][]float32{{0.1, 0}}
	committed := [][]uint32{{0, 1}}
	fresh := [][]float32{
		{0.05, 0},
		{9, 9},
		{0, 9},
	}
	freshTags := []uint32{4, 5, 6}
	if err := idx.BatchInsert(fresh, freshTags); err != nil {
		t.Fatalf("insert fresh nodes: %v", err)
	}
	joined, stats, err := idx.FreshJoinMerge(
		queries,
		committed,
		2,
		fresh,
		freshTags,
		4,
		FreshJoinParams{SmallFreshThreshold: 1},
	)
	if err != nil {
		t.Fatalf("fresh merge: %v", err)
	}
	if got := joined[0][0]; got != 4 {
		t.Fatalf("fresh graph-region candidate was not admitted first: got %d, row=%v", got, joined[0])
	}
	if stats.ExactedCandidates == 0 || stats.ExactedCandidates >= stats.FreshVectors {
		t.Fatalf("expected graph-region filter to exact a strict subset, stats=%+v", stats)
	}
}

func TestANNchorM2FreshMergeGraphRegionUnpublishedFreshExactsAll(t *testing.T) {
	t.Setenv("ANNCHOR_M2_REGION_CAP", "1")
	t.Setenv("ANNCHOR_M2_REGION_OVERLAP", "0")
	t.Setenv("ANNCHOR_M2_EDGE_ANCHOR_K", "2")
	t.Setenv("ANNCHOR_M2_REGION_MAX", "8")

	idx := NewIndex(IndexTypeANNchorM2, IndexParams{
		Dim:            2,
		MaxElements:    16,
		M:              4,
		EfConstruction: 20,
		Threads:        1,
		DataType:       DataTypeFloat,
		UseNodeLock:    true,
		Metric:         MetricL2,
	})
	defer idx.Close()

	base := [][]float32{
		{0, 0},
		{10, 0},
		{0, 10},
		{10, 10},
	}
	if err := idx.Build(base, []uint32{0, 1, 2, 3}); err != nil {
		t.Fatalf("build ANNchorM2: %v", err)
	}

	fresh := [][]float32{
		{0.05, 0},
		{9, 9},
		{0, 9},
	}
	joined, stats, err := idx.FreshJoinMerge(
		[][]float32{{0.1, 0}},
		[][]uint32{{0, 1}},
		2,
		fresh,
		[]uint32{4, 5, 6},
		4,
		FreshJoinParams{SmallFreshThreshold: 1},
	)
	if err != nil {
		t.Fatalf("fresh merge: %v", err)
	}
	if got := joined[0][0]; got != 4 {
		t.Fatalf("unpublished fresh vector was not exact-admitted first: got %d, row=%v", got, joined[0])
	}
	if stats.ExactedCandidates != stats.FreshVectors {
		t.Fatalf("expected unpublished fresh metadata to exact all fresh vectors, stats=%+v", stats)
	}
}
