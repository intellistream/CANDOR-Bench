package internal

import (
	"os"
	"testing"
)

func TestParseConfigBytesMidInsertOverrides(t *testing.T) {
	cfg, err := parseConfigBytes([]byte(`
data:
  dataset_name: sift
  data_path: ../data/sift/sift_base.bin
  incr_query_path: ../data/sift/sift_query_stream.bin
  overall_query_path: ../data/sift/sift_query.bin
index:
  index_type: annchor-preempt
  m: 16
  ef_construction: 200
search:
  ef_search: 50
  enable_mvcc: true
  enable_preempt_m2: false
  mid_insert_preempt_enable: true
  mid_insert_preempt_k: 64
  mid_insert_preempt_revalidate: false
  mid_insert_shadow_replan: true
  mid_insert_harm_guard: true
  mid_insert_harm_micro_replan: true
  mid_insert_harm_micro_probe_ef: 32
  mid_insert_preempt_max_wait_us: 0
  mid_insert_preempt_activate_after_commits: 50000
  mid_insert_preempt_every_n: 8
  mid_insert_preempt_max_inflight: 1
  mid_insert_harm_micro_pool_cap: 48
  mid_insert_harm_online_policy: true
  mid_insert_harm_busy_wait_commits: 24
  mid_insert_harm_search_backlog_threshold: 2
  mid_insert_harm_priority_search_threshold: 1
  mid_insert_harm_full_foreign_outrank: 3
  mid_insert_harm_full_selected_frontier_touched: 2
  mid_insert_harm_defer_enabled: true
  mid_insert_harm_defer_queue_cap: 512
  mid_insert_harm_defer_drain_budget: 2
  mid_insert_harm_defer_high_watermark_pct: 75
workload:
  batch_size: 20
  num_threads: 64
  query_mode: chasing
result:
  output_dir: ./result/test
`))
	if err != nil {
		t.Fatalf("parseConfigBytes returned error: %v", err)
	}

	if cfg.Search.MidInsertPreemptEnable == nil || !*cfg.Search.MidInsertPreemptEnable {
		t.Fatalf("mid_insert_preempt_enable not parsed")
	}
	if cfg.Search.MidInsertPreemptK == nil || *cfg.Search.MidInsertPreemptK != 64 {
		t.Fatalf("mid_insert_preempt_k not parsed: %+v", cfg.Search.MidInsertPreemptK)
	}
	if cfg.Search.MidInsertShadowReplan == nil || !*cfg.Search.MidInsertShadowReplan {
		t.Fatalf("mid_insert_shadow_replan not parsed")
	}
	if cfg.Search.MidInsertHarmGuard == nil || !*cfg.Search.MidInsertHarmGuard {
		t.Fatalf("mid_insert_harm_guard not parsed")
	}
	if cfg.Search.MidInsertHarmMicroReplan == nil || !*cfg.Search.MidInsertHarmMicroReplan {
		t.Fatalf("mid_insert_harm_micro_replan not parsed")
	}
	if cfg.Search.MidInsertHarmMicroProbeEf == nil || *cfg.Search.MidInsertHarmMicroProbeEf != 32 {
		t.Fatalf("mid_insert_harm_micro_probe_ef not parsed")
	}
	if cfg.Search.MidInsertPreemptMaxWaitUs == nil || *cfg.Search.MidInsertPreemptMaxWaitUs != 0 {
		t.Fatalf("mid_insert_preempt_max_wait_us not parsed")
	}
	if cfg.Search.MidInsertActivateAfterCommits == nil || *cfg.Search.MidInsertActivateAfterCommits != 50000 {
		t.Fatalf("mid_insert_preempt_activate_after_commits not parsed")
	}
	if cfg.Search.MidInsertPreemptEveryN == nil || *cfg.Search.MidInsertPreemptEveryN != 8 {
		t.Fatalf("mid_insert_preempt_every_n not parsed")
	}
	if cfg.Search.MidInsertPreemptMaxInflight == nil || *cfg.Search.MidInsertPreemptMaxInflight != 1 {
		t.Fatalf("mid_insert_preempt_max_inflight not parsed")
	}
	if cfg.Search.MidInsertHarmMicroPoolCap == nil || *cfg.Search.MidInsertHarmMicroPoolCap != 48 {
		t.Fatalf("mid_insert_harm_micro_pool_cap not parsed")
	}
	if cfg.Search.MidInsertHarmOnlinePolicy == nil || !*cfg.Search.MidInsertHarmOnlinePolicy {
		t.Fatalf("mid_insert_harm_online_policy not parsed")
	}
	if cfg.Search.MidInsertHarmBusyWaitCommits == nil || *cfg.Search.MidInsertHarmBusyWaitCommits != 24 {
		t.Fatalf("mid_insert_harm_busy_wait_commits not parsed")
	}
	if cfg.Search.MidInsertHarmSearchBacklogThreshold == nil || *cfg.Search.MidInsertHarmSearchBacklogThreshold != 2 {
		t.Fatalf("mid_insert_harm_search_backlog_threshold not parsed")
	}
	if cfg.Search.MidInsertHarmPrioritySearchThreshold == nil || *cfg.Search.MidInsertHarmPrioritySearchThreshold != 1 {
		t.Fatalf("mid_insert_harm_priority_search_threshold not parsed")
	}
	if cfg.Search.MidInsertHarmFullForeignOutrank == nil || *cfg.Search.MidInsertHarmFullForeignOutrank != 3 {
		t.Fatalf("mid_insert_harm_full_foreign_outrank not parsed")
	}
	if cfg.Search.MidInsertHarmFullSelectedFrontierTouched == nil || *cfg.Search.MidInsertHarmFullSelectedFrontierTouched != 2 {
		t.Fatalf("mid_insert_harm_full_selected_frontier_touched not parsed")
	}
	if cfg.Search.MidInsertHarmDeferEnabled == nil || !*cfg.Search.MidInsertHarmDeferEnabled {
		t.Fatalf("mid_insert_harm_defer_enabled not parsed")
	}
	if cfg.Search.MidInsertHarmDeferQueueCap == nil || *cfg.Search.MidInsertHarmDeferQueueCap != 512 {
		t.Fatalf("mid_insert_harm_defer_queue_cap not parsed")
	}
	if cfg.Search.MidInsertHarmDeferDrainBudget == nil || *cfg.Search.MidInsertHarmDeferDrainBudget != 2 {
		t.Fatalf("mid_insert_harm_defer_drain_budget not parsed")
	}
	if cfg.Search.MidInsertHarmDeferHighWatermarkPct == nil || *cfg.Search.MidInsertHarmDeferHighWatermarkPct != 75 {
		t.Fatalf("mid_insert_harm_defer_high_watermark_pct not parsed")
	}
}

func TestApplyMidInsertPreemptEnvRestoresState(t *testing.T) {
	oldEnable, hadEnable := os.LookupEnv("ANN_MID_INSERT_PREEMPT_ENABLE")
	oldK, hadK := os.LookupEnv("ANN_MID_INSERT_PREEMPT_K")
	oldShadow, hadShadow := os.LookupEnv("ANN_MID_INSERT_PREEMPT_SHADOW_REPLAN")
	oldGuard, hadGuard := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_GUARD")
	oldMicro, hadMicro := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_MICRO_REPLAN")
	oldMicroProbe, hadMicroProbe := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_MICRO_PROBE_EF")
	oldMicroCap, hadMicroCap := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_MICRO_POOL_CAP")
	oldOnline, hadOnline := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_ONLINE_POLICY")
	oldBusy, hadBusy := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_BUSY_WAIT_COMMITS")
	oldSearchBacklog, hadSearchBacklog := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_SEARCH_BACKLOG_THRESHOLD")
	oldPrioritySearch, hadPrioritySearch := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_PRIORITY_SEARCH_THRESHOLD")
	oldFullOutrank, hadFullOutrank := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_FULL_FOREIGN_OUTRANK")
	oldFullTouch, hadFullTouch := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_FULL_SELECTED_FRONTIER_TOUCHED")
	oldDefer, hadDefer := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_ENABLED")
	oldDeferCap, hadDeferCap := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_QUEUE_CAP")
	oldDeferBudget, hadDeferBudget := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_DRAIN_BUDGET")
	oldDeferPct, hadDeferPct := os.LookupEnv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_HIGH_WATERMARK_PCT")
	t.Cleanup(func() {
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_ENABLE", oldEnable, hadEnable)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_K", oldK, hadK)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_SHADOW_REPLAN", oldShadow, hadShadow)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_GUARD", oldGuard, hadGuard)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_MICRO_REPLAN", oldMicro, hadMicro)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_MICRO_PROBE_EF", oldMicroProbe, hadMicroProbe)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_MICRO_POOL_CAP", oldMicroCap, hadMicroCap)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_ONLINE_POLICY", oldOnline, hadOnline)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_BUSY_WAIT_COMMITS", oldBusy, hadBusy)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_SEARCH_BACKLOG_THRESHOLD", oldSearchBacklog, hadSearchBacklog)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_PRIORITY_SEARCH_THRESHOLD", oldPrioritySearch, hadPrioritySearch)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_FULL_FOREIGN_OUTRANK", oldFullOutrank, hadFullOutrank)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_FULL_SELECTED_FRONTIER_TOUCHED", oldFullTouch, hadFullTouch)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_DEFER_ENABLED", oldDefer, hadDefer)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_DEFER_QUEUE_CAP", oldDeferCap, hadDeferCap)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_DEFER_DRAIN_BUDGET", oldDeferBudget, hadDeferBudget)
		restoreEnvForTest("ANN_MID_INSERT_PREEMPT_HARM_DEFER_HIGH_WATERMARK_PCT", oldDeferPct, hadDeferPct)
	})

	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_ENABLE", "0"); err != nil {
		t.Fatalf("Setenv enable baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_K", "3"); err != nil {
		t.Fatalf("Setenv k baseline failed: %v", err)
	}
	if err := os.Unsetenv("ANN_MID_INSERT_PREEMPT_SHADOW_REPLAN"); err != nil {
		t.Fatalf("Unsetenv shadow baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_GUARD", "0"); err != nil {
		t.Fatalf("Setenv guard baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_MICRO_REPLAN", "0"); err != nil {
		t.Fatalf("Setenv micro baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_MICRO_PROBE_EF", "12"); err != nil {
		t.Fatalf("Setenv micro probe baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_MICRO_POOL_CAP", "16"); err != nil {
		t.Fatalf("Setenv micro cap baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_ONLINE_POLICY", "0"); err != nil {
		t.Fatalf("Setenv online baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_BUSY_WAIT_COMMITS", "11"); err != nil {
		t.Fatalf("Setenv busy baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_SEARCH_BACKLOG_THRESHOLD", "5"); err != nil {
		t.Fatalf("Setenv search backlog baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_PRIORITY_SEARCH_THRESHOLD", "2"); err != nil {
		t.Fatalf("Setenv priority search baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_FULL_FOREIGN_OUTRANK", "4"); err != nil {
		t.Fatalf("Setenv full outrank baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_FULL_SELECTED_FRONTIER_TOUCHED", "5"); err != nil {
		t.Fatalf("Setenv full touch baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_ENABLED", "0"); err != nil {
		t.Fatalf("Setenv defer baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_QUEUE_CAP", "64"); err != nil {
		t.Fatalf("Setenv defer cap baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_DRAIN_BUDGET", "3"); err != nil {
		t.Fatalf("Setenv defer budget baseline failed: %v", err)
	}
	if err := os.Setenv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_HIGH_WATERMARK_PCT", "60"); err != nil {
		t.Fatalf("Setenv defer pct baseline failed: %v", err)
	}

	enable := true
	k := 16
	shadow := true
	guard := true
	micro := true
	microProbe := 32
	microCap := 48
	online := true
	busy := 24
	searchBacklog := 2
	prioritySearch := 1
	fullOutrank := 3
	fullTouch := 2
	deferEnabled := true
	deferCap := 512
	deferBudget := 2
	deferPct := 75
	cfg := &Config{}
	cfg.Search.MidInsertPreemptEnable = &enable
	cfg.Search.MidInsertPreemptK = &k
	cfg.Search.MidInsertShadowReplan = &shadow
	cfg.Search.MidInsertHarmGuard = &guard
	cfg.Search.MidInsertHarmMicroReplan = &micro
	cfg.Search.MidInsertHarmMicroProbeEf = &microProbe
	cfg.Search.MidInsertHarmMicroPoolCap = &microCap
	cfg.Search.MidInsertHarmOnlinePolicy = &online
	cfg.Search.MidInsertHarmBusyWaitCommits = &busy
	cfg.Search.MidInsertHarmSearchBacklogThreshold = &searchBacklog
	cfg.Search.MidInsertHarmPrioritySearchThreshold = &prioritySearch
	cfg.Search.MidInsertHarmFullForeignOutrank = &fullOutrank
	cfg.Search.MidInsertHarmFullSelectedFrontierTouched = &fullTouch
	cfg.Search.MidInsertHarmDeferEnabled = &deferEnabled
	cfg.Search.MidInsertHarmDeferQueueCap = &deferCap
	cfg.Search.MidInsertHarmDeferDrainBudget = &deferBudget
	cfg.Search.MidInsertHarmDeferHighWatermarkPct = &deferPct

	restore, err := applyMidInsertPreemptEnv(cfg)
	if err != nil {
		t.Fatalf("applyMidInsertPreemptEnv returned error: %v", err)
	}

	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_ENABLE"); got != "1" {
		t.Fatalf("enable env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_K"); got != "16" {
		t.Fatalf("k env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_SHADOW_REPLAN"); got != "1" {
		t.Fatalf("shadow env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_GUARD"); got != "1" {
		t.Fatalf("guard env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_MICRO_REPLAN"); got != "1" {
		t.Fatalf("micro env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_MICRO_PROBE_EF"); got != "32" {
		t.Fatalf("micro probe env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_MICRO_POOL_CAP"); got != "48" {
		t.Fatalf("micro cap env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_ONLINE_POLICY"); got != "1" {
		t.Fatalf("online env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_BUSY_WAIT_COMMITS"); got != "24" {
		t.Fatalf("busy env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_SEARCH_BACKLOG_THRESHOLD"); got != "2" {
		t.Fatalf("search backlog env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_PRIORITY_SEARCH_THRESHOLD"); got != "1" {
		t.Fatalf("priority search env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_FULL_FOREIGN_OUTRANK"); got != "3" {
		t.Fatalf("full outrank env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_FULL_SELECTED_FRONTIER_TOUCHED"); got != "2" {
		t.Fatalf("full touch env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_ENABLED"); got != "1" {
		t.Fatalf("defer env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_QUEUE_CAP"); got != "512" {
		t.Fatalf("defer cap env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_DRAIN_BUDGET"); got != "2" {
		t.Fatalf("defer budget env mismatch: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_HIGH_WATERMARK_PCT"); got != "75" {
		t.Fatalf("defer pct env mismatch: got %q", got)
	}

	restore()

	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_ENABLE"); got != "0" {
		t.Fatalf("enable env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_K"); got != "3" {
		t.Fatalf("k env not restored: got %q", got)
	}
	if _, exists := os.LookupEnv("ANN_MID_INSERT_PREEMPT_SHADOW_REPLAN"); exists {
		t.Fatalf("shadow env should have been unset after restore")
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_GUARD"); got != "0" {
		t.Fatalf("guard env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_MICRO_REPLAN"); got != "0" {
		t.Fatalf("micro env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_MICRO_PROBE_EF"); got != "12" {
		t.Fatalf("micro probe env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_MICRO_POOL_CAP"); got != "16" {
		t.Fatalf("micro cap env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_ONLINE_POLICY"); got != "0" {
		t.Fatalf("online env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_BUSY_WAIT_COMMITS"); got != "11" {
		t.Fatalf("busy env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_SEARCH_BACKLOG_THRESHOLD"); got != "5" {
		t.Fatalf("search backlog env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_PRIORITY_SEARCH_THRESHOLD"); got != "2" {
		t.Fatalf("priority search env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_FULL_FOREIGN_OUTRANK"); got != "4" {
		t.Fatalf("full outrank env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_FULL_SELECTED_FRONTIER_TOUCHED"); got != "5" {
		t.Fatalf("full touch env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_ENABLED"); got != "0" {
		t.Fatalf("defer env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_QUEUE_CAP"); got != "64" {
		t.Fatalf("defer cap env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_DRAIN_BUDGET"); got != "3" {
		t.Fatalf("defer budget env not restored: got %q", got)
	}
	if got := os.Getenv("ANN_MID_INSERT_PREEMPT_HARM_DEFER_HIGH_WATERMARK_PCT"); got != "60" {
		t.Fatalf("defer pct env not restored: got %q", got)
	}
}

func restoreEnvForTest(key, value string, existed bool) {
	if existed {
		_ = os.Setenv(key, value)
		return
	}
	_ = os.Unsetenv(key)
}
