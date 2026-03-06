package internal

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v2"
)

type Config struct {
	VariantName string `yaml:"-"`
	Data        struct {
		DatasetName      string `yaml:"dataset_name"`
		MaxElements      int    `yaml:"max_elements"`
		BeginNum         int    `yaml:"begin_num"`
		DataType         string `yaml:"data_type"`
		DataPath         string `yaml:"data_path"`
		QueryPath        string `yaml:"query_path"`
		IncrQueryPath    string `yaml:"incr_query_path"`
		OverallQueryPath string `yaml:"overall_query_path"`
		OverallGtPath    string `yaml:"overall_gt_path"`
		IncrGtPath       string `yaml:"incr_gt_path"`
	} `yaml:"data"`

	Index struct {
		IndexType      string  `yaml:"index_type"`
		M              int     `yaml:"m"`
		EfConstruction int     `yaml:"ef_construction"`
		LevelM         float32 `yaml:"level_m"`
		Alpha          float32 `yaml:"alpha"`
		VisitLimit     int     `yaml:"visit_limit"`

		SealedType    string `yaml:"sealed_type"`
		SealThreshold int    `yaml:"seal_threshold"`
	} `yaml:"index"`

	Search struct {
		RecallAt    uint32  `yaml:"recall_at"`
		EfSearch    uint32  `yaml:"ef_search"`
		BeamWidth   uint32  `yaml:"beam_width"`
		Alpha       float32 `yaml:"alpha"`
		VisitLimit  uint32  `yaml:"visit_limit"`
		UseNodeLock           *bool    `yaml:"use_node_lock"`
		EnableMvcc            *bool    `yaml:"enable_mvcc"`
		EnableS3                  *bool    `yaml:"enable_s3"`
		S3ProximityThreshold      *float32 `yaml:"s3_proximity_threshold"`
		LazyRecoveryRatio         *float64 `yaml:"lazy_recovery_ratio"`
		RecoveryExpandCutoff      *int     `yaml:"recovery_expand_cutoff"`
		EnableOnlineGc            *bool    `yaml:"enable_online_gc"`
		EnablePathSkip            *bool    `yaml:"enable_path_skip"`
		EnableCandidateInjection  *bool    `yaml:"enable_candidate_injection"`
		S3AdaptiveEnabled         *bool    `yaml:"s3_adaptive_enabled"`
		S3AdaptiveThreshold       *float32 `yaml:"s3_adaptive_threshold"`
		EnableWarmStart           *bool    `yaml:"enable_warm_start"`
		EnableSearchSharing       *bool    `yaml:"enable_search_sharing"`
		SearchSharingCheckInterval *int    `yaml:"search_sharing_check_interval"`
	} `yaml:"search"`

	Workload struct {
		BatchSize          int     `yaml:"batch_size"`
		NumThreads         int     `yaml:"num_threads"`
		QueueSize          int     `yaml:"queue_size"`
		InsertEventRate    float64 `yaml:"insert_event_rate"`
		SearchEventRate    float64 `yaml:"search_event_rate"`
		WithExternalRWLock bool    `yaml:"with_external_rw_lock"`
		QueryMode          string  `yaml:"query_mode"`
		ZipfianSkew        float64 `yaml:"zipfian_skew"`
		PerQueryLatency    *bool   `yaml:"per_query_latency"`
	} `yaml:"workload"`

	ThroughputSweep struct {
		Enabled        bool    `yaml:"enabled"`
		StartRate      float64 `yaml:"start_rate"`
		EndRate        float64 `yaml:"end_rate"`
		Steps          int     `yaml:"steps"`
		WarmupSeconds  int     `yaml:"warmup_seconds"`
		MeasureSeconds int     `yaml:"measure_seconds"`
	} `yaml:"throughput_sweep"`

	Compare struct {
		Enabled  bool            `yaml:"enabled"`
		Simulate *SimulateConfig `yaml:"simulate"`
	} `yaml:"compare"`

	Profile struct {
		MemoryMonitorInterval int  `yaml:"memory_monitor_interval"`
		EnableMemoryProfile   bool `yaml:"enable_memory_profile"`
	} `yaml:"profile"`

	Result struct {
		OutputDir string `yaml:"output_dir"`
	} `yaml:"result"`
}

const (
	queryModeRoundRobin = "round_robin"
	queryModeChasing    = "chasing"
	queryModePeeking    = "peeking"
	queryModeZipfian    = "zipfian"

	defaultOverallRecallTool = "../utils/build/calc_recall"
	defaultIncrRecallTool    = "../utils/build/calc_incr_recall"
)

type PartialUpdateConfig struct {
	Enabled bool  `yaml:"enabled"`
	Seed    int64 `yaml:"seed"`
	Mode    int   `yaml:"mode"`
}

type SimulateConfig struct {
	Enabled bool                `yaml:"enabled"`
	Partial PartialUpdateConfig `yaml:"partial_update"`
}

type rateGroup struct {
	InsertEventRate float64
	SearchEventRate float64
}

func loadConfigVariants(filename string) ([]*Config, error) {
	buf, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading config file: %v", err)
	}

	var raw interface{}
	if err := yaml.Unmarshal(buf, &raw); err != nil {
		return nil, fmt.Errorf("error parsing config file: %v", err)
	}

	baseName := strings.TrimSuffix(filepath.Base(filename), filepath.Ext(filename))
	rateGroups := extractRateGroups(raw)
	params := collectSweepParams(raw, nil)
	if !hasSweep(params) && len(rateGroups) == 0 {
		cfg, err := parseConfigBytes(buf)
		if err != nil {
			return nil, err
		}
		cfg.VariantName = baseName
		return []*Config{cfg}, nil
	}

	combos := enumerateSweepCombos(params)
	if len(combos) == 0 {
		combos = [][]int{{}}
	}

	hasRateSweep := len(rateGroups) > 0

	var configs []*Config
	for _, combo := range combos {
		if hasRateSweep {
			for _, group := range rateGroups {
				materialized := materializeSweep(raw, params, combo)
				workload := ensureWorkloadMap(materialized)
				workload["insert_event_rate"] = group.InsertEventRate
				workload["search_event_rate"] = group.SearchEventRate
				data, err := yaml.Marshal(materialized)
				if err != nil {
					return nil, fmt.Errorf("failed to encode variant: %w", err)
				}
				cfg, err := parseConfigBytes(data)
				if err != nil {
					return nil, err
				}
				rateLabel := buildRateLabel(group.InsertEventRate, group.SearchEventRate)
				variantLabel := rateLabel
				if sweepLabel := buildVariantLabel(params, combo); sweepLabel != "" {
					if variantLabel != "" {
						variantLabel += "__"
					}
					variantLabel += sweepLabel
				}
				cfg.VariantName = variantLabel
				configs = append(configs, cfg)
			}
			continue
		}

		materialized := materializeSweep(raw, params, combo)
		data, err := yaml.Marshal(materialized)
		if err != nil {
			return nil, fmt.Errorf("failed to encode variant: %w", err)
		}
		cfg, err := parseConfigBytes(data)
		if err != nil {
			return nil, err
		}
		variantLabel := buildVariantLabel(params, combo)
		cfg.VariantName = variantLabel
		configs = append(configs, cfg)
	}

	return configs, nil
}

func parseConfigBytes(buf []byte) (*Config, error) {
	config := &Config{}
	err := yaml.Unmarshal(buf, config)
	if err != nil {
		return nil, fmt.Errorf("error parsing config file: %v", err)
	}

	if config.Data.IncrQueryPath == "" {
		if config.Data.QueryPath != "" {
			config.Data.IncrQueryPath = config.Data.QueryPath
		} else if config.Data.OverallQueryPath != "" {
			config.Data.IncrQueryPath = config.Data.OverallQueryPath
		}
	}
	if config.Data.OverallQueryPath == "" {
		if config.Data.QueryPath != "" {
			config.Data.OverallQueryPath = config.Data.QueryPath
		} else if config.Data.IncrQueryPath != "" {
			config.Data.OverallQueryPath = config.Data.IncrQueryPath
		}
	}

	if config.Data.IncrQueryPath == "" {
		return nil, fmt.Errorf("incr_query_path (or query_path/overall_query_path) must be provided")
	}
	if config.Data.OverallQueryPath == "" && config.Data.OverallGtPath != "" {
		return nil, fmt.Errorf("overall_query_path (or query_path/incr_query_path) must be provided when overall recall is enabled")
	}

	if config.Data.OverallQueryPath == "" {
		config.Data.OverallQueryPath = config.Data.IncrQueryPath
	}

	if config.Workload.BatchSize <= 0 {
		return nil, fmt.Errorf("workload.batch_size must be set and greater than 0")
	}

	if config.Workload.QueryMode == "" {
		config.Workload.QueryMode = queryModeRoundRobin
	}

	if err := validateConfig(config); err != nil {
		return nil, fmt.Errorf("config validation failed: %v", err)
	}

	return config, nil
}

type sweepParam struct {
	Path   []string
	Key    string
	Values []interface{}
}

func collectSweepParams(node interface{}, path []string) []sweepParam {
	var params []sweepParam
	collectSweep(node, path, &params)
	return params
}

func collectSweep(node interface{}, path []string, params *[]sweepParam) {
	switch val := node.(type) {
	case map[interface{}]interface{}:
		keys := make([]string, 0, len(val))
		keyMap := make(map[string]interface{}, len(val))
		for k, v := range val {
			keyStr := fmt.Sprintf("%v", k)
			keys = append(keys, keyStr)
			keyMap[keyStr] = v
		}
		sort.Strings(keys)
		for _, key := range keys {
			collectSweep(keyMap[key], append(path, key), params)
		}
	case []interface{}:
		if len(val) == 0 {
			return
		}
		if isScalarList(val) {
			key := strings.Join(path, ".")
			*params = append(*params, sweepParam{
				Path:   append([]string(nil), path...),
				Key:    key,
				Values: val,
			})
		}
	}
}

func hasSweep(params []sweepParam) bool {
	for _, p := range params {
		if len(p.Values) > 1 {
			return true
		}
	}
	return false
}

func isScalarList(values []interface{}) bool {
	for _, v := range values {
		switch v.(type) {
		case map[interface{}]interface{}, []interface{}:
			return false
		}
	}
	return true
}

func enumerateSweepCombos(params []sweepParam) [][]int {
	if len(params) == 0 {
		return nil
	}
	var combos [][]int
	var dfs func(idx int, current []int)
	dfs = func(idx int, current []int) {
		if idx == len(params) {
			combination := append([]int(nil), current...)
			combos = append(combos, combination)
			return
		}
		for i := range params[idx].Values {
			current = append(current, i)
			dfs(idx+1, current)
			current = current[:len(current)-1]
		}
	}
	dfs(0, make([]int, 0, len(params)))
	return combos
}

func materializeSweep(node interface{}, params []sweepParam, combo []int) interface{} {
	selection := make(map[string]int, len(params))
	for i, p := range params {
		selection[p.Key] = combo[i]
	}
	return applySelection(node, selection, nil)
}

func ensureWorkloadMap(node interface{}) map[interface{}]interface{} {
	root, ok := node.(map[interface{}]interface{})
	if !ok {
		return nil
	}
	workload, ok := root["workload"].(map[interface{}]interface{})
	if !ok {
		workload = make(map[interface{}]interface{})
		root["workload"] = workload
	}
	return workload
}

func toFloat64(value interface{}) float64 {
	switch v := value.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case uint64:
		return float64(v)
	case string:
		f, _ := strconv.ParseFloat(v, 64)
		return f
	}
	return 0
}

func extractRateGroups(node interface{}) []rateGroup {
	root, ok := node.(map[interface{}]interface{})
	if !ok {
		return nil
	}
	workload, ok := root["workload"].(map[interface{}]interface{})
	if !ok {
		return nil
	}

	rawGroups, _ := workload["rate_groups(r/w)"].([]interface{})
	if len(rawGroups) == 0 {
		return nil
	}

	var groups []rateGroup
	for _, entry := range rawGroups {
		switch val := entry.(type) {
		case map[interface{}]interface{}:
			group := rateGroup{}
			if v, exists := val["insert"]; exists {
				group.InsertEventRate = toFloat64(v)
			} else if v, exists := val["write"]; exists {
				group.InsertEventRate = toFloat64(v)
			} else if v, exists := val["w"]; exists {
				group.InsertEventRate = toFloat64(v)
			}
			if v, exists := val["search"]; exists {
				group.SearchEventRate = toFloat64(v)
			} else if v, exists := val["read"]; exists {
				group.SearchEventRate = toFloat64(v)
			} else if v, exists := val["r"]; exists {
				group.SearchEventRate = toFloat64(v)
			}
			groups = append(groups, group)
		case []interface{}:
			if len(val) >= 2 {
				group := rateGroup{
					InsertEventRate: toFloat64(val[1]),
					SearchEventRate: toFloat64(val[0]),
				}
				groups = append(groups, group)
			}
		}
	}
	delete(workload, "rate_groups(r/w)")
	delete(workload, "rate_groups")
	delete(workload, "insert_event_rate")
	delete(workload, "search_event_rate")
	if len(groups) == 0 {
		return nil
	}
	return groups
}

func applySelection(node interface{}, selection map[string]int, path []string) interface{} {
	switch val := node.(type) {
	case map[interface{}]interface{}:
		out := make(map[interface{}]interface{}, len(val))
		for k, v := range val {
			keyStr := fmt.Sprintf("%v", k)
			out[k] = applySelection(v, selection, append(path, keyStr))
		}
		return out
	case []interface{}:
		key := strings.Join(path, ".")
		if idx, ok := selection[key]; ok {
			if idx >= 0 && idx < len(val) {
				return applySelection(val[idx], selection, path)
			}
			return nil
		}
		cloned := make([]interface{}, len(val))
		for i, elem := range val {
			cloned[i] = applySelection(elem, selection, append(path, fmt.Sprintf("%d", i)))
		}
		return cloned
	default:
		return val
	}
}

func buildRateLabel(insertRate, searchRate float64) string {
	return fmt.Sprintf("w%s-r%s", formatVariantValue(insertRate), formatVariantValue(searchRate))
}

func formatVariantValue(v interface{}) string {
	value := fmt.Sprintf("%v", v)
	value = strings.ReplaceAll(value, "/", "-")
	value = strings.ReplaceAll(value, string(os.PathSeparator), "-")
	value = strings.ReplaceAll(value, " ", "")
	value = strings.ReplaceAll(value, ":", "_")
	value = strings.ReplaceAll(value, ".", "p")
	return value
}

func buildVariantLabel(params []sweepParam, combo []int) string {
	if len(params) == 0 || len(combo) == 0 {
		return ""
	}
	var parts []string
	for i, p := range params {
		if len(p.Values) <= 1 {
			continue
		}
		if i >= len(combo) {
			continue
		}
		val := p.Values[combo[i]]
		key := sanitizeParamKey(p.Key)
		if key == "compare_enabled" {
			continue
		}
		if key == "use_node_lock" || key == "enable_mvcc" {
			parts = append(parts, fmt.Sprintf("%s-%s", key, formatVariantValue(val)))
		} else {
			parts = append(parts, fmt.Sprintf("%s%s", key, formatVariantValue(val)))
		}
	}
	return strings.Join(parts, "__")
}

func sanitizeParamKey(k string) string {
	k = strings.ReplaceAll(k, ".", "_")
	k = strings.ReplaceAll(k, "/", "_")
	k = strings.TrimPrefix(k, "index_")
	k = strings.TrimPrefix(k, "search_")
	k = strings.TrimPrefix(k, "workload_")
	return k
}

func validateConfig(config *Config) error {
	if config.Workload.BatchSize <= 0 {
		return fmt.Errorf("batch_size must be set and greater than 0")
	}
	if config.Workload.BatchSize%10 != 0 {
		return fmt.Errorf("batch_size (%d) must be a multiple of 10", config.Workload.BatchSize)
	}

	if config.Workload.InsertEventRate < 0 {
		return fmt.Errorf("insert_event_rate cannot be negative")
	}
	if config.Workload.SearchEventRate < 0 {
		return fmt.Errorf("search_event_rate cannot be negative")
	}

	switch config.Workload.QueryMode {
	case queryModeRoundRobin, queryModeChasing, queryModePeeking, queryModeZipfian:
	default:
		return fmt.Errorf("invalid workload.query_mode %s (allowed: %s, %s, %s, %s)",
			config.Workload.QueryMode, queryModeRoundRobin, queryModeChasing, queryModePeeking, queryModeZipfian)
	}

	return nil
}

func checkRecallConfig(config *Config) {
	hasOverallGt := config.Data.OverallGtPath != ""
	hasOverallTool := defaultOverallRecallTool != ""
	if hasOverallGt && hasOverallTool {
		log.Printf("Overall recall: ENABLED")
	} else {
		log.Printf("Overall recall: DISABLED (missing overall_gt_path or recall tool)")
	}

	hasIncrGt := config.Data.IncrGtPath != ""
	hasIncrTool := defaultIncrRecallTool != ""
	if hasIncrGt && hasIncrTool {
		log.Printf("Incremental recall: ENABLED")
	} else {
		log.Printf("Incremental recall: DISABLED (missing incr_gt_path or recall tool)")
	}
}

func (c *Config) ShouldSimulate() bool {
	if !c.Compare.Enabled {
		return false
	}
	if c.Compare.Simulate == nil {
		return false
	}
	return c.Compare.Simulate.Enabled
}

func (c *Config) StageQueryWindow() int {
	if c.Workload.BatchSize > 0 {
		return c.Workload.BatchSize
	}
	return 0
}

func (c *Config) QueryModeSuffix() string {
	switch c.Workload.QueryMode {
	case queryModeRoundRobin:
		return "_rr"
	case queryModeChasing:
		return "_ch"
	case queryModePeeking:
		return "_pk"
	case queryModeZipfian:
		return "_zf"
	default:
		return ""
	}
}

func (c *Config) InsertRate() float64 {
	return c.Workload.InsertEventRate
}

func (c *Config) SearchRate() float64 {
	return c.Workload.SearchEventRate
}
