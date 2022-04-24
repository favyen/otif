package lib

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
)

func ParseDims(dims string) [2]int {
	parts := strings.Split(dims, "x")
	if len(parts) != 2 {
		panic(fmt.Errorf("bad dims %v", dims))
	}
	return [2]int{
		ParseInt(parts[0]),
		ParseInt(parts[1]),
	}
}

type Config struct {
	Label string
	Classes map[string]bool
	OrigDims [2]int
	DetectorDims [][2]int // assuming parameters are loaded from OrigDims; otherwise need to re-scale accordingly
	Freqs []int // freqs to use for RNN
}

func GetConfig(dataRoot string, label string) Config {
	var cfg Config
	ReadJsonFile(filepath.Join(dataRoot, "dataset", label, "cfg.json"), &cfg)
	return cfg
}

type DetectorConfig struct {
	Name string
	Dims [2]int
	Sizes [][2]int
	Threshold float64
}

func ParseDetectorConfig(mainCfg Config, s string) DetectorConfig {
	var cfg DetectorConfig
	if strings.HasPrefix(s, "{") {
		JsonUnmarshal([]byte(s), &cfg)
		return cfg
	}
	parts := strings.Split(s, "-")
	cfg.Name = parts[0]
	cfg.Dims = ParseDims(parts[1])
	cfg.Threshold = 0.25
	cfg.Sizes = [][2]int{
		cfg.Dims,
	}
	for _, dims := range mainCfg.DetectorDims {
		x := dims[0] * cfg.Dims[0] / mainCfg.OrigDims[0]
		x = ((x+31) / 32) * 32
		y := dims[1] * cfg.Dims[1] / mainCfg.OrigDims[1]
		y = ((y+31) / 32) * 32
		cfg.Sizes = append(cfg.Sizes, [2]int{x, y})
	}
	return cfg
}

func (cfg DetectorConfig) Dir() string {
	return fmt.Sprintf("%s-%dx%d", cfg.Name, cfg.Dims[0], cfg.Dims[1])
}

func (cfg DetectorConfig) String() string {
	return string(JsonMarshal(cfg))
}

type SegmentationConfig struct {
	Dims [2]int
	Threshold float64
}

func ParseSegmentationConfig(s string) SegmentationConfig {
	var cfg SegmentationConfig
	parts := strings.Split(s, "_")
	cfg.Dims[0] = ParseInt(parts[0])
	cfg.Dims[1] = ParseInt(parts[1])
	if len(parts) >= 3 {
		cfg.Threshold, _ = strconv.ParseFloat(parts[2], 64)
	}
	return cfg
}

func (cfg SegmentationConfig) Dir() string {
	return fmt.Sprintf("%d_%d", cfg.Dims[0], cfg.Dims[1])
}

func (cfg SegmentationConfig) String() string {
	return fmt.Sprintf("%d_%d_%v", cfg.Dims[0], cfg.Dims[1], cfg.Threshold)
}

type TrackerConfig struct {
	Profile TrackerProfile
}

func ParseTrackerConfig(s string) TrackerConfig {
	var profile TrackerProfile
	JsonUnmarshal([]byte(s), &profile)
	return TrackerConfig{
		Profile: profile,
	}
}

func (cfg TrackerConfig) String() string {
	return string(JsonMarshal(cfg.Profile))
}
