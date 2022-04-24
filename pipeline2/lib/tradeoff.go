package lib

// these functions take a configuration and spit out next configuration to try

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func readSpeedFile(fname string) int {
	bytes, err := ioutil.ReadFile(fname)
	if err != nil {
		panic(err)
	}
	return ParseInt(strings.TrimSpace(string(bytes)))
}

func DetectorTradeoff(dataRoot string, cfg Config, prevDetectorCfg DetectorConfig) *DetectorConfig {
	validPath := filepath.Join(dataRoot, "dataset", cfg.Label, "valid")
	curDetectorTime := readSpeedFile(filepath.Join(validPath, prevDetectorCfg.Dir(), "speed.txt"))
	targetTime := curDetectorTime // original (just next faster detector)
	//targetTime := curDetectorTime*(10-readSpeedFile("/tmp/experiment.txt"))/10 // experiment
	var nextDetector DetectorConfig
	nextDetectorTime := -1
	files, err := ioutil.ReadDir(validPath)
	if err != nil {
		panic(err)
	}
	// pick the slowest detector that is faster (less runtime) than target time
	for _, fi := range files {
		speedFname := filepath.Join(validPath, fi.Name(), "speed.txt")
		if _, err := os.Stat(speedFname); err != nil {
			continue
		}
		t := readSpeedFile(speedFname)
		if t >= targetTime {
			continue
		}
		if nextDetectorTime < 0 || t > nextDetectorTime {
			nextDetector = ParseDetectorConfig(cfg, fi.Name())
			nextDetectorTime = t
		}
	}
	if nextDetectorTime < 0 {
		return nil
	}
	return &nextDetector
}

func SegmentationTradeoff(dataRoot string, mode string, cfg Config, prev SegmentationConfig, detectorCfg DetectorConfig) *SegmentationConfig {
	var cmd *exec.Cmd
	rootPath := filepath.Join(dataRoot, "dataset", cfg.Label)
	if mode == "blazeit" {
		cmd = exec.Command(
			"python", "../blazeit/iter_param.py", rootPath,
			fmt.Sprintf("%v", prev.Threshold), detectorCfg.String(),
		)
	} else if mode == "default" {
		cmd = exec.Command(
			"python", "../model/iter_param.py", rootPath,
			prev.Dir(), fmt.Sprintf("%v", prev.Threshold), detectorCfg.String(),
		)
	}
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(string(output))
		panic(err)
	}
	for _, line := range strings.Split(string(output), "\n") {
		if !strings.HasPrefix(line, "iter") {
			continue
		}
		cfg := ParseSegmentationConfig(line[4:])
		return &cfg
	}
	return nil
}

func TrackerTradeoff(dataRoot string, cfg Config, prev TrackerConfig) *TrackerConfig {
	var profiles []TrackerProfile
	ReadJsonFile(filepath.Join(dataRoot, "dataset", cfg.Label, "train/tracks/profile.json"), &profiles)
	var nextProfile *TrackerProfile
	for _, profile := range profiles {
		if profile.NumFrames >= prev.Profile.NumFrames*2/3 {
			continue
		} else if nextProfile != nil && profile.NumFrames < nextProfile.NumFrames {
			continue
		}
		nextProfile = new(TrackerProfile)
		*nextProfile = profile
	}
	if nextProfile == nil {
		return nil
	}
	return &TrackerConfig{
		Profile: *nextProfile,
	}
}

// used for chameleon
func TrackerTradeoffSimple(cfg Config, prev TrackerConfig) *TrackerConfig {
	profile := prev.Profile
	var nthresholds []float64
	var lastZero int
	for i, t := range profile.Thresholds {
		nthresholds = append(nthresholds, t)
		if t == 0 {
			lastZero = i
		}
	}
	if lastZero == len(nthresholds)-1 {
		return nil
	}
	nthresholds[lastZero+1] = 0
	return &TrackerConfig{
		Profile: TrackerProfile{Thresholds: nthresholds},
	}
}
