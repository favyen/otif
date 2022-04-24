package main

import (
	"./lib"
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

type Configs struct {
	Detector lib.DetectorConfig
	Segmentation lib.SegmentationConfig
	Tracker lib.TrackerConfig
}

func main() {
	dataRoot := os.Args[1]
	label := os.Args[2]
	mode := os.Args[3]
	validfile := os.Args[4]
	logfile := os.Args[5]

	cfg := lib.GetConfig(dataRoot, label)
	opts := lib.GetExecOptions(mode, true)

	bytes, err := ioutil.ReadFile(validfile)
	if err != nil {
		panic(err)
	}
	for i, line := range strings.Split(string(bytes), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, "\t")
		detectCfg := lib.ParseDetectorConfig(cfg, parts[0])
		segmentCfg := lib.ParseSegmentationConfig(parts[1])
		trackerCfg := lib.ParseTrackerConfig(parts[2])
		validAcc, _ := strconv.ParseFloat(parts[5], 64)

		// create directory where output tracks will be stored
		outDir := fmt.Sprintf("./outputs/%s/%s/%d/", label, mode, i)
		os.MkdirAll(outDir, 0755)

		t, acc, stderr := lib.Exec2(dataRoot, cfg, "test", outDir, detectCfg, segmentCfg, trackerCfg, opts)

		logStr := fmt.Sprintf("%v\t%v\t%v\t%v\t%v\t%v\t%v", detectCfg, segmentCfg, trackerCfg, t, validAcc, acc, stderr)
		fmt.Println(logStr)
		file, err := os.OpenFile(logfile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			panic(err)
		}
		if _, err := file.Write([]byte(logStr+"\n")); err != nil {
			panic(err)
		}
		file.Close()
	}
}
