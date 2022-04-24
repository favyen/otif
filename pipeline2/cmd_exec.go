package main

import (
	"./lib"
	"fmt"
	"os"
)

func main() {
	dataRoot := os.Args[1]
	label := os.Args[2]
	cfg := lib.GetConfig(dataRoot, label)

	mode := os.Args[3]
	trainval := os.Args[4]
	outPath := os.Args[5]
	detectorCfg := lib.ParseDetectorConfig(cfg, os.Args[6])
	segmentationCfg := lib.ParseSegmentationConfig(os.Args[7])
	trackerCfg := lib.ParseTrackerConfig(os.Args[8])

	opts := lib.GetExecOptions(mode, trainval == "test")
	msPerSample, acc, stderr := lib.Exec2(dataRoot, cfg, trainval, outPath, detectorCfg, segmentationCfg, trackerCfg, opts)
	fmt.Printf("sample runtime %d ms; accuracy=%v, stderr=%v\n", msPerSample, acc, stderr)
}
