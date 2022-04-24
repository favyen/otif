package main

import (
	"./lib"
	"fmt"
	"os"
	"time"
)

type Configs struct {
	Detector lib.DetectorConfig
	Segmentation lib.SegmentationConfig
	Tracker lib.TrackerConfig
}

func main() {
	dataRoot := os.Args[1]
	label := os.Args[2]
	cfg := lib.GetConfig(dataRoot, label)

	mode := os.Args[3]
	prevDetector := lib.ParseDetectorConfig(cfg, os.Args[4])
	prevSegmentation := lib.ParseSegmentationConfig(os.Args[5])
	prevTracker := lib.ParseTrackerConfig(os.Args[6])
	iterations := lib.ParseInt(os.Args[7])
	logfile := os.Args[8]

	configs := Configs{
		Detector: prevDetector,
		Segmentation: prevSegmentation,
		Tracker: prevTracker,
	}
	opts := lib.GetExecOptions(mode, false)

	var tradeoffFuncs []func(lib.Config, Configs) *Configs

	// detector tradeoff
	if mode == "ours" || mode == "ours-simple" || mode == "chameleon-good" || mode == "chameleon-bad" || mode == "naive" || mode == "ours1" || mode == "ours2" || mode == "ours3" || mode == "ours-segment" {
		tradeoffFuncs = append(tradeoffFuncs, func(cfg lib.Config, configs Configs) *Configs {
			newDetector := lib.DetectorTradeoff(dataRoot, cfg, configs.Detector)
			if newDetector == nil {
				return nil
			}
			return &Configs{*newDetector, configs.Segmentation, configs.Tracker}
		})
	}

	// segmentation tradeoff
	if mode == "ours" || mode == "ours-simple" || mode == "blazeit" || mode == "catdet" || mode == "ours-segment" {
		tradeoffFuncs = append(tradeoffFuncs, func(cfg lib.Config, config Configs) *Configs {
			newSegmentation := lib.SegmentationTradeoff(dataRoot, opts.GetSegmentMode(), cfg, configs.Segmentation, configs.Detector)
			if newSegmentation == nil {
				return nil
			}
			return &Configs{configs.Detector, *newSegmentation, configs.Tracker}
		})
	}

	// tracker tradeoff
	if mode == "ours" {
		tradeoffFuncs = append(tradeoffFuncs, func(cfg lib.Config, config Configs) *Configs {
			newTracker := lib.TrackerTradeoff(dataRoot, cfg, configs.Tracker)
			if newTracker == nil {
				return nil
			}
			return &Configs{configs.Detector, configs.Segmentation, *newTracker}
		})
	} else if mode == "ours-simple" || mode == "chameleon-good" || mode == "chameleon-bad" || mode == "miris" || mode == "ours2" || mode == "ours3" || mode == "ours-tracker" {
		tradeoffFuncs = append(tradeoffFuncs, func(cfg lib.Config, config Configs) *Configs {
			newTracker := lib.TrackerTradeoffSimple(cfg, configs.Tracker)
			if newTracker == nil {
				return nil
			}
			return &Configs{configs.Detector, configs.Segmentation, *newTracker}
		})
	}

	for i := 0; i < iterations; i++ {
		var bestConfigs Configs
		var bestTime int
		var bestAcc float64 = -1
		var actualAcc float64

		for _, f := range tradeoffFuncs {
			tryConfigs := f(cfg, configs)
			if tryConfigs == nil {
				continue
			}
			t, acc, _ := lib.Exec2(dataRoot, cfg, "valid", "/tmp/y1/", tryConfigs.Detector, tryConfigs.Segmentation, tryConfigs.Tracker, opts)
			time.Sleep(5*time.Second)
			fmt.Printf("[%v]: time=%d accuracy=%v\n", *tryConfigs, t, acc)
			if acc > bestAcc {
				bestConfigs = *tryConfigs
				bestTime = t
				bestAcc = acc

				// bestAcc may be from different eval function (for chameleon-bad)
				// so here we eval again with actual accuracy
				actualAcc, _ = lib.Eval(dataRoot, cfg, "valid", "/tmp/y1/")
			}
		}
		if bestAcc == -1 {
			break
		}

		configs = bestConfigs
		logStr := fmt.Sprintf("%v\t%v\t%v\t%v\t%v\t%v", configs.Detector, configs.Segmentation, configs.Tracker, bestTime, bestAcc, actualAcc)
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
