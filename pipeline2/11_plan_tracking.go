package main

import (
	"./lib"

	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"
)

func main() {
	label := os.Args[1]
	cfg := lib.GetConfig(label)
	detector := lib.ParseDetectorConfig(cfg, os.Args[2])

	videoPath := cfg.RootPath + "train/video/"
	detectPath := cfg.RootPath + "train/" + detector.Dir() + "/"
	trackPath := cfg.RootPath + "train/tracks/"
	tracker := lib.NewTracker("rnn", cfg)
	defer tracker.Close()

	// apply full framerate tracker on the detections from best detector
	files, err := ioutil.ReadDir(videoPath)
	if err != nil {
		panic(err)
	}
	for _, fi := range files {
		if !strings.HasSuffix(fi.Name(), ".mp4") {
			continue
		}
		id := lib.ParseInt(strings.Split(fi.Name(), ".mp4")[0])

		var detections [][]lib.Detection
		lib.ReadJsonFile(detectPath + fmt.Sprintf("%d.json", id), &detections)
		detections = lib.FilterDetectionsByClass(detections, cfg.Classes)
		//lib.RescaleDetections(detections, cfg.OrigDims, detector.Dims)

		vreader := lib.ReadFfmpeg(videoPath + fi.Name(), detector.Dims[0], detector.Dims[1])
		var outDetections [][]lib.Detection
		for frameIdx := 0; ; frameIdx++ {
			im, err := vreader.Read()
			if err == io.EOF {
				break
			} else if err != nil {
				panic(err)
			}
			dlist, _ := tracker.Infer(id, frameIdx, im, detections[frameIdx])
			outDetections = append(outDetections, dlist)
		}
		vreader.Close()
		tracker.End(id)
		//lib.RescaleDetections(outDetections, detector.Dims, cfg.OrigDims)
		bytes := lib.JsonMarshal(outDetections)
		if err := ioutil.WriteFile(trackPath + fmt.Sprintf("%d.json", id), bytes, 0644); err != nil {
			panic(err)
		}
	}

	// get the profiling outputs
	profiles := tracker.GetThresholds(videoPath, trackPath, cfg.OrigDims, detector.Dims, cfg.Freqs)
	var profileList []lib.TrackerProfile
	for _, profile := range profiles {
		profileList = append(profileList, profile)
	}
	bytes := lib.JsonMarshal(profileList)
	if err := ioutil.WriteFile(trackPath + "profile.json", bytes, 0644); err != nil {
		panic(err)
	}
}
