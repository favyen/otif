package main

import (
	"./lib"

	"io"
	"io/ioutil"
	"os"
	"strings"
	"sync"
)

func simpleTracker(tracker *lib.Tracker, detections [][]lib.Detection, vreader *lib.FfmpegReader) [][]lib.Detection {
	var outDetections [][]lib.Detection
	for frameIdx := 0; ; frameIdx++ {
		im, err := vreader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
		dlist, _ := tracker.Infer(0, frameIdx, im, detections[frameIdx])
		outDetections = append(outDetections, dlist)
	}
	vreader.Close()
	tracker.End(0)
	return outDetections
}

func actualTracker(dataRoot string, cfg lib.Config, trackerMode string, detections [][]lib.Detection, vreader *lib.FfmpegReader, profile lib.TrackerProfile) [][]lib.Detection {
	defer vreader.Close()

	tracker := lib.NewTracker(dataRoot, trackerMode, cfg)
	defer tracker.End(0)

	// keep track of whether we hit EOF since loop below may
	// call nextIm multiple times after EOF
	eof := false
	nextIm := func() (lib.Image, bool) {
		if eof {
			return lib.Image{}, true
		}
		im, err := vreader.Read()
		if err == io.EOF {
			eof = true
			return lib.Image{}, true
		} else if err != nil {
			panic(err)
		}
		return im, false
	}

	// initialize all reasonable skips
	// don't use a skip if corresponding threshold is too high
	var skips []int
	x := 1
	for _, threshold := range profile.Thresholds {
		if threshold > 0.9 {
			break
		}
		skips = append(skips, x)
		x *= 2
	}

	outDetections := make([][]lib.Detection, len(detections))
	im, done := nextIm()
	if done {
		return outDetections
	}
	outDetections[0], _ = tracker.Infer(0, 0, im, detections[0])

	lastFrame := 0
	nextRead := 1 // equal to # frames read so far
	frameBuffer := make(map[int]lib.Image)

	for {
		updated := false
		// iterate through skips until we get high enough confidence on some frame
		for i := len(skips)-1; i >= 0; i-- {
			// read frames if needed
			curFrame := lastFrame + skips[i]
			for nextRead <= curFrame {
				im, done := nextIm()
				if done {
					break
				}
				frameBuffer[nextRead] = im
				nextRead++
			}
			// quit if current skip goes beyond the end of the file
			if nextRead <= curFrame {
				continue
			}
			dlist, conf := tracker.Infer(0, curFrame, frameBuffer[curFrame], detections[curFrame])
			if i != 0 && conf < profile.Thresholds[i] {
				continue
			}
			// update our state
			outDetections[curFrame] = dlist
			for frameIdx := lastFrame+1; frameIdx < curFrame; frameIdx++ {
				delete(frameBuffer, frameIdx)
			}
			lastFrame = curFrame
			updated = true
			break
		}
		if !updated {
			break
		}
	}

	return outDetections
}

func main() {
	dataRoot := os.Args[1]
	label := os.Args[2]
	videoPath := os.Args[3]
	jsonPath := os.Args[4]
	outPath := os.Args[5]
	trackerMode := os.Args[6]
	classList := strings.Split(os.Args[7], ",")
	var profile lib.TrackerProfile
	lib.JsonUnmarshal([]byte(os.Args[8]), &profile)

	var fnames []string
	if len(os.Args) >= 10 {
		fnames = []string{os.Args[9]}
	} else {
		files, err := ioutil.ReadDir(videoPath)
		if err != nil {
			panic(err)
		}
		for _, fi := range files {
			if !strings.HasSuffix(fi.Name(), ".mp4") {
				continue
			}
			fnames = append(fnames, fi.Name())
		}
	}

	cfg := lib.GetConfig(dataRoot, label)

	// use specific class list instead of cfg since this script can be used to
	// track all objects instead of just specific class
	classSet := make(map[string]bool)
	for _, class := range classList {
		class = strings.TrimSpace(class)
		if class == "" {
			continue
		}
		classSet[class] = true
	}

	nthreads := 16
	ch := make(chan string)
	var wg sync.WaitGroup

	for i := 0; i < nthreads; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for fname := range ch {
				label := strings.Split(fname, ".mp4")[0]

				var detections [][]lib.Detection
				lib.ReadJsonFile(jsonPath+label+".json", &detections)
				if len(classSet) > 0 {
					detections = lib.FilterDetectionsByClass(detections, classSet)
				}
				vreader := lib.ReadFfmpeg(videoPath+label+".mp4", 640, 352)
				outDetections := actualTracker(dataRoot, cfg, trackerMode, detections, vreader, profile)
				bytes := lib.JsonMarshal(outDetections)
				if err := ioutil.WriteFile(outPath+label+".json", bytes, 0644); err != nil {
					panic(err)
				}
			}
		}()
	}
	for _, fname := range fnames {
		ch <- fname
	}
	close(ch)
	wg.Wait()
}
