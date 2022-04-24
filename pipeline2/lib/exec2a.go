package lib

import (
	"fmt"
	"encoding/json"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type ExecOptions struct {
	TrackerMode string
	SegmentMode string
	Refinement bool
	EvalFunc func(dataRoot string, cfg Config, validtest string, path string) (float64, float64)
}

func (opts ExecOptions) GetTrackerMode() string {
	if opts.TrackerMode == "" {
		return "rnn"
	}
	return opts.TrackerMode
}

func (opts ExecOptions) GetSegmentMode() string {
	if opts.SegmentMode == "" {
		return "default"
	}
	return opts.SegmentMode
}

func (opts ExecOptions) GetEvalFunc() func(string, Config, string, string) (float64, float64) {
	if opts.EvalFunc == nil {
		return Eval
	}
	return opts.EvalFunc
}

func GetExecOptions(mode string, istest bool) ExecOptions {
	var opts ExecOptions
	if mode == "chameleon-good" || mode == "chameleon-bad" || mode == "blazeit" || mode == "naive" || mode == "ours1" || mode == "ours2" || mode == "catdet" || mode == "ours-segment" {
		opts.TrackerMode = "iou"
	}
	if mode == "miris" {
		opts.TrackerMode = "miris"
	}
	if mode == "blazeit" {
		opts.SegmentMode = "blazeit"
	}
	if mode == "ours" || mode == "ours-simple" {
		opts.Refinement = true
	}

	if mode == "chameleon-bad" && !istest {
		opts.EvalFunc = EvalDetection
	}

	return opts
}

/*var segTime time.Duration
var detectTime time.Duration
var decodeTime time.Duration
var trackTime time.Duration
var timeMu sync.Mutex*/

func Exec2(dataRoot string, cfg Config, trainval string, outPath string, detectorCfg DetectorConfig, segmentationCfg SegmentationConfig, trackerCfg TrackerConfig, opts ExecOptions) (int, float64, float64) {
	fmt.Printf("exec config (%v, %v, %v) to %v\n", detectorCfg, segmentationCfg, trackerCfg, outPath)

	os.MkdirAll(outPath, 0755)

	// get clusters from training tracks if needed for refinement
	var clusters []Cluster
	if opts.Refinement {
		fmt.Printf("load clusters for refinement...\n")
		var trainSet [][]TrackDetection

		trainDir := filepath.Join(dataRoot, "dataset", cfg.Label, "tracker/tracks/")
		files, err := ioutil.ReadDir(trainDir)
		if err != nil {
			panic(err)
		}
		for _, fi := range files {
			if !strings.HasSuffix(fi.Name(), ".json") {
				continue
			}
			bytes, err := ioutil.ReadFile(filepath.Join(trainDir, fi.Name()))
			if err != nil {
				panic(err)
			}
			var detections [][]Detection
			if err := json.Unmarshal(bytes, &detections); err != nil {
				panic(err)
			}
			detections = FilterDetectionsByClass(detections, cfg.Classes)
			tracks := GetTracks(detections)
			trainSet = append(trainSet, tracks...)
		}

		clusters, _ = ClusterTracks(trainSet)
		fmt.Printf("... done (%d train tracks, %d clusters)\n", len(trainSet), len(clusters))
	}

	t0 := time.Now()
	any64 := false // reset t0 when the first thread gets to frame 64 or higher
	var durationSamples []int

	videoPath := filepath.Join(dataRoot, "dataset", cfg.Label, trainval, "video/")
	fnames, err := ioutil.ReadDir(videoPath)
	if err != nil {
		panic(err)
	}
	var videoFnames []string
	for _, fi := range fnames {
		if !strings.HasSuffix(fi.Name(), ".mp4") {
			continue
		}
		videoFnames = append(videoFnames, fi.Name())
	}
	numTotal := len(videoFnames)

	var mu sync.Mutex
	respCond := sync.NewCond(&mu)
	type ModelMeta struct {
		Size [2]int
		Cond *sync.Cond
		BatchSize int
	}
	type PendingJob struct {
		// video ID and frame index
		ID int
		FrameIdx int
		// cropped window
		Image Image
		// for detect job: offset from topleft (to correct detection coordinates)
		Offset [2]int
		// for detect job: windows that should contain the detection
		Cells [][4]int
	}
	type DetectorOutput struct {
		// how many detector jobs are still needed
		Needed int
		// flat detection list
		Detections []Detection
	}
	// model size -> pending jobs
	pendingJobs := make(map[[2]int][]PendingJob)
	// (video ID, frame idx) -> priority
	priorities := make(map[[2]int]int)
	// model outputs
	detectOutputs := make(map[[2]int]*DetectorOutput)
	// number of videos that are waiting for some frame to be processed
	var numWaiting int = 0

	// initialize models
	done := false
	var modelWg sync.WaitGroup
	segSize := [2]int{-1, -1}
	var modelMetas []ModelMeta
	wakeupModels := func() {
		for _, meta := range modelMetas {
			if len(pendingJobs[meta.Size]) < meta.BatchSize && (numWaiting < numTotal || len(pendingJobs[meta.Size]) == 0) && !done {
				continue
			}
			meta.Cond.Broadcast()
		}
	}
	modelLoop := func(sz [2]int, batchSize int, f func([]PendingJob), cleanup func()) {
		cond := sync.NewCond(&mu)
		modelMetas = append(modelMetas, ModelMeta{
			Size: sz,
			Cond: cond,
			BatchSize: batchSize,
		})
		modelWg.Add(1)
		go func() {
			defer modelWg.Done()
			defer cleanup()
			for {
				mu.Lock()
				// wait for batch to arrive
				// but if all the jobs are blocked, then we need to process it immediately
				for len(pendingJobs[sz]) < batchSize && (numWaiting < numTotal || len(pendingJobs[sz]) == 0) && !done {
					cond.Wait()
				}
				if done {
					mu.Unlock()
					break
				}

				// pick the jobs with lowest (most) priority
				sort.Slice(pendingJobs[sz], func(i, j int) bool {
					job1 := pendingJobs[sz][i]
					job2 := pendingJobs[sz][j]
					prio1 := priorities[[2]int{job1.ID, job1.FrameIdx}]
					prio2 := priorities[[2]int{job2.ID, job2.FrameIdx}]
					return prio1 < prio2 || (prio1 == prio2 && job1.FrameIdx < job2.FrameIdx)
				})
				var jobs []PendingJob
				if len(pendingJobs[sz]) < batchSize {
					jobs = pendingJobs[sz]
					pendingJobs[sz] = nil
					fmt.Println(sz, "undersized job", len(jobs), "/", batchSize)
				} else {
					jobs = append([]PendingJob{}, pendingJobs[sz][0:batchSize]...)
					n := copy(pendingJobs[sz][0:], pendingJobs[sz][batchSize:])
					pendingJobs[sz] = pendingJobs[sz][0:n]
					fmt.Println(sz, "full job", len(jobs), "/", batchSize)
				}

				mu.Unlock()
				f(jobs)
			}
		}()
	}

	seg := NewSegmentationModel(dataRoot, opts.GetSegmentMode(), 32, cfg, segmentationCfg, detectorCfg)
	modelLoop(segSize, 32, func(jobs []PendingJob) {
		var indices []string
		for _, job := range jobs {
			indices = append(indices, fmt.Sprintf("%d/%d", job.ID, job.FrameIdx))
		}
		//fmt.Println("applying segmentation on jobs", indices)

		var images []Image
		for _, job := range jobs {
			images = append(images, job.Image)
		}

		//t0 := time.Now()
		windows := seg.GetWindows(images)
		/*timeMu.Lock()
		segTime += time.Now().Sub(t0)
		timeMu.Unlock()*/

		mu.Lock()
		for i, job := range jobs {
			detectOutputs[[2]int{job.ID, job.FrameIdx}] = &DetectorOutput{
				Needed: len(windows[i]),
			}
			for _, window := range windows[i] {
				crop := jobs[i].Image.Crop(window.Bounds[0], window.Bounds[1], window.Bounds[2], window.Bounds[3])
				sz := [2]int{window.Bounds[2]-window.Bounds[0], window.Bounds[3]-window.Bounds[1]}
				pendingJobs[sz] = append(pendingJobs[sz], PendingJob{
					ID: job.ID,
					FrameIdx: job.FrameIdx,
					Offset: [2]int{window.Bounds[0], window.Bounds[1]},
					Image: crop,
					Cells: window.Cells,
				})
			}
		}
		wakeupModels()
		respCond.Broadcast() // needed in case any jobs had no windows output
		mu.Unlock()
	}, func() {
		seg.Close()
	})

	for szIdx, sz := range detectorCfg.Sizes {
		batchSize := 4
		count := 1
		if szIdx == 0 {
			count = 2
		}

		for i := 0; i < count; i++ {
			// here, we set origDims=sz because we will handle re-scaling ourselves
			// since anyway we e.g. need to run 128x128 detector on a crop of the image
			yolo := NewYolov3(dataRoot, batchSize, sz, detectorCfg.Dims, sz, detectorCfg.Threshold, cfg.Classes, cfg.Label)
			modelLoop(sz, batchSize, func(jobs []PendingJob) {
				var indices []string
				for _, job := range jobs {
					indices = append(indices, fmt.Sprintf("%d/%d", job.ID, job.FrameIdx))
				}
				//fmt.Println(sz, "applying detector on jobs", indices)
				var images []Image
				for _, job := range jobs {
					images = append(images, job.Image)
				}

				//t0 := time.Now()
				outputs := yolo.Detect(images)
				/*timeMu.Lock()
				detectTime += time.Now().Sub(t0)
				timeMu.Unlock()*/

				mu.Lock()
				for i, job := range jobs {
					dlist := []Detection{}
					for _, d := range outputs[i] {
						// first, add the offset to get actual position in detector coordinates
						d.Left = d.Left+job.Offset[0]
						d.Top = d.Top+job.Offset[1]
						d.Right = d.Right+job.Offset[0]
						d.Bottom = d.Bottom+job.Offset[1]

						// if it doesn't match any cell, then skip it
						good := false
						cx, cy := (d.Left+d.Right)/2, (d.Top+d.Bottom)/2
						for _, cell := range job.Cells {
							if cx < cell[0] || cx >= cell[2] || cy < cell[1] || cy >= cell[3] {
								continue
							}
							good = true
							break
						}
						if !good {
							continue
						}

						// now change the coordinate system to the original resolution
						d.Left = d.Left*cfg.OrigDims[0]/detectorCfg.Dims[0]
						d.Top = d.Top*cfg.OrigDims[1]/detectorCfg.Dims[1]
						d.Right = d.Right*cfg.OrigDims[0]/detectorCfg.Dims[0]
						d.Bottom = d.Bottom*cfg.OrigDims[1]/detectorCfg.Dims[1]

						dlist = append(dlist, d)
					}

					output := detectOutputs[[2]int{job.ID, job.FrameIdx}]
					output.Detections = append(output.Detections, dlist...)
					output.Needed--
				}
				respCond.Broadcast()
				mu.Unlock()
			}, func() {
				yolo.Close()
			})
		}
	}

	// process the videos in parallel
	var wg sync.WaitGroup
	trackers := make([]*Tracker, 2)
	for i := range trackers {
		trackers[i] = NewTracker(dataRoot, opts.GetTrackerMode(), cfg)
	}
	for _, fname := range videoFnames {
		wg.Add(1)
		go func(fname string) {
			defer wg.Done()
			id := ParseInt(strings.Split(fname, ".mp4")[0])
			tracker := trackers[id%len(trackers)]

			// detections that were submitted to job queue
			submitted := make(map[int]bool)

			detectorFunc := func(frameIdx int, im Image, extras []int, extraImages []Image) []Detection {
				submit := func(idx int, im Image, priority int) {
					priorities[[2]int{id, idx}] = priority
					if submitted[idx] {
						return
					}
					pendingJobs[segSize] = append(pendingJobs[segSize], PendingJob{
						ID: id,
						FrameIdx: idx,
						Image: im,
					})
					submitted[idx] = true
				}

				mu.Lock()
				defer mu.Unlock()

				submit(frameIdx, im, 0)
				for i, idx := range extras {
					submit(idx, extraImages[i], 1+i)
				}

				numWaiting++
				wakeupModels()
				//fmt.Println(id, "waiting for frame output", frameIdx)

				k := [2]int{id, frameIdx}
				for detectOutputs[k] == nil || detectOutputs[k].Needed > 0 {
					respCond.Wait()
				}

				// reset t0 if we're the first thread to reach frameIdx>=64
				if frameIdx >= 64 && !any64 {
					t0 = time.Now()
					any64 = true
				}

				numWaiting--
				return detectOutputs[k].Detections
			}

			detections := execTrackerLoop2(id, tracker, filepath.Join(videoPath, fname), detectorCfg.Dims, trackerCfg.Profile, detectorFunc)

			mu.Lock()
			numWaiting++
			wakeupModels()
			durationSamples = append(durationSamples, int(time.Now().Sub(t0)/time.Millisecond))
			mu.Unlock()

			// refine using clusters computed earlier if desired
			if opts.Refinement && cfg.Label != "amsterdam" && cfg.Label != "jackson" {
				tracks := GetTracks(detections)
				tracks = Postprocess(clusters, tracks)
				detections = DetectionsFromTracks(tracks)
			}
			if cfg.Label == "amsterdam" || cfg.Label == "jackson" {
				goodTracks := GetGoodTracks("amsterdam", detections)[0]
				detections = DetectionsFromTracks(goodTracks)
			}

			// save all of the detections
			bytes := JsonMarshal(detections)
			outFname := fmt.Sprintf("%s/%d.json", outPath, id)
			if err := ioutil.WriteFile(outFname, bytes, 0644); err != nil {
				panic(err)
			}
		}(fname)
	}
	wg.Wait()

	// close models
	mu.Lock()
	done = true
	wakeupModels()
	mu.Unlock()
	modelWg.Wait()
	for _, tracker := range trackers {
		tracker.Close()
	}

	//fmt.Printf("seg=%v detect=%v track=%v decode=%v\n", segTime, detectTime, trackTime, decodeTime)

	var acc, stderr float64
	if trainval == "valid" || trainval == "test" {
		acc, stderr = opts.GetEvalFunc()(dataRoot, cfg, trainval, outPath)
	}
	//return int(time.Now().Sub(t0)/time.Millisecond)/len(videoFnames), acc
	var durationSum int
	for _, duration := range durationSamples {
		durationSum += duration
	}
	return durationSum/len(durationSamples)/len(videoFnames), acc, stderr
}

type BufferedFfmpegReader struct {
	mu sync.Mutex
	cond *sync.Cond
	buffer []Image
	offset int
	extras []Image
	done bool
}

func NewBufferedFfmpegReader(vreader *FfmpegReader, size int) *BufferedFfmpegReader {
	bfr := &BufferedFfmpegReader{}
	bfr.cond = sync.NewCond(&bfr.mu)
	for i := 0; i < size; i++ {
		bfr.extras = append(bfr.extras, NewImage(vreader.Width, vreader.Height))
	}

	go func() {
		bfr.mu.Lock()
		for {
			for len(bfr.extras) == 0 {
				bfr.cond.Wait()
			}
			im := bfr.extras[len(bfr.extras)-1]
			bfr.extras = bfr.extras[0:len(bfr.extras)-1]
			bfr.mu.Unlock()

			err := vreader.ReadInto(im)
			if err == io.EOF {
				bfr.mu.Lock()
				bfr.done = true
				bfr.cond.Broadcast()
				bfr.mu.Unlock()
				return
			} else if err != nil {
				panic(err)
			}

			bfr.mu.Lock()
			bfr.buffer = append(bfr.buffer, im)
			bfr.cond.Broadcast()
		}
	}()

	return bfr
}

// Returns (image, false), or if EOF then (..., true)
// Blocks until EOF or image is available.
func (bfr *BufferedFfmpegReader) GetFrame(frameIdx int) (Image, bool) {
	bfr.mu.Lock()
	defer bfr.mu.Unlock()

	for !bfr.done && bfr.offset+len(bfr.buffer) <= frameIdx {
		bfr.cond.Wait()
	}

	if frameIdx < bfr.offset+len(bfr.buffer) {
		return bfr.buffer[frameIdx-bfr.offset], false
	}

	return Image{}, true
}

// Discard frames below frameIdx
func (bfr *BufferedFfmpegReader) Discard(frameIdx int) {
	bfr.mu.Lock()
	defer bfr.mu.Unlock()

	if frameIdx <= bfr.offset {
		return
	}

	// first index in buffer that will NOT be discarded
	pos := frameIdx - bfr.offset

	discarded := bfr.buffer[0:pos]
	for _, im := range discarded {
		bfr.extras = append(bfr.extras, im)
	}
	n := copy(bfr.buffer[0:], bfr.buffer[pos:])
	bfr.buffer = bfr.buffer[0:n]
	bfr.offset = frameIdx

	bfr.cond.Broadcast()
}

// Buffer is valid until the next Discard call.
func (bfr *BufferedFfmpegReader) GetBuffer() ([]Image, int) {
	bfr.mu.Lock()
	defer bfr.mu.Unlock()

	return bfr.buffer, bfr.offset
}

func execTrackerLoop2(id int, tracker *Tracker, videoFname string, detectorDims [2]int, profile TrackerProfile, detectorFunc func(frameIdx int, im Image, extras []int, extraImages []Image) []Detection) [][]Detection {
	defer tracker.End(id)

	// initialize all reasonable skips
	// don't use a skip if corresponding threshold is too high
	var skips []int
	x := 1
	for _, threshold := range profile.Thresholds {
		if threshold > 0.9 {
			break
		}
		if threshold == 0 {
			skips = nil
		}
		skips = append(skips, x)
		x *= 2
	}
	minSkip := skips[0]
	maxSkip := skips[len(skips)-1]

	var outDetections [][]Detection
	setDetections := func(frameIdx int, dlist []Detection) {
		for frameIdx >= len(outDetections) {
			outDetections = append(outDetections, []Detection{})
		}
		outDetections[frameIdx] = dlist
	}

	vreader := ReadFfmpeg(videoFname, detectorDims[0], detectorDims[1])
	vreader.Skip = minSkip
	defer vreader.Close()
	bfr := NewBufferedFfmpegReader(vreader, 192)

	im, done := bfr.GetFrame(0)
	if done {
		return nil
	}
	dlist := detectorFunc(0, im, nil, nil)
	dlist, _ = tracker.Infer(id, 0, im, dlist)
	setDetections(0, dlist)

	lastFrame := 0

	getDetections := func(frameIdx int, im Image) []Detection {
		var extras []int
		var extraImages []Image
		buffer, offset := bfr.GetBuffer()
		for i, extraIm := range buffer {
			idx := (offset+i) * minSkip
			if idx == frameIdx || idx % maxSkip != 0 {
				continue
			}
			extras = append(extras, idx)
			extraImages = append(extraImages, extraIm)
		}
		return detectorFunc(frameIdx, im, extras, extraImages)
	}

	for {
		updated := false
		// iterate through skips until we get high enough confidence on some frame
		for i := len(skips)-1; i >= 0; i-- {
			// we must be aligned with the skip
			// e.g. process 0 and 4, then 2, next should be 4 then 8, not 6
			if lastFrame % skips[i] != 0 {
				continue
			}
			curFrame := lastFrame + skips[i]
			//t0 := time.Now()
			im, done := bfr.GetFrame(curFrame/minSkip)
			/*timeMu.Lock()
			decodeTime += time.Now().Sub(t0)
			timeMu.Unlock()*/
			if done {
				continue
			}

			dlist := getDetections(curFrame, im)
			//t0 = time.Now()
			tracked, conf := tracker.Infer(id, curFrame, im, dlist)
			/*timeMu.Lock()
			trackTime += time.Now().Sub(t0)
			timeMu.Unlock()*/
			if i != 0 && conf < profile.Thresholds[i] {
				continue
			}
			// update our state
			setDetections(curFrame, tracked)
			lastFrame = curFrame
			bfr.Discard(lastFrame/minSkip)
			updated = true
			break
		}
		if !updated {
			fmt.Println(id, "didn't update, lastframe=", lastFrame)
			break
		}
	}

	return outDetections
}
