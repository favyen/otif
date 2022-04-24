package lib

/*
for 1 fps

* threshold means we look at more frames when confidence is less than threshold
* threshold=0: don't look at any more frames
* threshold=1: don't track at this rate (just start tracking at next framerate)

pre-process:
(1) track over the video at 1 fps; but update RNN features using ground truth tracks
(2) for each track, it yields:
	(a) 1 - (2nd highest) / (1st highest): so if threshold is higher, then we would look at extra frame
	(b) minimum threshold needed to get it correct, i.e., 0 if correct option had highest prob, or 1 - (correct score) / (1st highest) otherwise
(3) using (b), compute minimum threshold needed to recover 80% contiguous segment of each track, call this (c)

setting threshold:
(1) iterate over alpha [0.99, 0.98, 0.96, 0.92, ...]
(2) pick alpha-percentile value in the (c) values
(3) estimate how many extra frames we would use based on the (a) values
*/

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
)

type Tracker struct {
	cmd *exec.Cmd
	stdin io.WriteCloser
	rd *bufio.Reader
	mu sync.Mutex
}

type TrackerPacket struct {
	ID int `json:"id"`
	FrameIdx int `json:"frame_idx"`
	Detections []Detection `json:"detections"`
	GT map[int]*int `json:"gt"`
	Type string `json:"type"`
}

type TrackerResponse struct {
	Outputs []int `json:"outputs"`
	Conf float64 `json:"conf"`
	T map[int]float64 `json:"t"`
}

func NewTracker(dataRoot string, mode string, cfg Config) *Tracker {
	var pythonFname string
	if mode == "iou" {
		pythonFname = "tracker.py"
	} else if mode == "rnn" {
		pythonFname = "../rnn/tracker.py"
	} else if mode == "miris" {
		pythonFname = "../tracker-miris/tracker.py"
	} else {
		panic(fmt.Errorf("bad tracker mode %s", mode))
	}
	cmd := exec.Command(
		"python", pythonFname,
		dataRoot, cfg.Label,
		strconv.Itoa(cfg.OrigDims[0]), strconv.Itoa(cfg.OrigDims[1]),
	)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		panic(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		panic(err)
	}
	if true {
		cmd.Stderr = os.Stderr
	}
	if err := cmd.Start(); err != nil {
		panic(err)
	}
	rd := bufio.NewReader(stdout)
	return &Tracker{
		cmd: cmd,
		stdin: stdin,
		rd: rd,
	}
}

func (tracker *Tracker) do(id int, frameIdx int, im Image, detections []Detection, gt map[int]*int) TrackerResponse {
	packet := TrackerPacket{
		ID: id,
		FrameIdx: frameIdx,
		Detections: detections,
		GT: gt,
		Type: "job",
	}
	bytes := JsonMarshal(packet)

	tracker.mu.Lock()
	defer tracker.mu.Unlock()

	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(bytes)))
	tracker.stdin.Write(header)
	tracker.stdin.Write(bytes)

	header = make([]byte, 12)
	binary.BigEndian.PutUint32(header[0:4], uint32(len(im.Bytes)))
	binary.BigEndian.PutUint32(header[4:8], uint32(im.Width))
	binary.BigEndian.PutUint32(header[8:12], uint32(im.Height))
	tracker.stdin.Write(header)
	tracker.stdin.Write(im.Bytes)

	var line string
	for {
		var err error
		line, err = tracker.rd.ReadString('\n')
		if err != nil {
			panic(err)
		}
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "json") {
			continue
		}
		break
	}

	jsonBytes := []byte(line[4:])
	var response TrackerResponse
	if err := json.Unmarshal(jsonBytes, &response); err != nil {
		panic(err)
	}
	return response
}

func (tracker *Tracker) Infer(id int, frameIdx int, im Image, detections []Detection) ([]Detection, float64) {
	response := tracker.do(id, frameIdx, im, detections, nil)

	var outputs []Detection
	for i, d := range detections {
		d.TrackID = new(int)
		*d.TrackID = response.Outputs[i]
		outputs = append(outputs, d)
	}
	return outputs, response.Conf
}

func (tracker *Tracker) End(id int) {
	packet := TrackerPacket{
		ID: id,
		Type: "end",
	}
	bytes := JsonMarshal(packet)

	tracker.mu.Lock()
	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(bytes)))
	tracker.stdin.Write(header)
	tracker.stdin.Write(bytes)
	tracker.mu.Unlock()
}

type TrackerProfile struct {
	NumFrames int
	Thresholds []float64
}

func (prof TrackerProfile) MaxGap() int {
	if len(prof.Thresholds) < 2 {
		return 1
	}
	gap := 1
	for _, threshold := range prof.Thresholds[1:] {
		if threshold == 1 {
			return gap
		}
		gap *= 2
	}
	return gap
}

// Returns map from alpha value to tracker profile
func (tracker *Tracker) GetThresholds(videoPath string, jsonPath string, origDims [2]int, detectorDims [2]int, freqs []int) map[float64]TrackerProfile {
	// returns length of longest contiguous, non-disconnected segment
	getLongestSegment := func(disconnects []bool) int {
		discList := []int{-1}
		for i, bad := range disconnects {
			if !bad {
				continue
			}
			discList = append(discList, i)
		}
		discList = append(discList, len(disconnects)-1)

		longest := 0
		for i := 0; i < len(discList)-1; i++ {
			l := discList[i+1]-discList[i]
			if l > longest {
				longest = l
			}
		}
		return longest
	}

	// get the minimum threshold needed to recover this track
	getMinThreshold := func(confs []float64) float64 {
		// iteratively disconnect the track at edges with smallest confidence
		// repeat until longest contiguous segment is <80% of the original track
		disconnects := make([]bool, len(confs))
		var threshold float64
		for {
			// find highest confidence
			var bestConf float64 = 0.01
			var bestIdx int = -1
			for i, score := range confs {
				if disconnects[i] {
					continue
				}
				if score > bestConf {
					bestIdx = i
					bestConf = score
				}
			}
			if bestIdx == -1 {
				break
			}

			// disconnect and check longest segment length
			disconnects[bestIdx] = true
			l := getLongestSegment(disconnects)
			if l > len(disconnects)*8/10 {
				continue
			}
			// it doesn't work, so threshold must be at least as high as bestConf
			threshold = bestConf
			break
		}
		return threshold
	}

	// determine whether inferred detections match up for a given gt track
	isMatchingTrack := func(gtID int, freq int, gt [][]Detection, detections [][]Detection) bool {
		// find the track ID in detections that matches most with gtID
		hits := make(map[int]int)
		for frameIdx := range gt {
			if frameIdx%freq != 0 {
				continue
			}
			for i, d := range gt[frameIdx] {
				if *d.TrackID != gtID {
					continue
				}
				hits[*detections[frameIdx][i].TrackID]++
			}
		}
		var bestID int = -1
		for trackID := range hits {
			if hits[trackID] < 2 {
				continue
			}
			if bestID == -1 || hits[trackID] > hits[bestID] {
				bestID = trackID
			}
		}
		if bestID == -1 {
			return false
		}

		// see if inferred track matches at 80% of the detections
		var matches int = 0
		var misses int = 0
		for frameIdx := range gt {
			if frameIdx%freq != 0 {
				continue
			}
			for i, d := range gt[frameIdx] {
				if *d.TrackID != gtID {
					continue
				}
				if *detections[frameIdx][i].TrackID == bestID {
					matches++
				} else {
					misses++
				}
			}
		}

		return matches >= 4*misses
	}

	getConfidences := func(id int, freq int) (trackMinThreshs []float64, frameConfidences []float64) {
		var detections [][]Detection
		ReadJsonFile(jsonPath+fmt.Sprintf("%d.json", id), &detections)

		// re-scale detections and compute the range of frames each track spans
		//RescaleDetections(detections, origDims, detectorDims)
		trackRanges := make(map[int][2]int)
		for i := range detections {
			for j := range detections[i] {
				trackID := *detections[i][j].TrackID
				if _, ok := trackRanges[trackID]; !ok {
					trackRanges[trackID] = [2]int{i, i}
				} else {
					trackRanges[trackID] = [2]int{trackRanges[trackID][0], i}
				}
			}
		}
		activeTracks := make([][]int, len(detections))
		for trackID, rng := range trackRanges {
			if rng[1]-rng[0] < 96 {
				// this track is rather noisy so we shouldn't use it in scoring
				continue
			}
			for frameIdx := rng[0]; frameIdx <= rng[1]; frameIdx++ {
				activeTracks[frameIdx] = append(activeTracks[frameIdx], trackID)
			}
		}

		// run the tracker to get probabilities but force it to update using gt tracks
		// although we also separately collect the inferred tracks without gt
		confByTrackID := make(map[int][]float64)
		inferred := make([][]Detection, len(detections))
		vreader := ReadFfmpeg(filepath.Join(videoPath, fmt.Sprintf("%d.mp4", id)), detectorDims[0], detectorDims[1])
		for frameIdx := 0; ; frameIdx++ {
			im, err := vreader.Read()
			if err == io.EOF {
				break
			} else if err != nil {
				panic(err)
			}

			if frameIdx%freq != 0 {
				continue
			}

			gt := make(map[int]*int)
			for _, trackID := range activeTracks[frameIdx] {
				gt[trackID] = nil
			}
			for i, d := range detections[frameIdx] {
				if _, ok := gt[*d.TrackID]; !ok {
					// track was too short
					continue
				}
				gt[*d.TrackID] = new(int)
				*gt[*d.TrackID] = i
			}

			response := tracker.do(id, frameIdx, im, detections[frameIdx], gt)
			frameConfidences = append(frameConfidences, response.Conf)
			for trackID, thresh := range response.T {
				confByTrackID[trackID] = append(confByTrackID[trackID], thresh)
			}

			// also get the inferred tracks
			inferred[frameIdx], _ = tracker.Infer(-1, frameIdx, im, detections[frameIdx])
		}
		tracker.End(id)
		tracker.End(-1)
		vreader.Close()

		// get min thresholds for the tracks
		// if inferred track overlaps >= 80% with gt track, then we mark it okay
		// otherwise threshold is based on the probabilities when running it with gt updates
		for gtID, confs := range confByTrackID {
			var minThreshold float64
			if isMatchingTrack(gtID, freq, detections, inferred) {
				minThreshold = 0
			} else {
				minThreshold = getMinThreshold(confs)
			}
			trackMinThreshs = append(trackMinThreshs, minThreshold)
		}

		bytes := JsonMarshal(inferred)
		ioutil.WriteFile("/tmp/x.json", bytes, 0644)

		return
	}

	getWorkload := func(frameConfidences []float64, threshold float64) int {
		var numFrames int
		for _, conf := range frameConfidences {
			if conf > threshold {
				continue
			}
			numFrames++
		}
		return numFrames
	}

	// list videos
	var ids []int
	files, err := ioutil.ReadDir(videoPath)
	if err != nil {
		panic(err)
	}
	for _, fi := range files {
		if !strings.HasSuffix(fi.Name(), ".mp4") {
			continue
		}
		id := ParseInt(strings.Split(fi.Name(), ".mp4")[0])
		ids = append(ids, id)
	}

	// first get min thresholds and frame confidences at each freq
	minThresholds := make([][]float64, len(freqs))
	frameConfidences := make([][]float64, len(freqs))
	for i, freq := range freqs {
		for _, id := range ids {
			fmt.Printf("process %d.mp4 at freq=%d\n", id, freq)
			curThreshs, curConfidences := getConfidences(id, freq)
			minThresholds[i] = append(minThresholds[i], curThreshs...)
			frameConfidences[i] = append(frameConfidences[i], curConfidences...)
		}
		sort.Float64s(minThresholds[i])
	}

	// now, profile each alpha
	profiles := make(map[float64]TrackerProfile)
	for _, alpha := range []float64{1.0, 0.998, 0.995, 0.99, 0.98, 0.96, 0.92, 0.84, 0.68} {
		var profile TrackerProfile
		for i := range freqs {
			// get alpha-percentile lenient (highest) threshold
			idx := int(float64(len(minThresholds[i]))*alpha)
			if idx >= len(minThresholds[i]) {
				idx = len(minThresholds[i])-1
			}
			threshold := minThresholds[i][idx]

			profile.Thresholds = append(profile.Thresholds, threshold)
			profile.NumFrames += getWorkload(frameConfidences[i], threshold)
		}
		profiles[alpha] = profile
	}

	return profiles
}

func (tracker *Tracker) Close() {
	tracker.stdin.Close()
	tracker.cmd.Wait()
}
