package main

import (
	goslgraph "github.com/cpmech/gosl/graph"
	gomapinfer "github.com/mitroadmaps/gomapinfer/common"

	"encoding/json"
	"io/ioutil"
	"os"
	"strconv"
)

type Detection struct {
	FrameIdx int `json:"frame_idx"`
	TrackID int `json:"track_id"`
	Left int `json:"left"`
	Top int `json:"top"`
	Right int `json:"right"`
	Bottom int `json:"bottom"`
	Score float64 `json:"score,omitempty"`
	Class string `json:"class,omitempty"`
}

func (d Detection) Bounds() gomapinfer.Rectangle {
	return gomapinfer.Rectangle{
		Min: gomapinfer.Point{float64(d.Left), float64(d.Top)},
		Max: gomapinfer.Point{float64(d.Right), float64(d.Bottom)},
	}
}

type Track struct {
	ID int
	Detections []Detection
}

func (t *Track) Last() Detection {
	return t.Detections[len(t.Detections) - 1]
}

const MaxAge = 20

func main() {
	fname := os.Args[1]
	outFname := os.Args[2]
	class := os.Args[3]
	threshold, _ := strconv.ParseFloat(os.Args[4], 64)

	var allDetections [][]Detection // indexed by frame idx
	bytes, err := ioutil.ReadFile(fname)
	if err != nil {
		panic(err)
	}
	if err := json.Unmarshal(bytes, &allDetections); err != nil {
		panic(err)
	}

	var tracks []*Track
	activeTracks := make(map[int]*Track)
	for frameIdx := 0; frameIdx < len(allDetections); frameIdx++ {
		var detections []Detection
		for _, d := range allDetections[frameIdx] {
			if d.Class != class {
				continue
			} else if d.Score < threshold {
				continue
			}
			d.FrameIdx = frameIdx
			detections = append(detections, d)
		}

		matches := hungarianMatcher(activeTracks, detections)
		unmatched := make(map[int]Detection)
		for detectionIdx := range detections {
			unmatched[detectionIdx] = detections[detectionIdx]
		}
		for trackID, detectionIdx := range matches {
			delete(unmatched, detectionIdx)
			detection := detections[detectionIdx]
			detection.TrackID = trackID
			tracks[trackID].Detections = append(tracks[trackID].Detections, detection)
		}
		for _, detection := range unmatched {
			trackID := len(tracks)
			detection.TrackID = trackID
			track := &Track{
				ID: trackID,
				Detections: []Detection{detection},
			}
			tracks = append(tracks, track)
			activeTracks[track.ID] = track
		}
		for _, track := range activeTracks {
			if frameIdx - track.Last().FrameIdx < MaxAge {
				continue
			}
			delete(activeTracks, track.ID)
		}
	}

	var trackedDetections [][]Detection // indexed by frame idx
	for _, track := range tracks {
		for _, detection := range track.Detections {
			for len(trackedDetections) <= detection.FrameIdx {
				trackedDetections = append(trackedDetections, nil)
			}
			trackedDetections[detection.FrameIdx] = append(trackedDetections[detection.FrameIdx], detection)
		}
	}
	bytes, err = json.Marshal(trackedDetections)
	if err != nil {
		panic(err)
	}
	if err := ioutil.WriteFile(outFname, bytes, 0644); err != nil {
		panic(err)
	}
}

// Returns map from track idx to detection idx that should be added corresponding to that track.
func hungarianMatcher(activeTracks map[int]*Track, detections []Detection) map[int]int {
	if len(activeTracks) == 0 || len(detections) == 0 {
		return nil
	}

	var trackList []*Track
	for _, track := range activeTracks {
		trackList = append(trackList, track)
	}

	// create cost matrix for hungarian algorithm
	// rows: active tracks (trackList)
	// cols: current detections (detections)
	// values: 1-IoU if overlap is non-zero, or 1.5 otherwise
	costMatrix := make([][]float64, len(trackList))
	for i, track := range trackList {
		costMatrix[i] = make([]float64, len(detections))
		trackRect := track.Last().Bounds()

		for j, detection := range detections {
			curRect := detection.Bounds()
			iou := trackRect.IOU(curRect)
			if detection.Class != track.Detections[0].Class {
				iou = 0
			}
			var cost float64
			if iou > 0.99 {
				cost = 0.01
			} else if iou > 0.1 {
				cost = 1 - iou
			} else {
				cost = 1.5
			}
			costMatrix[i][j] = cost
		}
	}

	munkres := &goslgraph.Munkres{}
	munkres.Init(len(trackList), len(detections))
	munkres.SetCostMatrix(costMatrix)
	munkres.Run()

	matches := make(map[int]int)
	for i, j := range munkres.Links {
		track := trackList[i]
		if j < 0 || costMatrix[i][j] > 0.9 {
			continue
		}
		matches[track.ID] = j
	}
	return matches
}
