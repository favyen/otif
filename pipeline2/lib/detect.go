package lib

import (
	"github.com/mitroadmaps/gomapinfer/common"

	"bufio"
	"encoding/json"
	"io"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
)

type Detection struct {
	Left int `json:"left"`
	Top int `json:"top"`
	Right int `json:"right"`
	Bottom int `json:"bottom"`
	Class string `json:"class"`
	Score float64 `json:"score"`
	TrackID *int `json:"track_id,omitempty"`
}

type Point struct {
	X int
	Y int
}

func RescaleDetections(detections [][]Detection, origDims [2]int, newDims [2]int) {
	for frameIdx := range detections {
		for i := range detections[frameIdx] {
			detections[frameIdx][i].Left = detections[frameIdx][i].Left*newDims[0]/origDims[0]
			detections[frameIdx][i].Right = detections[frameIdx][i].Right*newDims[0]/origDims[0]
			detections[frameIdx][i].Top = detections[frameIdx][i].Top*newDims[1]/origDims[1]
			detections[frameIdx][i].Bottom = detections[frameIdx][i].Bottom*newDims[1]/origDims[1]
		}
	}
}

func FilterDetectionsByClass(detections [][]Detection, cls map[string]bool) [][]Detection {
	ndetections := make([][]Detection, len(detections))
	for frameIdx, dlist := range detections {
		for _, d := range dlist {
			if !cls[d.Class] {
				continue
			}
			ndetections[frameIdx] = append(ndetections[frameIdx], d)
		}
	}
	return ndetections
}

type TrackDetection struct {
	Detection
	FrameIdx int
}

func GetTracks(detections [][]Detection) [][]TrackDetection {
	tracks := make(map[int][]TrackDetection)
	for frameIdx, dlist := range detections {
		for _, d := range dlist {
			tracks[*d.TrackID] = append(tracks[*d.TrackID], TrackDetection{
				Detection: d,
				FrameIdx: frameIdx,
			})
		}
	}
	var trackList [][]TrackDetection
	for _, track := range tracks {
		trackList = append(trackList, track)
	}
	return trackList
}

func DetectionsFromTracks(tracks [][]TrackDetection) [][]Detection {
	var detections [][]Detection
	for _, track := range tracks {
		for _, d := range track {
			for len(detections) <= d.FrameIdx {
				detections = append(detections, nil)
			}
			detections[d.FrameIdx] = append(detections[d.FrameIdx], d.Detection)
		}
	}
	return detections
}

func (d Detection) Center() Point {
	return Point{
		X: (d.Left+d.Right)/2,
		Y: (d.Top+d.Bottom)/2,
	}
}

func (d Detection) Rectangle() common.Rectangle {
	return common.Rectangle{
		common.Point{float64(d.Left), float64(d.Top)},
		common.Point{float64(d.Right), float64(d.Bottom)},
	}
}

func (p Point) Distance(o Point) float64 {
	dx := p.X-o.X
	dy := p.Y-o.Y
	return math.Sqrt(float64(dx*dx+dy*dy))
}

type Yolov3 struct {
	cmd *exec.Cmd
	stdin io.WriteCloser
	rd *bufio.Reader
	mu sync.Mutex

	batchSize int
	dims [2]int
	origDims [2]int
	zeroImage []byte
}

// paramDims indicate the uncropped image dimensions (which determines the model parameters that we load)
func NewYolov3(dataRoot string, batchSize int, dims [2]int, paramDims [2]int, origDims [2]int, threshold float64, classes map[string]bool, label string) *Yolov3 {
	var classList []string
	for cls := range classes {
		classList = append(classList, cls)
	}
	cmd := exec.Command(
		"python", "yolov3.py",
		dataRoot,
		strconv.Itoa(batchSize),
		strconv.Itoa(dims[0]), strconv.Itoa(dims[1]), strconv.Itoa(paramDims[0]), strconv.Itoa(paramDims[1]),
		"0.25", strings.Join(classList, ","), label,
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
	return &Yolov3{
		cmd: cmd,
		stdin: stdin,
		rd: rd,
		batchSize: batchSize,
		dims: dims,
		origDims: origDims,
		zeroImage: make([]byte, dims[0]*dims[1]*3),
	}
}

func (yolo *Yolov3) Detect(images []Image) [][]Detection {
	yolo.mu.Lock()
	for _, im := range images {
		yolo.stdin.Write(im.Bytes)
	}
	for i := len(images); i < yolo.batchSize; i++ {
		yolo.stdin.Write(yolo.zeroImage)
	}
	var line string
	for {
		var err error
		line, err = yolo.rd.ReadString('\n')
		if err != nil {
			panic(err)
		}
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "json") {
			continue
		}
		break
	}
	yolo.mu.Unlock()
	jsonBytes := []byte(line[4:])
	var detections [][]Detection
	if err := json.Unmarshal(jsonBytes, &detections); err != nil {
		panic(err)
	}
	for i := range detections {
		for j := range detections[i] {
			detections[i][j].Left = detections[i][j].Left*yolo.origDims[0]/yolo.dims[0]
			detections[i][j].Top = detections[i][j].Top*yolo.origDims[1]/yolo.dims[1]
			detections[i][j].Right = detections[i][j].Right*yolo.origDims[0]/yolo.dims[0]
			detections[i][j].Bottom = detections[i][j].Bottom*yolo.origDims[1]/yolo.dims[1]
		}
	}
	return detections[0:len(images)]
}

func (yolo *Yolov3) Close() {
	yolo.stdin.Close()
	yolo.cmd.Wait()
}
