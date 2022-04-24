package lib

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

type SegmentationModel struct {
	cmd *exec.Cmd
	stdin io.WriteCloser
	rd *bufio.Reader
	mu sync.Mutex
	batchSize int
	zeroImage []byte
}

type Window struct {
	Bounds [4]int
	Cells [][4]int
}

func NewSegmentationModel(dataRoot string, mode string, batchSize int, cfg Config, segmentation SegmentationConfig, detector DetectorConfig) *SegmentationModel {
	var cmd *exec.Cmd
	if mode == "blazeit" {
		cmd = exec.Command(
			"python", "../blazeit/apply_dyn.py",
			strconv.Itoa(batchSize), strconv.Itoa(segmentation.Dims[0]), strconv.Itoa(segmentation.Dims[1]), fmt.Sprintf("%v", segmentation.Threshold),
			strconv.Itoa(detector.Dims[0]), strconv.Itoa(detector.Dims[1]),
			filepath.Join(dataRoot, "dataset", cfg.Label, "blazeit_model/model"),
		)
	} else if mode == "default" {
		cmd = exec.Command(
			"python", "../model/apply_dyn.py",
			strconv.Itoa(batchSize), strconv.Itoa(segmentation.Dims[0]), strconv.Itoa(segmentation.Dims[1]), fmt.Sprintf("%v", segmentation.Threshold),
			strconv.Itoa(detector.Dims[0]), strconv.Itoa(detector.Dims[1]), string(JsonMarshal(detector.Sizes)),
			filepath.Join(dataRoot, "dataset", cfg.Label, "segmentation_models", fmt.Sprintf("%d_%d", segmentation.Dims[0], segmentation.Dims[1]), "model"),
		)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		panic(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		panic(err)
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		panic(err)
	}
	rd := bufio.NewReader(stdout)
	return &SegmentationModel{
		cmd: cmd,
		stdin: stdin,
		rd: rd,
		batchSize: batchSize,
		zeroImage: make([]byte, detector.Dims[0]*detector.Dims[1]*3),
	}
}

func (m *SegmentationModel) GetWindows(images []Image) [][]Window {
	m.mu.Lock()
	for _, im := range images {
		m.stdin.Write(im.Bytes)
	}
	for i := len(images); i < m.batchSize; i++ {
		m.stdin.Write(m.zeroImage)
	}
	var line string
	for {
		var err error
		line, err = m.rd.ReadString('\n')
		if err != nil {
			panic(err)
		}
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "json") {
			continue
		}
		break
	}
	m.mu.Unlock()
	jsonBytes := []byte(line[4:])
	var windows [][]Window
	if err := json.Unmarshal(jsonBytes, &windows); err != nil {
		panic(err)
	}
	return windows[0:len(images)]
}

func (m *SegmentationModel) Close() {
	m.stdin.Close()
	m.cmd.Wait()
}
