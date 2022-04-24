package main

import (
	"./lib"

	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

func main() {
	dataRoot := os.Args[1]
	label := os.Args[2]
	videoPath := os.Args[3]
	outPath := os.Args[4]
	detectDims := lib.ParseDims(os.Args[5])

	os.MkdirAll(outPath, 0755)

	cfg := lib.GetConfig(dataRoot, label)
	batchSize := 16
	yolo := lib.NewYolov3(dataRoot, batchSize, detectDims, detectDims, cfg.OrigDims, 0.25, map[string]bool{}, label)

	ch := make(chan string)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for fname := range ch {
				label := strings.Split(fname, ".mp4")[0]
				vreader := lib.ReadFfmpeg(videoPath+fname, detectDims[0], detectDims[1])
				var detections [][]lib.Detection
				for {
					var images []lib.Image
					for i := 0; i < batchSize; i++ {
						im, err := vreader.Read()
						if err == io.EOF {
							break
						} else if err != nil {
							panic(err)
						}
						images = append(images, im)
					}
					if len(images) > 0 {
						cur := yolo.Detect(images)
						detections = append(detections, cur...)
					}
					if len(images) < batchSize {
						break
					}
				}
				vreader.Close()
				bytes, err := json.Marshal(detections)
				if err != nil {
					panic(err)
				}
				if err := ioutil.WriteFile(filepath.Join(outPath, label+".json"), bytes, 0644); err != nil {
					panic(err)
				}
			}
		}()
	}

	fnames, err := ioutil.ReadDir(videoPath)
	if err != nil {
		panic(err)
	}
	t0 := time.Now()
	for _, fname := range fnames {
		if !strings.HasSuffix(fname.Name(), ".mp4") {
			continue
		}
		ch <- fname.Name()
	}
	close(ch)
	wg.Wait()
	duration := time.Now().Sub(t0)
	msPerSample := int(duration/time.Millisecond)/len(fnames)
	if err := ioutil.WriteFile(filepath.Join(outPath, "speed.txt"), []byte(fmt.Sprintf("%d", msPerSample)), 0644); err != nil {
		panic(err)
	}

	yolo.Close()
}
