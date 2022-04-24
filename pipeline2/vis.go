package main

import (
	"./lib"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strconv"
)

var Colors = [][3]uint8{
	[3]uint8{255, 0, 0},
	[3]uint8{0, 255, 0},
	[3]uint8{0, 0, 255},
	[3]uint8{255, 255, 0},
	[3]uint8{0, 255, 255},
	[3]uint8{255, 0, 255},
	[3]uint8{0, 51, 51},
	[3]uint8{51, 153, 153},
	[3]uint8{102, 0, 51},
	[3]uint8{102, 51, 204},
	[3]uint8{102, 153, 204},
	[3]uint8{102, 255, 204},
	[3]uint8{153, 102, 102},
	[3]uint8{204, 102, 51},
	[3]uint8{204, 255, 102},
	[3]uint8{255, 255, 204},
	[3]uint8{121, 125, 127},
	[3]uint8{69, 179, 157},
	[3]uint8{250, 215, 160},
}

func main() {
	videoFname := os.Args[1]
	jsonFname := os.Args[2]
	outPath := os.Args[3]
	skip, err := strconv.Atoi(os.Args[4])
	if err != nil {
		panic(err)
	}

	var detections [][]lib.Detection
	lib.ReadJsonFile(jsonFname, &detections)
	lib.RescaleDetections(detections, [2]int{1280, 720}, [2]int{640, 352})

	rd := lib.ReadFfmpeg(videoFname, 640, 352)
	for frameIdx := 0; ; frameIdx++ {
		im, err := rd.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
		if frameIdx%skip != 0 {
			continue
		}
		for _, d := range detections[frameIdx] {
			var color [3]uint8
			if d.TrackID == nil {
				color = Colors[0]
			} else {
				color = Colors[*d.TrackID % len(Colors)]
			}
			im.DrawRectangle(d.Left, d.Top, d.Right, d.Bottom, 1, color)
		}
		bytes := im.AsJPG()
		if err := ioutil.WriteFile(fmt.Sprintf("%s/%d.jpg", outPath, frameIdx), bytes, 0644); err != nil {
			panic(err)
		}
	}
}
