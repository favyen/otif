package main

import (
	"github.com/mitroadmaps/gomapinfer/common"

	"./lib"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	dataRoot := os.Args[1]
	label := os.Args[2]
	fnames := strings.Split(os.Args[3], ",")

	cfg := lib.GetConfig(dataRoot, label)

	readTracks := func(fname string) [][]lib.TrackDetection {
		bytes, err := ioutil.ReadFile(fname)
		if err != nil {
			panic(err)
		}
		var detections [][]lib.Detection
		if err := json.Unmarshal(bytes, &detections); err != nil {
			panic(err)
		}
		tracks := lib.GetTracks(detections)
		return tracks
	}

	var trainSet [][]lib.TrackDetection
	trainDir := filePath.Join(dataRoot, "dataset", cfg.Label, "tracker/tracks/")
	files, err := ioutil.ReadDir(trainDir)
	if err != nil {
		panic(err)
	}
	for _, fi := range files {
		if !strings.HasSuffix(fi.Name(), ".json") {
			continue
		}
		tracks := readTracks(trainDir + fi.Name())
		trainSet = append(trainSet, tracks...)
	}

	var tracks [][]lib.TrackDetection
	for _, fname := range fnames {
		tracks = append(tracks, readTracks(fname)...)
	}

	fmt.Printf("clustering %d tracks\n", len(trainSet))
	clusters, _ := lib.ClusterTracks(trainSet)
	fmt.Printf("go from %d training set to %d clusters\n", len(trainSet), len(clusters))

	rd := lib.ReadFfmpeg(filepath.Join(dataRoot, "dataset", label, "tracker/video/0.mp4"), cfg.OrigDims[0], cfg.OrigDims[1])
	im, err := rd.Read()
	if err != nil {
		panic(err)
	}
	rd.Close()

	drawTrack := func(track []lib.TrackDetection, width int, color [3]uint8) {
		for i := 1; i < len(track); i++ {
			prev := track[i-1]
			cur := track[i]
			start := prev.Rectangle().Center()
			end := cur.Rectangle().Center()
			for _, p := range common.DrawLineOnCells(int(start.X), int(start.Y), int(end.X), int(end.Y), im.Width, im.Height) {
				im.FillRectangle(p[0]-width, p[1]-width, p[0]+width+1, p[1]+width+1, color)
			}
		}
	}

	/*for _, cluster := range clusters {
		drawTrack(cluster.Track, [3]uint8{255, 0, 0})
	}*/

	// find a long cluster
	/*var bigClusters []int
	for idx, cluster := range clusters {
		start := cluster.Track[0]
		end := cluster.Track[len(cluster.Track)-1]
		d := start.Rectangle().Center().Distance(end.Rectangle().Center())
		if d > 600 {
			bigClusters = append(bigClusters, idx)
		}
	}
	i1, i2 := bigClusters[0], bigClusters[1]

	for _, track := range members[i1] {
		drawTrack(track, 0, [3]uint8{255, 255, 0})
	}
	drawTrack(clusters[i1].Track, 2, [3]uint8{255, 0, 0})
	for _, track := range members[i2] {
		drawTrack(track, 0, [3]uint8{0, 255, 0})
	}
	drawTrack(clusters[i2].Track, 2, [3]uint8{0, 0, 255})*/

	for _, cluster := range clusters {
		start := cluster.Track[0]
		end := cluster.Track[len(cluster.Track)-1]
		displacement := start.Rectangle().Center().Distance(end.Rectangle().Center())
		if displacement < 600 {
			continue
		}
		drawTrack(cluster.Track, 0, [3]uint8{0, 0, 255})
	}


	// find long tracks
	var bigTracks []int
	for idx, track := range tracks {
		d := track[0].Rectangle().Center().Distance(track[len(track)-1].Rectangle().Center())
		if d > 300 && d < 600 {
			bigTracks = append(bigTracks, idx)
		}
	}
	fmt.Println(len(bigTracks), bigTracks)
	i1, i2, i3 := bigTracks[1], bigTracks[6], bigTracks[13]
	orig1, orig2, orig3 := tracks[i1], tracks[i2], tracks[i3]
	//i4 := bigTracks[13] // 1, 6, 13
	//orig4 := tracks[i4]
	fmt.Printf("processing %d tracks\n", len(tracks))
	tracks = lib.Postprocess(clusters, tracks)
	drawTrack(tracks[i1], 1, [3]uint8{255, 255, 0})
	drawTrack(tracks[i2], 1, [3]uint8{255, 255, 0})
	drawTrack(tracks[i3], 1, [3]uint8{255, 255, 0})
	//drawTrack(tracks[i4], 1, [3]uint8{255, 255, 0})

	drawTrack(orig1, 1, [3]uint8{255, 0, 0})
	drawTrack(orig2, 1, [3]uint8{255, 0, 0})
	drawTrack(orig3, 1, [3]uint8{255, 0, 0})
	//drawTrack(orig4, 1, [3]uint8{255, 0, 0})

	bytes := im.AsJPG()
	if err := ioutil.WriteFile("/home/ubuntu/vis/x.jpg", bytes, 0644); err != nil {
		panic(err)
	}
}
