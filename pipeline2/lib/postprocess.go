package lib

import (
	"github.com/mitroadmaps/gomapinfer/common"
	"log"
	"math"
)

func InterpolateTrack(track []TrackDetection) []TrackDetection {
	var output []TrackDetection
	for _, detection := range track {
		if len(output) > 0 {
			prev := output[len(output)-1]
			next := detection
			jump := next.FrameIdx - prev.FrameIdx
			for i := 1; i < jump; i++ {
				prevWeight := int(1000 * float64(jump - i) / float64(jump))
				nextWeight := int(1000 * float64(i) / float64(jump))
				interp := TrackDetection{
					Detection: Detection{
						Left: (prev.Left*prevWeight+next.Left*nextWeight)/1000,
						Top: (prev.Top*prevWeight+next.Top*nextWeight)/1000,
						Right: (prev.Right*prevWeight+next.Right*nextWeight)/1000,
						Bottom: (prev.Bottom*prevWeight+next.Bottom*nextWeight)/1000,
						Class: prev.Class,
						Score: prev.Score,
						TrackID: prev.TrackID,
					},
					FrameIdx: prev.FrameIdx + i,
				}
				output = append(output, interp)
			}
		}
		output = append(output, detection)
	}
	return output
}

func SampleNormalizedPoints(track []TrackDetection) []common.Point {
	// sample twenty points along the track
	var trackLength float64 = 0
	for i := 0; i < len(track) - 1; i++ {
		trackLength += track[i].Rectangle().Center().Distance(track[i+1].Rectangle().Center())
	}
	pointFreq := trackLength / 20

	points := []common.Point{track[0].Rectangle().Center()}
	remaining := pointFreq
	for i := 0; i < len(track) - 1; i++ {
		segment := common.Segment{track[i].Rectangle().Center(), track[i+1].Rectangle().Center()}
		for segment.Length() > remaining {
			vector := segment.Vector()
			p := segment.Start.Add(vector.Scale(remaining / segment.Length()))
			points = append(points, p)
			segment = common.Segment{p, segment.End}
			remaining = pointFreq
		}
		remaining -= segment.Length()
	}
	for len(points) < 20 {
		points = append(points, track[len(track)-1].Rectangle().Center())
	}
	return points[0:20]
}

func TrackDistances(tracks1 [][]TrackDetection, tracks2 [][]TrackDetection) [][]float64 {
	log.Printf("\ttrack distances: sampling points for %d tracks1 and %d tracks2", len(tracks1), len(tracks2))
	points1 := make([][]common.Point, len(tracks1))
	points2 := make([][]common.Point, len(tracks2))
	for i, track := range tracks1 {
		points1[i] = SampleNormalizedPoints(track)
	}
	for i, track := range tracks2 {
		points2[i] = SampleNormalizedPoints(track)
	}

	dist := func(idx1, idx2 int) float64 {
		var maxDistance float64 = 0
		for i := range points1[idx1] {
			d := points1[idx1][i].Distance(points2[idx2][i])
			if d > maxDistance {
				maxDistance = d
			}
		}
		return maxDistance
	}

	log.Printf("\ttrack distances: compute %dx%d distances", len(tracks1), len(tracks2))
	distances := make([][]float64, len(points1))
	for i := range distances {
		distances[i] = make([]float64, len(points2))
		for j := range distances[i] {
			distances[i][j] = dist(i, j)
		}
	}
	return distances
}

// another distance metric
// track1 must be finely interpolated
func TrackDistance2(track1 []TrackDetection, track2 []TrackDetection) float64 {
	var maxDistance float64 = 0
	var lastIdx int = 0 // if track2[i] matches track1[j], track2[i+1] cannot match with any track1[k] for k<j
	for _, d2 := range track2 {
		var best float64 = 9999
		for idx, d1 := range track1 {
			if idx < lastIdx {
				continue
			}
			d := d1.Rectangle().Center().Distance(d2.Rectangle().Center())
			if d < best {
				best = d
				lastIdx = idx
			}
		}
		if best > maxDistance {
			maxDistance = best
		}
	}
	return maxDistance
}

type Cluster struct {
	TrackIdx int
	Track []TrackDetection
	Count int
}

// Cluster the training tracks.
func ClusterTracks(tracks [][]TrackDetection) ([]Cluster, [][][]TrackDetection) {
	distances := TrackDistances(tracks, tracks)
	var threshold float64 = 50

	var clusters []Cluster
	var members [][][]TrackDetection
	for trackIdx, track := range tracks {
		var bestCluster int = -1
		var bestDistance float64
		for clusterIdx, cluster := range clusters {
			d := distances[trackIdx][cluster.TrackIdx]
			if d >= threshold {
				continue
			}
			if bestCluster == -1 || d < bestDistance {
				bestCluster = clusterIdx
				bestDistance = d
			}
		}
		if bestCluster != -1 {
			clusters[bestCluster].Count++
			members[bestCluster] = append(members[bestCluster], track)
			continue
		}

		// didn't match to any cluster, so add a new one
		// make sure cluster representative is fully interpolated
		clusters = append(clusters, Cluster{trackIdx, InterpolateTrack(track), 1})
		members = append(members, [][]TrackDetection{track})
	}

	return clusters, members
}

// Add prefix/suffix detections to each track to refine them.
// We leverage the provided clusters to do this.
func Postprocess(clusters []Cluster, tracks [][]TrackDetection) [][]TrackDetection {
	// returns index of detection at least 100 pixels away either forward/backward from the given index
	// or -1
	nextDetection := func(track []TrackDetection, idx int, direction int) int {
		p := track[idx].Rectangle().Center()
		for i := idx + direction; i >= 0 && i < len(track); i += direction {
			if track[i].Rectangle().Center().Distance(p) > 100 {
				return i
			}
		}
		return -1
	}

	maxFrame := 0
	for _, track := range tracks {
		for _, d := range track {
			if d.FrameIdx > maxFrame {
				maxFrame = d.FrameIdx
			}
		}
	}

	for i, track := range tracks {
		// find closest cluster
		var closestCluster Cluster
		var bestDistance float64 = -1
		for _, cluster := range clusters {
			d := TrackDistance2(cluster.Track, track)
			if bestDistance == -1 || d < bestDistance {
				closestCluster = cluster
				bestDistance = d
			}
		}
		if bestDistance == -1 || bestDistance > 75 || len(closestCluster.Track) < 10 {
			continue
		}

		// compare vector to prefix/suffix with velocity
		prefix := closestCluster.Track[0]
		suffix := closestCluster.Track[len(closestCluster.Track)-1]

		pnext := nextDetection(track, 0, 1)
		if pnext != -1 && track[0].FrameIdx > 0 {
			vector1 := prefix.Rectangle().Center().Sub(track[0].Rectangle().Center())
			vector2 := track[0].Rectangle().Center().Sub(track[pnext].Rectangle().Center())
			angle := vector1.SignedAngle(vector2)
			if math.Abs(angle) < math.Pi/4 {
				prefix.FrameIdx = track[0].FrameIdx-1
				prefix.TrackID = track[0].TrackID
				tracks[i] = append([]TrackDetection{prefix}, tracks[i]...)
			}
		}

		snext := nextDetection(track, len(track)-1, -1)
		if snext != -1 && track[len(track)-1].FrameIdx < maxFrame {
			vector1 := suffix.Rectangle().Center().Sub(track[len(track)-1].Rectangle().Center())
			vector2 := track[len(track)-1].Rectangle().Center().Sub(track[snext].Rectangle().Center())
			angle := vector1.SignedAngle(vector2)
			if math.Abs(angle) < math.Pi/4 {
				suffix.FrameIdx = track[len(track)-1].FrameIdx+1
				suffix.TrackID = track[0].TrackID
				tracks[i] = append(tracks[i], suffix)
			}
		}
	}

	return tracks
}
