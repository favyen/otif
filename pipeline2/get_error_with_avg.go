package main

import (
	"./lib"

	"fmt"
	"io/ioutil"
	"math"
	"os"
	"strings"
)

func main() {
	label := os.Args[1]

	cfg := lib.GetConfig(label)
	gtFname := cfg.RootPath+"/test/gt.txt"
	gtCounts := make(map[int][]int)
	bytes, err := ioutil.ReadFile(gtFname)
	if err != nil {
		panic(err)
	}
	var columns int
	for _, line := range strings.Split(string(bytes), "\n") {
		parts := strings.Split(strings.TrimSpace(line), "\t")
		if len(parts) < 2 || parts[0] == "#" {
			continue
		}
		id := lib.ParseInt(parts[0])
		var counts []int
		for _, part := range parts[1:] {
			counts = append(counts, lib.ParseInt(part))
		}
		gtCounts[id] = counts
		columns = len(counts)
	}

	sums := make([]int, columns)
	for _, counts := range gtCounts {
		for i := 0; i < columns; i++ {
			sums[i] += counts[i]
		}
	}

	means := make([]float64, columns)
	for i := 0; i < columns; i++ {
		means[i] = float64(sums[i]) / float64(len(gtCounts))
	}

	var accs []float64
	for _, counts := range gtCounts {
		for i := 0; i < columns; i++ {
			acc := math.Abs(means[i]-float64(counts[i]))
			accs = append(accs, acc)
		}
	}
	var accsum float64
	for _, acc := range accs {
		accsum += acc
	}
	fmt.Println(accsum/float64(len(accs)))
}
