package lib

import (
	"encoding/json"
	"math"
	"io/ioutil"
	"strconv"
)

func ParseInt(s string) int {
	x, err := strconv.Atoi(s)
	if err != nil {
		panic(err)
	}
	return x
}

func JsonMarshal(x interface{}) []byte {
	bytes, err := json.Marshal(x)
	if err != nil {
		panic(err)
	}
	return bytes
}

func JsonUnmarshal(bytes []byte, x interface{}) {
	err := json.Unmarshal(bytes, x)
	if err != nil {
		panic(err)
	}
}

func ReadJsonFile(fname string, x interface{}) {
	bytes, err := ioutil.ReadFile(fname)
	if err != nil {
		panic(err)
	}
	JsonUnmarshal(bytes, x)
}

func FloatsMean(floats []float64) float64 {
	var sum float64
	for _, x := range floats {
		sum += x
	}
	return sum / float64(len(floats))
}

// Returns the sample standard deviation.
func FloatsStddev(floats []float64) float64 {
	mean := FloatsMean(floats)
	var sqdevSum float64
	for _, x := range floats {
		sqdevSum += (x-mean) * (x-mean)
	}
	return math.Sqrt(sqdevSum / float64(len(floats)-1))
}

func FloatsStderr(floats []float64) float64 {
	return FloatsStddev(floats) / math.Sqrt(float64(len(floats)))
}
