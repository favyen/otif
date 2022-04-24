package main

import (
	"./lib"
	"fmt"
	"os"
)

func main() {
	dataRoot := os.Args[1]
	label := os.Args[2]
	validtest := os.Args[3]
	path := os.Args[4]
	fmt.Println(lib.Eval(dataRoot, lib.GetConfig(dataRoot, label), validtest, path))
}
