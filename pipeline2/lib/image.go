package lib

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"os"
)

type Image struct {
	Width int
	Height int
	Bytes []byte
}

func NewImage(width int, height int) Image {
	return Image{
		Width: width,
		Height: height,
		Bytes: make([]byte, 3*width*height),
	}
}

func ImageFromBytes(width int, height int, bytes []byte) Image {
	return Image{
		Width: width,
		Height: height,
		Bytes: bytes,
	}
}

func ImageFromJPGReader(rd io.Reader) Image {
	im, err := jpeg.Decode(rd)
	if err != nil {
		panic(err)
	}
	rect := im.Bounds()
	width := rect.Dx()
	height := rect.Dy()
	bytes := make([]byte, width*height*3)
	for i := 0; i < width; i++ {
		for j := 0; j < height; j++ {
			r, g, b, _ := im.At(i + rect.Min.X, j + rect.Min.Y).RGBA()
			bytes[(j*width+i)*3+0] = uint8(r >> 8)
			bytes[(j*width+i)*3+1] = uint8(g >> 8)
			bytes[(j*width+i)*3+2] = uint8(b >> 8)
		}
	}
	return Image{
		Width: width,
		Height: height,
		Bytes: bytes,
	}
}

func ImageFromFile(fname string) Image {
	file, err := os.Open(fname)
	if err != nil {
		panic(err)
	}
	im := ImageFromJPGReader(file)
	file.Close()
	return im
}

func (im Image) AsImage() image.Image {
	pixbuf := make([]byte, im.Width*im.Height*4)
	j := 0
	channels := 0
	for i := range im.Bytes {
		pixbuf[j] = im.Bytes[i]
		j++
		channels++
		if channels == 3 {
			pixbuf[j] = 255
			j++
			channels = 0
		}
	}
	img := &image.RGBA{
		Pix: pixbuf,
		Stride: im.Width*4,
		Rect: image.Rect(0, 0, im.Width, im.Height),
	}
	return img
}

func (im Image) AsJPG() []byte {
	buf := new(bytes.Buffer)
	if err := jpeg.Encode(buf, im.AsImage(), nil); err != nil {
		panic(err)
	}
	return buf.Bytes()
}

func (im Image) AsPNG() []byte {
	buf := new(bytes.Buffer)
	if err := png.Encode(buf, im.AsImage()); err != nil {
		panic(err)
	}
	return buf.Bytes()
}

func (im Image) ToBytes() []byte {
	return im.Bytes
}

func (im Image) SetRGB(i int, j int, color [3]uint8) {
	if i < 0 || i >= im.Width || j < 0 || j >= im.Height {
		return
	}
	for channel := 0; channel < 3; channel++ {
		im.Bytes[(j*im.Width+i)*3+channel] = color[channel]
	}
}

func (im Image) GetRGB(i int, j int) [3]uint8 {
	var color [3]uint8
	for channel := 0; channel < 3; channel++ {
		color[channel] = im.Bytes[(j*im.Width+i)*3+channel]
	}
	return color
}

func (im Image) FillRectangle(left, top, right, bottom int, color [3]uint8) {
	for i := left; i < right; i++ {
		for j := top; j < bottom; j++ {
			im.SetRGB(i, j, color)
		}
	}
}

func (im Image) Copy() Image {
	bytes := make([]byte, len(im.Bytes))
	copy(bytes, im.Bytes)
	return Image{
		Width: im.Width,
		Height: im.Height,
		Bytes: bytes,
	}
}

func (im Image) DrawRectangle(left, top, right, bottom int, width int, color [3]uint8) {
	im.FillRectangle(left-width, top, left+width, bottom, color)
	im.FillRectangle(right-width, top, right+width, bottom, color)
	im.FillRectangle(left, top-width, right, top+width, color)
	im.FillRectangle(left, bottom-width, right, bottom+width, color)
}

func (im Image) DrawImage(left int, top int, other Image) {
	for i := 0; i < other.Width; i++ {
		for j := 0; j < other.Height; j++ {
			im.SetRGB(left+i, top+j, other.GetRGB(i, j))
		}
	}
}

func (im Image) Crop(sx int, sy int, ex int, ey int) Image {
	other := NewImage(ex-sx, ey-sy)
	for i := 0; i < other.Width; i++ {
		for j := 0; j < other.Height; j++ {
			other.SetRGB(i, j, im.GetRGB(sx+i, sy+j))
		}
	}
	return other
}

// for image.Image

func (im Image) Set(i int, j int, c color.Color) {
	r, g, b, _ := c.RGBA()
	r = r >> 8
	g = g >> 8
	b = b >> 8
	im.SetRGB(i, j, [3]uint8{uint8(r), uint8(g), uint8(b)})
}

func (im Image) At(i int, j int) color.Color {
	c := im.GetRGB(i, j)
	return color.RGBA{c[0], c[1], c[2], 255}
}

func (im Image) ColorModel() color.Model {
	return color.RGBAModel
}

func (im Image) Bounds() image.Rectangle {
	return image.Rectangle{image.Point{0, 0}, image.Point{im.Width, im.Height}}
}
