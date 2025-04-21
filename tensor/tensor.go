package tensor

import "fmt"

type Tensor struct {
	Shape   []int
	Strides []int
	Data    []float64
}

// NewTensor creates a new tensor with any dimension.
func NewTensor(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	strides := computeStrides(shape)

	return &Tensor{
		Shape:   shape,
		Strides: strides,
		Data:    data,
	}
}

// computeStrides calculate strides at n-dimensional indexing
func computeStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

// Index converts multidimensional to 1D offset.
func (t *Tensor) Index(indices ...int) int {
	if len(indices) != len(t.Shape) {
		{
			panic("Dimension mismatch")
		}
	}
	offset := 0
	for i, idx := range indices {
		offset += idx * t.Strides[i]
	}
	return offset
}

// At reads a value from the tensor.
func (t *Tensor) At(indices ...int) float64 {
	return t.Data[t.Index(indices...)]
}

// Fill fills the tensor with a constant value.
func (t *Tensor) Fill(val float64) {
	for i := range t.Data {
		t.Data[i] = val
	}
}

func (t *Tensor) Reshape(newShape ...int) {
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}
	if newSize != len(t.Data) {
		panic("reshape: size mismatch")
	}
	t.Shape = newShape
	t.Strides = computeStrides(newShape)
}

// Copy returns a deep copy
func (t *Tensor) Copy() *Tensor {
	newData := make([]float64, len(t.Data))
	copy(newData, t.Data)
	return &Tensor{
		Shape:   append([]int{}, t.Shape...),
		Strides: append([]int{}, t.Strides...),
		Data:    newData,
	}
}
func (t *Tensor) Print() {
	fmt.Printf("Shape: %v\n", t.Shape)
	fmt.Println("Data:")
	for i, val := range t.Data {
		fmt.Printf("[%d]%.3f\n", i, val)
	}
}
