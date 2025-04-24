package layers

import (
	"awesomeTensoroperations/tensor"
	"math/rand"
)

type Dense struct {
	Weights *tensor.Tensor
	Biases  *tensor.Tensor
	Input   *tensor.Tensor
	Output  *tensor.Tensor
}

func NewDense(inFeatures, outFeatures int) *Dense {
	w := tensor.NewTensor(outFeatures, inFeatures)
	b := tensor.NewTensor(outFeatures)
	for i := range w.Data {
		w.Data[i] = rand.NormFloat64() * 0.1
	}
	return &Dense{Weights: w, Biases: b}
}

func (d *Dense) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 1 || x.Shape[0] != d.Weights.Shape[1] {
		panic("input dimension mismatch")
	}
	d.Input = x.Copy()
	out := tensor.NewTensor(d.Weights.Shape[0])
	for i := 0; i < d.Weights.Shape[0]; i++ {
		sum := 0.0
		for j := 0; j < d.Weights.Shape[1]; j++ {
			sum += d.Weights.At(i, j) * x.At(j)
		}
		out.Set(sum+d.Biases.At(i), i)
	}
	d.Output = out
	return out
}
