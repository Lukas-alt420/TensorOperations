package layers

import (
	"awesomeTensoroperations/tensor"
)

type ReLu struct {
	Input  *tensor.Tensor
	Output *tensor.Tensor
}

func NewReLu() *ReLu {
	return &ReLu{}
}

func (r *ReLu) Forward(x *tensor.Tensor) *tensor.Tensor {
	r.Input = x.Copy()
	out := tensor.NewTensor(x.Shape...)
	for i, val := range x.Data {
		if val > 0 {
			out.Data[i] = val
		} else {
			out.Data[i] = 0
		}
	}
	r.Output = out
	return out
}
