package tensor

import "math"

func Add(a, b *Tensor) *Tensor {
	shape := broadcastShape(a.Shape, b.Shape)
	out := NewTensor(shape...)

	for i := 0; i < len(out.Data); i++ {
		aIndices := unflattenIndex(i, shape, a.Shape)
		bIndices := unflattenIndex(i, shape, b.Shape)

		aVal := a.At(aIndices...)
		bVal := b.At(bIndices...)
		out.Data[i] = aVal + bVal
	}
	return out
}

func Sub(a, b *Tensor) *Tensor {
	shape := broadcastShape(a.Shape, b.Shape)
	out := NewTensor(shape...)

	for i := 0; i < len(out.Data); i++ {
		aIndices := unflattenIndex(i, shape, a.Shape)
		bIndices := unflattenIndex(i, shape, b.Shape)

		aVal := a.At(aIndices...)
		bVal := b.At(bIndices...)
		out.Data[i] = aVal - bVal
	}
	return out
}

func Mul(a, b *Tensor) *Tensor {
	shape := broadcastShape(a.Shape, b.Shape)
	out := NewTensor(shape...)

	for i := 0; i < len(out.Data); i++ {
		aIndecies := unflattenIndex(i, shape, a.Shape)
		bIndicies := unflattenIndex(i, shape, b.Shape)

		aVal := a.At(aIndecies...)
		bVal := b.At(bIndicies...)
		out.Data[i] = aVal * bVal
	}
	return out
}

// Relu Activation functions#
func Relu(t *Tensor) *Tensor {
	out := t.Copy()
	for i, v := range out.Data {
		if v < 0 {
			out.Data[i] = 0
		}
	}
	return out
}

func Sigmoid(t *Tensor) *Tensor {
	out := t.Copy()
	for i, v := range out.Data {
		out.Data[i] = 1 / (1 + math.Exp(-v))
	}
	return out
}

func Softmax(t *Tensor) *Tensor {
	maxVal := t.Data[0]
	for _, v := range t.Data {
		if v > maxVal {
			maxVal = v
		}
	}
	expSum := 0.0
	expVals := make([]float64, len(t.Data))
	for i, v := range t.Data {
		expVals[i] = math.Exp(v - maxVal)
		expSum += expVals[i]
	}
	out := NewTensor(t.Shape...)
	for i := range expVals {
		out.Data[i] = expVals[i] / expSum
	}
	return out
}

// unflattenIndex maps a flat index to original tensor index using shape and original
func unflattenIndex(flatIdx int, targetShape, originalShape []int) []int {
	targetIdx := make([]int, len(targetShape))
	s := 1
	for i := len(targetShape) - 1; i >= 0; i-- {
		targetIdx[i] = (flatIdx / s) % targetShape[i]
		s *= targetShape[i]
	}
	// broadcast smaller shapes
	idx := make([]int, len(originalShape))
	dimOffset := len(targetShape) - len(originalShape)
	for i := range originalShape {
		if originalShape[i] == 1 {
			idx[i] = 0
		} else {
			idx[i] = targetIdx[i+dimOffset]
		}
	}
	return idx
}

func broadcastShape(a, b []int) []int {
	maxLen := max(len(a), len(b))
	result := make([]int, maxLen)
	for i := 0; i < maxLen; i++ {
		aDim := getDimFromRight(a, i)
		bDim := getDimFromRight(b, i)
		if aDim != bDim && aDim != 1 && bDim != 1 {
			panic("broadcast: incompatible shapes")
		}
		result[maxLen-1-i] = max(aDim, bDim)
	}
	return result
}
func getDimFromRight(shape []int, idx int) int {
	if idx >= len(shape) {
		return 1
	}
	return shape[len(shape)-1-idx]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
