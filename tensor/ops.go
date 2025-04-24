package tensor

// MatMul: Matrix-multiplication(2D Tensors only)
func MatMul(a, b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("MatMul: only 2D tensors supported")
	}
	if a.Shape[1] != b.Shape[0] {
		panic("MatMul: incompatible dimensions")
	}
	m, n, k := a.Shape[0], a.Shape[1], b.Shape[1]
	out := NewTensor(m, k)

	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			sum := 0.0
			for l := 0; l < n; l++ {
				sum += a.At(i, l) * b.At(l, j)
			}
			out.Set(sum, i, j)
		}
	}
	return out
}

func sameShape(a, b *Tensor) bool {
	if len(a.Shape) != len(b.Shape) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	return true
}
