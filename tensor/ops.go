package tensor

func Add(a, b *Tensor) *Tensor {
	if !sameShape(a, b) {
		panic("Add: tensor shape do not match")
	}
	out :=
}