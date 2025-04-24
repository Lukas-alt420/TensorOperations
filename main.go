package main

import (
	"awesomeTensoroperations/tensor"
	"fmt"
)

func main() {
	a := tensor.NewTensor(2, 3)
	b := tensor.NewTensor(2, 3)
	a.Fill(1.0)
	a.Fill(2.0)

	sum := tensor.Add(a, b)
	mul := tensor.Mul(a, b)

	fmt.Println("A + B")
	sum.Print()

	fmt.Println("A + B:")
	mul.Print()

	x := tensor.NewTensor(2, 4)
	y := tensor.NewTensor(4, 3)
	x.Fill(1.0)
	y.Fill(0.5)

	out := tensor.MatMul(x, y)
	fmt.Println("MatMul:")
	out.Print()
}
