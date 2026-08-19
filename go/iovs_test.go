package iovs

import "testing"

func TestVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Fatal("empty version")
	}
	t.Log(v)
}

func TestGemm(t *testing.T) {
	a := []float32{1, 0, 0, 1}
	b := []float32{2, 3, 4, 5}
	c, err := Gemm(a, b, 2, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(c) != 4 {
		t.Fatalf("len %d", len(c))
	}
}
