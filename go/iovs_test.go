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

func l2sq(a, b []float32) float32 {
	var s float32
	for i := range a {
		t := a[i] - b[i]
		s += t * t
	}
	return s
}

func TestKMeans(t *testing.T) {
	n, dim := 20, 4
	data := make([]float32, n*dim)
	for i := range data {
		data[i] = float32((i*17)%100)/50 - 1
	}
	labels, dist, err := KMeansPredict(data, n, dim, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(labels) != n || len(dist) != n {
		t.Fatalf("shape %d %d", len(labels), len(dist))
	}
	t.Log("kmeans label0", labels[0])
}

func TestBruteForce(t *testing.T) {
	n, dim, k := 12, 4, 3
	data := make([]float32, n*dim)
	for i := range data {
		data[i] = float32((i*17)%100)/50 - 1
	}
	q := []float32{0.1, -0.2, 0.3, 0.0}
	nb, _, err := BruteSearch(data, n, dim, q, k)
	if err != nil {
		t.Fatal(err)
	}
	type pair struct {
		d float32
		i int64
	}
	ord := make([]pair, n)
	for i := 0; i < n; i++ {
		ord[i] = pair{l2sq(q, data[i*dim:(i+1)*dim]), int64(i)}
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if ord[j].d < ord[i].d {
				ord[i], ord[j] = ord[j], ord[i]
			}
		}
	}
	for tidx := 0; tidx < k; tidx++ {
		if nb[tidx] != ord[tidx].i {
			t.Fatalf("mismatch %d got %d expect %d", tidx, nb[tidx], ord[tidx].i)
		}
	}
}
