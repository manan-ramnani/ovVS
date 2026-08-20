# ioVS Python

ctypes bindings over `libiovs`. Set `IOVS_LIBRARY` to `iovs.dll` / `libiovs.so`.

```python
import iovs
res = iovs.Resources()
idx = iovs.neighbors.brute_force.build(dataset, dim=d, resources=res)
nb, ds = idx.search(queries, k=10)
```

`build`/`search` accept NumPy arrays and DLPack exporters.

Filtered search:

```python
nb, ds = idx.search(queries, k=10, allow_list=[0, 2, 4, 6])
# or a precomputed bitset from iovs.bitset_from_allow_list(n, ids)
```

Metrics: `METRIC_L2`, `METRIC_INNER_PRODUCT`, `METRIC_COSINE`, `METRIC_HAMMING`, `METRIC_LP`.
IVF-PQ / IVF-RaBitQ support `serialize` / `extend`.
