#pragma once

#include "export.h"
#include "status.h"
#include "version.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ovvsDevice {
  OVVS_DEVICE_AUTO = 0,
  OVVS_DEVICE_CPU = 1,
  OVVS_DEVICE_NPU = 2,
  OVVS_DEVICE_GPU = 3,
  OVVS_DEVICE_HETERO = 4
} ovvsDevice;

typedef enum ovvsPolicy {
  OVVS_POLICY_AUTO = 0,
  OVVS_POLICY_NPU_IF_FASTER = 1,
  OVVS_POLICY_GPU_IF_FASTER = 2,
  OVVS_POLICY_HETERO = 3,
  OVVS_POLICY_FORCE_NPU = 4,
  OVVS_POLICY_FORCE_GPU = 5,
  OVVS_POLICY_FORCE_CPU = 6
} ovvsPolicy;

typedef enum ovvsMetric {
  OVVS_METRIC_L2_EXPANDED = 0,
  OVVS_METRIC_L2_SQRT_EXPANDED = 1,
  OVVS_METRIC_INNER_PRODUCT = 2,
  OVVS_METRIC_COSINE_EXPANDED = 3,
  OVVS_METRIC_BITWISE_HAMMING = 4,
  OVVS_METRIC_LP_UNEXPANDED = 5
} ovvsMetric;

typedef enum ovvsDType {
  OVVS_DTYPE_F32 = 0,
  OVVS_DTYPE_F16 = 1,
  OVVS_DTYPE_I8 = 2,
  OVVS_DTYPE_U8 = 3,
  OVVS_DTYPE_F8E4M3 = 4,
  OVVS_DTYPE_F8E5M2 = 5,
  OVVS_DTYPE_F4E2M1 = 6
} ovvsDType;

typedef enum ovvsCagraBuildAlgo {
  OVVS_CAGRA_BUILD_NN_DESCENT = 0,
  OVVS_CAGRA_BUILD_IVF_PQ = 1,
  OVVS_CAGRA_BUILD_ITERATIVE = 2
} ovvsCagraBuildAlgo;

typedef struct ovvsResourcesImpl* ovvsResources_t;
typedef struct ovvsBruteForceIndexImpl* ovvsBruteForceIndex_t;
typedef struct ovvsIvfFlatIndexImpl* ovvsIvfFlatIndex_t;
typedef struct ovvsIvfPqIndexImpl* ovvsIvfPqIndex_t;
typedef struct ovvsIvfRabitqIndexImpl* ovvsIvfRabitqIndex_t;
typedef struct ovvsCagraIndexImpl* ovvsCagraIndex_t;
typedef struct ovvsNnDescentGraphImpl* ovvsNnDescentGraph_t;
typedef struct ovvsVamanaIndexImpl* ovvsVamanaIndex_t;
typedef struct ovvsScannIndexImpl* ovvsScannIndex_t;
typedef struct ovvsHnswIndexImpl* ovvsHnswIndex_t;
typedef struct ovvsKMeansModelImpl* ovvsKMeansModel_t;
typedef struct ovvsSlinkModelImpl* ovvsSlinkModel_t;
typedef struct ovvsSpectralModelImpl* ovvsSpectralModel_t;
typedef struct ovvsPcaModelImpl* ovvsPcaModel_t;
typedef struct ovvsPqModelImpl* ovvsPqModel_t;
typedef struct ovvsSqModelImpl* ovvsSqModel_t;
typedef struct ovvsBinaryQuantizerImpl* ovvsBinaryQuantizer_t;
typedef struct ovvsSpectralEmbedImpl* ovvsSpectralEmbed_t;
typedef struct ovvsBatcherImpl* ovvsBatcher_t;

OVVS_API const char* ovvsGetVersion(void);
OVVS_API const char* ovvsStatusString(ovvsStatus status);

OVVS_API ovvsStatus ovvsResourcesCreate(ovvsResources_t* res);
OVVS_API ovvsStatus ovvsResourcesDestroy(ovvsResources_t res);
OVVS_API ovvsStatus ovvsResourcesSetPolicy(ovvsResources_t res, ovvsPolicy policy);
OVVS_API ovvsStatus ovvsResourcesGetPolicy(ovvsResources_t res, ovvsPolicy* policy);
OVVS_API ovvsStatus ovvsResourcesNpuAvailable(ovvsResources_t res, int32_t* available);
OVVS_API ovvsStatus ovvsResourcesGpuAvailable(ovvsResources_t res, int32_t* available);
OVVS_API ovvsStatus ovvsResourcesSku(ovvsResources_t res, char* buf, int32_t len);
OVVS_API ovvsStatus ovvsResourcesNpuCompileFails(ovvsResources_t res, int32_t* count);
OVVS_API ovvsStatus ovvsResourcesNpuFallbacks(ovvsResources_t res, int32_t* count);
OVVS_API ovvsStatus ovvsResourcesLastDevice(ovvsResources_t res, ovvsDevice* device);
OVVS_API ovvsStatus ovvsResourcesLastComputeDtype(ovvsResources_t res, ovvsDType* dtype);
/* Resource-local cumulative counters for completed GPU CAGRA walk calls. One call may contain a
   query batch. A direct call requires both index buffers to be GPU-accessible. Upload calls count
   invocations that upload either the dataset or graph; upload bytes sum only those index buffers,
   excluding query, output, and bitset transfers. Counters saturate at INT64_MAX. */
OVVS_API ovvsStatus ovvsResourcesCagraTransferStats(ovvsResources_t res, int64_t* walks,
                                                    int64_t* direct_walks,
                                                    int64_t* upload_calls,
                                                    int64_t* upload_bytes);
OVVS_API ovvsStatus ovvsResourcesSetNpuBusy(ovvsResources_t res, int32_t busy);
OVVS_API int32_t ovvsSyclEnabled(void);
/* Package energy in microjoules from Linux RAPL sysfs, Windows EMI (intelppm RAPL),
   Energy Meter PDH, or Intel Power Gadget. Else UNSUPPORTED. `res` may be null. */
OVVS_API ovvsStatus ovvsResourcesEnergyUj(ovvsResources_t res, int64_t* uj);

OVVS_API void ovvsShaveTopkSmallest(const float* scores, int32_t cols, int32_t k, int32_t* idx,
                                    float* val);
OVVS_API float ovvsShavePqAdc(const float* tables, const uint8_t* code, int32_t pq_m, int32_t ks);
OVVS_API ovvsStatus ovvsPqAdcBatch(ovvsResources_t res, const float* tables, int32_t pq_m, int32_t ks,
                                   const uint8_t* codes, int64_t ncodes, float* out);

OVVS_API ovvsStatus ovvsProbeJson(char* buf, int32_t len);

/* Primitives — shipped entry points used by tests and algorithms. */
OVVS_API ovvsStatus ovvsGemm(ovvsResources_t res, const float* a, const float* b, float* c,
                             int64_t m, int64_t n, int64_t k, int32_t trans_b);
/* Host buffers stay fp32. `compute` is the device math type (f16/i8/f8/f4). */
OVVS_API ovvsStatus ovvsGemmEx(ovvsResources_t res, const float* a, const float* b, float* c,
                               int64_t m, int64_t n, int64_t k, int32_t trans_b, ovvsDType compute);
OVVS_API ovvsStatus ovvsTopk(ovvsResources_t res, const float* scores, int64_t rows, int64_t cols,
                             int64_t k, int64_t* indices, float* values, int32_t largest);
OVVS_API ovvsStatus ovvsGatherRows(ovvsResources_t res, const float* src, int64_t src_rows,
                                   int64_t dim, const int64_t* idx, int64_t nidx, float* out);
OVVS_API ovvsStatus ovvsPairwiseDistance(ovvsResources_t res, ovvsMetric metric, const float* x,
                                         int64_t nx, const float* y, int64_t ny, int64_t dim,
                                         float* out, float metric_arg);
OVVS_API ovvsStatus ovvsKSelection(ovvsResources_t res, const float* scores, int64_t rows,
                                   int64_t cols, int64_t k, int64_t* indices, float* values,
                                   int32_t largest);

/* Brute-force */
OVVS_API ovvsStatus ovvsBruteForceBuild(ovvsResources_t res, const float* dataset, int64_t n,
                                        int64_t dim, ovvsMetric metric, ovvsBruteForceIndex_t* index);
OVVS_API ovvsStatus ovvsBruteForceBuildTyped(ovvsResources_t res, const void* dataset, int64_t n,
                                             int64_t dim, ovvsMetric metric, ovvsDType dtype,
                                             ovvsBruteForceIndex_t* index);
OVVS_API ovvsStatus ovvsBruteForceSearch(ovvsResources_t res, ovvsBruteForceIndex_t index,
                                         const float* queries, int64_t nq, int64_t k,
                                         const uint8_t* bitset, int64_t* neighbors, float* distances);
OVVS_API ovvsStatus ovvsBruteForceDestroy(ovvsBruteForceIndex_t index);

/* IVF-Flat */
OVVS_API ovvsStatus ovvsIvfFlatBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                     ovvsMetric metric, int32_t nlist, ovvsIvfFlatIndex_t* index);
OVVS_API ovvsStatus ovvsIvfFlatSearch(ovvsResources_t res, ovvsIvfFlatIndex_t index,
                                      const float* queries, int64_t nq, int64_t k, int32_t nprobe,
                                      const uint8_t* bitset, int64_t* neighbors, float* distances);
OVVS_API ovvsStatus ovvsIvfFlatSerialize(ovvsIvfFlatIndex_t index, const char* path);
OVVS_API ovvsStatus ovvsIvfFlatDeserialize(ovvsResources_t res, const char* path,
                                           ovvsIvfFlatIndex_t* index);
OVVS_API ovvsStatus ovvsIvfFlatExtend(ovvsResources_t res, ovvsIvfFlatIndex_t index, const float* extra,
                                      int64_t nextra);
OVVS_API ovvsStatus ovvsIvfFlatDestroy(ovvsIvfFlatIndex_t index);

/* IVF-PQ */
OVVS_API ovvsStatus ovvsIvfPqBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                   ovvsMetric metric, int32_t nlist, int32_t pq_m, int32_t pq_nbits,
                                   ovvsIvfPqIndex_t* index);
OVVS_API ovvsStatus ovvsIvfPqSearch(ovvsResources_t res, ovvsIvfPqIndex_t index, const float* queries,
                                    int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
                                    const uint8_t* bitset, int64_t* neighbors, float* distances);
OVVS_API ovvsStatus ovvsIvfPqSerialize(ovvsIvfPqIndex_t index, const char* path);
OVVS_API ovvsStatus ovvsIvfPqDeserialize(ovvsResources_t res, const char* path, ovvsIvfPqIndex_t* index);
OVVS_API ovvsStatus ovvsIvfPqExtend(ovvsResources_t res, ovvsIvfPqIndex_t index, const float* extra,
                                    int64_t nextra);
OVVS_API ovvsStatus ovvsIvfPqDestroy(ovvsIvfPqIndex_t index);

/* IVF-RaBitQ */
OVVS_API ovvsStatus ovvsIvfRabitqBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                       ovvsMetric metric, int32_t nlist, ovvsIvfRabitqIndex_t* index);
OVVS_API ovvsStatus ovvsIvfRabitqSearch(ovvsResources_t res, ovvsIvfRabitqIndex_t index,
                                        const float* queries, int64_t nq, int64_t k, int32_t nprobe,
                                        int32_t krefine, const uint8_t* bitset, int64_t* neighbors,
                                        float* distances);
OVVS_API ovvsStatus ovvsIvfRabitqSerialize(ovvsIvfRabitqIndex_t index, const char* path);
OVVS_API ovvsStatus ovvsIvfRabitqDeserialize(ovvsResources_t res, const char* path,
                                             ovvsIvfRabitqIndex_t* index);
OVVS_API ovvsStatus ovvsIvfRabitqExtend(ovvsResources_t res, ovvsIvfRabitqIndex_t index, const float* extra,
                                        int64_t nextra);
OVVS_API ovvsStatus ovvsIvfRabitqDestroy(ovvsIvfRabitqIndex_t index);

/* NN-Descent */
OVVS_API ovvsStatus ovvsNnDescentBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                       ovvsMetric metric, int32_t graph_degree, int32_t iterations,
                                       ovvsNnDescentGraph_t* graph);
OVVS_API ovvsStatus ovvsNnDescentNeighbors(ovvsNnDescentGraph_t graph, const int32_t** ids, int64_t* n,
                                           int32_t* degree);
OVVS_API ovvsStatus ovvsNnDescentDestroy(ovvsNnDescentGraph_t graph);

/* CAGRA */
OVVS_API ovvsStatus ovvsCagraBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                   ovvsMetric metric, int32_t graph_degree, int32_t intermediate_degree,
                                   ovvsCagraIndex_t* index);
OVVS_API ovvsStatus ovvsCagraBuildEx(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                     ovvsMetric metric, int32_t graph_degree, int32_t intermediate_degree,
                                     ovvsCagraBuildAlgo algo, ovvsCagraIndex_t* index);
OVVS_API ovvsStatus ovvsCagraSearch(ovvsResources_t res, ovvsCagraIndex_t index, const float* queries,
                                    int64_t nq, int64_t k, int32_t itopk_size, int32_t search_width,
                                    const uint8_t* bitset, int64_t* neighbors, float* distances);
OVVS_API ovvsStatus ovvsCagraQuantize(ovvsResources_t res, ovvsCagraIndex_t index, int32_t pq_m,
                                      int32_t pq_nbits);
OVVS_API ovvsStatus ovvsCagraDetachDataset(ovvsCagraIndex_t index);
OVVS_API ovvsStatus ovvsCagraAttachDataset(ovvsCagraIndex_t index, const float* dataset, int64_t n,
                                           int64_t dim);
OVVS_API ovvsStatus ovvsCagraSerialize(ovvsCagraIndex_t index, const char* path);
OVVS_API ovvsStatus ovvsCagraSerializeEx(ovvsCagraIndex_t index, const char* path, int32_t include_dataset);
OVVS_API ovvsStatus ovvsCagraDeserialize(ovvsResources_t res, const char* path, ovvsCagraIndex_t* index);
OVVS_API ovvsStatus ovvsCagraExtend(ovvsResources_t res, ovvsCagraIndex_t index, const float* extra,
                                    int64_t nextra);
OVVS_API ovvsStatus ovvsCagraDestroy(ovvsCagraIndex_t index);

/* HNSW from CAGRA */
OVVS_API ovvsStatus ovvsHnswFromCagra(ovvsResources_t res, ovvsCagraIndex_t cagra, ovvsHnswIndex_t* index);
OVVS_API ovvsStatus ovvsHnswSearch(ovvsResources_t res, ovvsHnswIndex_t index, const float* queries,
                                   int64_t nq, int64_t k, int32_t ef, int64_t* neighbors, float* distances);
OVVS_API ovvsStatus ovvsHnswSerialize(ovvsHnswIndex_t index, const char* path);
OVVS_API ovvsStatus ovvsHnswDeserialize(ovvsResources_t res, const char* path, ovvsHnswIndex_t* index);
OVVS_API ovvsStatus ovvsHnswDestroy(ovvsHnswIndex_t index);

/* Vamana */
OVVS_API ovvsStatus ovvsVamanaBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                    ovvsMetric metric, int32_t graph_degree, float alpha,
                                    ovvsVamanaIndex_t* index);
OVVS_API ovvsStatus ovvsVamanaSearch(ovvsResources_t res, ovvsVamanaIndex_t index, const float* queries,
                                     int64_t nq, int64_t k, int32_t beam, const uint8_t* bitset,
                                     int64_t* neighbors, float* distances);
OVVS_API ovvsStatus ovvsVamanaSerialize(ovvsVamanaIndex_t index, const char* path);
OVVS_API ovvsStatus ovvsVamanaDeserialize(ovvsResources_t res, const char* path, ovvsVamanaIndex_t* index);
OVVS_API ovvsStatus ovvsVamanaMmap(ovvsResources_t res, const char* path, ovvsVamanaIndex_t* index);
OVVS_API ovvsStatus ovvsVamanaDestroy(ovvsVamanaIndex_t index);

/* ScaNN-like */
OVVS_API ovvsStatus ovvsScannBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                   ovvsMetric metric, int32_t nlist, int32_t pq_m, ovvsScannIndex_t* index);
OVVS_API ovvsStatus ovvsScannSearch(ovvsResources_t res, ovvsScannIndex_t index, const float* queries,
                                    int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
                                    int64_t* neighbors, float* distances);
OVVS_API ovvsStatus ovvsScannDestroy(ovvsScannIndex_t index);

/* All-neighbors */
OVVS_API ovvsStatus ovvsAllNeighbors(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                     ovvsMetric metric, int32_t k, int64_t* neighbors, float* distances);

/* Allow-list → bitset of length (n+7)/8. Missing ids are filtered out (bit 0). */
OVVS_API ovvsStatus ovvsBitsetFromAllowList(int64_t n, const int64_t* ids, int64_t nids, uint8_t* bitset);

/* K-means */
OVVS_API ovvsStatus ovvsKMeansFit(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                  int32_t nclusters, int32_t iters, ovvsKMeansModel_t* model);
OVVS_API ovvsStatus ovvsKMeansPredict(ovvsResources_t res, ovvsKMeansModel_t model, const float* x,
                                      int64_t n, int64_t* labels, float* distances);
OVVS_API ovvsStatus ovvsKMeansCentroids(ovvsKMeansModel_t model, const float** data, int32_t* nclusters,
                                        int64_t* dim);
OVVS_API ovvsStatus ovvsKMeansDestroy(ovvsKMeansModel_t model);

/* SLINK */
OVVS_API ovvsStatus ovvsSlinkFit(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                 int32_t nclusters, int32_t knn, ovvsSlinkModel_t* model);
OVVS_API ovvsStatus ovvsSlinkLabels(ovvsSlinkModel_t model, const int64_t** labels, int64_t* n);
OVVS_API ovvsStatus ovvsSlinkDestroy(ovvsSlinkModel_t model);

/* Spectral */
OVVS_API ovvsStatus ovvsSpectralFit(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                    int32_t nclusters, int32_t knn, ovvsSpectralModel_t* model);
OVVS_API ovvsStatus ovvsSpectralLabels(ovvsSpectralModel_t model, const int64_t** labels, int64_t* n);
OVVS_API ovvsStatus ovvsSpectralDestroy(ovvsSpectralModel_t model);

/* Quantizers / PCA */
OVVS_API ovvsStatus ovvsSqFit(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                              ovvsSqModel_t* model);
OVVS_API ovvsStatus ovvsSqEncode(ovvsSqModel_t model, const float* x, int64_t n, uint8_t* codes);
OVVS_API ovvsStatus ovvsSqDecode(ovvsSqModel_t model, const uint8_t* codes, int64_t n, float* x);
OVVS_API ovvsStatus ovvsSqDestroy(ovvsSqModel_t model);

OVVS_API ovvsStatus ovvsPqFit(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                              int32_t pq_m, int32_t pq_nbits, ovvsPqModel_t* model);
OVVS_API ovvsStatus ovvsPqEncode(ovvsPqModel_t model, const float* x, int64_t n, uint8_t* codes);
OVVS_API ovvsStatus ovvsPqDecode(ovvsPqModel_t model, const uint8_t* codes, int64_t n, float* x);
OVVS_API ovvsStatus ovvsPqDestroy(ovvsPqModel_t model);

OVVS_API ovvsStatus ovvsBinaryFit(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                  ovvsBinaryQuantizer_t* model);
OVVS_API ovvsStatus ovvsBinaryEncode(ovvsBinaryQuantizer_t model, const float* x, int64_t n, uint8_t* codes);
OVVS_API ovvsStatus ovvsBinaryDestroy(ovvsBinaryQuantizer_t model);

OVVS_API ovvsStatus ovvsPcaFit(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                               int32_t ncomp, ovvsPcaModel_t* model);
OVVS_API ovvsStatus ovvsPcaTransform(ovvsPcaModel_t model, const float* x, int64_t n, float* out);
OVVS_API ovvsStatus ovvsPcaDestroy(ovvsPcaModel_t model);

/* Spectral embedding (preprocess); SpectralFit remains clustering on the embedding. */
OVVS_API ovvsStatus ovvsSpectralEmbedFit(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                         int32_t ncomp, int32_t knn, ovvsSpectralEmbed_t* model);
OVVS_API ovvsStatus ovvsSpectralEmbedData(ovvsSpectralEmbed_t model, const float** data, int64_t* n,
                                          int32_t* ncomp);
OVVS_API ovvsStatus ovvsSpectralEmbedDestroy(ovvsSpectralEmbed_t model);

/* Dynamic batcher over a brute-force index: coalesces small queries up to max_batch / max_wait_ms. */
OVVS_API ovvsStatus ovvsBatcherCreate(ovvsResources_t res, ovvsBruteForceIndex_t index, int32_t max_batch,
                                      int32_t max_wait_ms, ovvsBatcher_t* out);
OVVS_API ovvsStatus ovvsBatcherSearch(ovvsBatcher_t batcher, const float* queries, int64_t nq, int64_t k,
                                      int64_t* neighbors, float* distances);
OVVS_API ovvsStatus ovvsBatcherLastBatchSize(ovvsBatcher_t batcher, int32_t* nq);
OVVS_API ovvsStatus ovvsBatcherDestroy(ovvsBatcher_t batcher);

#ifdef __cplusplus
}
#endif
