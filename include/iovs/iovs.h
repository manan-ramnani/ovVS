#pragma once

#include "export.h"
#include "status.h"
#include "version.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum iovsDevice {
  IOVS_DEVICE_AUTO = 0,
  IOVS_DEVICE_CPU = 1,
  IOVS_DEVICE_NPU = 2,
  IOVS_DEVICE_GPU = 3,
  IOVS_DEVICE_HETERO = 4
} iovsDevice;

typedef enum iovsPolicy {
  IOVS_POLICY_AUTO = 0,
  IOVS_POLICY_NPU_IF_FASTER = 1,
  IOVS_POLICY_GPU_IF_FASTER = 2,
  IOVS_POLICY_HETERO = 3,
  IOVS_POLICY_FORCE_NPU = 4,
  IOVS_POLICY_FORCE_GPU = 5,
  IOVS_POLICY_FORCE_CPU = 6
} iovsPolicy;

typedef enum iovsMetric {
  IOVS_METRIC_L2_EXPANDED = 0,
  IOVS_METRIC_L2_SQRT_EXPANDED = 1,
  IOVS_METRIC_INNER_PRODUCT = 2,
  IOVS_METRIC_COSINE_EXPANDED = 3,
  IOVS_METRIC_BITWISE_HAMMING = 4,
  IOVS_METRIC_LP_UNEXPANDED = 5
} iovsMetric;

typedef enum iovsDType {
  IOVS_DTYPE_F32 = 0,
  IOVS_DTYPE_F16 = 1,
  IOVS_DTYPE_I8 = 2,
  IOVS_DTYPE_U8 = 3
} iovsDType;

typedef enum iovsCagraBuildAlgo {
  IOVS_CAGRA_BUILD_NN_DESCENT = 0,
  IOVS_CAGRA_BUILD_IVF_PQ = 1,
  IOVS_CAGRA_BUILD_ITERATIVE = 2
} iovsCagraBuildAlgo;

typedef struct iovsResourcesImpl* iovsResources_t;
typedef struct iovsBruteForceIndexImpl* iovsBruteForceIndex_t;
typedef struct iovsIvfFlatIndexImpl* iovsIvfFlatIndex_t;
typedef struct iovsIvfPqIndexImpl* iovsIvfPqIndex_t;
typedef struct iovsIvfRabitqIndexImpl* iovsIvfRabitqIndex_t;
typedef struct iovsCagraIndexImpl* iovsCagraIndex_t;
typedef struct iovsNnDescentGraphImpl* iovsNnDescentGraph_t;
typedef struct iovsVamanaIndexImpl* iovsVamanaIndex_t;
typedef struct iovsScannIndexImpl* iovsScannIndex_t;
typedef struct iovsHnswIndexImpl* iovsHnswIndex_t;
typedef struct iovsKMeansModelImpl* iovsKMeansModel_t;
typedef struct iovsSlinkModelImpl* iovsSlinkModel_t;
typedef struct iovsSpectralModelImpl* iovsSpectralModel_t;
typedef struct iovsPcaModelImpl* iovsPcaModel_t;
typedef struct iovsPqModelImpl* iovsPqModel_t;
typedef struct iovsSqModelImpl* iovsSqModel_t;
typedef struct iovsBinaryQuantizerImpl* iovsBinaryQuantizer_t;
typedef struct iovsSpectralEmbedImpl* iovsSpectralEmbed_t;
typedef struct iovsBatcherImpl* iovsBatcher_t;

IOVS_API const char* iovsGetVersion(void);
IOVS_API const char* iovsStatusString(iovsStatus status);

IOVS_API iovsStatus iovsResourcesCreate(iovsResources_t* res);
IOVS_API iovsStatus iovsResourcesDestroy(iovsResources_t res);
IOVS_API iovsStatus iovsResourcesSetPolicy(iovsResources_t res, iovsPolicy policy);
IOVS_API iovsStatus iovsResourcesGetPolicy(iovsResources_t res, iovsPolicy* policy);
IOVS_API iovsStatus iovsResourcesNpuAvailable(iovsResources_t res, int32_t* available);
IOVS_API iovsStatus iovsResourcesGpuAvailable(iovsResources_t res, int32_t* available);
IOVS_API iovsStatus iovsResourcesSku(iovsResources_t res, char* buf, int32_t len);
IOVS_API iovsStatus iovsResourcesNpuCompileFails(iovsResources_t res, int32_t* count);
IOVS_API iovsStatus iovsResourcesNpuFallbacks(iovsResources_t res, int32_t* count);
IOVS_API iovsStatus iovsResourcesLastDevice(iovsResources_t res, iovsDevice* device);
IOVS_API iovsStatus iovsResourcesSetNpuBusy(iovsResources_t res, int32_t busy);
IOVS_API int32_t iovsSyclEnabled(void);

IOVS_API void iovsShaveTopkSmallest(const float* scores, int32_t cols, int32_t k, int32_t* idx,
                                    float* val);
IOVS_API float iovsShavePqAdc(const float* tables, const uint8_t* code, int32_t pq_m, int32_t ks);

IOVS_API iovsStatus iovsProbeJson(char* buf, int32_t len);

/* Primitives — shipped entry points used by tests and algorithms. */
IOVS_API iovsStatus iovsGemm(iovsResources_t res, const float* a, const float* b, float* c,
                             int64_t m, int64_t n, int64_t k, int32_t trans_b);
IOVS_API iovsStatus iovsTopk(iovsResources_t res, const float* scores, int64_t rows, int64_t cols,
                             int64_t k, int64_t* indices, float* values, int32_t largest);
IOVS_API iovsStatus iovsGatherRows(iovsResources_t res, const float* src, int64_t src_rows,
                                   int64_t dim, const int64_t* idx, int64_t nidx, float* out);
IOVS_API iovsStatus iovsPairwiseDistance(iovsResources_t res, iovsMetric metric, const float* x,
                                         int64_t nx, const float* y, int64_t ny, int64_t dim,
                                         float* out, float metric_arg);
IOVS_API iovsStatus iovsKSelection(iovsResources_t res, const float* scores, int64_t rows,
                                   int64_t cols, int64_t k, int64_t* indices, float* values,
                                   int32_t largest);

/* Brute-force */
IOVS_API iovsStatus iovsBruteForceBuild(iovsResources_t res, const float* dataset, int64_t n,
                                        int64_t dim, iovsMetric metric, iovsBruteForceIndex_t* index);
IOVS_API iovsStatus iovsBruteForceBuildTyped(iovsResources_t res, const void* dataset, int64_t n,
                                             int64_t dim, iovsMetric metric, iovsDType dtype,
                                             iovsBruteForceIndex_t* index);
IOVS_API iovsStatus iovsBruteForceSearch(iovsResources_t res, iovsBruteForceIndex_t index,
                                         const float* queries, int64_t nq, int64_t k,
                                         const uint8_t* bitset, int64_t* neighbors, float* distances);
IOVS_API iovsStatus iovsBruteForceDestroy(iovsBruteForceIndex_t index);

/* IVF-Flat */
IOVS_API iovsStatus iovsIvfFlatBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                     iovsMetric metric, int32_t nlist, iovsIvfFlatIndex_t* index);
IOVS_API iovsStatus iovsIvfFlatSearch(iovsResources_t res, iovsIvfFlatIndex_t index,
                                      const float* queries, int64_t nq, int64_t k, int32_t nprobe,
                                      const uint8_t* bitset, int64_t* neighbors, float* distances);
IOVS_API iovsStatus iovsIvfFlatSerialize(iovsIvfFlatIndex_t index, const char* path);
IOVS_API iovsStatus iovsIvfFlatDeserialize(iovsResources_t res, const char* path,
                                           iovsIvfFlatIndex_t* index);
IOVS_API iovsStatus iovsIvfFlatExtend(iovsResources_t res, iovsIvfFlatIndex_t index, const float* extra,
                                      int64_t nextra);
IOVS_API iovsStatus iovsIvfFlatDestroy(iovsIvfFlatIndex_t index);

/* IVF-PQ */
IOVS_API iovsStatus iovsIvfPqBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                   iovsMetric metric, int32_t nlist, int32_t pq_m, int32_t pq_nbits,
                                   iovsIvfPqIndex_t* index);
IOVS_API iovsStatus iovsIvfPqSearch(iovsResources_t res, iovsIvfPqIndex_t index, const float* queries,
                                    int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
                                    const uint8_t* bitset, int64_t* neighbors, float* distances);
IOVS_API iovsStatus iovsIvfPqDestroy(iovsIvfPqIndex_t index);

/* IVF-RaBitQ */
IOVS_API iovsStatus iovsIvfRabitqBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                       iovsMetric metric, int32_t nlist, iovsIvfRabitqIndex_t* index);
IOVS_API iovsStatus iovsIvfRabitqSearch(iovsResources_t res, iovsIvfRabitqIndex_t index,
                                        const float* queries, int64_t nq, int64_t k, int32_t nprobe,
                                        int32_t krefine, const uint8_t* bitset, int64_t* neighbors,
                                        float* distances);
IOVS_API iovsStatus iovsIvfRabitqDestroy(iovsIvfRabitqIndex_t index);

/* NN-Descent */
IOVS_API iovsStatus iovsNnDescentBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                       iovsMetric metric, int32_t graph_degree, int32_t iterations,
                                       iovsNnDescentGraph_t* graph);
IOVS_API iovsStatus iovsNnDescentNeighbors(iovsNnDescentGraph_t graph, const int32_t** ids, int64_t* n,
                                           int32_t* degree);
IOVS_API iovsStatus iovsNnDescentDestroy(iovsNnDescentGraph_t graph);

/* CAGRA */
IOVS_API iovsStatus iovsCagraBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                   iovsMetric metric, int32_t graph_degree, int32_t intermediate_degree,
                                   iovsCagraIndex_t* index);
IOVS_API iovsStatus iovsCagraBuildEx(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                     iovsMetric metric, int32_t graph_degree, int32_t intermediate_degree,
                                     iovsCagraBuildAlgo algo, iovsCagraIndex_t* index);
IOVS_API iovsStatus iovsCagraSearch(iovsResources_t res, iovsCagraIndex_t index, const float* queries,
                                    int64_t nq, int64_t k, int32_t itopk_size, int32_t search_width,
                                    const uint8_t* bitset, int64_t* neighbors, float* distances);
IOVS_API iovsStatus iovsCagraQuantize(iovsResources_t res, iovsCagraIndex_t index, int32_t pq_m,
                                      int32_t pq_nbits);
IOVS_API iovsStatus iovsCagraDetachDataset(iovsCagraIndex_t index);
IOVS_API iovsStatus iovsCagraAttachDataset(iovsCagraIndex_t index, const float* dataset, int64_t n,
                                           int64_t dim);
IOVS_API iovsStatus iovsCagraSerialize(iovsCagraIndex_t index, const char* path);
IOVS_API iovsStatus iovsCagraSerializeEx(iovsCagraIndex_t index, const char* path, int32_t include_dataset);
IOVS_API iovsStatus iovsCagraDeserialize(iovsResources_t res, const char* path, iovsCagraIndex_t* index);
IOVS_API iovsStatus iovsCagraExtend(iovsResources_t res, iovsCagraIndex_t index, const float* extra,
                                    int64_t nextra);
IOVS_API iovsStatus iovsCagraDestroy(iovsCagraIndex_t index);

/* HNSW from CAGRA */
IOVS_API iovsStatus iovsHnswFromCagra(iovsResources_t res, iovsCagraIndex_t cagra, iovsHnswIndex_t* index);
IOVS_API iovsStatus iovsHnswSearch(iovsResources_t res, iovsHnswIndex_t index, const float* queries,
                                   int64_t nq, int64_t k, int32_t ef, int64_t* neighbors, float* distances);
IOVS_API iovsStatus iovsHnswSerialize(iovsHnswIndex_t index, const char* path);
IOVS_API iovsStatus iovsHnswDeserialize(iovsResources_t res, const char* path, iovsHnswIndex_t* index);
IOVS_API iovsStatus iovsHnswDestroy(iovsHnswIndex_t index);

/* Vamana */
IOVS_API iovsStatus iovsVamanaBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                    iovsMetric metric, int32_t graph_degree, float alpha,
                                    iovsVamanaIndex_t* index);
IOVS_API iovsStatus iovsVamanaSearch(iovsResources_t res, iovsVamanaIndex_t index, const float* queries,
                                     int64_t nq, int64_t k, int32_t beam, const uint8_t* bitset,
                                     int64_t* neighbors, float* distances);
IOVS_API iovsStatus iovsVamanaSerialize(iovsVamanaIndex_t index, const char* path);
IOVS_API iovsStatus iovsVamanaDeserialize(iovsResources_t res, const char* path, iovsVamanaIndex_t* index);
IOVS_API iovsStatus iovsVamanaMmap(iovsResources_t res, const char* path, iovsVamanaIndex_t* index);
IOVS_API iovsStatus iovsVamanaDestroy(iovsVamanaIndex_t index);

/* ScaNN-like */
IOVS_API iovsStatus iovsScannBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                   iovsMetric metric, int32_t nlist, int32_t pq_m, iovsScannIndex_t* index);
IOVS_API iovsStatus iovsScannSearch(iovsResources_t res, iovsScannIndex_t index, const float* queries,
                                    int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
                                    int64_t* neighbors, float* distances);
IOVS_API iovsStatus iovsScannDestroy(iovsScannIndex_t index);

/* All-neighbors */
IOVS_API iovsStatus iovsAllNeighbors(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                     iovsMetric metric, int32_t k, int64_t* neighbors, float* distances);

/* K-means */
IOVS_API iovsStatus iovsKMeansFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                  int32_t nclusters, int32_t iters, iovsKMeansModel_t* model);
IOVS_API iovsStatus iovsKMeansPredict(iovsResources_t res, iovsKMeansModel_t model, const float* x,
                                      int64_t n, int64_t* labels, float* distances);
IOVS_API iovsStatus iovsKMeansCentroids(iovsKMeansModel_t model, const float** data, int32_t* nclusters,
                                        int64_t* dim);
IOVS_API iovsStatus iovsKMeansDestroy(iovsKMeansModel_t model);

/* SLINK */
IOVS_API iovsStatus iovsSlinkFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                 int32_t nclusters, int32_t knn, iovsSlinkModel_t* model);
IOVS_API iovsStatus iovsSlinkLabels(iovsSlinkModel_t model, const int64_t** labels, int64_t* n);
IOVS_API iovsStatus iovsSlinkDestroy(iovsSlinkModel_t model);

/* Spectral */
IOVS_API iovsStatus iovsSpectralFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                    int32_t nclusters, int32_t knn, iovsSpectralModel_t* model);
IOVS_API iovsStatus iovsSpectralLabels(iovsSpectralModel_t model, const int64_t** labels, int64_t* n);
IOVS_API iovsStatus iovsSpectralDestroy(iovsSpectralModel_t model);

/* Quantizers / PCA */
IOVS_API iovsStatus iovsSqFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                              iovsSqModel_t* model);
IOVS_API iovsStatus iovsSqEncode(iovsSqModel_t model, const float* x, int64_t n, uint8_t* codes);
IOVS_API iovsStatus iovsSqDecode(iovsSqModel_t model, const uint8_t* codes, int64_t n, float* x);
IOVS_API iovsStatus iovsSqDestroy(iovsSqModel_t model);

IOVS_API iovsStatus iovsPqFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                              int32_t pq_m, int32_t pq_nbits, iovsPqModel_t* model);
IOVS_API iovsStatus iovsPqEncode(iovsPqModel_t model, const float* x, int64_t n, uint8_t* codes);
IOVS_API iovsStatus iovsPqDecode(iovsPqModel_t model, const uint8_t* codes, int64_t n, float* x);
IOVS_API iovsStatus iovsPqDestroy(iovsPqModel_t model);

IOVS_API iovsStatus iovsBinaryFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                  iovsBinaryQuantizer_t* model);
IOVS_API iovsStatus iovsBinaryEncode(iovsBinaryQuantizer_t model, const float* x, int64_t n, uint8_t* codes);
IOVS_API iovsStatus iovsBinaryDestroy(iovsBinaryQuantizer_t model);

IOVS_API iovsStatus iovsPcaFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                               int32_t ncomp, iovsPcaModel_t* model);
IOVS_API iovsStatus iovsPcaTransform(iovsPcaModel_t model, const float* x, int64_t n, float* out);
IOVS_API iovsStatus iovsPcaDestroy(iovsPcaModel_t model);

/* Spectral embedding (preprocess); SpectralFit remains clustering on the embedding. */
IOVS_API iovsStatus iovsSpectralEmbedFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                         int32_t ncomp, int32_t knn, iovsSpectralEmbed_t* model);
IOVS_API iovsStatus iovsSpectralEmbedData(iovsSpectralEmbed_t model, const float** data, int64_t* n,
                                          int32_t* ncomp);
IOVS_API iovsStatus iovsSpectralEmbedDestroy(iovsSpectralEmbed_t model);

/* Dynamic batcher over a brute-force index: coalesces small queries up to max_batch / max_wait_ms. */
IOVS_API iovsStatus iovsBatcherCreate(iovsResources_t res, iovsBruteForceIndex_t index, int32_t max_batch,
                                      int32_t max_wait_ms, iovsBatcher_t* out);
IOVS_API iovsStatus iovsBatcherSearch(iovsBatcher_t batcher, const float* queries, int64_t nq, int64_t k,
                                      int64_t* neighbors, float* distances);
IOVS_API iovsStatus iovsBatcherDestroy(iovsBatcher_t batcher);

#ifdef __cplusplus
}
#endif
