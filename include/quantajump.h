/* quantajump: two-stage cascading vector index.
 *
 * Stage 1 routes through an HNSW-style graph over 1-bit sign vectors;
 * Stage 2 reranks candidates with 3-bit TurboQuant payloads.
 *
 * The vector dimension is fixed at library build time (`zig build -Dc-dim=N`,
 * default 1536); query it at runtime with qj_dim().
 */

#ifndef QUANTAJUMP_H
#define QUANTAJUMP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qj_index qj_index;
typedef struct qj_context qj_context;

/* The vector dimension this library was compiled for. */
size_t qj_dim(void);

/* Creates an empty index. ef_construction controls build-time beam width
 * (e.g. 2x the graph degree); seed parameterizes the random rotation.
 * Returns NULL on allocation failure. */
qj_index *qj_index_create(size_t ef_construction, uint64_t seed);
void qj_index_destroy(qj_index *index);

/* Adds a vector of qj_dim() floats under the given id.
 * Returns 0 on success, -1 on failure (allocation or duplicate id). */
int32_t qj_index_add(qj_index *index, uint64_t id, const float *coords);

/* Multi-threaded bulk ingest of n vectors (row-major, n*qj_dim() floats). */
int32_t qj_index_add_batch(qj_index *index, const uint64_t *ids,
                           const float *coords, size_t n, size_t threads);

/* Tombstones a vector by id in O(1). Returns 0 on success, -1 if unknown. */
int32_t qj_index_remove(qj_index *index, uint64_t id);

/* Number of live (non-removed) vectors. */
size_t qj_index_len(const qj_index *index);

/* Persistence (.tq format). qj_index_load returns NULL on failure. */
int32_t qj_index_save(const qj_index *index, const char *path);
qj_index *qj_index_load(const char *path);

/* Creates a reusable search context sized for the index's current contents;
 * recreate it after further inserts. m is the stage-1 candidate count
 * (e.g. 64). Returns NULL on allocation failure. */
qj_context *qj_context_create(const qj_index *index, size_t m);
void qj_context_destroy(qj_context *ctx);

/* Writes up to k results (k capped at 256) into out_ids/out_scores, sorted
 * by descending inner-product score. Returns the result count.
 * Performs no allocation. */
size_t qj_search(const qj_index *index, qj_context *ctx, const float *query,
                 size_t k, uint64_t *out_ids, float *out_scores);

/* Restricts results to an allowlist of ids (exact scoring over the
 * allowlist; unknown/removed ids are skipped). */
size_t qj_search_filtered(const qj_index *index, qj_context *ctx,
                          const float *query, const uint64_t *allowed,
                          size_t allowed_len, size_t k, uint64_t *out_ids,
                          float *out_scores);

/* Multi-threaded batch search over n queries (row-major). out_ids and
 * out_scores hold n*k slots, out_counts n entries. Returns 0 on success. */
int32_t qj_search_batch(const qj_index *index, const float *queries, size_t n,
                        size_t k, size_t m, size_t threads, uint64_t *out_ids,
                        float *out_scores, size_t *out_counts);

#ifdef __cplusplus
}
#endif

#endif /* QUANTAJUMP_H */
