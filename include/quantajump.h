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
 * Returns 0 on success, -1 on allocation failure. */
int32_t qj_index_add(qj_index *index, uint64_t id, const float *coords);
size_t qj_index_len(const qj_index *index);

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

#ifdef __cplusplus
}
#endif

#endif /* QUANTAJUMP_H */
