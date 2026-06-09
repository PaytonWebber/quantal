/* quantal: two-stage cascading vector index.
 *
 * Stage 1 routes through an HNSW-style graph over 1-bit sign vectors;
 * Stage 2 reranks candidates with 3-bit TurboQuant payloads.
 *
 * The vector dimension is fixed at library build time (`zig build -Dc-dim=N`,
 * default 1536); query it at runtime with quantal_dim().
 */

#ifndef QUANTAL_H
#define QUANTAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct quantal_index quantal_index;
typedef struct quantal_context quantal_context;

/* The vector dimension this library was compiled for. */
size_t quantal_dim(void);

/* The SimHash routing-code length (>= quantal_dim(); larger on low-dim builds). */
size_t quantal_routing_bits(void);

/* Creates an empty index. ef_construction controls build-time beam width
 * (e.g. 2x the graph degree); seed parameterizes the random rotation.
 * Returns NULL on allocation failure. */
quantal_index *quantal_index_create(size_t ef_construction, uint64_t seed);
void quantal_index_destroy(quantal_index *index);

/* Adds a vector of quantal_dim() floats under the given id.
 * Returns 0 on success; -2 if the id is already present, -3 on dimension
 * mismatch, -1 on any other failure (e.g. out of memory). */
int32_t quantal_index_add(quantal_index *index, uint64_t id, const float *coords);

/* Multi-threaded bulk ingest of n vectors (row-major, n*quantal_dim() floats). */
int32_t quantal_index_add_batch(quantal_index *index, const uint64_t *ids,
                           const float *coords, size_t n, size_t threads);

/* Tombstones a vector by id in O(1). Returns 0 on success, -1 if unknown. */
int32_t quantal_index_remove(quantal_index *index, uint64_t id);

/* Number of live (non-removed) vectors. */
size_t quantal_index_len(const quantal_index *index);

/* Exact heap bytes owned by the index (payloads, routing graph, rerank
 * store, bookkeeping) — precise accounting, not process RSS. */
size_t quantal_index_memory_bytes(const quantal_index *index);

/* Persistence (.tq format). quantal_index_load returns NULL on failure. */
int32_t quantal_index_save(const quantal_index *index, const char *path);
quantal_index *quantal_index_load(const char *path);

/* Creates a reusable search context sized for the index's current contents;
 * recreate it after further inserts. m is the stage-1 candidate count
 * (e.g. 64). Returns NULL on allocation failure. */
quantal_context *quantal_context_create(const quantal_index *index, size_t m);
void quantal_context_destroy(quantal_context *ctx);

/* Writes up to k results (k capped at 256) into out_ids/out_scores, sorted
 * by descending inner-product score. Returns the result count.
 * Performs no allocation. */
size_t quantal_search(const quantal_index *index, quantal_context *ctx, const float *query,
                 size_t k, uint64_t *out_ids, float *out_scores);

/* Restricts results to an allowlist of ids (exact scoring over the
 * allowlist; unknown/removed ids are skipped). */
size_t quantal_search_filtered(const quantal_index *index, quantal_context *ctx,
                          const float *query, const uint64_t *allowed,
                          size_t allowed_len, size_t k, uint64_t *out_ids,
                          float *out_scores);

/* Multi-threaded batch search over n queries (row-major). out_ids and
 * out_scores hold n*k slots, out_counts n entries. Returns 0 on success. */
int32_t quantal_search_batch(const quantal_index *index, const float *queries, size_t n,
                        size_t k, size_t m, size_t threads, uint64_t *out_ids,
                        float *out_scores, size_t *out_counts);

#ifdef __cplusplus
}
#endif

#endif /* QUANTAL_H */
