#ifndef LATTICE_SIGNATURE_H
#define LATTICE_SIGNATURE_H

#include <stddef.h>
#include <stdint.h>

#include "lattice/scheduler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lt_signature lt_signature_t;

/* A signature records request-local expert activations during prefill. It is
 * placement-only telemetry: predictions may change latency but never routing
 * decisions or model arithmetic. */
lt_signature_t *lt_signature_create(uint32_t layers, uint32_t experts);
void lt_signature_destroy(lt_signature_t *signature);
void lt_signature_reset(lt_signature_t *signature);

int lt_signature_observe(lt_signature_t *signature,
                         uint32_t layer,
                         const uint32_t *expert_ids,
                         const double *weights,
                         size_t count);

/* Multiply all evidence by factor in [0,1]. Useful when a signature is reused
 * across turns while allowing old evidence to fade. */
int lt_signature_decay(lt_signature_t *signature, double factor);

/* Deterministic highest-mass experts. Returns number written. Scores are
 * normalized probabilities for the requested layer. */
size_t lt_signature_topk(const lt_signature_t *signature,
                         uint32_t layer,
                         size_t k,
                         uint32_t *out_expert_ids,
                         double *out_scores);

/* Submit top candidates to the three-path scheduler. Tensor IDs are generated
 * as tensor_id_base + layer * experts + expert_id. Existing demand requests
 * are deduplicated/promoted by the scheduler. Returns submissions attempted,
 * or -1 on invalid arguments/scheduler failure. */
int lt_signature_schedule(const lt_signature_t *signature,
                          lt_scheduler_t *scheduler,
                          uint32_t layer,
                          size_t k,
                          uint64_t tensor_id_base,
                          uint64_t bytes_each,
                          uint64_t deadline_ns,
                          lt_transfer_path_t path,
                          double min_confidence,
                          char *error,
                          size_t error_cap);

uint32_t lt_signature_layers(const lt_signature_t *signature);
uint32_t lt_signature_experts(const lt_signature_t *signature);
double lt_signature_layer_mass(const lt_signature_t *signature, uint32_t layer);

/* Portable sparse text format. Loading requires matching geometry and fails
 * closed on malformed or out-of-range rows. */
int lt_signature_save(const lt_signature_t *signature,
                      const char *path,
                      char *error,
                      size_t error_cap);
int lt_signature_load(lt_signature_t *signature,
                      const char *path,
                      char *error,
                      size_t error_cap);

#ifdef __cplusplus
}
#endif

#endif /* LATTICE_SIGNATURE_H */
