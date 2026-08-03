#include "lattice/signature.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct lt_signature {
    uint32_t layers;
    uint32_t experts;
    double *mass;
    double *layer_total;
};

static void lt_sig_error(char *error, size_t cap, const char *message) {
    if (!error || cap == 0) return;
    snprintf(error, cap, "%s", message ? message : "signature error");
}

static int lt_sig_size(uint32_t layers, uint32_t experts, size_t *out) {
    if (layers == 0 || experts == 0 ||
        (size_t)layers > SIZE_MAX / (size_t)experts) return -1;
    *out = (size_t)layers * (size_t)experts;
    return 0;
}

lt_signature_t *lt_signature_create(uint32_t layers, uint32_t experts) {
    lt_signature_t *signature;
    size_t count;
    if (lt_sig_size(layers, experts, &count) != 0) return NULL;
    signature = (lt_signature_t *)calloc(1, sizeof(*signature));
    if (!signature) return NULL;
    signature->mass = (double *)calloc(count, sizeof(*signature->mass));
    signature->layer_total = (double *)calloc(layers, sizeof(*signature->layer_total));
    if (!signature->mass || !signature->layer_total) {
        lt_signature_destroy(signature);
        return NULL;
    }
    signature->layers = layers;
    signature->experts = experts;
    return signature;
}

void lt_signature_destroy(lt_signature_t *signature) {
    if (!signature) return;
    free(signature->mass);
    free(signature->layer_total);
    free(signature);
}

void lt_signature_reset(lt_signature_t *signature) {
    size_t count;
    if (!signature || lt_sig_size(signature->layers, signature->experts, &count) != 0) return;
    memset(signature->mass, 0, count * sizeof(*signature->mass));
    memset(signature->layer_total, 0, signature->layers * sizeof(*signature->layer_total));
}

int lt_signature_observe(lt_signature_t *signature,
                         uint32_t layer,
                         const uint32_t *expert_ids,
                         const double *weights,
                         size_t count) {
    size_t i;
    if (!signature || layer >= signature->layers || (!expert_ids && count)) return -1;
    for (i = 0; i < count; ++i) {
        double value = weights ? weights[i] : 1.0;
        uint32_t expert = expert_ids[i];
        if (expert >= signature->experts || !isfinite(value) || value < 0.0) return -1;
    }
    for (i = 0; i < count; ++i) {
        double value = weights ? weights[i] : 1.0;
        size_t index = (size_t)layer * signature->experts + expert_ids[i];
        signature->mass[index] += value;
        signature->layer_total[layer] += value;
    }
    return 0;
}

int lt_signature_decay(lt_signature_t *signature, double factor) {
    size_t count, i;
    if (!signature || !isfinite(factor) || factor < 0.0 || factor > 1.0 ||
        lt_sig_size(signature->layers, signature->experts, &count) != 0) return -1;
    for (i = 0; i < count; ++i) signature->mass[i] *= factor;
    for (i = 0; i < signature->layers; ++i) signature->layer_total[i] *= factor;
    return 0;
}

size_t lt_signature_topk(const lt_signature_t *signature,
                         uint32_t layer,
                         size_t k,
                         uint32_t *out_expert_ids,
                         double *out_scores) {
    size_t written = 0;
    uint32_t expert;
    double total;
    if (!signature || layer >= signature->layers || k == 0 ||
        !out_expert_ids || !out_scores) return 0;
    if (k > signature->experts) k = signature->experts;
    total = signature->layer_total[layer];
    if (!(total > 0.0) || !isfinite(total)) return 0;

    while (written < k) {
        uint32_t best = 0;
        double best_mass = -1.0;
        int found = 0;
        for (expert = 0; expert < signature->experts; ++expert) {
            size_t j;
            double value = signature->mass[(size_t)layer * signature->experts + expert];
            int used = 0;
            for (j = 0; j < written; ++j) {
                if (out_expert_ids[j] == expert) {
                    used = 1;
                    break;
                }
            }
            if (used || value <= 0.0) continue;
            if (!found || value > best_mass || (value == best_mass && expert < best)) {
                best = expert;
                best_mass = value;
                found = 1;
            }
        }
        if (!found) break;
        out_expert_ids[written] = best;
        out_scores[written] = best_mass / total;
        ++written;
    }
    return written;
}

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
                          size_t error_cap) {
    uint32_t *ids;
    double *scores;
    size_t n, i;
    int attempted = 0;
    if (!signature || !scheduler || layer >= signature->layers || k == 0 ||
        bytes_each == 0 || !isfinite(min_confidence) || min_confidence < 0.0 ||
        min_confidence > 1.0) {
        lt_sig_error(error, error_cap, "invalid signature scheduling arguments");
        return -1;
    }
    if (k > signature->experts) k = signature->experts;
    ids = (uint32_t *)malloc(k * sizeof(*ids));
    scores = (double *)malloc(k * sizeof(*scores));
    if (!ids || !scores) {
        free(ids);
        free(scores);
        lt_sig_error(error, error_cap, "out of memory scheduling signature");
        return -1;
    }
    n = lt_signature_topk(signature, layer, k, ids, scores);
    for (i = 0; i < n; ++i) {
        lt_transfer_request_t request;
        int result;
        if (scores[i] < min_confidence) continue;
        memset(&request, 0, sizeof(request));
        if ((uint64_t)layer > (UINT64_MAX - tensor_id_base - ids[i]) / signature->experts) {
            free(ids);
            free(scores);
            lt_sig_error(error, error_cap, "tensor ID calculation overflow");
            return -1;
        }
        request.tensor_id = tensor_id_base + (uint64_t)layer * signature->experts + ids[i];
        if (request.tensor_id == 0) {
            free(ids);
            free(scores);
            lt_sig_error(error, error_cap, "tensor ID zero is reserved");
            return -1;
        }
        request.bytes = bytes_each;
        request.layer = layer;
        request.deadline_ns = deadline_ns;
        request.confidence = scores[i];
        request.path = path;
        request.opaque = ids[i];
        result = lt_scheduler_submit(scheduler, &request, error, error_cap);
        if (result < 0) {
            free(ids);
            free(scores);
            return -1;
        }
        ++attempted;
    }
    free(ids);
    free(scores);
    lt_sig_error(error, error_cap, "");
    return attempted;
}

uint32_t lt_signature_layers(const lt_signature_t *signature) {
    return signature ? signature->layers : 0;
}

uint32_t lt_signature_experts(const lt_signature_t *signature) {
    return signature ? signature->experts : 0;
}

double lt_signature_layer_mass(const lt_signature_t *signature, uint32_t layer) {
    if (!signature || layer >= signature->layers) return 0.0;
    return signature->layer_total[layer];
}

int lt_signature_save(const lt_signature_t *signature,
                      const char *path,
                      char *error,
                      size_t error_cap) {
    FILE *fp;
    uint32_t layer, expert;
    if (!signature || !path || !*path) {
        lt_sig_error(error, error_cap, "signature and output path are required");
        return -1;
    }
    fp = fopen(path, "wb");
    if (!fp) {
        char message[256];
        snprintf(message, sizeof(message), "cannot open %s: %s", path, strerror(errno));
        lt_sig_error(error, error_cap, message);
        return -1;
    }
    fprintf(fp, "LTSIG1\t%u\t%u\n", signature->layers, signature->experts);
    for (layer = 0; layer < signature->layers; ++layer) {
        for (expert = 0; expert < signature->experts; ++expert) {
            double value = signature->mass[(size_t)layer * signature->experts + expert];
            if (value > 0.0) fprintf(fp, "%u\t%u\t%.17g\n", layer, expert, value);
        }
    }
    if (fclose(fp) != 0) {
        lt_sig_error(error, error_cap, "failed to finish signature file");
        return -1;
    }
    lt_sig_error(error, error_cap, "");
    return 0;
}

int lt_signature_load(lt_signature_t *signature,
                      const char *path,
                      char *error,
                      size_t error_cap) {
    FILE *fp;
    char line[512];
    unsigned layers, experts;
    double *mass = NULL, *totals = NULL;
    size_t count;
    unsigned long line_number = 1;
    if (!signature || !path || !*path ||
        lt_sig_size(signature->layers, signature->experts, &count) != 0) {
        lt_sig_error(error, error_cap, "signature and input path are required");
        return -1;
    }
    fp = fopen(path, "rb");
    if (!fp) {
        char message[256];
        snprintf(message, sizeof(message), "cannot open %s: %s", path, strerror(errno));
        lt_sig_error(error, error_cap, message);
        return -1;
    }
    if (!fgets(line, sizeof(line), fp) ||
        sscanf(line, "LTSIG1\t%u\t%u", &layers, &experts) != 2 ||
        layers != signature->layers || experts != signature->experts) {
        fclose(fp);
        lt_sig_error(error, error_cap, "signature header or geometry mismatch");
        return -1;
    }
    mass = (double *)calloc(count, sizeof(*mass));
    totals = (double *)calloc(signature->layers, sizeof(*totals));
    if (!mass || !totals) {
        free(mass);
        free(totals);
        fclose(fp);
        lt_sig_error(error, error_cap, "out of memory loading signature");
        return -1;
    }
    while (fgets(line, sizeof(line), fp)) {
        unsigned layer, expert;
        double value;
        char trailing;
        ++line_number;
        if (sscanf(line, "%u\t%u\t%lf %c", &layer, &expert, &value, &trailing) != 3 ||
            layer >= signature->layers || expert >= signature->experts ||
            !isfinite(value) || value <= 0.0) {
            char message[256];
            snprintf(message, sizeof(message), "malformed signature row at line %lu", line_number);
            free(mass);
            free(totals);
            fclose(fp);
            lt_sig_error(error, error_cap, message);
            return -1;
        }
        mass[(size_t)layer * signature->experts + expert] += value;
        totals[layer] += value;
    }
    if (ferror(fp)) {
        free(mass);
        free(totals);
        fclose(fp);
        lt_sig_error(error, error_cap, "error reading signature file");
        return -1;
    }
    fclose(fp);
    memcpy(signature->mass, mass, count * sizeof(*mass));
    memcpy(signature->layer_total, totals, signature->layers * sizeof(*totals));
    free(mass);
    free(totals);
    lt_sig_error(error, error_cap, "");
    return 0;
}
