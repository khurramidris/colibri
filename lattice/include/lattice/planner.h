#ifndef LATTICE_PLANNER_H
#define LATTICE_PLANNER_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LT_NAME_MAX 96
#define LT_ERROR_MAX 256

/* Placement only changes where an exact tensor is served from. It must never
 * change tensor bytes, routing semantics, or arithmetic. Approximate policies
 * belong in a separate layer and are intentionally absent from this API. */
typedef enum {
    LT_ROLE_DENSE = 0,
    LT_ROLE_ATTENTION,
    LT_ROLE_SHARED_EXPERT,
    LT_ROLE_ROUTED_EXPERT,
    LT_ROLE_EMBEDDING,
    LT_ROLE_KV_CACHE,
    LT_ROLE_RECURRENT_STATE,
    LT_ROLE_NORM,
    LT_ROLE_OTHER
} lt_role_t;

typedef enum {
    LT_TIER_NVME = 0,
    LT_TIER_RAM,
    LT_TIER_VRAM
} lt_tier_t;

enum {
    LT_CLASS_STREAMABLE  = 1u << 0,
    LT_CLASS_GPU_CAPABLE = 1u << 1,
    LT_CLASS_RAM_REQUIRED= 1u << 2,
    LT_CLASS_VRAM_REQUIRED=1u << 3
};

/* A class represents count equally-sized tensors ordered hottest-first.
 * touches_per_token is the expected total number of touches across the class.
 * skew=0 is uniform. Larger skew concentrates probability in low ranks.
 * cpu_ms_per_touch and gpu_ms_per_touch are optional measured costs; zero means
 * unknown and the planner optimizes movement only. */
typedef struct {
    char name[LT_NAME_MAX];
    lt_role_t role;
    uint64_t bytes_each;
    uint32_t count;
    double touches_per_token;
    double skew;
    double cpu_ms_per_touch;
    double gpu_ms_per_touch;
    uint32_t min_ram_count;
    uint32_t min_vram_count;
    unsigned flags;
} lt_tensor_class_t;

typedef struct {
    uint64_t ram_budget_bytes;
    uint64_t vram_budget_bytes;
    double nvme_read_gbps;
    double ram_read_gbps;
    double host_to_device_gbps;
    double overlap_efficiency; /* 0=sum, 1=perfect overlap of transfer/compute */
    double reserve_fraction;   /* leave this fraction of RAM and VRAM unused */
} lt_hardware_t;

typedef struct {
    uint32_t vram_count;
    uint32_t ram_count;
    uint32_t nvme_count;
    uint64_t vram_bytes;
    uint64_t ram_bytes;
    uint64_t nvme_bytes;
    double hot_mass_vram;
    double hot_mass_ram;
    double hot_mass_nvme;
    double ram_score_per_byte;
    double vram_score_per_byte;
} lt_class_plan_t;

typedef struct {
    lt_class_plan_t *classes;
    size_t nclasses;
    uint64_t ram_used_bytes;
    uint64_t vram_used_bytes;
    uint64_t nvme_resident_bytes;
    double nvme_bytes_per_token;
    double h2d_bytes_per_token;
    double cpu_compute_ms_per_token;
    double gpu_compute_ms_per_token;
    double movement_ms_per_token;
    double compute_ms_per_token;
    double predicted_ms_per_token;
    int feasible;
    char error[LT_ERROR_MAX];
} lt_plan_t;

const char *lt_role_name(lt_role_t role);
int lt_role_parse(const char *text, lt_role_t *out);
const char *lt_tier_name(lt_tier_t tier);

/* Returns 0 on success. On failure, out is initialized with feasible=0 and a
 * human-readable error. The plan owns out->classes; call lt_plan_free. */
int lt_plan_build(const lt_hardware_t *hw,
                  const lt_tensor_class_t *classes,
                  size_t nclasses,
                  lt_plan_t *out);

int lt_plan_validate(const lt_hardware_t *hw,
                     const lt_tensor_class_t *classes,
                     const lt_plan_t *plan,
                     char *error,
                     size_t error_cap);

void lt_plan_free(lt_plan_t *plan);
void lt_plan_print(FILE *fp,
                   const lt_hardware_t *hw,
                   const lt_tensor_class_t *classes,
                   const lt_plan_t *plan);
int lt_plan_write_json(FILE *fp,
                       const lt_hardware_t *hw,
                       const lt_tensor_class_t *classes,
                       const lt_plan_t *plan);

/* Dependency-free TSV manifest parser. Format:
 * name<TAB>role<TAB>bytes_each<TAB>count<TAB>touches_per_token<TAB>skew
 *   <TAB>cpu_ms<TAB>gpu_ms<TAB>flags<TAB>min_ram<TAB>min_vram
 * Flags are comma separated: streamable,gpu,ram_required,vram_required.
 * Blank lines and lines beginning with # are ignored. */
int lt_manifest_load_tsv(const char *path,
                         lt_tensor_class_t **out_classes,
                         size_t *out_nclasses,
                         char *error,
                         size_t error_cap);
void lt_manifest_free(lt_tensor_class_t *classes);

#ifdef __cplusplus
}
#endif

#endif /* LATTICE_PLANNER_H */
