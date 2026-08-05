#!/usr/bin/env python3
"""Materialize the reviewed native OLMoE acceptance contract.

This script is idempotent and tied to explicit source anchors. It fails closed
when the source revision no longer matches the reviewed contract.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8", newline="\n")


def replace_once_or_present(
    text: str, old: str, new: str, marker: str, label: str
) -> str:
    if marker in text:
        return text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one replacement, found {count}")
    return text.replace(old, new)


def replace_region_or_present(
    text: str,
    start: str,
    end: str,
    replacement: str,
    marker: str,
    label: str,
) -> str:
    if marker in text:
        return text
    start_count = text.count(start)
    end_count = text.count(end)
    if start_count != 1 or end_count != 1:
        raise SystemExit(
            f"{label}: expected unique anchors, found start={start_count} end={end_count}"
        )
    i = text.index(start)
    j = text.index(end, i)
    return text[:i] + replacement + text[j:]


path = "c/olmoe.c"
text = read(path)

text = replace_once_or_present(
    text,
    " *   EXPERT_DROP=0/1: 1=fadvise(DONTNEED) after each expert read (old behaviour,\n"
    " *                    for RAM-tight boxes); 0=keep pages cached (default)\n",
    " *   EXPERT_DROP=0/1: 1=fadvise(DONTNEED) after each expert read (old behaviour,\n"
    " *                    for RAM-tight boxes); 0=keep pages cached (default)\n"
    " *   OLMOE_IGNORE_PERSISTED_PINS=0/1: 1=do not load or save hot_pinned.bin\n",
    "OLMOE_IGNORE_PERSISTED_PINS=0/1",
    "environment documentation",
)
text = replace_once_or_present(
    text,
    "#include <math.h>\n#include <time.h>\n",
    "#include <math.h>\n#include <time.h>\n#include <limits.h>\n#include <stdint.h>\n",
    "#include <limits.h>",
    "bounded integer includes",
)
text = replace_once_or_present(
    text,
    "static int g_pilot = 0;\n",
    "static int g_pilot = 0;\nstatic int g_ignore_persisted_pins = 0;\n",
    "static int g_ignore_persisted_pins = 0;",
    "persisted pin global",
)

load_block = """    // Persistent pin state is outside acceptance identity unless explicitly enabled.
    if (!g_ignore_persisted_pins) {
        char pinpath[512];
        snprintf(pinpath, sizeof(pinpath), "%s/hot_pinned.bin", snap);
        FILE *pinf = fopen(pinpath, "rb");
        if (pinf) {
            size_t expected_size = (size_t)c->n_layers * c->n_experts;
            size_t got = fread(m->is_pinned, 1, expected_size, pinf);
            fclose(pinf);
            if (got == expected_size) {
                m->hot_pinned = 1;
                printf("[HOT] Loaded persistent pinning from %s\\n", pinpath);

                if (g_pilot) {
                    ensure_pilot_worker_started(m);
                    for (int l = 0; l < c->n_layers; l++) {
                        for (int e = 0; e < c->n_experts; e++) {
                            if (m->is_pinned[l * c->n_experts + e]) {
                                unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_RELAXED);
                                unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                                if (w - r < 4096) {
                                    pilot_q[w & 4095].l = l;
                                    pilot_q[w & 4095].e = e;
                                    pthread_mutex_lock(&g_pilot_mx);
                                    m->is_queued[l * c->n_experts + e] = 1;
                                    pthread_mutex_unlock(&g_pilot_mx);
                                    __atomic_store_n(&pilot_w, w + 1, __ATOMIC_RELEASE);
                                }
                            }
                        }
                    }
                    printf("[HOT] Pre-loading pinned experts into cache...\\n");
                    double t_wait = now_s();
                    while (1) {
                        unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                        unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_ACQUIRE);
                        if (r == w) break;
                        sleep_ms(2);
                    }
                    printf("[HOT] Pre-loaded in %.1fs!\\n", now_s() - t_wait);
                }
            } else {
                memset(m->is_pinned, 0, expected_size);
                printf("[HOT] Warning: invalid pin file size (got %zu, expected %zu)\\n",
                       got, expected_size);
            }
        }
    } else {
        printf("OLMOE_PERSISTED_PINS_DISABLED\\n");
    }
"""
text = replace_region_or_present(
    text,
    "    // Persistent Hot Pinning: try to load hot_pinned.bin\n",
    "}\n\nstatic void slot_ensure_allocated",
    load_block,
    "OLMOE_PERSISTED_PINS_DISABLED",
    "persistent pin load",
)

reader_block = """#define OLMOE_MAX_REFERENCE_BYTES (4u * 1024u * 1024u)
#define OLMOE_MAX_TOTAL_TOKENS 2048
#define OLMOE_MAX_NEW_TOKENS 512

static int *read_int_array(jval *o, const char *key, int *n_out) {
    *n_out = 0;
    if (!o || o->t != J_OBJ) return NULL;
    jval *a = json_get(o, key);
    if (!a || a->t != J_ARR || a->len < 1 || a->len > OLMOE_MAX_TOTAL_TOKENS) return NULL;
    if ((size_t)a->len > SIZE_MAX / sizeof(int)) return NULL;
    int *r = calloc((size_t)a->len, sizeof(int));
    if (!r) return NULL;
    for (int i = 0; i < a->len; i++) {
        jval *item = a->kids ? a->kids[i] : NULL;
        if (!item || item->t != J_NUM || !isfinite(item->num)
                || item->num < 0.0 || item->num > (double)INT_MAX
                || floor(item->num) != item->num) {
            free(r);
            return NULL;
        }
        r[i] = (int)item->num;
    }
    *n_out = a->len;
    return r;
}
"""
text = replace_region_or_present(
    text,
    "static int *read_int_array(jval *o, const char *key, int *n_out) {\n",
    "\nint main(int argc, char **argv) {",
    reader_block,
    "#define OLMOE_MAX_REFERENCE_BYTES",
    "bounded token array parser",
)

text = replace_once_or_present(
    text,
    '    g_pilot = getenv("PILOT") ? atoi(getenv("PILOT")) : 0;\n',
    '    g_pilot = getenv("PILOT") ? atoi(getenv("PILOT")) : 0;\n'
    '    g_ignore_persisted_pins = getenv("OLMOE_IGNORE_PERSISTED_PINS")\n'
    '        ? atoi(getenv("OLMOE_IGNORE_PERSISTED_PINS")) != 0 : 0;\n',
    'g_ignore_persisted_pins = getenv("OLMOE_IGNORE_PERSISTED_PINS")',
    "persisted pin environment",
)
text = replace_once_or_present(
    text,
    '    int cap    = argc > 1 ? atoi(argv[1]) : 16;\n'
    '    int bits   = argc > 2 ? atoi(argv[2]) : 8;\n'
    '    if (bits < 2 || bits > 8) {\n',
    '    int cap    = argc > 1 ? atoi(argv[1]) : 16;\n'
    '    int bits   = argc > 2 ? atoi(argv[2]) : 8;\n'
    '    if (cap < 1 || cap > 512) {\n'
    '        fprintf(stderr, "cache_cap must be 1..512 (got %d)\\n", cap);\n'
    '        return 1;\n'
    '    }\n'
    '    if (bits < 2 || bits > 8) {\n',
    "cache_cap must be 1..512",
    "positive cache capacity",
)

reference_block = """    FILE *f = fopen(refpath, "rb");
    if (!f) { perror(refpath); return 1; }
    if (fseek(f, 0, SEEK_END) != 0) { perror("fseek"); fclose(f); return 1; }
    long n = ftell(f);
    if (n <= 0 || (unsigned long)n > OLMOE_MAX_REFERENCE_BYTES) {
        fprintf(stderr, "reference JSON must be 1..%u bytes (got %ld)\\n",
                OLMOE_MAX_REFERENCE_BYTES, n);
        fclose(f);
        return 1;
    }
    if (fseek(f, 0, SEEK_SET) != 0) { perror("fseek"); fclose(f); return 1; }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fprintf(stderr, "reference allocation failed\\n"); fclose(f); return 1; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n || ferror(f)) {
        fprintf(stderr, "reference JSON read failed\\n");
        free(buf);
        fclose(f);
        return 1;
    }
    buf[n] = 0;
    fclose(f);
    char *arena = NULL;
    jval *ref = json_parse(buf, &arena);
    int np = 0, nfull = 0;
    int *prompt = read_int_array(ref, "prompt_ids", &np);
    int *full = read_int_array(ref, "full_ids", &nfull);
    if (!ref || !prompt || !full || np < 1 || nfull <= np
            || nfull > OLMOE_MAX_TOTAL_TOKENS || nfull - np > OLMOE_MAX_NEW_TOKENS) {
        fprintf(stderr, "reference JSON contains an invalid or oversized token path\\n");
        free(prompt);
        free(full);
        free(buf);
        free(arena);
        return 1;
    }
    for (int i = 0; i < np; i++) {
        if (full[i] != prompt[i]) {
            fprintf(stderr, "reference full_ids must begin with prompt_ids\\n");
            free(prompt);
            free(full);
            free(buf);
            free(arena);
            return 1;
        }
    }
    int n_new = nfull - np;

    Model m; model_init(&m, snap, cap, bits);
    for (int i = 0; i < nfull; i++) {
        if (full[i] < 0 || full[i] >= m.c.vocab) {
            fprintf(stderr, "reference token %d is outside vocab=%d\\n", full[i], m.c.vocab);
            free(prompt);
            free(full);
            free(buf);
            free(arena);
            return 1;
        }
    }
"""
text = replace_region_or_present(
    text,
    '    FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }\n',
    '    printf("resident weights loaded in %.1fs | RSS after load: %.2f GB\\n", m.dense_load_s, rss_gb());',
    reference_block,
    "OLMOE_MAX_REFERENCE_BYTES, n",
    "bounded reference parser",
)

text = replace_once_or_present(
    text,
    "        free(buf); free(arena);\n        return 0;\n    }\n\n"
    "    int *out = malloc((np + n_new) * sizeof(int));\n",
    "        free(prompt); free(full); free(buf); free(arena);\n"
    "        return 0;\n"
    "    }\n\n"
    "    int *out = calloc((size_t)nfull, sizeof(int));\n"
    "    if (!out) {\n"
    '        fprintf(stderr, "generation token allocation failed\\n");\n'
    "        free(prompt); free(full); free(buf); free(arena);\n"
    "        return 1;\n"
    "    }\n",
    "generation token allocation failed",
    "bounded output allocation",
)
text = replace_once_or_present(
    text,
    "    if (m.hot_pinned) {\n",
    "    if (!g_ignore_persisted_pins && m.hot_pinned) {\n",
    "if (!g_ignore_persisted_pins && m.hot_pinned) {",
    "persistent pin save guard",
)
text = replace_once_or_present(
    text,
    "    free(buf); free(arena);\n    return 0;\n}\n",
    "    free(out); free(prompt); free(full); free(buf); free(arena);\n"
    "    return 0;\n"
    "}\n",
    "free(out); free(prompt); free(full); free(buf); free(arena);",
    "reference cleanup",
)
write(path, text)

make_path = "c/Makefile"
make = read(make_path)
make = replace_once_or_present(
    make,
    "check:\n\t$(MAKE) clean\n\t$(MAKE) portable\n\t$(MAKE) test\n",
    "check:\n\t$(MAKE) clean\n\t$(MAKE) portable olmoe$(EXE)\n\t$(MAKE) test\n",
    "$(MAKE) portable olmoe$(EXE)",
    "check builds OLMoE",
)
write(make_path, make)

required = (
    "OLMOE_IGNORE_PERSISTED_PINS",
    "OLMOE_PERSISTED_PINS_DISABLED",
    "cache_cap must be 1..512",
    "#define OLMOE_MAX_REFERENCE_BYTES",
    "if (!g_ignore_persisted_pins && m.hot_pinned) {",
)
materialized = read(path)
missing = [marker for marker in required if marker not in materialized]
if missing:
    raise SystemExit(f"materialization incomplete: {missing}")

print("native OLMoE contract materialized")
