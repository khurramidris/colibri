# Optional BATS research binaries. Invoke from c/:
#   make -f Makefile.bats olmoe-bats
#
# Include the stock Makefile to inherit its platform/toolchain detection without
# changing the default build or release artifacts.
include Makefile

.PHONY: olmoe-bats

olmoe-bats: olmoe_bats$(EXE)

olmoe_bats$(EXE): olmoe_bats.c olmoe.c bats.h route_trace.h st.h json.h compat.h \
                  sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h
	$(CC) $(CFLAGS) olmoe_bats.c -o $@ $(LDFLAGS)
