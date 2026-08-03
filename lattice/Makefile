CC ?= cc
AR ?= ar
CSTD ?= -std=c11
WARN ?= -Wall -Wextra -Wpedantic -Werror
OPT ?= -O2
CPPFLAGS += -Iinclude
CFLAGS += $(CSTD) $(WARN) $(OPT) -pthread
LDLIBS += -lm -pthread

BUILD := build
BIN := $(BUILD)/lattice-plan
LIB := $(BUILD)/liblattice.a
LIB_OBJECTS := $(BUILD)/planner.o $(BUILD)/scheduler.o $(BUILD)/signature.o $(BUILD)/block_layout.o $(BUILD)/io_reader.o $(BUILD)/trunk_plan.o $(BUILD)/runtime.o
TESTS := $(BUILD)/test_planner $(BUILD)/test_scheduler $(BUILD)/test_signature $(BUILD)/test_block_layout $(BUILD)/test_io_reader $(BUILD)/test_trunk_plan $(BUILD)/test_runtime

.PHONY: all test check sanitize demo clean

all: $(BIN)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/planner.o: src/planner.c include/lattice/planner.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/scheduler.o: src/scheduler.c include/lattice/scheduler.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/signature.o: src/signature.c include/lattice/signature.h include/lattice/scheduler.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/block_layout.o: src/block_layout.c include/lattice/block_layout.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/io_reader.o: src/io_reader.c include/lattice/io_reader.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/trunk_plan.o: src/trunk_plan.c include/lattice/trunk_plan.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/runtime.o: src/runtime.c include/lattice/runtime.h include/lattice/io_reader.h include/lattice/scheduler.h include/lattice/signature.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(LIB): $(LIB_OBJECTS)
	$(AR) rcs $@ $^

$(BIN): src/main.c $(LIB) include/lattice/planner.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) src/main.c $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/test_planner: tests/test_planner.c $(LIB) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/test_scheduler: tests/test_scheduler.c $(LIB) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/test_signature: tests/test_signature.c $(LIB) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/test_block_layout: tests/test_block_layout.c $(LIB) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/test_io_reader: tests/test_io_reader.c $(LIB) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/test_trunk_plan: tests/test_trunk_plan.c $(LIB) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/test_runtime: tests/test_runtime.c $(LIB) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

test: $(TESTS)
	$(BUILD)/test_planner
	$(BUILD)/test_scheduler
	$(BUILD)/test_signature
	$(BUILD)/test_block_layout
	$(BUILD)/test_io_reader
	$(BUILD)/test_trunk_plan
	$(BUILD)/test_runtime

check: clean all test
	$(BIN) --manifest examples/small-moe.tsv --ram-gib 8 --vram-gib 4 --nvme-gbps 5 --quiet --json $(BUILD)/plan.json
	python3 -c 'import json; p=json.load(open("$(BUILD)/plan.json", encoding="utf-8")); assert p["schema"]=="lattice.plan.v1" and p["exact"] and p["feasible"]'

sanitize:
	$(MAKE) clean
	$(MAKE) test OPT='-O1 -g' CFLAGS='$(CSTD) $(WARN) -O1 -g -pthread -fsanitize=address,undefined -fno-omit-frame-pointer' LDFLAGS='-fsanitize=address,undefined'

demo: $(BIN)
	$(BIN) --demo --ram-gib 16 --vram-gib 8 --nvme-gbps 7

clean:
	rm -rf $(BUILD)
