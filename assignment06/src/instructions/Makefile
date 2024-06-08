SHELL:=/bin/bash
HOSTNAME:=$(shell hostname)
BIN:=instructions-$(HOSTNAME)
SRC:=instructions.cpp
CSV_FILE:=$(BIN).csv

CXX:=g++ -std=c++17
CXXFLAGS?=-O3 -g -fno-tree-vectorize

# setting of arithmetic operator
OP?=+
# setting of maximum number of dependency chains
CHAINS_MAX?=1U

all: $(BIN)

$(BIN): $(SRC) Makefile
	$(CXX) $(CXXFLAGS) -DOP=$(OP) -DCHAINS_MAX=$(CHAINS_MAX) $< -o $@

.PHONY: clean bench-instructions

bench-instructions:
	@echo dtype,op,chains,time > $(CSV_FILE)
	@for o in + "*" /; do \
		make $(BIN) -B OP="$$o"; \
		./$(BIN) 27 >> $(CSV_FILE); \
	done;

clean:
	$(RM) $(BIN)
