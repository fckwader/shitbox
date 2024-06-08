SHELL:=/bin/bash
HOSTNAME:=$(shell hostname)
BIN:=branch-$(HOSTNAME)
SRC:=branch.cpp

CXX:=g++ -std=c++17
CXXFLAGS?=-O3 -g -fno-tree-vectorize

all: $(BIN)

$(BIN): $(SRC) Makefile
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: clean

clean:
	$(RM) $(BIN)

