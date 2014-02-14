PROJECT := $(shell readlink $(dir $(lastword $(MAKEFILE_LIST))) -f)

BIN = $(PROJECT)/bin
SRC = $(PROJECT)/src
THIRD_PARTY = $(PROJECT)/third_party

THIRD_PARTY_SRC = $(THIRD_PARTY)/src
THIRD_PARTY_LIB = $(THIRD_PARTY)/lib
THIRD_PARTY_INCLUDE = $(THIRD_PARTY)/include

NEED_MKDIR = $(BIN) \
             $(THIRD_PARTY_SRC) \
             $(THIRD_PARTY_LIB) \
             $(THIRD_PARTY_INCLUDE)

CXX = g++
CXXFLAGS = -g -O4 -std=c++11 -fno-omit-frame-pointer
INCFLAGS = -I$(SRC) -I$(THIRD_PARTY_INCLUDE) 
LDFLAGS = -Wl,-rpath,$(THIRD_PARTY_LIB) \
          -L$(THIRD_PARTY_LIB) \
          -pthread \
          -lgflags \
          -lglog \
          -lboost_thread \
          -lboost_system

# ================= principal rules ==================

all: third_party

path: $(NEED_MKDIR)

$(NEED_MKDIR):
	mkdir -p $@

clean: 
	rm -rf $(BIN)

distclean: clean
	rm -rf $(filter-out $(THIRD_PARTY)/third_party.mk, \
		            $(wildcard $(THIRD_PARTY)/*))

.PHONY: all path clean distclean

include $(THIRD_PARTY)/third_party.mk

