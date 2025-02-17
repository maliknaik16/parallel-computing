# Compiler
NVCC = nvcc
CXX  = g++

# Compiler Flags
CXXFLAGS = -O2 -Wall -std=c++17
NVCCFLAGS = -arch=all-major -O2 --compiler-options "-Wall"

# Directories
SRC_DIR = src
BIN_DIR = bin

LDFLAGS = -L${LD_LIBRARY_PATH} -lcudart

# Find all projects (directories in exercises/)
PROJECTS = $(shell find $(SRC_DIR) -maxdepth 1 -type d | tail -n +2 | xargs -n 1 basename)

# Generate executable names (bin/project1, bin/project2, ...)
EXECUTABLES = $(addprefix $(BIN_DIR)/, $(PROJECTS))

all: $(EXECUTABLES)

# Rule to compile each project with nvcc
$(BIN_DIR)/%: $(SRC_DIR)/%/*.cu $(SRC_DIR)/%/*.cpp
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -x cu $(SRC_DIR)/$*/main.cpp $(SRC_DIR)/$*/kernel.cu -o $@


clean:
	rm -rf $(BIN_DIR)
