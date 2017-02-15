
# ===== Get the input parameters. ===== #
DEFAULT_EXE=Simple.py
DEFAULT_WORKERS=1
DEFAULT_SIZE=300
DEFAULT_ITERATIONS=20

EXEC=$(DEFAULT_EXE)
ifdef EXE
    EXEC=$(EXE)
endif

NWORKERS=$(DEFAULT_WORKERS)
ifdef WORKERS
    NWORKERS=$(WORKERS)
endif

DIM_SIZE=$(DEFAULT_SIZE)
ifdef SIZE
    DIM_SIZE=$(SIZE)
endif

N_ITE=$(DEFAULT_ITERATIONS)
ifdef ITERATIONS
    N_ITE=$(ITERATIONS)
endif
# ===================================== #

.PHONY: clean run_sequential run_parallel help

clean:
	rm -f *.so *.o *~ $(SOURCES)/$(SRC).cpp $(SOURCES)/$(SRC).o $(SOURCES)/$(SRC).so $(PARALLEL_SIMULATOR)/fdtd/$(SRC).so $(PARALLEL_SIMULATOR)/fdtd/*.pyc $(SEQUENTIAL_SIMULATOR)/fdtd/*.pyc

CXX=g++

PYTHON=/usr/include/python2.7

LDFLAGS=-I $(PYTHON) -I $$FF_ROOT
OPT_FLAGS= -O3 -msse2 -DNO_DEFAULT_MAPPING -finline-functions
FLAGS=-Wall -std=c++11 -pthread -fopenmp -fPIC

CFLAGS= $(FLAGS) -fno-strict-aliasing -DNDEBUG -fwrapv -fPIC -c -lm
LFLAGS= $(FLAGS) -shared -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -fwrapv \
        -D_FORTIFY_SOURCE=2 -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security -lm

SEQUENTIAL_SIMULATOR="Sequential Optical Simulator"
PARALLEL_SIMULATOR="Parallel Optical Simulator"
SOURCES=./MemoryView
RUN_CMD=python

SRC=parallel_curl

# Print the usage commands.
help:
	@echo "Usage: make run_sequential [EXE]           [SIZE] [ITERATIONS]"
	@echo "Usage: make run_parallel   [EXE] [WORKERS] [SIZE] [ITERATIONS]"
	@echo "Default parameters: EXE=$(DEFAULT_EXE), WORKERS=$(DEFAULT_WORKERS), SIZE=$(DEFAULT_SIZE), ITERATIONS=$(DEFAULT_ITERATIONS)"

# Create the C++ MemoryView .so library.
lib_parallel: $(SRC).so

$(SRC).so: $(SRC).o
	$(CXX) $(OPT_FLAGS) $(LDFLAGS) $(LFLAGS) $(SOURCES)/$(SRC).o -o $(SOURCES)/$(SRC).so
$(SRC).o: $(SRC).cpp
	$(CXX) $(OPT_FLAGS) $(LDFLAGS) $(CFLAGS) $(SOURCES)/$(SRC).cpp -o $(SOURCES)/$(SRC).o
$(SRC).cpp: $(SOURCES)/$(SRC).pyx
	python $$PYTHON_LIBS/cython.py --cplus $(SOURCES)/$(SRC).pyx

run_sequential:
	@echo "\n======================"
	@echo "RUNNING:"
	@echo "======================\n"
	$(RUN_CMD) $(SEQUENTIAL_SIMULATOR)/$(EXEC) $(DIM_SIZE) $(N_ITE)

run_parallel:
	make lib_parallel
	cp $(SOURCES)/$(SRC).so $(PARALLEL_SIMULATOR)/fdtd/
	@echo "\n======================"
	@echo "RUNNING:"
	@echo "======================\n"
	$(RUN_CMD) $(PARALLEL_SIMULATOR)/$(EXEC) $(NWORKERS) $(DIM_SIZE) $(N_ITE)

