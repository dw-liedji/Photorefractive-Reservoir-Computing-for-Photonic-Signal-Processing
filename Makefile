
# ===== Get the input parameters. ===== #
EXEC=Simple.py
ifdef EXE
    EXEC=$(EXE)
endif

NWORKERS=1
ifdef WORKERS
    NWORKERS=$(WORKERS)
endif

DIM_SIZE=300
ifdef SIZE
    DIM_SIZE=$(SIZE)
endif

ITERATIONS=Simple.py
ifdef N_ITE
    ITERATIONS=$(N_ITE)
endif
# ===================================== #

.PHONY: clean run help

clean:
	rm -f *.so *.o *~ $(SOURCES)/$(SRC).cpp $(SOURCES)/$(SRC).o $(SOURCES)/$(SRC).so fdtd/$(SRC).so fdtd/*.pyc

CXX=g++

PYTHON=/usr/include/python2.7

LDFLAGS=-I $(PYTHON) -I $$FF_ROOT
OPT_FLAGS= -O3 -msse2 -DNO_DEFAULT_MAPPING -finline-functions
FLAGS=-Wall -std=c++11 -pthread -fopenmp -fPIC

CFLAGS= $(FLAGS) -fno-strict-aliasing -DNDEBUG -fwrapv -fPIC -c -lm
LFLAGS= $(FLAGS) -shared -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -fwrapv \
        -D_FORTIFY_SOURCE=2 -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security -lm

SOURCES=./MemoryView
RUN_CMD=python

SRC=parallel_curl

# Print the usage command.
help:
	@echo "Usage: make run [EXE] [WORKERS] [SIZE] [N_ITE]"

# Create the C++ MemoryView .so library.
lib: $(SRC).so

$(SRC).so: $(SRC).o
	$(CXX) $(OPT_FLAGS) $(LDFLAGS) $(LFLAGS) $(SOURCES)/$(SRC).o -o $(SOURCES)/$(SRC).so
$(SRC).o: $(SRC).cpp
	$(CXX) $(OPT_FLAGS) $(LDFLAGS) $(CFLAGS) $(SOURCES)/$(SRC).cpp -o $(SOURCES)/$(SRC).o
$(SRC).cpp: $(SOURCES)/$(SRC).pyx
	python $$PYTHON_LIBS/cython.py --cplus $(SOURCES)/$(SRC).pyx

run:
	make lib
	cp $(SOURCES)/$(SRC).so fdtd/
	@echo "\n======================"
	@echo "RUNNING:"
	@echo "======================\n"
	$(RUN_CMD) $(EXEC) $(NWORKERS) $(DIM_SIZE) $(ITERATIONS)

