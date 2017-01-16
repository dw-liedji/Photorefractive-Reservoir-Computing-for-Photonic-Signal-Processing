
.PHONY: clean run lib test

CC=g++

PYTHON=/usr/include/python2.7

LDFLAGS=-I $(PYTHON) -I $$FF_ROOT
OPT_FLAGS= -O3 -msse2 #-DNO_DEFAULT_MAPPING
FLAGS=-Wall -std=c++11 -g -pthread -fopenmp -fPIC

CFLAGS= $(FLAGS) -fno-strict-aliasing -DNDEBUG -fwrapv -fPIC -c
LFLAGS= $(FLAGS) -shared -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -fwrapv \
        -D_FORTIFY_SOURCE=2 -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security

RUN_CMD=python
EXE=script.py

SRC=parallel_curl

THREADS=2
TEST_EXE=Test

# Create the .so library.
lib:
	cython --cplus $(SRC).pyx
	$(CC) $(OPT_FLAGS) $(LDFLAGS) $(CFLAGS) $(SRC).cpp -o $(SRC).o
	$(CC) $(OPT_FLAGS) $(LDFLAGS) $(LFLAGS) $(SRC).o   -o $(SRC).so

clean:
	rm -f *.so *.o *~ $(TEST_EXE) $(SRC).cpp

# Run the C++ test file.
test:
	$(CC) $(OPT_FLAGS) $(LDFLAGS) $(FLAGS) $(TEST_EXE).cpp -o $(TEST_EXE)
	./$(TEST_EXE) $(THREADS)

run:
	make clean
	make lib
	@echo -e "\n======================"
	@echo "RUNNING:"
	@echo -e "======================\n"
	$(RUN_CMD) $(EXE)
