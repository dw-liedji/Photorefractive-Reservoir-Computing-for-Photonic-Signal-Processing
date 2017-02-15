
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
clean:
	rm -f *.so *.o *~ $(TEST_EXE) $(SRC).cpp

# Create the C++ .so library.
lib: $(SRC).so

$(SRC).so: $(SRC).o
	$(CC) $(OPT_FLAGS) $(LDFLAGS) $(LFLAGS) $(SRC).o -o $(SRC).so
$(SRC).o: $(SRC).cpp
	$(CC) $(OPT_FLAGS) $(LDFLAGS) $(CFLAGS) $(SRC).cpp -o $(SRC).o
$(SRC).cpp: $(SRC).pyx
	cython --cplus $(SRC).pyx



# Compile and run the C++ test file.
test: $(TEST_EXE)
	./$(TEST_EXE) $(THREADS)

$(TEST_EXE): $(TEST_EXE).o
	$(CC) $(OPT_FLAGS) $(LDFLAGS) $(FLAGS) $(TEST_EXE).o -o $(TEST_EXE)
$(TEST_EXE).o: $(TEST_EXE).cpp
	$(CC) $(OPT_FLAGS) $(LDFLAGS) $(FLAGS) -c $(TEST_EXE).cpp

run:
	make clean
	make lib
	@echo -e "\n======================"
	@echo "RUNNING:"
	@echo -e "======================\n"
	$(RUN_CMD) $(EXE)
