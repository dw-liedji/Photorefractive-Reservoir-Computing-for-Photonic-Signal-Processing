
from timeit import default_timer as time
import sys
from time import sleep
from parallel_curl import *


##########
## INIT ##
##########
CU = 300 # #Cubes
M =  300 # #Matrices.
R =  300 # #Rows.
CO = 3   # #Columns.
Q =  20  # Number of timesteps (3000??).
# Threads.
Np = multiprocessing.cpu_count()
Np = 4

# Courant Number.
sc = 0.7

Type = np.float32


def new_fields():
    np.random.seed(6)
    E = np.random.rand(CU,M,R,CO).astype(dtype=Type)
    H = np.random.rand(CU,M,R,CO).astype(dtype=Type)
    D = np.random.rand(CU,M,R,CO).astype(dtype=Type)
    return E,H,D

if __name__ == '__main__':
    E, H, D = new_fields()
    
    init( Np )
    
    print "START SEQUENTIAL.."
    t = time()
    for i in xrange(Q):
        pySEQ_curl_E( E, H, D, sc )
    print "Py_SEQ:     ",time()-t
    
    curl = np.zeros((CU,M,CO)).astype(dtype=Type)
    
    print "START PARALLEL.."
    
    for T in xrange( 1, Np+1 ):
        for i in xrange(Q):
            #t = time()
            curl_H( E, H, D, curl, sc, T )
            #curl_H_FF( E, H, D, curl, sc, T )
        print "Cpp_PAR:    ",time()-t, "Np = ",T
