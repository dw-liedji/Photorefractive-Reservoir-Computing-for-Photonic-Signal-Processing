
import sys

import numpy as np
cimport numpy as np
from timeit import default_timer as time
import multiprocessing
from time import sleep
import cython

from cython import boundscheck, wraparound
from cython cimport parallel
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp cimport bool
from libcpp.vector cimport vector
cimport posix.unistd as uid_t
from libc.time cimport clock

from libc.stdint cimport *

ctypedef int64_t size_dim
ctypedef int64_t size_m

cdef extern from "parallel_curl.h" nogil:
    cdef cppclass ParallelCurl[T]:
        ParallelCurl()
        void curl_E_OPENMP( int, T*, T*, T*, T*, T, size_m, size_m, size_m, size_m )
        void curl_H_OPENMP( int, T*, T*, T*, T*, T, size_m, size_m, size_m, size_m )

# Define the type of the elements.
ctypedef float T


def seq_curl_E(E): # Transforms an E field into an H field by performing a curl
    ret = np.zeros(E.shape)
    
    # x - component
    ret[:,:-1,:,0] += (E[:,1:,:,2] - E[:,:-1,:,2])
    ret[:, -1,:,0] -=  E[:,-1,:,2]
    ret[:,:,:-1,0] -= (E[:,:,1:,1] - E[:,:,:-1,1])
    ret[:,:, -1,0] +=  E[:,:,-1,1]

    # y-component
    ret[:,:,:-1,1] += (E[:,:,1:,0] - E[:,:,:-1,0])
    ret[:,:, -1,1] -=  E[:,:,-1,0]
    ret[:-1,:,:,1] -= (E[1:,:,:,2] - E[:-1,:,:,2])
    ret[ -1,:,:,1] +=  E[-1,:,:,2]
    
    # z - component
    ret[:-1,:,:,2] += (E[1:,:,:,1] - E[:-1,:,:,1])
    ret[ -1,:,:,2] -=  E[-1,:,:,1]
    ret[:,:-1,:,2] -= (E[:,1:,:,0] - E[:,:-1,:,0])
    ret[:, -1,:,2] +=  E[:,-1,:,0]
    
    return ret


def pySEQ_curl_E( E, H, D, sc ):
    E += sc * D * seq_curl_E( H )


@boundscheck(False)
@wraparound(False)
def curl_E( np.ndarray[T, ndim=4] E, np.ndarray[T, ndim=4] H, np.ndarray[T, ndim=4] D, np.ndarray[T, ndim=3] curl, T sc, int Np ):
    # Get the size of the input data structures.
    cdef size_m CU = H.shape[0], M = H.shape[1], R = H.shape[2], CO = H.shape[3]
    cdef T _sc = sc
    cdef int _Np = Np
    
    cdef ParallelCurl[T] pCurl
    with nogil:
        pCurl.curl_E_OPENMP( _Np, &E[0,0,0,0], &H[0,0,0,0], &D[0,0,0,0], &curl[0,0,0], _sc, CU, M, R, CO )


@boundscheck(False)
@wraparound(False)
def curl_H( np.ndarray[T, ndim=4] E, np.ndarray[T, ndim=4] H, np.ndarray[T, ndim=4] D, np.ndarray[T, ndim=3] curl, T sc, int Np ):
    # Get the size of the input data structures.
    cdef size_m CU = H.shape[0], M = H.shape[1], R = H.shape[2], CO = H.shape[3]
    cdef T _sc = sc
    cdef int _Np = Np
    
    cdef ParallelCurl[T] pCurl
    with nogil:
        pCurl.curl_H_OPENMP( _Np, &E[0,0,0,0], &H[0,0,0,0], &D[0,0,0,0], &curl[0,0,0], _sc, CU, M, R, CO )
