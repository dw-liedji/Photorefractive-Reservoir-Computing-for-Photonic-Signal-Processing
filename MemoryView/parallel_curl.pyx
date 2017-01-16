
import sys

import numpy as np
cimport numpy as np
from timeit import default_timer as time
import multiprocessing
from time import sleep
import cython

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

cdef extern from "ff_curl.h" namespace "ff_curl_parallel" nogil:
    cdef cppclass FF_cube[T]:
        FF_cube()
        FF_cube( int )
        void curl_E( int, MemoryView[T]*, MemoryView[T]*, MemoryView[T]*, MemoryView[T]*, T, size_m, size_m, size_m, size_m )
        void curl_H( int, MemoryView[T]*, MemoryView[T]*, MemoryView[T]*, MemoryView[T]*, T, size_m, size_m, size_m, size_m )

cdef extern from "memory_view.h" namespace "algebra" nogil:
    cdef cppclass MemoryView[T]:
        MemoryView( vector[size_dim] )
        MemoryView( T*, vector[size_dim] )
        #MemoryView( MemoryView[T]* )
        
        MemoryView[T]* slice( size_m, size_m, size_m, size_m, size_m, size_m, size_m, size_m )
        MemoryView[T]* slice( vector[size_dim] )
        
        MemoryView[T]* operator[]( char* ) except +IndexError
        
        void curl_E( MemoryView[T]*, MemoryView[T]*, MemoryView[T]*, MemoryView[T]*, T, bool )
        void curl_H( MemoryView[T]*, MemoryView[T]*, MemoryView[T]*, MemoryView[T]*, T, bool )
        
        void print_out()


# Define the type of the elements.
ctypedef float T

# FastFlow object.
cdef FF_cube[T]* ff = NULL

def init( Np ): # Init the objects used in the computation.
    global ff
    ff = new FF_cube[T]( Np )


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


def curl_E( np.ndarray[T, ndim=4] E, np.ndarray[T, ndim=4] H, np.ndarray[T, ndim=4] D, np.ndarray[T, ndim=3] curl, sc, Np ):
    
    # Get the size of the input data structures.
    cdef size_m CU = H.shape[0], M = H.shape[1], R = H.shape[2], CO = H.shape[3]
    
    # 4D Vectors.
    cdef MemoryView[T]* _E = new MemoryView[T]( &E[0,0,0,0], [CU, M, R, CO] )
    cdef MemoryView[T]* _H = new MemoryView[T]( &H[0,0,0,0], [CU, M, R, CO] )
    cdef MemoryView[T]* _D = new MemoryView[T]( &D[0,0,0,0], [CU, M, R, CO] )
    
    cdef MemoryView[T]* _curl = new MemoryView[T]( &curl[0,0,0], [CU, M, 1, CO] )
    
    # Partitions.
    cdef int x, offset, last, start, stop
    cdef int c = CU/Np
    cdef int _offset = CU % Np
    cdef int _Np = Np
    cdef int _sc = sc
    
    with nogil:
        # Parallel cycle with OpenMP.
        for x in parallel.prange( _Np, schedule = 'static', num_threads = _Np ):
            offset = 1 if x < _offset else _offset
            last   = (x == _Np-1)
            start  = 0 if x == 0 else ((x * c) + offset)
            stop   = CU if last else ((x+1)*c + 1 + offset)
            
            _H.curl_E( _E.slice( start, stop, 0, M, 0, R, 0, CO ),
                       _H.slice( start, stop, 0, M, 0, R, 0, CO ),
                       _D.slice( start, stop, 0, M, 0, R, 0, CO ),
                       _curl.slice( start, stop, 0, M, 0, 1, 0, CO ),
                       _sc, last )
        
        free( _E ); free( _H ); free( _D );
        free( _curl )


def curl_H( np.ndarray[T, ndim=4] E, np.ndarray[T, ndim=4] H, np.ndarray[T, ndim=4] D, np.ndarray[T, ndim=3] curl, sc, Np ):
    
    # Get the size of the input data structures.
    cdef int CU = H.shape[0], M = H.shape[1], R = H.shape[2], CO = H.shape[3]
    
    # 4D Vectors.
    cdef MemoryView[T]* _E = new MemoryView[T]( &E[0,0,0,0], [CU, M, R, CO] )
    cdef MemoryView[T]* _H = new MemoryView[T]( &H[0,0,0,0], [CU, M, R, CO] )
    cdef MemoryView[T]* _D = new MemoryView[T]( &D[0,0,0,0], [CU, M, R, CO] )
    
    cdef MemoryView[T]* _curl = new MemoryView[T]( &curl[0,0,0], [CU, M, 1, CO] )
    
    # Partitions.
    cdef int x, offset, first, last, start, stop
    cdef int c = CU/Np, _offset = CU%Np
    cdef int _Np = Np
    cdef int _sc = sc
    
    
    with nogil:
        # Parallel cycle with OpenMP.
        for x in parallel.prange( _Np, schedule = 'static', num_threads = _Np ):
            offset = 1 if x < _offset else _offset
            first  = (x == 0)
            start  = 0 if first else ((x * c) - 1 + offset)
            stop   = CU if (x == _Np-1) else ((x+1) * c + offset)
            
            _H.curl_H( _H.slice( start, stop, 0, M, 0, R, 0, CO ),
                       _E.slice( start, stop, 0, M, 0, R, 0, CO ),
                       _D.slice( start, stop, 0, M, 0, R, 0, CO ),
                       _curl.slice( start, stop, 0, M, 0, 1, 0, CO ),
                       _sc, first )
        
        free( _E ); free( _H ); free( _D );
        free( _curl );

def curl_E_FF( np.ndarray[T, ndim=4] E, np.ndarray[T, ndim=4] H, np.ndarray[T, ndim=4] D, np.ndarray[T, ndim=3] curl, sc, Np ):
    
    # Get the size of the input data structures.
    cdef int CU = H.shape[0], M = H.shape[1], R = H.shape[2], CO = H.shape[3]
    cdef int _sc = sc
    cdef int _Np = Np
    
    # 4D Vectors.
    cdef MemoryView[T]* _E = new MemoryView[T]( &E[0,0,0,0], [CU, M, R, CO] )
    cdef MemoryView[T]* _H = new MemoryView[T]( &H[0,0,0,0], [CU, M, R, CO] )
    cdef MemoryView[T]* _D = new MemoryView[T]( &D[0,0,0,0], [CU, M, R, CO] )
    cdef MemoryView[T]* _curl = new MemoryView[T]( &curl[0,0,0], [CU, M, 1, CO] )
    
    with nogil:
        ff.curl_E( _Np, _E, _H, _D, _curl, _sc, CU, M, R, CO )
        free( _E ); free( _H ); free( _D ); free( _curl );


def curl_H_FF( np.ndarray[T, ndim=4] E, np.ndarray[T, ndim=4] H, np.ndarray[T, ndim=4] D, np.ndarray[T, ndim=3] curl, sc, Np ):
    
    # Get the size of the input data structures.
    cdef int CU = H.shape[0], M = H.shape[1], R = H.shape[2], CO = H.shape[3]
    cdef int _sc = sc
    cdef int _Np = Np
    
    # 4D Vectors.
    cdef MemoryView[T]* _E = new MemoryView[T]( &E[0,0,0,0], [CU, M, R, CO] )
    cdef MemoryView[T]* _H = new MemoryView[T]( &H[0,0,0,0], [CU, M, R, CO] )
    cdef MemoryView[T]* _D = new MemoryView[T]( &D[0,0,0,0], [CU, M, R, CO] )
    cdef MemoryView[T]* _curl = new MemoryView[T]( &curl[0,0,0], [CU, M, 1, CO] )
    
    with nogil:
        ff.curl_H( _Np, _E, _H, _D, _curl, _sc, CU, M, R, CO )
        free( _E ); free( _H ); free( _D ); free( _curl );
