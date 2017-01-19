
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <chrono>
#include <iostream>

#include <omp.h>

#define __DEBUG__ 0

//#include "ff_curl.h"
#include "memory_view.h"

//using namespace ff_curl_parallel;
using namespace algebra;

// Type of the elements.
typedef float V;

constexpr V sc = 0.7f;


template<typename T>
void curl_E_OPENMP( int Np, MemoryView<T>* E, MemoryView<T>* H, MemoryView<T>* D, MemoryView<T>* curl, int CU, int M, int R, int CO )
{
    const size_m c       = CU/Np;
    const size_m _offset = CU%Np;

    #pragma omp parallel num_threads(Np)
    {
        const int x = omp_get_thread_num();
        const int offset   = (x < _offset) ? 1 : _offset;
        const bool last    = (x == Np-1);
        const size_m start = (x * c) + offset;
        const size_m stop  = last ? CU : (((x+1) * c+1) + offset);
        
        H->curl_E( E->slice( { start, stop, 0, M, 0, R, 0, CO } ),
                   H->slice( { start, stop, 0, M, 0, R, 0, CO } ),
                   D->slice( { start, stop, 0, M, 0, R, 0, CO } ),
                   curl->slice( { start, stop, 0, M, 0, 1, 0, CO } ),
                   sc, last );
    }
}

template<typename T>
void curl_H_OPENMP( int Np, MemoryView<T>* E, MemoryView<T>* H, MemoryView<T>* D, MemoryView<T>* curl, int CU, int M, int R, int CO )
{
    const size_m c       = CU/Np;
    const size_m _offset = CU%Np;

    #pragma omp parallel num_threads(Np)
    {
        const int x = omp_get_thread_num();
        const int offset   = (x < _offset) ? 1 : _offset;
        const bool first   = (x == 0);
        const size_m start = (first) ? (x * c) : ((x * c) - 1) + offset;
        const size_m stop  = (x == Np-1) ? CU : ((x+1)*c + offset);
        
        H->curl_H( H->slice( { start, stop, 0, M, 0, R, 0, CO } ),
                   E->slice( { start, stop, 0, M, 0, R, 0, CO } ),
                   D->slice( { start, stop, 0, M, 0, R, 0, CO } ),
                   curl->slice( { start, stop, 0, M, 0, 1, 0, CO } ),
                   sc, first );
    }
}


int main( int argc, char* argv[] )
{
#if __DEBUG__
    const size_m CU = 2;
    const size_m M  = 2;
    const size_m R  = 3;
    const size_m CO = 3;
#else
    const size_m CU = 700;
    const size_m M  = 700;
    const size_m R  = 250;
    const size_m CO = 3;
#endif
    
    MemoryView<V>* H = new MemoryView<V>( { CU, M, R, CO } ); *H = 2;
    MemoryView<V>* E = new MemoryView<V>( { CU, M, R, CO } ); *E = 5;
    MemoryView<V>* D = new MemoryView<V>( { CU, M, R, CO } ); *D = 4;
    MemoryView<V>* curl = new MemoryView<V>( { CU, M, 1, CO } );
    
#if __DEBUG__
    /*MemoryView<V>* tmp1 = new MemoryView<V>( { CU, M, R, CO } ); *tmp1 = 3;
    MemoryView<V>* tmp2 = new MemoryView<V>( { CU, M, R } ); *tmp2 = 2;
    
    *tmp2 = (*tmp1)[":,:,:,1"];
    //printf("TMP2\n");
    //tmp2->print_out();
    
    *H = 2; *E = 5; *D = 4;
    
    *(*D)["1:,:,:,:"] = *((*H)["1:,:,:,:"]) + (*E)["1:,:,:,:"];
    *(*D)["1:,:,:,1"] -= (*((*H)["1:,:,:,2"]) -(*E)[":-1,:,:,2"]);
    *(*D)["1:,:,:,1"] -= (*H)["1:,:,:,2"];
    //D->print_out();*/
    
    MemoryView<V> F( { CU, M, R, CO } );
    *(F["1,1,1,1"]) = 5;
    //F.print_out();
    
    MemoryView<V>* tmp = new MemoryView<V>( { CU, M, R } );
    *tmp = F;
    tmp->print_out();
    
    //MemoryView<V>* G = new MemoryView<V>( &F );
    MemoryView<V> G( { CU, M, R, CO } );
    
    F = 5;
    *(G["1:"]) = F["1:"];
    G.print_out();
    
    //G.curl_E( H, E, D, curl, sc, true );
    
    //F["1:"]->print_out();
    //E->print_out();
    //printf( "\n" );
    //G["1:,1:,1:,1:"]->print_out();
#else
    const int Q = 1;
    const int Np = stoll( argv[1] );
    
    printf( "INIZIO IL CURL\n" );
    
    for(int n = 1; n <= Np; n++) {
        std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
        for(int i = 0; i < Q; i++) {
            #ifdef _OPENMP
            curl_E_OPENMP( n, H, E, D, curl, CU, M, R, CO );
            #else
            H->curl_E( H, E, D, curl, sc, true );
            #endif /* _OPENMP */
        } 
        std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s, N: " << n << "\n";
    }
    
    printf( "TERMINATO\n" );
#endif
    
    delete H;
    delete E;
    delete D;
    delete curl;
    
    return 0;
}
