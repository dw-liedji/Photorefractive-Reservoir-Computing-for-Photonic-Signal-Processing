
/*
 *  parallel_curl.h
 *
 *  Created on: 22 dec 2016
 *  Author: Stefano Ceccotti
 *  Author: Tommaso Catuogno
*/

#ifndef _PARALLEL_CURL_H
#define _PARALLEL_CURL_H

#include <omp.h>

#include "memory_view.h"

using namespace algebra;

template<typename T>
class ParallelCurl
{
    public:
        ParallelCurl() {}
        
        void curl_E_OPENMP( int Np, T* E, T* H, T* D, T* C, T sc, size_m CU, size_m M, size_m R, size_m CO )
        {
            MemoryView<T>* _E = new MemoryView<T>( E, { CU, M, R, CO } );
            MemoryView<T>* _H = new MemoryView<T>( H, { CU, M, R, CO } );
            MemoryView<T>* _D = new MemoryView<T>( D, { CU, M, R, CO } );
            MemoryView<T>* _C = new MemoryView<T>( C, { CU, M, CO } );
            
            const size_m c       = CU/Np;
            const size_m _offset = CU%Np;

            #pragma omp parallel num_threads(Np)
            {
                const int x = omp_get_thread_num();
                const int offset   = (x < _offset) ? 1 : _offset;
                const bool last    = (x == Np-1);
                const size_m start = (x * c) + offset;
                const size_m stop  = last ? CU : (((x+1) * c+1) + offset);
                
                curl_E( _E->slice( { start, stop } ), _H->slice( { start, stop } ),
                        _D->slice( { start, stop } ), _C->slice( { start, stop } ),
                        sc, last );
            }
            
            free( _E ); free( _H );
            free( _D ); free( _C );
        }
        
    private:
        inline void curl_E( MemoryView<T>* IN, MemoryView<T>* OUT, MemoryView<T>* D, MemoryView<T>* curl, const T sc, const bool last )
        {
            MemoryView<T> ret( IN->getDimensions() );
            if(last) {
                // X-component.
                *(ret[":,:-1,:,0"]) += (*(IN[0][":,1:,:,2"]) - IN[0][":,:-1,:,2"]);
                *(ret[":, -1,:,0"]) -=    IN[0][":,-1,:,2"];
                *(ret[":,:,:-1,0"]) -= (*(IN[0][":,:,1:,1"]) - IN[0][":,:,:-1,1"]);
                *(ret[":,:, -1,0"]) +=    IN[0][":,:,-1,1"];
                
                // Y-component.
                *(ret[":,:,:-1,1"]) += (*(IN[0][":,:,1:,0"]) - IN[0][":,:,:-1,0"]);
                *(ret[":,:, -1,1"]) -=    IN[0][":,:,-1,0"];
                *(ret[":-1,:,:,1"]) -= (*(IN[0]["1:,:,:,2"]) - IN[0][":-1,:,:,2"]);
                *(ret[" -1,:,:,1"]) +=    IN[0]["-1,:,:,2"];
                
                // Z-component.
                *(ret[":-1,:,:,2"]) += (*(IN[0]["1:,:,:,1"]) - IN[0][":-1,:,:,1"]);
                *(ret[" -1,:,:,2"]) -=    IN[0]["-1,:,:,1"];
                *(ret[":,:-1,:,2"]) -= (*(IN[0][":,1:,:,0"]) - IN[0][":,:-1,:,0"]);
                *(ret[":, -1,:,2"]) +=    IN[0][":,-1,:,0"];
                
                *curl = ret[":,:,1,:"];
                *OUT -= ret * ((*D) * sc);
            }
            else {
                // X-component.
                *(ret[":-1,:-1,:,0"]) += (*(IN[0][":-1,1:,:,2"]) - IN[0][":-1,:-1,:,2"]);
                *(ret[":-1, -1,:,0"]) -=    IN[0][":-1,-1,:,2"];
                *(ret[":-1,:,:-1,0"]) -= (*(IN[0][":-1,:,1:,1"]) - IN[0][":-1,:,:-1,1"]);
                *(ret[":-1,:, -1,0"]) +=    IN[0][":-1,:,-1,1"];
                
                // Y-component.
                *(ret[":-1,:,:-1,1"]) += (*(IN[0][":-1,:,1:,0"]) - IN[0][":-1,:,:-1,0"]);
                *(ret[":-1,:, -1,1"]) -=    IN[0][":-1,:,-1,0"];
                *(ret[":-1,:,:,  1"]) -= (*(IN[0]["1:, :, :,2"]) - IN[0][":-1,:,:  ,2"]);
                
                // Z-component.
                *(ret[":-1,:,:,  2"]) += (*(IN[0]["1: , :,:,1"]) - IN[0][":-1,:  ,:,1"]);
                *(ret[":-1,:-1,:,2"]) -= (*(IN[0][":-1,1:,:,0"]) - IN[0][":-1,:-1,:,0"]);
                *(ret[":-1, -1,:,2"]) +=    IN[0][":-1,-1,:,0"];
                
                *(curl[0][":-1"]) = ret[":-1,:,1,:"];
                *(OUT[0][":-1"]) += *ret[":-1"] * (*D[0][":-1"] * sc);
            }
        }
        
    public:
        void curl_H_OPENMP( int Np, T* E, T* H, T* D, T* C, T sc, size_m CU, size_m M, size_m R, size_m CO )
        {
            MemoryView<T>* _E = new MemoryView<T>( E, { CU, M, R, CO } );
            MemoryView<T>* _H = new MemoryView<T>( H, { CU, M, R, CO } );
            MemoryView<T>* _D = new MemoryView<T>( D, { CU, M, R, CO } );
            MemoryView<T>* _C = new MemoryView<T>( C, { CU, M, CO } );
            
            const size_m c       = CU/Np;
            const size_m _offset = CU%Np;

            #pragma omp parallel num_threads(Np)
            {
                const int x = omp_get_thread_num();
                const int offset   = (x < _offset) ? 1 : _offset;
                const bool first   = (x == 0);
                const size_m start = (first) ? (x * c) : ((x * c) - 1) + offset;
                const size_m stop  = (x == Np-1) ? CU : ((x+1)*c + offset);
                
                curl_H( _E->slice( { start, stop } ), _H->slice( { start, stop } ),
                        _D->slice( { start, stop } ), _C->slice( { start, stop } ),
                        sc, first );
            }
            
            free( _E ); free( _H );
            free( _D ); free( _C );
        }
        
    private:
        inline void curl_H( MemoryView<T>* IN, MemoryView<T>* OUT, MemoryView<T>* D, MemoryView<T>* curl, const T sc, const bool first )
        {
            MemoryView<T> ret( IN->getDimensions() );
            if(first) {
                // X-component.
                *(ret[":,1:,:,0"]) += (*(IN[0][":,1:,:,2"]) - IN[0][":,:-1,:,2"]);
                *(ret[":,0 ,:,0"]) +=    IN[0][":,0 ,:,2"];
                *(ret[":,:,1:,0"]) -= (*(IN[0][":,:,1:,1"]) - IN[0][":,:,:-1,1"]);
                *(ret[":,:,0 ,0"]) -=    IN[0][":,:,0 ,1"];
                
                // Y-component.
                *(ret[":,:,1:,1"]) += (*(IN[0][":,:,1:,0"]) - IN[0][":,:,:-1,0"]);
                *(ret[":,:,0 ,1"]) +=    IN[0][":,:,0 ,0"];
                *(ret["1:,:,:,1"]) -= (*(IN[0]["1:,:,:,2"]) - IN[0][":-1,:,:,2"]);
                *(ret["0 ,:,:,1"]) -=    IN[0]["0,:,:, 2"];
                
                // Z-component.
                *(ret["1:,:,:,2"]) += (*(IN[0]["1:,:,:,1"]) - IN[0][":-1,:,:,1"]);
                *(ret["0 ,:,:,2"]) +=    IN[0]["0 ,:,:,1"];
                *(ret[":,1:,:,2"]) -= (*(IN[0][":,1:,:,0"]) - IN[0][":,:-1,:,0"]);
                *(ret[":,0 ,:,2"]) -=    IN[0][":,0 ,:,0"];
                
                *curl = ret[":,:,1,:"];
                *OUT += ret * ((*D) * sc);
            }
            else {
                // X-component.
                *(ret["1:,1:,:,0"]) += (*(IN[0]["1:,1:,:,2"]) - IN[0]["1:,:-1,:,2"]);
                *(ret["1:,0 ,:,0"]) +=    IN[0]["1:,0 ,:,2"];
                *(ret["1:,:,1:,0"]) -= (*(IN[0]["1:,:,1:,1"]) - IN[0]["1:,:,:-1,1"]);
                *(ret["1:,:,0 ,0"]) -=    IN[0]["1:,:,0 ,1"];
                
                // Y-component.
                *(ret["1:,:,1:,1"]) += (*(IN[0]["1:,:,1:,0"]) - IN[0]["1:,:,:-1,0"]);
                *(ret["1:,:,0 ,1"]) +=    IN[0]["1:,:,0 ,0"];
                *(ret["1:,:,:, 1"]) -= (*(IN[0]["1:,:,:, 2"]) - IN[0][":-1,:,:, 2"]);
                
                // Z-component.
                *(ret["1:,:,:, 2"]) += (*(IN[0]["1:,:,:, 1"]) - IN[0][":-1,:,  :,1"]);
                *(ret["1:,1:,:,2"]) -= (*(IN[0]["1:,1:,:,0"]) - IN[0]["1: ,:-1,:,0"]);
                *(ret["1:,0 ,:,2"]) -=    IN[0]["1:,0 ,:,0"];
                
                *(curl[0]["1:"]) = ret["1:,:,1,:"];
                *(OUT[0]["1:"]) += *ret["1:"] * (*D[0]["1:"] * sc);
            }
        }
};

#endif
