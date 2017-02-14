
/*
 *  ff_curl.h
 *
 *  Created on: 10 oct 2016
 *  Author: Stefano Ceccotti
 *  Author: Tommaso Catuogno
*/

#ifndef _FF_CURL_H
#define _FF_CURL_H

#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>

#include <ff/parallel_for.hpp>

#include "memory_view.h"

#define VECTOR MemoryView<T>

using namespace ff;
using namespace algebra;

namespace ff_curl_parallel
{
    template<typename T>
    class FF_cube
    {
        private:
            int _Np;
            ParallelFor* pf;
            
        public:
            FF_cube()
            {
                _Np = sysconf( _SC_NPROCESSORS_ONLN );
                pf = new ParallelFor( _Np );
            }
            
            FF_cube( const int Np )
            {
                _Np = Np;
                pf = new ParallelFor( Np );
            }
            
            void curl_E( VECTOR* E, VECTOR* H, VECTOR* D, VECTOR* curl,
                         const int sc, const size_m CU, const size_m M, const size_m R, const size_m CO )
            {
                const size_m c       = CU/_Np;
                const size_m _offset = CU%_Np;
                
                // Invoked as: start, stop, step (increment variable), grain (iterations per worker).
                pf->parallel_for_idx( 0, _Np, 1, 0, [&]( const int start, const int stop, const int idx ) {
                    const int offset    = (idx < _offset) ? 1 : _offset;
                    const bool last     = (idx == _Np-1);
                    const size_m _start = (idx == 0) ? 0 : ((idx * c) + offset);
                    const size_m _stop  = last ? CU : ((idx+1)*c + 1 + offset);
                    
                    H->curl_E( E->slice( _start, _stop, 0, M, 0, R, 0, CO ),
                               H->slice( _start, _stop, 0, M, 0, R, 0, CO ),
                               D->slice( _start, _stop, 0, M, 0, R, 0, CO ),
                               curl->slice( _start, _stop, 0, M, 0, 1, 0, CO ),
                               sc, last );
                });
            }
            
            void curl_H( VECTOR* E, VECTOR* H, VECTOR* D, VECTOR* curl,
                         const int sc, const size_m CU, const size_m M, const size_m R, const size_m CO )
            {
                const size_m c       = CU/_Np;
                const size_m _offset = CU%_Np;
                
                // Invoked as: start, stop, step (increment variable), grain (iterations per worker).
                pf->parallel_for_idx( 0, _Np, 1, 0, [&]( const int start, const int stop, const int idx ) {
                    const int offset    = (idx < _offset) ? 1 : _offset;
                    const bool first    = (idx == 0);
                    const size_m _start = (first) ? 0 : ((idx * c) - 1 + offset);
                    const size_m _stop  = (idx == _Np-1) ? CU : ((idx+1)*c + offset);
                    
                    H->curl_H( H->slice( _start, _stop, 0, M, 0, R, 0, CO ),
                               E->slice( _start, _stop, 0, M, 0, R, 0, CO ),
                               D->slice( _start, _stop, 0, M, 0, R, 0, CO ),
                               curl->slice( _start, _stop, 0, M, 0, 1, 0, CO ),
                               sc, first );
                });
            }
            
            ~FF_cube()
            { delete pf; }
    };
}

#endif
