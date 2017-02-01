
/*
 *  range.h
 *
 *  Created on: 14 dec 2016
 *  Author: Stefano Ceccotti & Tommaso Catuogno
*/

#ifndef _RANGE_H
#define _RANGE_H

#include "settings.h"

#include <vector>
#include <utility>

class Range
{
    public:
        std::pair<size_m, size_m> _boundaries[MAX_DIMENSIONS * 2];
    private:
        size_dim _shapes[MAX_DIMENSIONS];
        size_dim _size = 0;
        size_dim _totalSize;
        bool _sliced = false;
    
    public:
        Range(){}
        
        template<bool sliced>
        inline void setRange( const std::vector<size_dim> boundaries )
        {
            _totalSize = 1;
            size_dim i = 0, index = 0;
            for(size_dim elem : boundaries) {
                if(index == 0)
                    _boundaries[i] = std::make_pair( elem, elem );
                else {
                    _boundaries[i].second = elem;
                    _shapes[i] = _boundaries[i].second - _boundaries[i].first;
                    _totalSize *= _shapes[i++];
                }
                
                index = (index+1) % 2;
            }
            
            _size = boundaries.size() / 2;
            _sliced = sliced;
        }
        
        inline void checkSize( const char* fun, const int& line, Range& in )
        {
            if(_totalSize != in._totalSize) {
                printf( "[%s, Line: %d] Different input shapes: (", fun, line ); printSize();
                printf( ") and (" ); in.printSize(); printf( ")\n" );
                throw;
            }
            
            size_dim offset = 0, in_offset = 0;
            for(size_dim i = 0; i < _size; i++) {
                if(shape(i+offset) == 1) { offset++; continue; }
                if(in.shape(i+in_offset) == 1) { in_offset++; continue; }
                if(shape(i + offset) != in.shape(i + in_offset)) {
                    printf( "[%s, Line: %d] Different input shapes: (", fun, line ); printSize();
                    printf( ") and (" ); in.printSize(); printf( ")\n" );
                    throw;
                }
            }
        }
        
        INLINE bool isSliced()
        { return _sliced; }
        
        inline size_dim getTotalSize()
        { return _totalSize; }
        
        template<int dim>
        inline size_m shape()
        { return _shapes[dim]; }
        
        INLINE size_m shape( const int& dim )
        { return _shapes[dim]; }
        
    private:
        inline void printSize()
        {
            for(size_dim i = 0; i < _size; i++) {
                if(i < _size - 1) printf( "%ld ", shape(i) );
                else printf( "%ld", shape(i) );
            }
        }
        
    public:
        ~Range(){}
};

#endif /* _RANGE_H */
