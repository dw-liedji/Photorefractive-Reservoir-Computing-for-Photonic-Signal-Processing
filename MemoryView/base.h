
/*
 *  base.h
 *
 *  Created on: 12 dec 2016
 *  Authors: Stefano Ceccotti & Tommaso Catuogno
*/

#ifndef _BASE_H
#define _BASE_H


#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <utility>
#include <vector>
#include <algorithm>

#include <typeinfo>

#include "range.h"
#include "functions.h"
#include "settings.h"

using namespace std;

#define _MAX(a,b) (((a) > (b)) ? (a) : (b))

// Get the index to access the 4D matrix.
#define GET_INDEX( cu, m, r, co ) \
    ((cu) * cSize) + ((m) * mSize) + ((r) * columns) + (co)

// Macro used to check the index boundaries during a slicing.
#define CHECK_INDEX( index, max ) \
    if(index >= max){ printf( "Out of bounds on buffer access\n" ); throw; }



namespace algebra
{
    template<typename T>
    class Base
    {
        public:
            T* _data = NULL;
            size_dim _index;
        private:
            Range _range;
            
            Function<Base,T>* fun = NULL;
            size_m   _dimensions[MAX_DIMENSIONS];
            size_dim _sizeDimensions[MAX_DIMENSIONS];
            
            size_m   _indices[MAX_DIMENSIONS];
            size_dim _positions[MAX_DIMENSIONS];
            
            size_dim _size;
        
        #ifdef NO_VECTORIALIZATION
            static const size_dim block = 1;
            static const bool VECTORIALIZATION = false;
        #else
            static const size_dim block = BLOCK / sizeof( T );
            static const bool VECTORIALIZATION = true;
        #endif
        
        public:
            Base() {}
            
            Base( const std::vector<size_dim> dimensions )
            {
                build<std::vector<size_dim>>( dimensions );
                _data = (T*) calloc( _range.getTotalSize(), sizeof( T ) );
                //_data = (T*) _mm_malloc( total_size * sizeof( T ), BLOCK );
            }
            
            Base( T* data, const std::vector<size_dim> dimensions )
            {
                build<std::vector<size_dim>>( dimensions );
                _data = data;
            }
            
            Base( T* data, const size_dim* dimensions, const int& size )
            {
                build<std::vector<size_dim>>( std::vector<size_dim>( dimensions, dimensions + size ) );
                _data = data;
            }
            
            Base( const Base<T>* other )
            {
                _size = other->_size;
                
                memcpy( _dimensions, other->_dimensions, sizeof( size_m ) * MAX_DIMENSIONS );
                memcpy( _sizeDimensions, other->_sizeDimensions, sizeof( size_dim ) * MAX_DIMENSIONS );
                
                memcpy( _indices, other->_indices, sizeof( size_m ) * MAX_DIMENSIONS );
                memcpy( _positions, other->_positions, sizeof( size_dim ) * MAX_DIMENSIONS );
                
                _data = other->_data;
                _range = other->_range;
            }
        
        private:
            template<class Container>
            void build( const Container& container )
            {
                _size = container.size();
                if(_size > MAX_DIMENSIONS) {
                    fprintf( stderr, "Sequence too large; cannot be greater than %d\n", MAX_DIMENSIONS );
                    throw;
                }
                
                std::vector<size_dim> boundaries( _size * 2 );
                
                size_dim i = 0, j = 0, total_size = 1;
                for(auto iterator = container.begin(); iterator != container.end(); ++iterator) {
                    size_dim elem = *iterator;
                    total_size *= elem;
                    _dimensions[j++] = elem;
                    boundaries[i++] = 0;
                    boundaries[i++] = elem;
                }
                
                for(size_dim i = _size - 1; i >= 0; i--)
                    _sizeDimensions[i] = (i == _size-1) ? 1 : (_sizeDimensions[i+1] * _dimensions[i+1]);
                _range.setRange<false>( boundaries );
            }
            
            
            // ======== UTILITY METHODS ======== //
            
            
            
            
        public:
            T* data() { return _data; }
            
            template<bool sliced>
            void setRange( const std::vector<size_dim> boundaries )
            { _range.setRange<sliced>( boundaries ); }
            
            Base<T>* slice( const std::vector<size_dim> boundaries )
            {
                Base<T>* _tmp = new Base<T>( _data, _dimensions, _size );
                _tmp->setRange<true>( boundaries );
                
                return _tmp;
            }
            
            T get( const std::vector<size_dim> positions )
            {
                size_dim i = 0, _pos = 0;
                for(size_dim pos : positions)
                    _pos += (_sizeDimensions[i++] * pos);
                
                return _data[_pos];
            }
            
            std::vector<size_dim> getDimensions()
            {
                std::vector<size_dim> dimensions( _size );
                for(size_dim i = 0; i < _size; i++)
                    dimensions[i] = _range.shape( i );
                
                return dimensions;
            }
            
            INLINE size_dim getIndex()
            { return _index; }
            
            INLINE void loadIndex()
            {
                _index = 0;
                for(auto i = 0; i < _size; i++)
                    _index += (_sizeDimensions[i] * _range._boundaries[i].first);
                
                for(auto i = 0; i < _size-2; i++)
                    _positions[i] = _index;
            }
            
            INLINE bool isSliced()
            { return _range.isSliced(); }
            
            INLINE bool isSubBlock()
            {
                for(size_dim i = 1; i < _size; i++)
                    if(_range.shape(i) != _dimensions[i]) return false;
                return true;
            }
            
            INLINE bool isAligned()
            { return ((uintptr_t) (_data + _index)) % BLOCK == 0; }
            
            INLINE bool isAlignable()
            { return ((uintptr_t) _data) % BLOCK == 0; }
            
            template<int offset>
            INLINE void update( const int dim )
            {
                switch( dim ) {
                    case( 1 ): _index += offset; break;
                    case( 2 ): _index += _sizeDimensions[_size-2]; break;
                    default: _index = _positions[_size-dim] += _sizeDimensions[_size-dim]; break;
                }
            }
            
            template<int dim, int offset>
            INLINE void update()
            {
                switch( dim ) {
                    case( 1 ): _index += offset; break;
                    case( 2 ): _index += _sizeDimensions[_size-2]; break;
                    default: _index = _positions[_size-dim] += _sizeDimensions[_size-dim]; break;
                }
            }
            
            // ========================== //
            
            
            
            // === VECTOR OPERATIONS === //
            virtual Base<T>* operator+( void );
            
            virtual Base<T>* operator+( Base<T>* in );
            
            virtual Base<T>* operator-( Base<T>* in );
            
            virtual Base<T>* operator*( Base<T>* in );
            
            virtual Base<T>* operator*( const T& val );
            
            virtual Base<T>* operator*=( Base<T>* in );
            
            virtual Base<T>* operator*=( const T& val );
            
            virtual Base<T>* operator=( Base<T>* in );
            
            virtual Base<T>* operator=( const T& val );
            
            virtual Base<T>* operator+=( Base<T>* in );
            
            virtual Base<T>* plus_equals( Base<T>* in );
            
            virtual Base<T>* operator-=( Base<T>* in );
            
            virtual Base<T>* minus_equals( Base<T>* in );
            
            virtual Base<T>* operator[]( const char* idx );
            
            void printSize()
            {
                printf( "RANGES = [" );
                for(size_dim i = 0; i < _size-1; i++)
                    printf( "(%d, %d), ", _range._boundaries[i].first, _range._boundaries[i].second );
                printf( "(%d, %d)", _range._boundaries[_size-1].first, _range._boundaries[_size-1].second );
                printf( "]\n");
            }
            
            void print_out()
            {
                printSize();
                
                printf( "[" );
                printDimension( 0, std::vector<size_dim>( _size ) );
                printf( "]\n" );
            }
            
        private:
            inline void printDimension( const size_m& dim, std::vector<size_dim> indices )
            {
                const size_dim from = _range._boundaries[dim].first;
                const size_dim to   = _range._boundaries[dim].second;
                
                indices[dim] = 0;
                
                if(dim == _size-1) {
                    for(size_dim i = from; i < to; i++) {
                        indices[dim] = i;
                        printf( (i < to-1) ? " %.8lf " : " %.8lf", get( indices ) );
                    }
                }
                else {
                    for(size_dim i = from; i < to; i++) {
                        if(i > from) {
                            for(size_dim i = 0; i <= dim; i++)
                                printf( " " );
                        }
                        printf( "[" );
                        
                        indices[dim] = i;
                        printDimension( dim+1, indices );
                        
                        printf( "]" );
                        if(i < to-1) {
                            for(size_dim i = 0; i < _size - dim - 1; i++)
                                printf( "\n" );
                        }
                    }
                }
            }
        
        public:
            virtual ~Base(){ if(_data != NULL) free( _data ); }
    };
}

#endif /* _BASE_H */
