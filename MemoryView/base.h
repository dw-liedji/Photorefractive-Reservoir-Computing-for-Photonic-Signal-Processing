
/*
 *  base.h
 *
 *  Created on: 12 dec 2016
 *  Author: Stefano Ceccotti
 *  Author: Tommaso Catuogno
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


namespace algebra
{
    template<typename Derived, typename T>
    class Base
    {
        public:
            T* _data = NULL;
            size_dim _index;
        #ifndef NO_VECTORIALIZATION
            MM_VECT(i)* _x_vec_i;
            MM_VECT()*  _x_vec_s;
            MM_VECT(d)* _x_vec_d;
        #endif
        
        protected:
            Range _range;
            
            size_m   _dimensions[MAX_DIMENSIONS];
            size_dim _sizeDimensions[MAX_DIMENSIONS];
            
            size_m   _indices[MAX_DIMENSIONS];
            size_dim _positions[MAX_DIMENSIONS];
            
            size_dim _size;
            
        #ifdef NO_VECTORIALIZATION
            const bool VECTORIALIZATION = false;
        #else
            const bool VECTORIALIZATION = true;
        #endif
        
        
        protected:
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
            
            Base( const Derived* other )
            { copyFrom( other ); }
        
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
            inline void setRange( const std::vector<size_dim> boundaries )
            { _range.setRange<sliced>( boundaries ); }
            
            Derived* slice( const std::vector<size_dim> boundaries )
            {
                Derived* _tmp = new Derived( _data, _dimensions, _size );
                _tmp->setRange<true>( boundaries );
                
                return _tmp;
            }
            
            inline Range getRange()
            { return _range; }

            inline T get( const std::vector<size_dim> positions )
            {
                size_dim i = 0, _pos = 0;
                for(size_dim pos : positions)
                    _pos += (_sizeDimensions[i++] * pos);
                
                return _data[_pos];
            }
            
            inline std::vector<size_dim> getDimensions()
            {
                std::vector<size_dim> dimensions( _size );
                for(size_dim i = 0; i < _size; i++)
                    dimensions[i] = _range.shape( i );
                
                return dimensions;
            }
            
            MV_INLINE void copyFrom( const Derived* other )
            {
                _size = other->_size;
                
                memcpy( _dimensions, other->_dimensions, sizeof( size_m ) * MAX_DIMENSIONS );
                memcpy( _sizeDimensions, other->_sizeDimensions, sizeof( size_dim ) * MAX_DIMENSIONS );
                
                memcpy( _indices, other->_indices, sizeof( size_m ) * MAX_DIMENSIONS );
                memcpy( _positions, other->_positions, sizeof( size_dim ) * MAX_DIMENSIONS );
                
                _data = other->_data;
                _range.copyFrom( other->_range );
            }
            
            MV_INLINE size_dim getIndex()
            { return _index; }
            
            MV_INLINE void loadIndex()
            {
                _index = 0;
                for(size_dim i = 0; i < _size; i++)
                    _index += (_sizeDimensions[i] * _range._boundaries[i].first);
                
                for(size_dim i = 0; i < _size-2; i++)
                    _positions[i] = _index;
            }
            
        #ifndef NO_VECTORIALIZATION
            MV_INLINE void loadVector()
            {
                if(IS_DOUBLE( T ))     _x_vec_d = (MM_VECT(d)*) (_data + _index);
                else if(IS_FLOAT( T )) _x_vec_s = (MM_VECT( )*) (_data + _index);
                else                   _x_vec_i = (MM_VECT(i)*) (_data + _index);
            }
        #endif
            
            MV_INLINE bool isSliced()
            { return _range.isSliced(); }
            
            MV_INLINE bool isSubBlock()
            {
                for(size_dim i = 1; i < _size; i++)
                    if(_range.shape(i) != _dimensions[i]) return false;
                return true;
            }
            
            MV_INLINE size_dim toAlignment()
			{ return (block - (_index % block)) % block; }
            
            MV_INLINE bool isAligned()
            { return _data != NULL && ((uintptr_t) &(_data[_index])) % BLOCK == 0; }
            
            MV_INLINE bool isAlignable()
            { return _data != NULL && (BLOCK - (((uintptr_t) &(_data[0])) % BLOCK)) % sizeof( T ) == 0; }
            
            template<int dim>
            MV_INLINE void update_dim( const int offset )
            {
                switch( dim ) {
                    case( 1 ): _index += offset; break;
                    case( 2 ): _index += _sizeDimensions[_size-2]; break;
                    default  : _index = _positions[_size-dim] += _sizeDimensions[_size-dim]; break;
                }
            }
            
            // ========================== //
            
            
            
            // === VECTOR OPERATIONS === //
            /*virtual Base<T>* operator+( void );
            
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
            
            // ========================== //*/
            
            inline Derived* operator[]( const char* idx )
            {
                string index = string( idx );
                // Remove the useless characters (whitespaces and tabs) from the beginning and from the end.
                index.erase( std::remove( index.begin(), index.end(), '\t' ), index.end() );
                index.erase( std::remove( index.begin(), index.end(), ' ' ), index.end() );
                
                std::vector<size_dim> boundaries( _size * 2 );
                
                size_t length = index.length(), dim = 0, curr = 0, next = 0;
                int16_t pos;
                
                // Retrieve the indices used for the slicing.
                do {
                    next = index.find( ',', curr );
                    if(next == string::npos) // Not found => last dimension.
                        next = length;
                    
                    size_t colon = index.find( ':', curr );
                    if(colon == string::npos || colon > next) { // Not found.
                        // Unary position.
                        pos = stoi( index.substr( curr, (colon-curr) ) );
                        if(pos >= _dimensions[dim]){ fprintf( stderr, "Out of bounds on buffer access (dimension %ld).\n", (dim+1) ); throw; };
                        
                        boundaries[2*dim] = (pos < 0) ? _range._boundaries[dim].second + pos : _range._boundaries[dim].first + pos;
                        boundaries[2*dim+1] = boundaries[2*dim] + 1;
                    }
                    else {
                        // All current dimension.
                        if(colon == curr) {
                            boundaries[2*dim]   = _range._boundaries[dim].first;
                            boundaries[2*dim+1] = _range._boundaries[dim].second;
                        }
                        
                        // Starting position.
                        if(colon > curr) {
                            pos = stoi( index.substr( curr, (colon-curr) ) );
                            if(pos < 0) boundaries[2*dim] = _range._boundaries[dim].second + pos;
                            else        boundaries[2*dim] = _range._boundaries[dim].first + pos;
                        }
                        
                        // Ending position.
                        if(colon == next-1)
                            boundaries[2*dim+1] = _range._boundaries[dim].second;
                        else {
                            pos = stoi( index.substr( colon+1, (next-colon-1) ) );
                            if(pos < 0) boundaries[2*dim+1] = _MAX( _range._boundaries[dim].first, _range._boundaries[dim].second + pos );
                            else        boundaries[2*dim+1] = _range._boundaries[dim].first + pos;
                        }
                    }
                    
                    dim++;
                    curr = next + 1;
                } while(curr < length);
                
                for(size_dim i = dim; i < _size; i++) {
                    boundaries[2*i]   = _range._boundaries[i].first;
                    boundaries[2*i+1] = _range._boundaries[i].second;
                }
                
                return slice( boundaries );
            }
            
            void print_out()
            {
                printSize();
                
                printf( "[" );
                printDimension( 0, std::vector<size_dim>( _size ) );
                printf( "]\n" );
            }
            
        private:
            inline void printSize()
            {
                printf( "RANGES = [" );
                for(size_dim i = 0; i < _size-1; i++)
                    printf( "(%ld, %ld), ", _range._boundaries[i].first, _range._boundaries[i].second );
                printf( "(%ld, %ld)", _range._boundaries[_size-1].first, _range._boundaries[_size-1].second );
                printf( "]\n");
            }
            
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
    };
}

#endif /* _BASE_H */
