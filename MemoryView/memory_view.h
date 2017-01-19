
/*
 *  memory_view.h
 *
 *  Created on: 12 dec 2016
 *  Authors: Stefano Ceccotti & Tommaso Catuogno
*/

#ifndef _MEMORY_VIEW_H
#define _MEMORY_VIEW_H


#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <utility>
#include <vector>
#include <chrono>
#include <algorithm>

#include <typeinfo>

#include "base.h"
#include "range.h"
#include "functions.h"
#include "settings.h"

using namespace std;

namespace algebra
{
    template<typename T>
    class MemoryView /*: public Base<T>*/
    {
        public:
            T* _data = NULL;
            size_dim _index;
        #ifndef NO_VECTORIALIZATION
            MM_VECT* _x_vec;
        #endif
        private:
            Range _range;
            
            Function<MemoryView,T>* _fun = NULL;
            size_m   _dimensions[MAX_DIMENSIONS];
            size_dim _sizeDimensions[MAX_DIMENSIONS];
            
            size_m   _indices[MAX_DIMENSIONS];
            size_dim _positions[MAX_DIMENSIONS];
            
            size_dim _size;
            
            size_dim block = BLOCK / sizeof( T );
        #ifdef NO_VECTORIALIZATION
            bool VECTORIALIZATION = false;
        #else
            bool VECTORIALIZATION = true;
        #endif
        
        public:
            MemoryView() /*: Base<T>()*/ {}
            
            MemoryView( const std::vector<size_dim> dimensions ) /*: Base<T>( dimensions )*/
            {
                build<std::vector<size_dim>>( dimensions );
                _data = (T*) _mm_malloc( _range.getTotalSize() * sizeof( T ), BLOCK );
            }
            
            MemoryView( T* data, const std::vector<size_dim> dimensions ) /*: Base<T>( data, dimensions )*/
            {
                build<std::vector<size_dim>>( dimensions );
                _data = data;
            }
            
            MemoryView( T* data, const size_dim* dimensions, const int& size ) /*: Base<T>( data, dimensions, size )*/
            {
                build<std::vector<size_dim>>( std::vector<size_dim>( dimensions, dimensions + size ) );
                _data = data;
            }
            
            MemoryView( const MemoryView<T>* other ) /*: Base<T>( other )*/
            {
                _size           = other->_size;
                
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
            
            MemoryView<T>* slice( const std::vector<size_dim> boundaries )
            {
                MemoryView<T>* _tmp = new MemoryView<T>( _data, _dimensions, _size );
                _tmp->setRange<true>( boundaries );
                
                return _tmp;
            }
            
            // Used only by Cython.
            MemoryView<T>* slice( const size_m fCu, const size_m tCu, const size_m fM, const size_m tM, const size_m fR, const size_m tR, const size_m fCo, const size_m tCo )
            {
                MemoryView<T>* _tmp = new MemoryView<T>( _data, _dimensions, _size );
                _tmp->setRange<true>( { fCu, tCu, fM, tM, fR, tR, fCo, tCo } );
                
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
                for(size_dim i = 0; i < _size; i++)
                    _index += (_sizeDimensions[i] * _range._boundaries[i].first);
                
                for(size_dim i = 0; i < _size-2; i++)
                    _positions[i] = _index;
            }
            
        #ifndef NO_VECTORIALIZATION
            INLINE void loadVector()
            { _x_vec = (MM_VECT*) (_data + _index); }
        #endif
            
            INLINE bool isSliced()
            { return _range.isSliced(); }
            
            INLINE bool isSubBlock()
            {
                for(size_dim i = 1; i < _size; i++)
                    if(_range.shape(i) != _dimensions[i]) return false;
                return true;
            }
            
            INLINE size_dim toAlignment()
			{ return (block - (_index % block)) % block; }
            
            INLINE bool isAligned()
            { return _data != NULL && ((uintptr_t) &(_data[_index])) % BLOCK == 0; }
            
            INLINE bool isAlignable()
            { return _data != NULL && (BLOCK - (((uintptr_t) &(_data[0])) % BLOCK)) % sizeof( T ) == 0; }
            
            template<int offset>
            INLINE void update( const int dim )
            {
                switch( dim ) {
                    case( 1 ): _index += offset; break;
                    case( 2 ): _index += _sizeDimensions[_size-2]; break;
                    default  : _index = _positions[_size-dim] += _sizeDimensions[_size-dim]; break;
                }
            }
            
            template<int dim, int offset>
            INLINE void update()
            {
                switch( dim ) {
                    case( 1 ): _index += offset; break;
                    case( 2 ): _index += _sizeDimensions[_size-2]; break;
                    default  : _index = _positions[_size-dim] += _sizeDimensions[_size-dim]; break;
                }
            }
            
            INLINE void update( const int dim, const int offset )
            {
                switch( dim ) {
                    case( 1 ): _index += offset; break;
                    case( 2 ): _index += _sizeDimensions[_size-2]; break;
                    default  : _index = _positions[_size-dim] += _sizeDimensions[_size-dim]; break;
                }
            }
            
            // ========================== //
            
            
            
            // === VECTOR OPERATIONS === //
            
            inline MemoryView<T>* operator+( void ){ return this; }
            
            inline MemoryView<T>* operator+( MemoryView<T>* in )
            {
                _range.checkSize( "operator+", __LINE__, in->_range );
                
                _fun = new Function<MemoryView,T>( new MemoryView<T>( this ),
                                                   new MemoryView<T>( in ),
                                                   Function<MemoryView,T>::sum );
                
            #ifndef NO_VECTORIALIZATION
                _fun->addVectFunction( Function<MemoryView,T>::v_sum );
            #endif
                
                _fun->addNext( in->_fun );
                
                return this;
            }
            
            inline MemoryView<T>* operator-( MemoryView<T>* in )
            {
                _range.checkSize( "operator-", __LINE__, in->_range );
                
                _fun = new Function<MemoryView,T>( new MemoryView<T>( this ),
                                                   new MemoryView<T>( in ),
                                                   Function<MemoryView,T>::sub );
                
            #ifndef NO_VECTORIALIZATION
                _fun->addVectFunction( Function<MemoryView,T>::v_sub );
            #endif
                
                _fun->addNext( in->_fun );
                
                return this;
            }
            
            inline MemoryView<T>* operator*( MemoryView<T>* in )
            {
                _range.checkSize( "operator*", __LINE__, in->_range );
                
                _fun = new Function<MemoryView,T>( new MemoryView<T>( this ),
                                                   new MemoryView<T>( in ),
                                                   Function<MemoryView,T>::mul );
                
            #ifndef NO_VECTORIALIZATION
                _fun->addVectFunction( Function<MemoryView,T>::v_mul );
            #endif
                
                _fun->addNext( in->_fun );
                
                return this;
            }
            
            inline MemoryView<T>* operator*( const T& value )
            {
                _fun = new Function<MemoryView,T>( new MemoryView<T>( this ), NULL,
                                                   Function<MemoryView,T>::mul, value );
                
            #ifndef NO_VECTORIALIZATION
                _fun->addConstFunction( Function<MemoryView,T>::v_mul );
            #endif
                
                return this;
            }
            
            inline MemoryView<T>* operator*=( MemoryView<T>* in )
            {
                _range.checkSize( "operator*=", __LINE__, in->_range );
                
                COMPUTE_OP_MULTI( *=, MM_MUL( _x_vec[i], fun->apply_vect( i ) ), MM_MUL( _x_vec[i], in->_x_vec[i] ) );
                return this;
            }
            
            inline MemoryView<T>* operator*=( const T& val )
            {
                const T value ALIGN = val;
                COMPUTE_OP_CONST( *=, MM_MUL( _x_vec[i], x_c_vec ) );
                return this;
            }
            
            inline MemoryView<T>* operator=( MemoryView<T>* in )
            {
                if(in->_fun == NULL && !isSliced()) {
                    // TODO Set the values of the input matrix into the destination matrix.
                    return in;
                }
                
                _range.checkSize( "operator=", __LINE__, in->_range );
                
                COMPUTE_OP_MULTI( =, fun->apply_vect( i ), in->_x_vec[i] );
                return this;
            }
            
            inline MemoryView<T>* operator=( const T& val )
            {
                const T value ALIGN = val;
                COMPUTE_OP_CONST( =, x_c_vec );
                return this;
            }
            
            inline MemoryView<T>* operator+=( MemoryView<T>* in )
            {
                _range.checkSize( "operator+=", __LINE__, in->_range );
                
                COMPUTE_OP_MULTI( +=, MM_ADD( _x_vec[i], fun->apply_vect( i ) ), MM_ADD( _x_vec[i], in->_x_vec[i] ) );
                return this;
            }
            
            inline MemoryView<T>* operator-=( MemoryView<T>* in )
            {
                _range.checkSize( "operator-=", __LINE__, in->_range );
                
                COMPUTE_OP_MULTI( -=, MM_SUB( _x_vec[i], fun->apply_vect( i ) ), MM_SUB( _x_vec[i], in->_x_vec[i] ) );
                return this;
            }
            
            inline MemoryView<T>* operator[]( const char* idx )
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
                        CHECK_INDEX( pos, _dimensions[dim] );
                        
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
            
            void curl_H( MemoryView<T>* IN, MemoryView<T>* OUT, MemoryView<T>* D, MemoryView<T>* curl, const T sc, const bool first )
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
            
            void curl_E( MemoryView<T>* IN, MemoryView<T>* OUT, MemoryView<T>* D, MemoryView<T>* curl, const T sc, const bool last )
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
            
            void printSize()
            {
                printf( "RANGES = [" );
                for(size_dim i = 0; i < _size-1; i++)
                    printf( "(%ld, %ld), ", _range._boundaries[i].first, _range._boundaries[i].second );
                printf( "(%ld, %ld)", _range._boundaries[_size-1].first, _range._boundaries[_size-1].second );
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
            ~MemoryView()
            { if(_data != NULL) _mm_free( _data ); }
    };
}

#endif /* _MEMORY_VIEW_H */
