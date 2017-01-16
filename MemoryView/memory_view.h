
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
                //_data = (T*) calloc( _range.getTotalSize(), sizeof( T ) );
                _data = (T*) _mm_malloc( total_size * sizeof( T ), BLOCK );
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
                _size = other->_size;
                
                memcpy( _dimensions, other->_dimensions, sizeof( size_m ) * MAX_DIMENSIONS );
                memcpy( _sizeDimensions, other->_sizeDimensions, sizeof( size_dim ) * MAX_DIMENSIONS );
                
                memcpy( _indices, other->_indices, sizeof( size_m ) * MAX_DIMENSIONS );
                memcpy( _positions, other->_positions, sizeof( size_dim ) * MAX_DIMENSIONS );
                
                _data = other->_data;
                memcpy( &_range, &other->_range, sizeof( Range ) );
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
                if(typeid(T) == typeid(double))     _fun->addFunction_d( Function<MemoryView,T>::v_sum_d );
                else if(typeid(T) == typeid(float)) _fun->addFunction_f( Function<MemoryView,T>::v_sum_f );
                //else if(typeid(T) == typeid(int))   _fun->addFunction_f( Function<MemoryView,T>::v_sum_i );
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
                if(typeid(T) == typeid(double))     _fun->addFunction_d( Function<MemoryView,T>::v_sub_d );
                else if(typeid(T) == typeid(float)) _fun->addFunction_f( Function<MemoryView,T>::v_sub_f );
                //else if(typeid(T) == typeid(int))   _fun->addFunction_i( Function<MemoryView,T>::v_sub_i );
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
                if(typeid(T) == typeid(double))     _fun->addFunction_d( Function<MemoryView,T>::v_mul_d );
                else if(typeid(T) == typeid(float)) _fun->addFunction_f( Function<MemoryView,T>::v_mul_f );
                //else if(typeid(T) == typeid(int))   _fun->addFunction_i( Function<MemoryView,T>::v_mul_i );
            #endif
                
                _fun->addNext( in->_fun );
                
                return this;
            }
            
            inline MemoryView<T>* operator*( const T& value )
            {
                _fun = new Function<MemoryView,T>( new MemoryView<T>( this ), NULL,
                                                   Function<MemoryView,T>::mul, value );
                
            #ifndef NO_VECTORIALIZATION
                if(typeid(T) == typeid(double))     _fun->addConstFunction_d( Function<MemoryView,T>::v_mul_d );
                else if(typeid(T) == typeid(float)) _fun->addConstFunction_f( Function<MemoryView,T>::v_mul_f );
                //else if(typeid(T) == typeid(int))   _fun->addConstFunction_i( Function<MemoryView,T>::v_sum_i );
            #endif
                
                return this;
            }
            
            inline MemoryView<T>* operator*=( MemoryView<T>* in )
            {
                _range.checkSize( "operator*=", __LINE__, in->_range );
                
                if(typeid(T) == typeid(float)) {       COMPUTE_OP_MULTI( *=, MM_MUL( s )( x_vec, FUN_CALL_POINTER( fun, f ) ), MM_MUL( s )( x_vec1, x_vec2 ), float,   , s ); }
                else if(typeid(T) == typeid(double)) { COMPUTE_OP_MULTI( *=, MM_MUL( d )( x_vec, FUN_CALL_POINTER( fun, d ) ), MM_MUL( d )( x_vec1, x_vec2 ), double, d, d ); }
                //else if(typeid(T) == typeid(int)) {    COMPUTE_OP_MULTI( *=, MM_MUL( s )( x_vec, FUN_CALL_POINTER( fun, i ) ), MM_MUL( s )( x_vec1, x_vec2 ), int,    i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator*=( const T& val )
            {
                const T value ALIGN = val;
                
                if(typeid(T) == typeid(float)) {       COMPUTE_OP_CONST( *=, MM_MUL( s )( x_vec1, x_vec2 ), float,   , s ); }
                else if(typeid(T) == typeid(double)) { COMPUTE_OP_CONST( *=, MM_MUL( d )( x_vec1, x_vec2 ), double, d, d ); }
                //else if(typeid(T) == typeid(int)) {    COMPUTE_OP_CONST( *=, MM_MUL( i )( x_vec1, x_vec2 ), int,    i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator=( MemoryView<T>* in )
            {
                _range.checkSize( "operator=", __LINE__, in->_range );
                
                if(_fun == NULL && !isSliced())
                    return in;
                
                //COMPUTE_OP_MULTI( OP, VECT_OP, BINARY_OP, V, type_suffix, op_suffix )
                if(typeid(T) == typeid(float)) {       COMPUTE_OP_MULTI( =, FUN_CALL_POINTER( fun, f ), x_vec1 = x_vec2, float,   , s ); }
                else if(typeid(T) == typeid(double)) { COMPUTE_OP_MULTI( =, FUN_CALL_POINTER( fun, d ), x_vec1 = x_vec2, double, d, d ); }
                //else if(typeid(T) == typeid(int)) {    COMPUTE_OP_MULTI( =, FUN_CALL_POINTER( fun, i ), x_vec1 = x_vec2, int,    i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator=( const T& val )
            {
                const T value ALIGN = val;
                
                if(typeid(T) == typeid(float)) {       COMPUTE_OP_CONST( =, x_vec2, float,   , s ); }
                else if(typeid(T) == typeid(double)) { COMPUTE_OP_CONST( =, x_vec2, double, d, d ); }
                //else if(typeid(T) == typeid(int)) {    COMPUTE_OP_CONST( *=, x_vec2, int,    i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator+=( MemoryView<T>* in )
            { return plus_equals( in ); }
            
            inline MemoryView<T>* plus_equals( MemoryView<T>* in )
            {
                _range.checkSize( "operator+=", __LINE__, in->_range );
                if(typeid(T) == typeid(float)) {       COMPUTE_OP_MULTI( +=, MM_ADD( s )( x_vec, FUN_CALL_POINTER( fun, f ) ), MM_ADD( s )( x_vec1, x_vec2 ), float,   , s ); }
                else if(typeid(T) == typeid(double)) { COMPUTE_OP_MULTI( +=, MM_ADD( d )( x_vec, FUN_CALL_POINTER( fun, d ) ), MM_ADD( d )( x_vec1, x_vec2 ), double, d, d ); }
                //else if(typeid(T) == typeid(int)) {    COMPUTE_OP_MULTI( +=, MM_ADD( s )( x_vec, FUN_CALL_POINTER( fun, i ) ), MM_ADD( s )( x_vec1, x_vec2 ), int,    i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator-=( MemoryView<T>* in )
            { return minus_equals( in ); }
            
            inline MemoryView<T>* minus_equals( MemoryView<T>* in )
            {
                _range.checkSize( "operator-=", __LINE__, in->_range );
                if(typeid(T) == typeid(float)) {       COMPUTE_OP_MULTI( -=, MM_SUB( s )( x_vec, FUN_CALL_POINTER( fun, f ) ), MM_SUB( s )( x_vec1, x_vec2 ), float,   , s ); }
                else if(typeid(T) == typeid(double)) { COMPUTE_OP_MULTI( -=, MM_SUB( d )( x_vec, FUN_CALL_POINTER( fun, d ) ), MM_SUB( d )( x_vec1, x_vec2 ), double, d, d ); }
                //else if(typeid(T) == typeid(int)) {    COMPUTE_OP_MULTI( -=, MM_SUB( s )( x_vec, FUN_CALL_POINTER( fun, i ) ), MM_SUB( s )( x_vec1, x_vec2 ), int,    i, i ); }
                
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
            
            void curl_H( MemoryView<T>* IN, MemoryView<T>* OUT, MemoryView<T>* D,
                         MemoryView<T>* curl, const T sc, const bool first )
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
            
            void curl_E( MemoryView<T>* IN, MemoryView<T>* OUT, MemoryView<T>* D,
                         MemoryView<T>* curl, const T sc, const bool last )
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
