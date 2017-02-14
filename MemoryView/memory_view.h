
/*
 *  memory_view.h
 *
 *  Created on: 12 dec 2016
 *  Author: Stefano Ceccotti
 *  Author: Tommaso Catuogno
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
using namespace functors;

namespace algebra
{
    template<typename T>
    class MemoryView /*: public Base<T>*/
    {
        public:
            T* _data = NULL;
            size_dim _index;
        #ifndef NO_VECTORIALIZATION
            MM_VECT(i)* _x_vec_i;
            MM_VECT()*  _x_vec_s;
            MM_VECT(d)* _x_vec_d;
        #endif
        
        private:
            Range _range;
            
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
                _data = (T*) calloc( _range.getTotalSize(), sizeof( T ) );
                //_data = (T*) _mm_malloc( total_size * sizeof( T ), BLOCK );
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
                memcpy( &_range, &(other->_range), sizeof( Range) );
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
            inline void setRange( const std::vector<size_dim> boundaries )
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
            
            MV_INLINE void copyFrom( MemoryView* other )
            {
                _size = other->_size;
                
                memcpy( _dimensions, other->_dimensions, sizeof( size_m ) * MAX_DIMENSIONS );
                memcpy( _sizeDimensions, other->_sizeDimensions, sizeof( size_dim ) * MAX_DIMENSIONS );
                
                memcpy( _indices, other->_indices, sizeof( size_m ) * MAX_DIMENSIONS );
                memcpy( _positions, other->_positions, sizeof( size_dim ) * MAX_DIMENSIONS );
                
                _data = other->_data;
                _range = other->_range;
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
            INLINE void loadVector()
            {
                if(IS_DOUBLE( T )) _x_vec_d = (MM_VECT(d)*) (_data + _index);
                else if(IS_FLOAT( T )) _x_vec_s = (MM_VECT()*) (_data + _index);
                else _x_vec_i = (MM_VECT(i)*) (_data + _index);
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
            
            template<int offset>
            MV_INLINE void update( const int dim )
            {
                switch( dim ) {
                    case( 1 ): _index += offset; break;
                    case( 2 ): _index += _sizeDimensions[_size-2]; break;
                    default  : _index = _positions[_size-dim] += _sizeDimensions[_size-dim]; break;
                }
            }
            
            template<int dim, int offset>
            MV_INLINE void update()
            {
                switch( dim ) {
                    case( 1 ): _index += offset; break;
                    case( 2 ): _index += _sizeDimensions[_size-2]; break;
                    default  : _index = _positions[_size-dim] += _sizeDimensions[_size-dim]; break;
                }
            }
            
            MV_INLINE void update( const int dim, const int offset )
            {
                switch( dim ) {
                    case( 1 ): _index += offset; break;
                    case( 2 ): _index += _sizeDimensions[_size-2]; break;
                    default  : _index = _positions[_size-dim] += _sizeDimensions[_size-dim]; break;
                }
            }
            
            // ========================== //
            
            
            
            // === VECTOR OPERATIONS === //
            
            inline MemoryView<T>* operator+( void )
            { return this; }
            
            inline SumEnd<MemoryView,T>* operator+( MemoryView<T>* in )
            {
                _range.checkSize( "operator+", __LINE__, in->_range );
                return new SumEnd<MemoryView,T>( new MemoryView<T>( this ), new MemoryView<T>( in ) );
            }
            
            template<typename Function>
            inline Sum<Function,MemoryView,T>* operator+( Function* f )
            {
                //TODO _range.checkSize( "operator+", __LINE__, f->getRange() );
                return new Sum<Function,MemoryView,T>( new MemoryView<T>( this ), f );
            }
            
            
            
            inline SubEnd<MemoryView,T>* operator-( MemoryView<T>* in )
            {
                _range.checkSize( "operator-", __LINE__, in->_range );
                return new SubEnd<MemoryView,T>( new MemoryView<T>( this ), new MemoryView<T>( in ) );
            }
            
            template<typename Function>
            inline Sub<Function,MemoryView,T>* operator-( Function* f )
            {
                //TODO _range.checkSize( "operator+", __LINE__, f->getRange() );
                return new Sub<Function,MemoryView,T>( new MemoryView<T>( this ), f );
            }
            
            
            
            inline MulEnd<MemoryView,T>* operator*( MemoryView<T>* in )
            {
                _range.checkSize( "operator*", __LINE__, in->_range );
                return new MulEnd<MemoryView,T>( new MemoryView<T>( this ), new MemoryView<T>( in ) );
            }
            
            template<typename Function>
            inline Mul<Function,MemoryView,T>* operator*( Function* f )
            {
                //TODO _range.checkSize( "operator+", __LINE__, f->getRange() );
                return new Mul<Function,MemoryView,T>( new MemoryView<T>( this ), f );
            }
            
            inline MulConst<MemoryView,T>* operator*( const T& value )
            { return new MulConst<MemoryView,T>( new MemoryView<T>( this ), value ); }
            
            
            
            inline MemoryView<T>* operator*=( MemoryView<T>* in )
            {
                _range.checkSize( "operator*=", __LINE__, in->_range );
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_BINARY( *=, MM_MULs()( L_VECT(s)[i], in->L_VECT(s)[i] ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_BINARY( *=, MM_MULd()( L_VECT(d)[i], in->L_VECT(d)[i] ), d, d ); }
                // TODO completare anche per short e char..
                //else if(typeid(T) == typeid(int)) {    COMPUTE_OP_BINARY( *=, MM_MUL( s )( x_vec, FUN_CALL_POINTER( fun, i ) ), MM_MUL( s )( x_vec1, x_vec2 ), i, i ); }
                
                return this;
            }
            
            template<typename Function>
            inline MemoryView<T>* operator*=( Function* fun )
            {
                // TODO _range.checkSize( "operator*=", __LINE__, fun->getRange() );
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_MULTI( *=, MM_MULs()( L_VECT(s)[i], fun->apply_vect_f( i ) ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_MULTI( *=, MM_MULd()( L_VECT(d)[i], fun->apply_vect_d( i ) ), d, d ); }
                // TODO completare anche per short e char..
                //else if(IS_SHORT(T)) {    COMPUTE_OP_MULTI( *=, MM_MUL(i,16)( L_VECT(i)[i], in->apply_vect_i16( i ) ), i, i ); }
                //else if(IS_CHAR(T)) {    COMPUTE_OP_MULTI( *=, MM_MUL(i,8)( L_VECT(i)[i], in->apply_vect_i8( i ) ), i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator*=( const T& val )
            {
                const T value ALIGN = val;
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_CONST( *=, MM_MULs()( L_VECT(s)[i], x_c_vec ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_CONST( *=, MM_MULd()( L_VECT(d)[i], x_c_vec ), d, d ); }
                //else if(typeid(T) == typeid(int)) {    COMPUTE_OP_CONST( *=, MM_MUL(i,8)( x_vec1, x_vec2 ), i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator=( MemoryView<T>* in )
            {
                _range.checkSize( "operator=", __LINE__, in->_range );
                
                if(!isSliced())
                    this->copyFrom( in );
                else {
                    if(IS_FLOAT( T )) {       COMPUTE_OP_BINARY( =, in->L_VECT(s)[i],  , s ); }
                    else if(IS_DOUBLE( T )) { COMPUTE_OP_BINARY( =, in->L_VECT(d)[i], d, d ); }
                    //else if(typeid(T) == typeid(int)) {    COMPUTE_OP_MULTI( =, x_vec1 = x_vec2, i, i ); }
                }
                
                return this;
            }
            
            template<typename Function>
            inline MemoryView<T>* operator=( Function* fun )
            {
                // TODO _range.checkSize( "operator*=", __LINE__, fun->getRange() );
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_MULTI( =, fun->apply_vect_f( i ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_MULTI( =, fun->apply_vect_d( i ), d, d ); }
                // TODO completare anche per short e char..
                //else if(IS_SHORT(T)) {    COMPUTE_OP_MULTI( =, fun->apply_vect_i16( i ), i, i ); }
                //else if(IS_CHAR(T)) {    COMPUTE_OP_MULTI( =, fun->apply_vect_i8( i ), i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator=( const T& val )
            {
                const T value ALIGN = val;
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_CONST( =, x_c_vec,  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_CONST( =, x_c_vec, d, d ); }
                //else if(typeid(T) == typeid(int)) {    COMPUTE_OP_CONST( *=, x_c_vec, i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator+=( MemoryView<T>* in )
            { 
                _range.checkSize( "operator+=", __LINE__, in->_range );
                if(IS_FLOAT( T )) {       COMPUTE_OP_BINARY( +=, MM_ADD(s,)( L_VECT(s)[i], in->L_VECT(s)[i] ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_BINARY( +=, MM_ADD(d,)( L_VECT(d)[i], in->L_VECT(d)[i] ), d, d ); }
                //else if(typeid(T) == typeid(int)) {    COMPUTE_OP_BINARY( +=, MM_ADD( s )( x_vec, FUN_CALL_POINTER( fun, i ) ), MM_ADD( s )( x_vec1, x_vec2 ), i, i ); }
                
                return this;
            }
            
            template<typename Function>
            inline MemoryView<T>* operator+=( Function* fun )
            {
                // TODO _range.checkSize( "operator*=", __LINE__, f->getRange() );
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_MULTI( +=, MM_ADD(s,)( L_VECT(s)[i], fun->apply_vect_f(i) ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_MULTI( +=, MM_ADD(d,)( L_VECT(d)[i], fun->apply_vect_d(i) ), d, d ); }
                // TODO completare anche per short e char..
                //else if(IS_SHORT(T)) {    COMPUTE_OP_MULTI( +=, MM_ADD(i,16)( L_VECT(i)[i], fun->apply_vect_i16( i ) ), i, i ); }
                //else if(IS_CHAR(T)) {    COMPUTE_OP_MULTI( +=, MM_ADD(i,8)( L_VECT(i)[i], fun->apply_vect_i8( i ) ), i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator-=( MemoryView<T>* in )
            {
                _range.checkSize( "operator-=", __LINE__, in->_range );
                if(IS_FLOAT( T )) {       COMPUTE_OP_BINARY( -=, MM_SUB(s,)( L_VECT(s)[i], in->L_VECT(s)[i] ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_BINARY( -=, MM_SUB(d,)( L_VECT(d)[i], in->L_VECT(d)[i] ), d, d ); }
                //else if(typeid(T) == typeid(int)) {    COMPUTE_OP_BINARY( -=, MM_SUB( s )( x_vec, FUN_CALL_POINTER( fun, i ) ), MM_SUB( s )( x_vec1, x_vec2 ), i, i ); }
                
                return this;
            }
            
            template<typename Function>
            inline MemoryView<T>* operator-=( Function* fun )
            {
                // TODO _range.checkSize( "operator*=", __LINE__, f->getRange() );
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_MULTI( -=, MM_SUB(s,)( L_VECT(s)[i], fun->apply_vect_f(i) ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_MULTI( -=, MM_SUB(d,)( L_VECT(d)[i], fun->apply_vect_d(i) ), d, d ); }
                // TODO completare anche per short e char..
                //else if(IS_SHORT(T)) {    COMPUTE_OP_MULTI( -=, MM_SUB(i,16)( L_VECT(i)[i], fun->apply_vect_i16( i ) ), i, i ); }
                //else if(IS_CHAR(T)) {    COMPUTE_OP_MULTI( -=, MM_SUB(i,8)( L_VECT(i)[i], fun->apply_vect_i8( i ) ), i, i ); }
                
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
                        T value = get( indices );
                        if(value < 0) printf( (i < to-1) ? "%.8lf " : "%.8lf", value );
                        else printf( (i < to-1) ? " %.8lf " : " %.8lf", value );
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
            ~MemoryView(){ if(_data != NULL) free( _data ); }
    };
}

#endif /* _MEMORY_VIEW_H */
