
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
    class MemoryView : public Base<MemoryView<T>,T>
    {
        public:
            using Base<MemoryView<T>,T>::_data;
            using Base<MemoryView<T>,T>::_index;
        #ifndef NO_VECTORIALIZATION
            using Base<MemoryView<T>,T>::_x_vec_i;
            using Base<MemoryView<T>,T>::_x_vec_s;
            using Base<MemoryView<T>,T>::_x_vec_d;
        #endif
            using Base<MemoryView<T>,T>::_range;
            
            using Base<MemoryView<T>,T>::_dimensions;
            using Base<MemoryView<T>,T>::_sizeDimensions;
            
            using Base<MemoryView<T>,T>::_indices;
            using Base<MemoryView<T>,T>::_positions;
            
            using Base<MemoryView<T>,T>::_size;
            
            using Base<MemoryView<T>,T>::VECTORIALIZATION;
        
        public:
            MemoryView() : Base<MemoryView<T>,T>() {}
            
            MemoryView( const std::vector<size_dim> dimensions ) : Base<MemoryView<T>,T>( dimensions )
            {}
            
            MemoryView( T* data, const std::vector<size_dim> dimensions ) : Base<MemoryView<T>,T>( data, dimensions )
            {}
            
            MemoryView( T* data, const size_dim* dimensions, const int& size ) : Base<MemoryView<T>,T>( data, dimensions, size )
            {}
            
            MemoryView( const MemoryView<T>* other ) : Base<MemoryView<T>,T>( other )
            {}
            
            // ======== UTILITY METHODS ======== //
            
            using Base<MemoryView<T>,T>::data;
            using Base<MemoryView<T>,T>::setRange;
            using Base<MemoryView<T>,T>::slice;
            using Base<MemoryView<T>,T>::getRange;
            using Base<MemoryView<T>,T>::get;
            using Base<MemoryView<T>,T>::getDimensions;
            using Base<MemoryView<T>,T>::copyFrom;
            using Base<MemoryView<T>,T>::getIndex;
            using Base<MemoryView<T>,T>::loadIndex;
            
        #ifndef NO_VECTORIALIZATION
            using Base<MemoryView<T>,T>::loadVector;
        #endif
            
            using Base<MemoryView<T>,T>::isSliced;
            using Base<MemoryView<T>,T>::isSubBlock;
            using Base<MemoryView<T>,T>::toAlignment;
            using Base<MemoryView<T>,T>::isAligned;
            using Base<MemoryView<T>,T>::isAlignable;
            //using Base<MemoryView<T>,T>::update;
            //using Base<MemoryView<T>,T>::template update<int offset>;
            //using Base<MemoryView<T>,T>::template<int,int> update<int,int>;
            
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
            
            using Base<MemoryView<T>,T>::update_dim;
            
            using Base<MemoryView<T>,T>::operator[];
            using Base<MemoryView<T>,T>::print_out;
            
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
                _range.checkSize( "operator+", __LINE__, f->getRange() );
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
                _range.checkSize( "operator+", __LINE__, f->getRange() );
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
                _range.checkSize( "operator+", __LINE__, f->getRange() );
                return new Mul<Function,MemoryView,T>( new MemoryView<T>( this ), f );
            }
            
            inline MulConst<MemoryView,T>* operator*( const T& value )
            { return new MulConst<MemoryView,T>( new MemoryView<T>( this ), value ); }
            
            
            
            inline MemoryView<T>* operator*=( MemoryView<T>* in )
            {
                _range.checkSize( "operator*=", __LINE__, in->_range );
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_BINARY( *=, MM_MUL(s,  )( L_VECT(s)[i], in->L_VECT(s)[i] ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_BINARY( *=, MM_MUL(d,  )( L_VECT(d)[i], in->L_VECT(d)[i] ), d, d ); }
                else if(IS_SHORT(T)) {    COMPUTE_OP_BINARY( *=, MM_MUL(i,16)( L_VECT(i)[i], in->L_VECT(i)[i] ), i, i ); }
                else if(IS_CHAR(T)) {     COMPUTE_OP_BINARY( *=, MM_MUL(i,8 )( L_VECT(i)[i], in->L_VECT(i)[i] ), i, i ); }
                
                return this;
            }
            
            template<typename Function>
            inline MemoryView<T>* operator*=( Function* fun )
            {
                _range.checkSize( "operator*=", __LINE__, fun->getRange() );
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_MULTI( *=, MM_MUL(s,  )( L_VECT(s)[i], fun->apply_vect_f( i ) ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_MULTI( *=, MM_MUL(d,  )( L_VECT(d)[i], fun->apply_vect_d( i ) ), d, d ); }
                else if(IS_SHORT(T)) {    COMPUTE_OP_MULTI( *=, MM_MUL(i,16)( L_VECT(i)[i], fun->apply_vect_i_16( i ) ), i, i ); }
                else if(IS_CHAR(T)) {     COMPUTE_OP_MULTI( *=, MM_MUL(i,8 )( L_VECT(i)[i], fun->apply_vect_i_8( i ) ), i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator*=( const T& val )
            {
                const T value ALIGN = val;
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_CONST( *=, MM_MUL(s,  )( L_VECT(s)[i], x_c_vec ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_CONST( *=, MM_MUL(d,  )( L_VECT(d)[i], x_c_vec ), d, d ); }
                else if(IS_SHORT(T)) {    COMPUTE_OP_CONST( *=, MM_MUL(i,16)( L_VECT(i)[i], x_c_vec ), i, i ); }
                else if(IS_CHAR(T)) {     COMPUTE_OP_CONST( *=, MM_MUL(i,8 )( L_VECT(i)[i], x_c_vec ), i, i ); }
                
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
                    else { /*Short and Char*/ COMPUTE_OP_BINARY( =, in->L_VECT(i)[i], i, i ); }
                }
                
                return this;
            }
            
            template<typename Function>
            inline MemoryView<T>* operator=( Function* fun )
            {
                _range.checkSize( "operator*=", __LINE__, fun->getRange() );
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_MULTI( =, fun->apply_vect_f( i ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_MULTI( =, fun->apply_vect_d( i ), d, d ); }
                else if(IS_SHORT(T)) {    COMPUTE_OP_MULTI( =, fun->apply_vect_i_16( i ), i, i ); }
                else if(IS_CHAR(T)) {     COMPUTE_OP_MULTI( =, fun->apply_vect_i_8( i ), i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator=( const T& val )
            {
                const T value ALIGN = val;
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_CONST( =, x_c_vec,  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_CONST( =, x_c_vec, d, d ); }
                else { /*Short and Char*/ COMPUTE_OP_CONST( =, x_c_vec, i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator+=( MemoryView<T>* in )
            { 
                _range.checkSize( "operator+=", __LINE__, in->_range );
                if(IS_FLOAT( T )) {       COMPUTE_OP_BINARY( +=, MM_ADD(s,  )( L_VECT(s)[i], in->L_VECT(s)[i] ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_BINARY( +=, MM_ADD(d,  )( L_VECT(d)[i], in->L_VECT(d)[i] ), d, d ); }
                else if(IS_SHORT(T)) {    COMPUTE_OP_BINARY( +=, MM_ADD(i,16)( L_VECT(i)[i], in->L_VECT(i)[i] ), i, i ); }
                else if(IS_CHAR(T)) {     COMPUTE_OP_BINARY( +=, MM_ADD(i, 8)( L_VECT(i)[i], in->L_VECT(i)[i] ), i, i ); }
                
                return this;
            }
            
            template<typename Function>
            inline MemoryView<T>* operator+=( Function* fun )
            {
                _range.checkSize( "operator*=", __LINE__, fun->getRange() );
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_MULTI( +=, MM_ADD(s,  )( L_VECT(s)[i], fun->apply_vect_f(i) ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_MULTI( +=, MM_ADD(d,  )( L_VECT(d)[i], fun->apply_vect_d(i) ), d, d ); }
                else if(IS_SHORT(T)) {    COMPUTE_OP_MULTI( +=, MM_ADD(i,16)( L_VECT(i)[i], fun->apply_vect_i_16( i ) ), i, i ); }
                else if(IS_CHAR(T)) {     COMPUTE_OP_MULTI( +=, MM_ADD(i, 8)( L_VECT(i)[i], fun->apply_vect_i_8( i ) ), i, i ); }
                
                return this;
            }
            
            inline MemoryView<T>* operator-=( MemoryView<T>* in )
            {
                _range.checkSize( "operator-=", __LINE__, in->_range );
                if(IS_FLOAT( T )) {       COMPUTE_OP_BINARY( -=, MM_SUB(s,  )( L_VECT(s)[i], in->L_VECT(s)[i] ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_BINARY( -=, MM_SUB(d,  )( L_VECT(d)[i], in->L_VECT(d)[i] ), d, d ); }
                else if(IS_SHORT(T)) {    COMPUTE_OP_BINARY( -=, MM_SUB(i,16)( L_VECT(i)[i], in->L_VECT(i)[i] ), i, i ); }
                else if(IS_CHAR(T)) {     COMPUTE_OP_BINARY( -=, MM_SUB(i, 8)( L_VECT(i)[i], in->L_VECT(i)[i] ), i, i ); }
                
                return this;
            }
            
            template<typename Function>
            inline MemoryView<T>* operator-=( Function* fun )
            {
                _range.checkSize( "operator*=", __LINE__, fun->getRange() );
                
                if(IS_FLOAT( T )) {       COMPUTE_OP_MULTI( -=, MM_SUB(s,  )( L_VECT(s)[i], fun->apply_vect_f(i) ),  , s ); }
                else if(IS_DOUBLE( T )) { COMPUTE_OP_MULTI( -=, MM_SUB(d,  )( L_VECT(d)[i], fun->apply_vect_d(i) ), d, d ); }
                else if(IS_SHORT(T)) {    COMPUTE_OP_MULTI( -=, MM_SUB(i,16)( L_VECT(i)[i], fun->apply_vect_i_16( i ) ), i, i ); }
                else if(IS_CHAR(T)) {     COMPUTE_OP_MULTI( -=, MM_SUB(i,8 )( L_VECT(i)[i], fun->apply_vect_i_8( i ) ), i, i ); }
                
                return this;
            }
            
            // ========================== //
        
        public:
            ~MemoryView()
            { if(_data != NULL) free( _data ); }
    };
}

#endif /* _MEMORY_VIEW_H */
