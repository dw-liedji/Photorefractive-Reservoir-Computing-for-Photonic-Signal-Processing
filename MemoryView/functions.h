
/*
 *  functions.h
 *
 *  Created on: 13 dec 2016
 *  Authors: Stefano Ceccotti & Tommaso Catuogno
*/

#ifndef _FUNCTIONS_H
#define _FUNCTIONS_H

#include <typeinfo>

#include "settings.h"



#define NON_LINEAR_SEQUENTIAL_OP( VALUE, OP, UPDATE )         \
    const size_dim SIZE = _range.shape( _size-1 );            \
    size_dim i;                                               \
    for(i = 0; i < _size-1; i++)                              \
        _indices[i] = _range._boundaries[i].first;            \
                                                              \
    while(i >= 0) {                                           \
        /* Scan the last dimension.*/                         \
        for(size_dim j = 0; j < SIZE; j++)                    \
            _data[_index + j] OP VALUE;                       \
                                                              \
        for(i = _size-2; i >= 0; i--) {                       \
            update<0>( _size-i );                             \
            UPDATE;                                           \
            if(++_indices[i] < _range._boundaries[i].second)  \
                break;                                        \
            else /* Reached the end restart the index. */     \
                _indices[i] = _range._boundaries[i].first;    \
        }                                                     \
    }
	
/* Iterations performed in a linear sequential manner. */
#define LINEAR_SEQUENTIAL_OP( OP, ITERATIONS, R_VALUE ) \
    for(size_dim i = 0; i < ITERATIONS; i++)            \
        _data[_index++] OP R_VALUE;


#ifdef NO_VECTORIALIZATION
	// Empty declarations.
	#define COMPUTE_OP_VECTORIALIZED( VECT_OP, R_VALUE, OP, UPDATE ) ;
	#define COMPUTE_OP_CONST_VECTORIALIZED( VECT_OP, R_VALUE, OP, UPDATE ) ;
	#define COMPUTE_OP_BINARY_VECTORIALIZED( VECT_OP, OP ) ;
#else
    /* Align the vector to the BLOCK-th byte. */
    #define ALIGNMENT( OP, R_VALUE, UPDATE )    \
        size_dim i;                             \
        const size_dim toAlign = toAlignment(); \
        for(i = 0; i < toAlign; i++)            \
            _data[_index++] OP R_VALUE;         \
        UPDATE;
    
    #define COMPUTE_OP_VECTORIALIZED( VECT_OP, R_VALUE, OP, UPDATE ) \
        ALIGNMENT( OP, R_VALUE, (fun->update( 1, i )) )              \
                                                                     \
        const size_dim n    = (N - toAlign) / block;                 \
        const size_dim rest = (N - toAlign) % block;                 \
        loadVector();                                                \
        fun->loadVector();                                           \
        for(i = 0; i < n; i++) {                                     \
            _x_vec[i] = VECT_OP;                                     \
            UPDATE;                                                  \
        }                                                            \
                                                                     \
        LINEAR_SEQUENTIAL_OP( OP, rest, R_VALUE );
	
    #define COMPUTE_OP_CONST_VECTORIALIZED( VECT_OP, R_VALUE, OP ) \
        ALIGNMENT( OP, R_VALUE,  )                                 \
                                                                   \
        const size_dim n    = (N - toAlign) / block;               \
        const size_dim rest = (N - toAlign) % block;               \
        _x_vec = (MM_VECT*) (_data + _index);                      \
        T _a_val[sizeof(T)] ALIGN;                                 \
        for(uint64_t i = 0; i < sizeof(T); i++) _a_val[i] = value; \
        MM_VECT x_c_vec = ((MM_VECT*) _a_val)[0];                  \
        for(i = 0; i < n; i++, _index += block)                    \
            _x_vec[i] = VECT_OP;                                   \
                                                                   \
        LINEAR_SEQUENTIAL_OP( OP, rest, R_VALUE );
    
    #define COMPUTE_OP_BINARY_VECTORIALIZED( VECT_OP, OP )         \
        ALIGNMENT( OP, in->_data[in->_index++],  )                 \
                                                                   \
        const size_dim n    = (N - toAlign) / block;               \
        const size_dim rest = (N - toAlign) % block;               \
        fun->loadVector();                                         \
        size_dim inIndex = in->_index;                             \
        for(i = 0; i < n; i++, _index += block, inIndex += block)  \
            _x_vec[i] = VECT_OP;                                   \
                                                                   \
        LINEAR_SEQUENTIAL_OP( OP, rest, in->_data[in->_index++] );
#endif


// Operation used when all the involved elements are vectors.
#define COMPUTE_OP_MULTI( OP, VECT_OP, BINARY_OP )                                                   \
    loadIndex();                                                                                     \
    Function<MemoryView,T>* fun = in->_fun;                                                          \
    if(fun != NULL) {                                                                                \
        fun->init();                                                                                 \
        if(!isSubBlock() || !fun->allSubBlocks()) {                                                  \
            /* Not linear sequential. */                                                             \
            NON_LINEAR_SEQUENTIAL_OP( fun->apply( j ), OP, fun->update<0>( _size-i ) );              \
        }                                                                                            \
        else{                                                                                        \
            /* Get the total size of the section. */                                                 \
            const size_dim N = _range.getTotalSize();                                                \
            if(!VECTORIALIZATION || !isAlignable() || !fun->allAlignable( toAlignment() )) {         \
                LINEAR_SEQUENTIAL_OP( OP, N, fun->apply( i ) );                                      \
            }                                                                                        \
            else {                                                                                   \
                /* Vectorialized solution. */                                                        \
                COMPUTE_OP_VECTORIALIZED( VECT_OP, fun->apply( i ), OP, (fun->update( 1, block )) ); \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
    else {                                                                                           \
        in->loadIndex();                                                                             \
        if(!isSubBlock() || !in->isSubBlock()) {                                                     \
            NON_LINEAR_SEQUENTIAL_OP( in->_data[in->_index+j], OP, in->update<0>( _size-i ) );       \
        }                                                                                            \
        else {                                                                                       \
            /* Get the total size of the section. */                                                 \
            const size_dim N = _range.getTotalSize();                                                \
            if(!VECTORIALIZATION || !isAlignable() || in->toAlignment() != toAlignment()) {          \
                LINEAR_SEQUENTIAL_OP( OP, N, in->_data[in->_index++] );                              \
            }                                                                                        \
            else {                                                                                   \
                /* Vectorialized binary solution. */                                                 \
                COMPUTE_OP_BINARY_VECTORIALIZED( BINARY_OP, OP );                                    \
            }                                                                                        \
        }                                                                                            \
    }


// Operation used when an involved element is a number.
#define COMPUTE_OP_CONST( OP, VECT_OP )                            \
    loadIndex();                                                   \
    if(!isSubBlock()) {                                            \
        NON_LINEAR_SEQUENTIAL_OP( value, OP,  );                   \
    }                                                              \
    else {                                                         \
        /* Get the total size of the section. */                   \
        const size_dim N = _range.getTotalSize();                  \
        if(!VECTORIALIZATION || !isAlignable()) {                  \
            LINEAR_SEQUENTIAL_OP( OP, N, value );                  \
        }                                                          \
        else {                                                     \
            /* Vectorialized solution. */                          \
            COMPUTE_OP_CONST_VECTORIALIZED( VECT_OP, value, OP );  \
        }                                                          \
    }




template<class M, typename T>
class Function
{
    public:
        M* _m1 = NULL; M* _m2 = NULL;
        T (*_f)( T, T );
    #ifndef NO_VECTORIALIZATION
        MM_VECT (*_vf)( MM_VECT, MM_VECT );
    #endif
    private:
        Function<M,T>* _next = NULL;
        T _value = -1;
        
    #ifndef NO_VECTORIALIZATION
        MM_VECT _c_vect ALIGN;
    #endif
    
    
    public:
        static INLINE T sum( T a, T b ) { return a+b; }
    #ifndef NO_VECTORIALIZATION
        static INLINE MM_VECT v_sum( MM_VECT a, MM_VECT b ){ return MM_ADD( a, b ); }
    #endif

        static INLINE T sub( T a, T b ) { return a-b; }
    #ifndef NO_VECTORIALIZATION
        static INLINE MM_VECT v_sub( MM_VECT a, MM_VECT b ){ return MM_SUB( a, b ); }
    #endif

        static INLINE T mul( T a, T b ) { return a*b; }
    #ifndef NO_VECTORIALIZATION
        static INLINE MM_VECT v_mul( MM_VECT a, MM_VECT b ){ return MM_MUL( a, b ); }
    #endif
    
    public:
        inline Function( M* m1, M* m2, T (*f)( T, T ), const T& val = 0 )
        { _m1 = m1; _m2 = m2; _f = f; _value = val; }
        
    #ifndef NO_VECTORIALIZATION
        inline void addVectFunction( MM_VECT (*vf)( MM_VECT, MM_VECT ) )
        { _vf = vf; }
        
        inline void addConstFunction( MM_VECT (*vf)( MM_VECT, MM_VECT ) )
        {
	        addVectFunction( vf );
	        T _a_val[sizeof(T)] ALIGN;
	        for(uint64_t i = 0; i < sizeof(T); i++) _a_val[i] = _value;
	        _c_vect = *((MM_VECT*) _a_val);
        }
    #endif
        
        inline void addNext( Function<M,T>* fun )
        { _next = fun; }
        
        
        
        
        
        
        inline void init()
        {
            _m1->loadIndex();
            if(_m2 != NULL) _m2->loadIndex();
            if(_next != NULL) _next->init();
        }
        
        inline bool allSubBlocks()
        {
            return _m1->isSubBlock() &&
                   ((_m2 != NULL) ? _m2->isSubBlock() : true) &&
                   ((_next != NULL) ? _next->allSubBlocks() : true);
        }
        
        /** Returns the alignment of each vector involved in the function. */
        inline bool allAligned()
        {
            return _m1->isAligned() &&
                   ((_m2 != NULL) ? _m2->isAligned() : true) &&
                   ((_next != NULL) ? _next->allAligned() : true);
        }
		
        inline bool allAlignable( const size_dim& toAlign )
        {
            return (_m1->isAlignable() && _m1->toAlignment() == toAlign) &&
                   ((_m2 != NULL) ? (_m2->isAlignable() && _m2->toAlignment() == toAlign) : true) &&
                   ((_next != NULL) ? _next->allAlignable( toAlign ) : true);
        }
        
    #ifndef NO_VECTORIALIZATION
        inline void loadVector()
        {
            _m1->loadVector();
            if(_m2 != NULL) _m2->loadVector();
            if(_next != NULL) _next->loadVector();
        }
    #endif
        
        template<int dim, int offset>
        inline void update()
        {
            if(_next != NULL) _next->update<dim, offset>();
            _m1->update<offset>( dim );
            if(_m2 != NULL) _m2->update<offset>( dim );
        }
        
        template<int offset>
        inline void update( const int &dim )
        {
            if(_next != NULL) _next->update<offset>( dim );
            _m1->update<offset>( dim );
            if(_m2 != NULL) _m2->update<offset>( dim );
        }
        
        inline void update( const int &dim, const int& offset )
        {
            if(_next != NULL) _next->update( dim, offset );
            _m1->update( dim, offset );
            if(_m2 != NULL) _m2->update( dim, offset );
        }
        
        
        
        inline T apply( const size_dim& offset )
        {
            if(_next == NULL) return _f( _m1->_data[_m1->_index + offset],
                                         (_m2 == NULL) ? _value :
                                         _m2->_data[_m2->_index + offset] );
            else return _f( _m1->_data[_m1->_index + offset], _next->apply( offset ) );
        }
    
    #ifndef NO_VECTORIALIZATION
        inline MM_VECT apply_vect( const int& offset )
        {
            if(_next == NULL) return _vf( _m1->_x_vec[offset],
                                          (_m2 == NULL) ? _c_vect :
                                          _m2->_x_vec[offset] );
            else return _vf( _m1->_x_vec[offset], _next->apply_vect( offset ) );
        }
    #endif
};

#endif /* _FUNCTIONS_H */
