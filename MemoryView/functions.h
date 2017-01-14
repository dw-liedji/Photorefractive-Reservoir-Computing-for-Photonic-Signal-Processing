
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
	
/* Iterations performed in a linearly sequential manner. */
#define LINEAR_SEQUENTIAL_OP( OP, ITERATIONS, R_VALUE ) \
    for(size_dim i = 0; i < ITERATIONS; i++)            \
        _data[_index++] OP R_VALUE;


#ifdef NO_VECTORIALIZATION
	// Empty declarations.
	#define COMPUTE_OP_VECTORIALIZED( VECT_OP, R_VALUE, OP, V, UPDATE, type_suffix, op_suffix ) ;
	#define COMPUTE_OP_CONST_VECTORIALIZED( VECT_OP, R_VALUE, OP, V, type_suffix, op_suffix ) ;
	#define COMPUTE_OP_BINARY_VECTORIALIZED( VECT_OP, OP, V, type_suffix, op_suffix ) ;
#else
    /* Align the vector to the BLOCK-th byte. */
    #define ALIGNMENT( OP, R_VALUE, UPDATE )    \
        size_dim i;                             \
        const size_dim toAlign = toAlignment(); \
        for(i = 0; i < toAlign; i++)            \
            _data[_index++] OP R_VALUE;         \
        UPDATE;
    
	#define COMPUTE_OP_VECTORIALIZED( VECT_OP, R_VALUE, OP, V, UPDATE, type_suffix, op_suffix ) \
        ALIGNMENT( OP, R_VALUE, (fun->update( 1, i )) )                                         \
                                                                                                \
		const size_dim n    = (N - toAlign) / block;                                            \
		const size_dim rest = (N - toAlign) % block;                                            \
		MM_VECT(type_suffix) x_vec;                                                             \
		for(i = 0; i < n; i++, _index += block) {                                               \
			x_vec = MM_LOAD(op_suffix)( (V*) (_data + _index) );                                \
			x_vec = VECT_OP;                                                                    \
			MM_STORE(op_suffix)( (V*) (_data + _index), x_vec );                                \
			UPDATE;                                                                             \
		}                                                                                       \
                                                                                                \
		LINEAR_SEQUENTIAL_OP( OP, rest, R_VALUE );
	
	#define COMPUTE_OP_CONST_VECTORIALIZED( VECT_OP, R_VALUE, OP, V, type_suffix, op_suffix )   \
	    ALIGNMENT( OP, R_VALUE,  )                                                              \
		                                                                                        \
		const size_dim n    = (N - toAlign) / block;                                            \
		const size_dim rest = (N - toAlign) % block;                                            \
		MM_VECT(type_suffix) x_vec1, x_vec2 = MM_SET1(op_suffix)( value );                      \
		for(i = 0; i < n; i++, _index += block) {                                               \
			x_vec1 = MM_LOAD(op_suffix)( (V*) (_data + _index) );                               \
			x_vec1 = VECT_OP;                                                                   \
			MM_STORE(op_suffix)( (V*) (_data + _index), x_vec1 );                               \
		}                                                                                       \
																			                    \
		LINEAR_SEQUENTIAL_OP( OP, rest, R_VALUE );

	#define COMPUTE_OP_BINARY_VECTORIALIZED( VECT_OP, OP, V, type_suffix, op_suffix )       \
	    ALIGNMENT( OP, in->_data[in->_index++],  )                                          \
		                                                                                    \
		const size_dim n    = (N - toAlign) / block;                                        \
		const size_dim rest = (N - toAlign) % block;                                        \
		size_dim inIndex = in->_index;                                                      \
		for(i = 0; i < n; i++, _index += block, inIndex += block) {                         \
			MM_VECT(type_suffix) x_vec1 = MM_LOAD(op_suffix)( (V*) (_data + _index) );      \
			MM_VECT(type_suffix) x_vec2 = MM_LOAD(op_suffix)( (V*) (in->_data + inIndex) ); \
			MM_STORE(op_suffix)( (V*) (_data + _index), VECT_OP );                          \
		}                                                                                   \
                                                                                            \
		LINEAR_SEQUENTIAL_OP( OP, rest, in->_data[in->_index++] );
#endif


// Operation used when all the involved elements are vectors.
#define COMPUTE_OP_MULTI( OP, VECT_OP, BINARY_OP, V, type_suffix, op_suffix )                  \
    loadIndex();                                                                               \
    Function<MemoryView,T>* fun = in->_fun;                                                    \
    if(fun != NULL) {                                                                          \
        fun->init();                                                                           \
        if(!isSubBlock() || !fun->allSubBlocks()) {                                            \
	        /* Not linear sequential. */                                                       \
            NON_LINEAR_SEQUENTIAL_OP( fun->apply( j ), OP, fun->update<0>( _size-i ) );        \
        }                                                                                      \
        else{                                                                                  \
            /* Get the total size of the section. */                                           \
            const size_dim N = _range.getTotalSize();                                          \
            if(!VECTORIALIZATION || !isAlignable() || !fun->allAlignable( toAlignment() )) {   \
                LINEAR_SEQUENTIAL_OP( OP, N, fun->apply( i ) );                                \
            }                                                                                  \
            else {                                                                             \
		        /* Vectorialized solution. */                                                  \
                COMPUTE_OP_VECTORIALIZED( VECT_OP, fun->apply( i ),                            \
                                          OP, V, (fun->update( 1, block )),                    \
                                          type_suffix, op_suffix );                            \
            }                                                                                  \
        }                                                                                      \
    }                                                                                          \
    else {                                                                                     \
        in->loadIndex();                                                                       \
        if(!isSubBlock() || !in->isSubBlock()) {                                               \
            NON_LINEAR_SEQUENTIAL_OP( in->_data[in->_index+j], OP, in->update<0>( _size-i ) ); \
        }                                                                                      \
        else {                                                                                 \
            /* Get the total size of the section. */                                           \
            const size_dim N = _range.getTotalSize();                                          \
            if(!VECTORIALIZATION || !isAlignable() || in->toAlignment() != toAlignment()) {    \
                LINEAR_SEQUENTIAL_OP( OP, N, in->_data[in->_index++] );                        \
            }                                                                                  \
            else {                                                                             \
                /* Vectorialized binary solution. */                                           \
                COMPUTE_OP_BINARY_VECTORIALIZED( BINARY_OP, OP, V, type_suffix, op_suffix );   \
            }                                                                                  \
        }                                                                                      \
    }



// Operation used when an involved element is a number.
#define COMPUTE_OP_CONST( OP, VECT_OP, V, type_suffix, op_suffix )                           \
    loadIndex();                                                                             \
    if(!isSubBlock()) {                                                                      \
        NON_LINEAR_SEQUENTIAL_OP( value, OP,  );                                             \
    }                                                                                        \
    else {                                                                                   \
        /* Get the total size of the section. */                                             \
        const size_dim N = _range.getTotalSize();                                            \
        if(!VECTORIALIZATION || !isAlignable()) {                                            \
            LINEAR_SEQUENTIAL_OP( OP, N, value );                                            \
        }                                                                                    \
        else {                                                                               \
            /* Vectorialized solution. */                                                    \
            COMPUTE_OP_CONST_VECTORIALIZED( VECT_OP, value, OP, V, type_suffix, op_suffix ); \
        }                                                                                    \
    }

// MM_TYPE must be one among: 'i', 'f' and 'd'.
#define FUN_CALL_POINTER( OBJ, MM_TYPE ) OBJ->apply_vect_ ## MM_TYPE()
#define FUN_CALL( OBJ, MM_TYPE ) OBJ.apply_vect_ ## MM_TYPE()


template<class M, typename T>
class Function
{
    public:
        M* _m1 = NULL; M* _m2 = NULL;
        T (*_f)( T, T );
    #ifndef NO_VECTORIALIZATION
        MM_VECT()  (*_vf_f)( MM_VECT(),  MM_VECT()  );
        MM_VECT(d) (*_vf_d)( MM_VECT(d), MM_VECT(d) );
        MM_VECT(i) (*_vf_i)( MM_VECT(i), MM_VECT(i) );
    #endif
    private:
        Function<M,T>* _next = NULL;
        T _value = -1;
        
    #ifndef NO_VECTORIALIZATION
        // Vectors used for the constant number Functions.
        MM_VECT()  _c_vect_f ALIGN;
        MM_VECT(i) _c_vect_i ALIGN;
        MM_VECT(d) _c_vect_d ALIGN;
    #endif
    
    
    public:
        static inline T sum( T a, T b ) { return a+b; }
    #ifndef NO_VECTORIALIZATION
        //FIXME non esiste MM_ADD(i).
        //static inline MM_VECT(i) v_sum_i( MM_VECT(i) a, MM_VECT(i) b ){ return MM_ADD(i)( a, b ); }
        static inline MM_VECT()  v_sum_f( MM_VECT()  a, MM_VECT()  b ){ return MM_ADD(s)( a, b ); }
        static inline MM_VECT(d) v_sum_d( MM_VECT(d) a, MM_VECT(d) b ){ return MM_ADD(d)( a, b ); }
    #endif

        static inline T sub( T a, T b ) { return a-b; }
    #ifndef NO_VECTORIALIZATION
        //FIXME non esiste MM_SUB(i).
        //static inline MM_VECT(i) v_sub_i( MM_VECT(i) a, MM_VECT(i) b ){ return MM_SUB(i)( a, b ); }
        static inline MM_VECT()  v_sub_f( MM_VECT()  a, MM_VECT()  b ){ return MM_SUB(s)( a, b ); }
        static inline MM_VECT(d) v_sub_d( MM_VECT(d) a, MM_VECT(d) b ){ return MM_SUB(d)( a, b ); }
    #endif

        static inline T mul( T a, T b ) { return a*b; }
    #ifndef NO_VECTORIALIZATION
        //FIXME non esiste MM_MUL(i).
        //static inline MM_VECT(i) v_mul_i( MM_VECT(i) a, MM_VECT(i) b ){ return MM_MUL(i)( a, b ); }
        static inline MM_VECT()  v_mul_f( MM_VECT()  a, MM_VECT()  b ){ return MM_MUL(s)( a, b ); }
        static inline MM_VECT(d) v_mul_d( MM_VECT(d) a, MM_VECT(d) b ){ return MM_MUL(d)( a, b ); }
    #endif
    
    public:
        inline Function( M* m1, M* m2, T (*f)( T, T ), const T& val = 0 )
        { _m1 = m1; _m2 = m2; _f = f; _value = val; }
        
    #ifndef NO_VECTORIALIZATION
        inline void addFunction_f( MM_VECT() (*vf)( MM_VECT(), MM_VECT() ) )
        { _vf_f = vf; }
        
        inline void addFunction_d( MM_VECT(d) (*vf)( MM_VECT(d), MM_VECT(d) ) )
        { _vf_d = vf; }
        
        inline void addFunction_i( MM_VECT(i) (*vf)( MM_VECT(i), MM_VECT(i) ) )
        { _vf_i = vf; }
    #endif
        
        inline void addConstFunction_f( MM_VECT() (*vf)( MM_VECT(), MM_VECT() ) )
        {
        #ifndef NO_VECTORIALIZATION
            addFunction_f( vf );
            const T _val ALIGN = _value;
            _c_vect_f = MM_SET1(s)( _val );
        #endif
        }
        
        inline void addConstFunction_d( MM_VECT(d) (*vf)( MM_VECT(d), MM_VECT(d) ) )
        {
        #ifndef NO_VECTORIALIZATION
            addFunction_d( vf );
            const T _val ALIGN = _value;
            _c_vect_d = MM_SET1(d)( _val );
        #endif
        }
        
        //FIXME non esiste MM_SET1(i).
        inline void addConstFunction_i( MM_VECT(i) (*vf)( MM_VECT(i), MM_VECT(i) ) )
        {
        #ifndef NO_VECTORIALIZATION
            addFunction_i( vf );
            //const T _val ALIGN = _value;
            //_c_vect_d = MM_SET1(i)( _val );
        #endif
        }
        
        inline void addNext( Function<M,T>* fun  )
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
        
        //inline T apply( const size_dim& offset )
        //{ return _f( offset ); }
    
    #ifndef NO_VECTORIALIZATION
        // TODO implementare..
        /*inline MM_VECT(i) apply_vect_i()
        {
            if(next == NULL) return _vf_i( MM_LOAD(i)( (int*) (_m1->_data + _m1->_index) ),
                                           (_m2 == NULL) ? c_vect_i :
                                           MM_LOAD(i)( (float*) (_m2->_data + _m2->_index) ) );
            else return _vf_i( MM_LOAD(i)( (int*) (_m1->_data + _m1->_index) ),
                               next->apply_vect_i() );
        }*/
        
        inline MM_VECT() apply_vect_f()
        {
            if(_next == NULL) return _vf_f( MM_LOAD(s)( (float*) (_m1->_data + _m1->_index) ),
                                            (_m2 == NULL) ? _c_vect_f :
                                            MM_LOAD(s)( (float*) (_m2->_data + _m2->_index) ) );
            else return _vf_f( MM_LOAD(s)( (float*) (_m1->_data + _m1->getIndex()) ),
                               _next->apply_vect_f() );
        }
        
        inline MM_VECT(d) apply_vect_d()
        {
            if(_next == NULL) return _vf_d( MM_LOAD(d)( (double*) (_m1->_data + _m1->_index) ),
                                            (_m2 == NULL) ? _c_vect_d :
                                            MM_LOAD(d)( (double*) (_m2->_data + _m2->_index) ) );
            else return _vf_d( MM_LOAD(d)( (double*) (_m1->_data + _m1->_index) ),
                               _next->apply_vect_d() );
        }
    #endif
};

#endif /* _FUNCTIONS_H */
