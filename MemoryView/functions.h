
/*
 *  functions.h
 *
 *  Created on: 13 dec 2016
 *  Author: Stefano Ceccotti
 *  Author: Tommaso Catuogno
*/

#ifndef _FUNCTIONS_H
#define _FUNCTIONS_H

#include <typeinfo>

#include "range.h"
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
	#define COMPUTE_OP_VECTORIALIZED( VECT_OP, R_VALUE, OP, UPDATE, type_suffix, op_suffix ) ;
	#define COMPUTE_OP_CONST_VECTORIALIZED( VECT_OP, R_VALUE, OP, type_suffix, op_suffix ) ;
	#define COMPUTE_OP_BINARY_VECTORIALIZED( VECT_OP, OP, type_suffix, op_suffix ) ;
#else
    /* Align the vector to the BLOCK-th byte. */
    #define ALIGNMENT( OP, R_VALUE, UPDATE )    \
        size_dim i;                             \
        const size_dim toAlign = toAlignment(); \
        for(i = 0; i < toAlign; i++)            \
            _data[_index++] OP R_VALUE;         \
        UPDATE;
    
	#define COMPUTE_OP_VECTORIALIZED( VECT_OP, R_VALUE, OP, UPDATE, type_suffix, op_suffix ) \
        ALIGNMENT( OP, R_VALUE, (fun->update( 1, i )) )                                      \
                                                                                             \
		const size_dim n    = (N - toAlign) / block;                                         \
		const size_dim rest = (N - toAlign) % block;                                         \
		loadVector();                                                                        \
        fun->loadVector();                                                                   \
		for(i = 0; i < n; i++, _index += block) {                                            \
			L_VECT(op_suffix)[i] = VECT_OP;                                                  \
			UPDATE;                                                                          \
		}                                                                                    \
                                                                                             \
		LINEAR_SEQUENTIAL_OP( OP, rest, R_VALUE );
	
	#define COMPUTE_OP_CONST_VECTORIALIZED( VECT_OP, R_VALUE, OP, type_suffix, op_suffix ) \
	    ALIGNMENT( OP, R_VALUE,  )                                                         \
		                                                                                   \
		const size_dim n    = (N - toAlign) / block;                                       \
		const size_dim rest = (N - toAlign) % block;                                       \
		loadVector();                                                                      \
        T _a_val[sizeof(T)] ALIGN;                                                         \
        for(uint64_t i = 0; i < sizeof(T); i++) _a_val[i] = value;                         \
        MM_VECT( type_suffix ) x_c_vec = *((MM_VECT( type_suffix )*) _a_val);              \
		for(i = 0; i < n; i++, _index += block)                                            \
			L_VECT(op_suffix)[i] = VECT_OP;                                                \
																			               \
		LINEAR_SEQUENTIAL_OP( OP, rest, R_VALUE );
    
	#define COMPUTE_OP_BINARY_VECTORIALIZED( VECT_OP, OP, type_suffix, op_suffix ) \
	    ALIGNMENT( OP, in->_data[in->_index++],  )                                 \
		                                                                           \
		const size_dim n    = (N - toAlign) / block;                               \
		const size_dim rest = (N - toAlign) % block;                               \
		loadVector(); in->loadVector();                                            \
		for(i = 0; i < n; i++, _index += block, in->_index += block)               \
			L_VECT(op_suffix)[i] = VECT_OP;                                        \
                                                                                   \
		LINEAR_SEQUENTIAL_OP( OP, rest, in->_data[in->_index++] );
#endif


// Operation used when all the involved elements are vectors.
#define COMPUTE_OP_MULTI( OP, VECT_OP, type_suffix, op_suffix )                          \
    loadIndex();                                                                         \
    fun->init();                                                                         \
    if(!isSubBlock() || !fun->allSubBlocks()) {                                          \
        /* Not linear sequential. */                                                     \
        NON_LINEAR_SEQUENTIAL_OP( fun->apply( j ), OP, fun->update<0>( _size-i ) );      \
    }                                                                                    \
    else{                                                                                \
        /* Get the total size of the section. */                                         \
        const size_dim N = _range.getTotalSize();                                        \
        if(!VECTORIALIZATION || !isAlignable() || !fun->allAlignable( toAlignment() )) { \
            LINEAR_SEQUENTIAL_OP( OP, N, fun->apply( i ) );                              \
        }                                                                                \
        else {                                                                           \
	        /* Vectorialized solution. */                                                \
            COMPUTE_OP_VECTORIALIZED( VECT_OP, fun->apply( i ),                          \
                                      OP, (fun->update( 1, block )),                     \
                                      type_suffix, op_suffix );                          \
        }                                                                                \
    }

#define COMPUTE_OP_BINARY( OP, VECT_OP, type_suffix, op_suffix )                           \
    loadIndex();                                                                           \
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
            COMPUTE_OP_BINARY_VECTORIALIZED( VECT_OP, OP, type_suffix, op_suffix );        \
        }                                                                                  \
    }



// Operation used when an involved element is a number.
#define COMPUTE_OP_CONST( OP, VECT_OP, type_suffix, op_suffix )                           \
    loadIndex();                                                                          \
    if(!isSubBlock()) {                                                                   \
        NON_LINEAR_SEQUENTIAL_OP( value, OP,  );                                          \
    }                                                                                     \
    else {                                                                                \
        /* Get the total size of the section. */                                          \
        const size_dim N = _range.getTotalSize();                                         \
        if(!VECTORIALIZATION || !isAlignable()) {                                         \
            LINEAR_SEQUENTIAL_OP( OP, N, value );                                         \
        }                                                                                 \
        else {                                                                            \
            /* Vectorialized solution. */                                                 \
            COMPUTE_OP_CONST_VECTORIALIZED( VECT_OP, value, OP, type_suffix, op_suffix ); \
        }                                                                                 \
    }



namespace functors
{
    template<typename Function, class M, typename T>
    class Operation
    {
        protected:
            M* _m = NULL;
            Function* _next = NULL;
            
        #ifndef NO_VECTORIALIZATION
            // Vectors used for constant functions.
            MM_VECT()  _c_vect_s ALIGN;
            MM_VECT(i) _c_vect_i ALIGN;
            MM_VECT(d) _c_vect_d ALIGN;
        #endif
        
            Operation( M* m, Function* f )
            { _m = m; _next = f; }
        
        public:
    		inline void init()
            {  _m->loadIndex(); _next->init(); }

    		inline Range getRange()
            { return _m->getRange(); }
            
            inline bool allSubBlocks()
            { return _m->isSubBlock() && _next->allSubBlocks(); }
            
            /** Returns the alignment of each vector involved in the function. */
            inline bool allAligned()
            { return _m->isAligned() && _next->allAligned(); }
		
            inline bool allAlignable( const size_dim& toAlign )
            { return _m->isAlignable() && _m->toAlignment() == toAlign && _next->allAlignable( toAlign ); }
            
            template<int dim, int offset>
            inline void update()
            { _next->update<dim, offset>(); _m->update<offset>( dim ); }
            
            template<int offset>
            inline void update( const int &dim )
            { _next->update<offset>( dim ); _m->update<offset>( dim ); }
            
            inline void update( const int &dim, const int& offset )
            { _next->update( dim, offset ); _m->update( dim, offset ); }
            
        #ifndef NO_VECTORIALIZATION
            inline void loadVector()
            { _m->loadVector(); _next->loadVector(); }
        #endif
    		
            virtual MV_INLINE T apply( const size_dim& offset ) = 0;
        #ifndef NO_VECTORIALIZATION
            virtual MV_INLINE MM_VECT()  apply_vect_f( const size_dim& offset ) = 0;
            virtual MV_INLINE MM_VECT(d) apply_vect_d( const size_dim& offset ) = 0;
            //virtual MV_INLINE MM_VECT(i) apply_vect_i( const size_dim& offset ) = 0;
        #endif
    };
    
    
    template<class M, typename T>
    class BinaryOperation
    {
        protected:
            M* _m1 = NULL;
            M* _m2 = NULL;
            
        public:
            BinaryOperation( M* m1, M* m2 )
            { _m1 = m1; _m2 = m2; }
            
            inline void init() { _m1->loadIndex(); _m2->loadIndex(); }
            inline Range getRange() { return _m1->getRange(); }
            inline bool allSubBlocks()
            { return _m1->isSubBlock() && _m2->isSubBlock(); }
            /** Returns the alignment of each vector involved in the function. */
            inline bool allAligned()
            { return _m1->isAligned() && _m2->isAligned(); }
		    inline bool allAlignable( const size_dim& toAlign )
		    { return _m1->isAlignable() && _m1->toAlignment() == toAlign &&
		             _m2->isAlignable() && _m2->toAlignment() == toAlign; }
            
            template<int dim, int offset>
            inline void update()
            { _m1->update<offset>( dim ); _m2->update<offset>( dim ); }
            
            template<int offset>
            inline void update( const int &dim )
            { _m1->update<offset>( dim ); _m2->update<offset>( dim ); }
            
            inline void update( const int &dim, const int& offset )
            { _m1->update( dim, offset ); _m2->update( dim, offset ); }
            
        #ifndef NO_VECTORIALIZATION
            inline void loadVector()
            { _m1->loadVector(); _m2->loadVector(); }
        #endif
        
            virtual MV_INLINE T apply( const size_dim& offset ) = 0;
        #ifndef NO_VECTORIALIZATION
            virtual MV_INLINE MM_VECT()  apply_vect_f( const size_dim& offset ) = 0;
            virtual MV_INLINE MM_VECT(d) apply_vect_d( const size_dim& offset ) = 0;
            //virtual MV_INLINE MM_VECT(i) apply_vect_i( const size_dim& offset ) = 0;
        #endif
    };
    
    template<class M, typename T>
    class ConstOperation
    {
        protected:
            M* _m = NULL;
            T _value = -1;
            
        #ifndef NO_VECTORIALIZATION
            // Vectors used for the constant number Functions.
            MM_VECT()  _c_vect_s ALIGN;
            MM_VECT(i) _c_vect_i ALIGN;
            MM_VECT(d) _c_vect_d ALIGN;
        #endif
            
        public:
            ConstOperation( M* m, const T& val = 0 )
            { _m = m; _value = val; }
            
            inline void init() { _m->loadIndex(); }
            inline Range getRange() { return _m->getRange(); }
            inline bool allSubBlocks()
            { return _m->isSubBlock(); }
            /** Returns the alignment of each vector involved in the function. */
            inline bool allAligned()
            { return _m->isAligned(); }
		    inline bool allAlignable( const size_dim& toAlign )
		    { return _m->isAlignable() && _m->toAlignment() == toAlign; }
            
            template<int dim, int offset>
            inline void update()
            { _m->update<offset>( dim ); }
            
            template<int offset>
            inline void update( const int &dim )
            { _m->update<offset>( dim ); }
            
            inline void update( const int &dim, const int& offset )
            { _m->update( dim, offset ); }
            
        #ifndef NO_VECTORIALIZATION
            inline void loadVector()
            { _m->loadVector(); }
        #endif
        
            virtual MV_INLINE T apply( const size_dim& offset ) = 0;
        #ifndef NO_VECTORIALIZATION
            virtual MV_INLINE MM_VECT()  apply_vect_f( const size_dim& offset ) = 0;
            virtual MV_INLINE MM_VECT(d) apply_vect_d( const size_dim& offset ) = 0;
            //virtual MV_INLINE MM_VECT(i) apply_vect_i( const size_dim& offset ) = 0;
        #endif
    };
    
    
    
    template<typename Function, class M, typename T>
    class Sum : public Operation<Function,M,T>
    {
        public:
            Sum( M* m, Function f ) : Operation<Function,M,T>( m, f )
            {}
            
            MV_INLINE T apply( const size_dim& offset )
            { return this->_m->_data[this->_m1->_index + offset] + this->_next->apply( offset ); }
            
        #ifndef NO_VECTORIALIZATION
            MV_INLINE MM_VECT() apply_vect_f( const size_dim& offset )
            { return MM_ADD(s,)( this->_m->_x_vec_s[offset], this->_next->apply_vect_f( offset ) ); }
            
            MV_INLINE MM_VECT(d) apply_vect_d( const size_dim& offset )
            { return MM_ADD(d,)( this->_m->_x_vec_d[offset], this->_next->apply_vect_d( offset ) ); }
        #endif
    };
    
    template<class M, typename T>
    class SumEnd : public BinaryOperation<M,T>
    {
        public:
            SumEnd( M* m1, M* m2 ) : BinaryOperation<M,T>( m1, m2 )
            {}
            
            MV_INLINE T apply( const size_dim& offset )
            { return this->_m1->_data[this->_m1->_index + offset] + this->_m2->_data[this->_m2->_index + offset]; }
            
        #ifndef NO_VECTORIALIZATION
            MV_INLINE MM_VECT() apply_vect_f( const size_dim& offset )
            { return MM_ADD(s,)( this->_m1->_x_vec_s[offset], this->_m2->_x_vec_s[offset] ); }
            
            MV_INLINE MM_VECT(d) apply_vect_d( const size_dim& offset )
            { return MM_ADD(d,)( this->_m1->_x_vec_d[offset], this->_m2->_x_vec_d[offset] ); }
        #endif
    };
    
    template<class M, typename T>
    class SumConst : public ConstOperation<M,T>
    {
        public:
            SumConst( M* m, const T& val ) : ConstOperation<M,T>( m, val )
            {
            #ifndef NO_VECTORIALIZATION
                const T _val ALIGN = val;
                T _a_val[sizeof(T)] ALIGN;
	            for(uint64_t i = 0; i < sizeof(T); i++) _a_val[i] = _val;
	            if(IS_DOUBLE(T)) this->_c_vect_d = *((MM_VECT(d)*) _a_val);
	            if(IS_FLOAT(T))  this->_c_vect_s = *((MM_VECT( )*) _a_val);
            #endif
            }
            
            MV_INLINE T apply( const size_dim& offset )
            { return this->_m->_data[this->_m->_index + offset] + this->_value; }
            
        #ifndef NO_VECTORIALIZATION
            MV_INLINE MM_VECT() apply_vect_f( const size_dim& offset )
            { return MM_ADD(s,)( this->_m->_x_vec_s[offset], this->_c_vect_s ); }
            
            MV_INLINE MM_VECT(d) apply_vect_d( const size_dim& offset )
            { return MM_ADD(d,)( this->_m->_x_vec_d[offset], this->_c_vect_d ); }
        #endif
    };
    
    template<typename Function, class M, typename T>
    class Sub : public Operation<Function,M,T>
    {
        public:
            Sub( M* m, Function* f ) : Operation<Function,M,T>( m, f )
            {}
            
            MV_INLINE T apply( const size_dim& offset )
            { return this->_m->_data[this->_m->_index + offset] - this->_next->apply( offset ); }
            
        #ifndef NO_VECTORIALIZATION
            MV_INLINE MM_VECT() apply_vect_f( const size_dim& offset )
            { return MM_SUB(s,)( this->_m->_x_vec_s[offset], this->_next->apply_vect_f( offset ) ); }
            
            MV_INLINE MM_VECT(d) apply_vect_d( const size_dim& offset )
            { return MM_SUB(d,)( this->_m->_x_vec_d[offset], this->_next->apply_vect_d( offset ) ); }
        #endif
    };
    
    template<class M, typename T>
    class SubEnd : public BinaryOperation<M,T>
    {
        public:
            SubEnd( M* m1, M* m2 ) : BinaryOperation<M,T>( m1, m2 )
            {}
            
            MV_INLINE T apply( const size_dim& offset )
            { return this->_m1->_data[this->_m1->_index + offset] - this->_m2->_data[this->_m2->_index + offset]; }
            
        #ifndef NO_VECTORIALIZATION
            MV_INLINE MM_VECT() apply_vect_f( const size_dim& offset )
            { return MM_SUB(s,)( this->_m1->_x_vec_s[offset], this->_m2->_x_vec_s[offset] ); }
            
            MV_INLINE MM_VECT(d) apply_vect_d( const size_dim& offset )
            { return MM_SUB(d,)( this->_m1->_x_vec_d[offset], this->_m2->_x_vec_d[offset] ); }
        #endif
    };
    
    template<class M, typename T>
    class SubConst : public ConstOperation<M,T>
    {
        public:
            SubConst( M* m, const T& val ) : ConstOperation<M,T>( m, val )
            {}
            
            MV_INLINE T apply( const size_dim& offset )
            { return this->_m->_data[this->_m->_index + offset] - this->_value; }
            
        #ifndef NO_VECTORIALIZATION
            MV_INLINE MM_VECT() apply_vect_f( const size_dim& offset )
            { return MM_SUB(s,)( this->_m->_x_vec_s[offset], this->_c_vect_s ); }
            
            MV_INLINE MM_VECT(d) apply_vect_d( const size_dim& offset )
            { return MM_SUB(d,)( this->_m->_x_vec_d[offset], this->_c_vect_d ); }
        #endif
    };
    
    template<typename Function, class M, typename T>
    class Mul : public Operation<Function,M,T>
    {
        public:
            Mul( M* m, Function* f ) : Operation<Function,M,T>( m, f )
            {}
            
            MV_INLINE T apply( const size_dim& offset )
            { return this->_m->_data[this->_m->_index + offset] * this->_next->apply( offset ); }
            
        #ifndef NO_VECTORIALIZATION
            MV_INLINE MM_VECT() apply_vect_f( const size_dim& offset )
            { return MM_MUL(s,)( this->_m->_x_vec_s[offset], this->_next->apply_vect_f( offset ) ); }
            
            MV_INLINE MM_VECT(d) apply_vect_d( const size_dim& offset )
            { return MM_MUL(d,)( this->_m->_x_vec_d[offset], this->_next->apply_vect_d( offset ) ); }
        #endif
    };
    
    template<class M, typename T>
    class MulEnd : public BinaryOperation<M,T>
    {
        public:
            MulEnd( M* m1, M* m2 ) : BinaryOperation<M,T>( m1, m2 )
            {}
            
            MV_INLINE T apply( const size_dim& offset )
            { return this->_m1->_data[this->_m1->_index + offset] * this->_m2->_data[this->_m2->_index + offset]; }
            
        #ifndef NO_VECTORIALIZATION
            MV_INLINE MM_VECT() apply_vect_f( const size_dim& offset )
            { return MM_MUL(s,)( this->_m1->_x_vec_s[offset], this->_m2->_x_vec_s[offset] ); }
            
            MV_INLINE MM_VECT(d) apply_vect_d( const size_dim& offset )
            { return MM_MUL(d,)( this->_m1->_x_vec_d[offset], this->_m2->_x_vec_d[offset] ); }
        #endif
    };
    
    template<class M, typename T>
    class MulConst : public ConstOperation<M,T>
    {
        public:
            MulConst( M* m, const T& val ) : ConstOperation<M,T>( m, val )
            {
            #ifndef NO_VECTORIALIZATION
                const T _val ALIGN = val;
                T _a_val[sizeof(T)] ALIGN;
	            for(uint64_t i = 0; i < sizeof(T); i++) _a_val[i] = _val;
	            if(IS_DOUBLE(T)) this->_c_vect_d = *((MM_VECT(d)*) _a_val);
	            if(IS_FLOAT(T))  this->_c_vect_s = *((MM_VECT( )*) _a_val);
            #endif
            }
            
            MV_INLINE T apply( const size_dim& offset )
            { return this->_m->_data[this->_m->_index + offset] * this->_value; }
            
        #ifndef NO_VECTORIALIZATION
            MV_INLINE MM_VECT() apply_vect_f( const size_dim& offset )
            { return MM_MUL(s,)( this->_m->_x_vec_s[offset], this->_c_vect_s ); }
            
            MV_INLINE MM_VECT(d) apply_vect_d( const size_dim& offset )
            { return MM_MUL(d,)( this->_m->_x_vec_d[offset], this->_c_vect_d ); }
        #endif
    };
    
    /*template<class M, typename T>
    class Div : public Operation<M,T>
    {
        public:
            Div( M* m1, M* m2, const T& val = 0 ) : Operation<M,T>( m1, m2, val )
            {}
            
            MV_INLINE T apply( const size_dim& offset )
            { return this->_m1->_data[this->_m1->_index + offset] / this->_next->apply( offset ); }
            
        #ifndef NO_VECTORIALIZATION
            MV_INLINE MM_VECT() apply_vect_f( const size_dim& offset )
            { return MM_DIV(s,)( this->_m1->_x_vec_s[offset], this->_next->apply_vect_f( offset ) ); }
            
            MV_INLINE MM_VECT(d) apply_vect_d( const size_dim& offset )
            { return MM_DIV(d,)( this->_m1->_x_vec_d[offset], this->_next->apply_vect_d( offset ) ); }
        #endif
    };*/
}

#endif /* _FUNCTIONS_H */
