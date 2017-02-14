
/*
 *  settings.h
 *
 *  Created on: 11 oct 2016
 *  Authors: Stefano Ceccotti & Tommaso Catuogno
*/

#ifndef _SETTINGS_H
#define _SETTINGS_H



// Dimension of the elements used by the data structures.
typedef int64_t size_dim;
typedef int64_t size_m;


#define MV_INLINE __attribute__((always_inline))


// Define the Maximum number of accepted dimensions.
#undef MAX_DIMENSIONS
#define MAX_DIMENSIONS 32

#define _MAX(a,b) (((a) > (b)) ? (a) : (b))

// Check the size of the actual type.
#define IS_DOUBLE( T ) sizeof(T) == sizeof( double )
#define IS_FLOAT( T )  sizeof(T) == sizeof( float )
#define IS_SHORT( T )  sizeof(T) == sizeof( short )
#define IS_CHAR( T )   sizeof(T) == sizeof( char )

// The vector pointer of the matrix.
#define L_VECT( suffix ) _x_vec_ ## suffix

#ifndef INLINE
    #define INLINE __attribute__((always_inline))
#endif

// Used to include all the possible vectorialized operations.
//#include <x86intrin.h>




#if defined( __AVX512F__ ) || defined( __AVX512ER__ ) || defined( __AVX512BW__ ) || defined( __AVX512CD__ ) || \
    defined( __AVX512PF__ ) || defined( __AVX512VL__ ) || defined( __AVX512DQ__ )
    #include <zmmintrin.h>
    #define BLOCK               64
    #define MM_VECT(suffix) __m512 ## suffix
    
    #define MM_ADD(prefix,bits) MM_ADD ## prefix(bits)
    #define MM_ADDi(bits) _mm512_add_epi ## bits
    #define MM_ADDs() _mm512_add_ps
    #define MM_ADDd() _mm512_add_pd
    
    #define MM_SUB(prefix,bits) MM_SUB ## prefix(bits)
    #define MM_SUBi(bits) _mm512_add_epi ## bits
    #define MM_SUBs() _mm512_sub_ps
    #define MM_SUBd() _mm512_sub_pd
    
    #define MM_MUL(prefix,bits) MM_MUL ## prefix(bits)
    #define MM_MULi(bits) _mm512_mullo_epi ## bits
    #define MM_MULs() _mm512_mul_ps
    #define MM_MULd() _mm512_mul_pd
    
	#define MM_DIV(suffix) _mm512_div_p ## suffix
	
    #define MM_SET1(prefix,bits) MM_SET1 ## prefix
    #define MM_SET1i(bits) _mm512_set1_epi ## bits
    #define MM_SET1s() _mm512_set1_ps
    #define MM_SET1d() _mm512_set1_pd
#elif defined( __AVX__ ) || defined( __AVX2__ )
    #include <immintrin.h>
    #define BLOCK               32
    #define MM_VECT(suffix) __m256 ## suffix
    
    #define MM_ADD(prefix,bits) MM_ADD ## prefix(bits)
    #define MM_ADDi(bits) _mm256_add_epi ## bits
    #define MM_ADDs() _mm256_add_ps
    #define MM_ADDd() _mm256_add_pd
    
    #define MM_SUB(prefix,bits) MM_SUB ## prefix(bits)
    #define MM_SUBi(bits) _mm256_add_epi ## bits
    #define MM_SUBs() _mm256_sub_ps
    #define MM_SUBd() _mm256_sub_pd
    
    #define MM_MUL(prefix,bits) MM_MUL ## prefix(bits)
    #define MM_MULi(bits) _mm256_mullo_epi ## bits
    #define MM_MULs() _mm256_mul_ps
    #define MM_MULd() _mm256_mul_pd
    
	#define MM_DIV(suffix) _mm256_div_p ## suffix
	
    #define MM_SET1(prefix,bits) MM_SET1 ## prefix
    #define MM_SET1i(bits) _mm256_set1_epi ## bits
    #define MM_SET1s() _mm256_set1_ps
    #define MM_SET1d() _mm256_set1_pd
#elif defined( __SSE__ ) || defined( __SSE2__ ) || defined( __SSE3__ ) || defined( __SSE4__ )
    #include <xmmintrin.h>
    #define BLOCK               16
    #define MM_VECT(suffix)     __m128 ## suffix
    
    #define MM_ADD(prefix,bits) MM_ADD ## prefix(bits)
    #define MM_ADDi(bits) _mm_add_epi ## bits
    #define MM_ADDs() _mm_add_ps
    #define MM_ADDd() _mm_add_pd
    
    #define MM_SUB(prefix,bits) MM_SUB ## prefix(bits)
    #define MM_SUBi(bits) _mm_add_epi ## bits
    #define MM_SUBs() _mm_sub_ps
    #define MM_SUBd() _mm_sub_pd
    
    #define MM_MUL(prefix,bits) MM_MUL ## prefix(bits)
    #define MM_MULi(bits) _mm_mullo_epi ## bits
    #define MM_MULs() _mm_mul_ps
    #define MM_MULd() _mm_mul_pd
    
	#define MM_DIV(suffix) _mm_div_p ## suffix
	
    #define MM_SET1(prefix,bits) MM_SET1 ## prefix
    #define MM_SET1i(bits) _mm_set1_epi ## bits
    #define MM_SET1s() _mm_set1_ps
    #define MM_SET1d() _mm_set1_pd
#else
	#define BLOCK               sizeof( T )
    #define MM_VECT(suffix)     T
    #define NO_VECTORIALIZATION
#endif



#if defined(_MSC_VER) // Windows
    #define ALIGN __declspec(align(BLOCK)))
#elif defined(__GNUC__) //  GCC, ICC, ICPC,..
    #define ALIGN __attribute__ ((aligned(BLOCK)))
#else // Empty declaration.
    #define ALIGN
#endif



#endif /* _SETTINGS_H */
