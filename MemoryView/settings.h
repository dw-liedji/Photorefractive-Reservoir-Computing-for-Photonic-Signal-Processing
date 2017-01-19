
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


#ifndef INLINE
    #define INLINE __attribute__((always_inline))
#endif


// Define the Maximum number of accepted dimensions.
#undef MAX_DIMENSIONS
#define MAX_DIMENSIONS 32


// Used to include all the possible vectorialized operations.
//#include <x86intrin.h>




#if defined( __AVX512F__ ) || defined( __AVX512ER__ ) || defined( __AVX512BW__ ) || defined( __AVX512CD__ ) || \
    defined( __AVX512PF__ ) || defined( __AVX512VL__ ) || defined( __AVX512DQ__ )
    #include <zmmintrin.h>
    #define BLOCK       64
    #define MM_VECT     __m512d
    #define MM_LOAD     _mm512_load_pd
    #define MM_STORE    _mm512_store_pd
    #define MM_ADD      _mm512_add_pd
    #define MM_SUB      _mm512_sub_pd
    #define MM_MUL      _mm512_mul_pd
	#define MM_DIV      _mm512_div_pd
    #define MM_SET1     _mm512_set1_pd
#elif defined( __AVX__ ) || defined( __AVX2__ )
    #include <immintrin.h>
    #define BLOCK       32
    #define MM_VECT     __m256d
    #define MM_LOAD     _mm256_load_pd
    #define MM_STORE    _mm256_store_pd
    #define MM_ADD      _mm256_add_pd
    #define MM_SUB      _mm256_sub_pd
    #define MM_MUL      _mm256_mul_pd
	#define MM_DIV      _mm256_div_pd
    #define MM_SET1     _mm256_set1_pd
#elif defined( __SSE__ ) || defined( __SSE2__ ) || defined( __SSE3__ ) || defined( __SSE4__ )
    #include <xmmintrin.h>
    #define BLOCK       16
    #define MM_VECT     __m128d
    #define MM_LOAD     _mm_load_pd
    #define MM_STORE    _mm_store_pd
    #define MM_ADD      _mm_add_pd
    #define MM_SUB      _mm_sub_pd
    #define MM_MUL      _mm_mul_pd
	#define MM_DIV      _mm_div_pd
    #define MM_SET1     _mm_set1_pd
#else
	#define BLOCK       sizeof( T )
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
