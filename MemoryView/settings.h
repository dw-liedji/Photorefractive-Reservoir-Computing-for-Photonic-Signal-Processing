
/*
 *  settings.h
 *
 *  Created on: 11 oct 2016
 *  Author: Stefano Ceccotti
 *  Author: Tommaso Catuogno
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
#define _MIN(a,b) (((a) < (b)) ? (a) : (b))

// Check the size of the actual type.
#define IS_DOUBLE(T) sizeof(T) == sizeof( double )
#define IS_FLOAT(T)  sizeof(T) == sizeof( float )
#define IS_SHORT(T)  sizeof(T) == sizeof( short )
#define IS_CHAR(T)   sizeof(T) == sizeof( char )

// The vector pointer of the matrix.
#define L_VECT( suffix ) _x_vec_ ## suffix

// Used to include all the possible vectorialized operations.
//#include <x86intrin.h>




#if defined( __AVX512F__ ) || defined( __AVX512ER__ ) || defined( __AVX512BW__ ) || defined( __AVX512CD__ ) || \
    defined( __AVX512PF__ ) || defined( __AVX512VL__ ) || defined( __AVX512DQ__ )
    #include <zmmintrin.h>
    #define BLOCK               64
    #define MM_VECT(suffix) __m512 ## suffix
    
    #define MM_ADD(prefix,bits) MM_ADD ## prefix(bits)
    #define MM_ADDi(bits) _mm512_add_epi ## bits
    #define MM_ADDs()     _mm512_add_ps
    #define MM_ADDd()     _mm512_add_pd
    
    #define MM_SUB(prefix,bits) MM_SUB ## prefix(bits)
    #define MM_SUBi(bits) _mm512_sub_epi ## bits
    #define MM_SUBs()     _mm512_sub_ps
    #define MM_SUBd()     _mm512_sub_pd
    
    #define MM_MUL(prefix,bits) MM_MUL ## prefix(bits)
    #define MM_MULi(bits) _mm_mul_pi ## bits /* Call the 8 or 16-bits vector function. */
    #define MM_MULs()     _mm512_mul_ps
    #define MM_MULd()     _mm512_mul_pd
    #define MM_MUL_LO     _mm512_mullo_epi16
    #define MM_MUL_HI     _mm512_mulhi_epi16
    
    #define MM_SR_LI  _mm512_srli_epi16
    #define MM_SL_LI  _mm512_slli_epi16
    #define MM_AND_SI _mm512_and_si512
    #define MM_OR_SI  _mm512_or_si512
    
	#define MM_DIV(suffix) _mm512_div_p ## suffix
	
    #define MM_SET1(prefix,bits) MM_SET1 ## prefix(bits)
    #define MM_SET1i(bits) _mm512_set1_epi ## bits
    #define MM_SET1s()     _mm512_set1_ps
    #define MM_SET1d()     _mm512_set1_pd
    
#elif defined( __AVX__ ) || defined( __AVX2__ )
    #include <immintrin.h>
    #define BLOCK               32
    #define MM_VECT(suffix) __m256 ## suffix
    
    #define MM_ADD(prefix,bits) MM_ADD ## prefix(bits)
    #define MM_ADDi(bits) _mm256_add_epi ## bits
    #define MM_ADDs()     _mm256_add_ps
    #define MM_ADDd()     _mm256_add_pd
    
    #define MM_SUB(prefix,bits) MM_SUB ## prefix(bits)
    #define MM_SUBi(bits) _mm256_sub_epi ## bits
    #define MM_SUBs()     _mm256_sub_ps
    #define MM_SUBd()     _mm256_sub_pd
    
    #define MM_MUL(prefix,bits) MM_MUL ## prefix(bits)
    #define MM_MULi(bits) _mm_mul_pi ## bits /* Call the 8 or 16-bits vector function. */
    #define MM_MULs()     _mm256_mul_ps
    #define MM_MULd()     _mm256_mul_pd
    #define MM_MUL_LO     _mm256_mullo_epi16
    #define MM_MUL_HI     _mm256_mulhi_epi16
    
    #define MM_SR_LI  _mm256_srli_epi16
    #define MM_SL_LI  _mm256_slli_epi16
    #define MM_AND_SI _mm256_and_si256
    #define MM_OR_SI  _mm256_or_si256
    
	#define MM_DIV(suffix) _mm256_div_p ## suffix
	
    #define MM_SET1(prefix,bits) MM_SET1 ## prefix(bits)
    #define MM_SET1i(bits) _mm256_set1_epi ## bits
    #define MM_SET1s()     _mm256_set1_ps
    #define MM_SET1d()     _mm256_set1_pd
    
#elif defined( __SSE__ ) || defined( __SSE2__ ) || defined( __SSE3__ ) || defined( __SSE4__ )
    #include <xmmintrin.h>
    #define BLOCK               16
    #define MM_VECT(suffix)     __m128 ## suffix
    
    #define MM_ADD(prefix,bits) MM_ADD ## prefix(bits)
    #define MM_ADDi(bits) _mm_add_epi ## bits
    #define MM_ADDs()     _mm_add_ps
    #define MM_ADDd()     _mm_add_pd
    
    #define MM_SUB(prefix,bits) MM_SUB ## prefix(bits)
    #define MM_SUBi(bits) _mm_add_epi ## bits
    #define MM_SUBs()     _mm_sub_ps
    #define MM_SUBd()     _mm_sub_pd
    
    #define MM_MUL(prefix,bits) MM_MUL ## prefix(bits)
    #define MM_MULi(bits) _mm_mul_pi ## bits /* Call the 8 or 16-bits vector function. */
    #define MM_MULs()     _mm_mul_ps
    #define MM_MULd()     _mm_mul_pd
    #define MM_MUL_LO     _mm_mullo_epi16
    #define MM_MUL_HI     _mm_mulhi_epi16
    
    #define MM_SR_LI  _mm_srli_epi16
    #define MM_SL_LI  _mm_slli_epi16
    #define MM_AND_SI _mm_and_si128
    #define MM_OR_SI  _mm_or_si128
    
	#define MM_DIV(suffix) _mm_div_p ## suffix
	
    #define MM_SET1(prefix,bits) MM_SET1 ## prefix(bits)
    #define MM_SET1i(bits) _mm_set1_epi ## bits
    #define MM_SET1s()     _mm_set1_ps
    #define MM_SET1d()     _mm_set1_pd
    
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

// Current block dimension.
#define block (BLOCK/sizeof(T))



#define ASSERT_CONCAT_(a, b) a##b
#define ASSERT_CONCAT(a, b) ASSERT_CONCAT_(a, b)
/* These can't be used after statements in c89. */
#ifdef __COUNTER__ // Windows
  #define STATIC_ASSERT(e,m) \
    ;enum { ASSERT_CONCAT(static_assert_, __COUNTER__) = 1/(int)(!!(e)) }
#else
  /* This can't be used twice on the same line so ensure if using in headers
   * that the headers are not included twice (by wrapping in #ifndef...#endif)
   * Note it doesn't cause an issue when used on same line of separate modules
   * compiled with gcc -combine -fwhole-program.  */
  #define STATIC_ASSERT(e,m) \
    ;enum { ASSERT_CONCAT(assert_line_, __LINE__) = 1/(int)(!!(e)) }
#endif



#endif /* _SETTINGS_H */
