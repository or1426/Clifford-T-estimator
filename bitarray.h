#ifndef BITARRAY_H
#define BITARRAY_H
#include <stdint.h>

//typedef unsigned __int128 uint_bitarray_t;
//#define popcount(x) popcount_unsigned__int128(x)
//#define parity(x) parity_unsigned__int128(x)
typedef uint_fast64_t uint_bitarray_t;
#define popcount(x) popcount_uint_fast64_t(x)
#define parity(x) parity_uint_fast64_t(x)

//#define popcount(x) popcount_generic(x)
//#define parity(x) parity_generic(x)

unsigned int parity_generic(uint_bitarray_t x);
unsigned int parity_uint_fast64_t(uint_fast64_t x);
unsigned int parity_unsigned__int128(unsigned __int128 x);

unsigned int popcount_generic(uint_bitarray_t x);
unsigned int popcount_uint_fast64_t(uint_fast64_t x);
unsigned int popcount_unsigned__int128(unsigned __int128 x);

uint_bitarray_t bitarray_rand();

void printBits(uint_bitarray_t x, int n);


extern const uint_bitarray_t ONE; //defined in bitarray.c
#endif
