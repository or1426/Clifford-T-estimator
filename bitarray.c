#include <stdio.h>
#include <stdlib.h>
#include "bitarray.h"

const uint_bitarray_t ZERO = 0;
const uint_bitarray_t ONE = 1;
const uint_bitarray_t MAX = -1;

unsigned int parity_generic(uint_bitarray_t x){
    unsigned int acc = 0;
    for(int i = 0; i < sizeof(x)*8; i++){
        acc ^= ((x>>i)&ONE);
    }
    return acc & ONE;
}
unsigned int parity_uint_fast64_t(uint_fast64_t x){
    //uint_fast64_t mask = 0xFFFFFFFFFFFFFFFFu;
    return __builtin_parityll(x/*&mask*/);
}

unsigned int parity_unsigned__int128(unsigned __int128 x){
    unsigned __int128 mask = ((unsigned __int128)0xFFFFFFFFFFFFFFFFu);
    return __builtin_parityll((unsigned long long)(x & (mask<<64))) ^ __builtin_parityll((unsigned long long)( (x&mask)  >>64));
}

unsigned int popcount_generic(uint_bitarray_t x){
    unsigned int acc = 0;
    for(int i = 0; i < sizeof(x)*8; i++){
        acc += ((unsigned int)((x>>i) & ONE));
    }
    return acc;
}
unsigned int popcount_uint_fast64_t(uint_fast64_t x){
    return __builtin_popcountll(x);
}

unsigned int popcount_unsigned__int128(unsigned __int128 x){
    unsigned __int128 mask = ((unsigned __int128)0xFFFFFFFFFFFFFFFFu);
    return __builtin_popcountll((unsigned long long)(x & (mask<<64))) + __builtin_popcountll((unsigned long long)( (x&mask)  >>64));
}


uint_bitarray_t bitarray_rand(){
    uint_bitarray_t val = 0;
    for(int i = 0; i < sizeof(uint_bitarray_t); i++){
        val ^= (((uint_bitarray_t)(((unsigned char)rand() % 256))) << (8*i));
    }
    return val;
}

void printBits(uint_bitarray_t x, int n){
    uint_bitarray_t ONE = 1;
    for(int i = 0; i < n; i++){
        printf("%u", (unsigned int)((x>>i) & ONE));
    }
}
