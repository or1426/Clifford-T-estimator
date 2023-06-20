#ifndef BIGNUM_H
#define BIGNUM_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "bitarray.h" // for uint_bitarray_t definition

struct BigNum
{
  size_t num_words; //number of ints making up our bignum
  uint_bitarray_t * arr;
};

typedef struct BigNum BigNum;


void init_zero_bignum(BigNum x, size_t num_words){
  x.num_words = num_words;
  arr = calloc(num_words, sizeof(uint_bitarray_t));
}

void free_bignum(BigNum x){
  num_words = 0;
  free(x.arr);    
}

void incr_bignum(BigNum x){
  for(size_t i = 0; i < x.num_words; i++){
    if(x.arr[i] != MAX){
      x.arr[i] += 1;
      break;
    }
    else{
      x.arr[i] = 0;
    }
  }
}


//return 1 if x < 2^n otherwise 0
unsigned char test_less_than_2_power_bignum(BigNum x, size_t n){
  size_t quotient = n / sizeof(uint_bitarray_t);
  size_t remainder = n % sizeof(uint_bitarray_t);

  if(n >=  x.num_words*sizeof(uint_bitarray_t)){
    return 1;
  }

  for(size_t i = quotient + 1; i < x.num_words; i++){
    if(x.arr[i] != ZERO){
      return 0;
    }
  }
	 
  
  for(size_t i = 0; i < quotient; i++){
    if(x.arr[i] != MAX){
      return 1;
    }
  }
  
}

#endif
