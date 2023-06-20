#include "binary_expression.h"

void BE_shrink_to_contents(BinaryExpression * poly){
  poly->capacity = poly->length;
  poly->data = realloc(poly->data, sizeof(uint_bitarray_t)*poly->capacity);
}


int compare_monomials(const void* a, const void* b)
{
  uint_bitarray_t arg1 = *(const uint_bitarray_t*)a;
  uint_bitarray_t arg2 = *(const uint_bitarray_t*)b;

  if (arg1 < arg2) return -1;
  if (arg1 > arg2) return 1;
  return 0;

  // return (arg1 > arg2) - (arg1 < arg2); // possible shortcut
  // return arg1 - arg2; // erroneous shortcut (fails if INT_MIN is present)
}

void BE_normalize(BinaryExpression * poly){
  if(poly->length > 0){
    qsort(poly->data, poly->length, sizeof(uint_bitarray_t), compare_monomials);

    int new_length = 0;
    int running_copies_count = 1;

    uint_bitarray_t current_monomial = poly->data[0];

    for(int i = 1; i < poly->length; i++){
      if(current_monomial == poly->data[i]){
        running_copies_count += 1;
      }else{
        if((running_copies_count % 2) == 1){
          poly->data[new_length] = current_monomial;
          new_length += 1;
        }
        current_monomial = poly->data[i];
        running_copies_count = 1;
      }
    }
    if((running_copies_count % 2) == 1){
      poly->data[new_length] = current_monomial;
      new_length += 1;
    }

    /*
      for(int i = 1; i < poly->length; i++){
      if(poly->data[i] == poly->data[i-1]){
      running_copies_count += 1;
      }else{
      if(running_copies_count % 2 == 1){
      poly->data[new_length] = poly->data[i];
      new_length += 1;
      }
      running_copies_count = 1;
      }
      }
      if(running_copies_count % 2 == 1){
      poly->data[new_length] = poly->data[poly->length - 1];
      new_length += 1;
      }
    */
    poly->length = new_length;
    if(poly->length < poly->capacity/4 || poly->length == 0){
      BE_shrink_to_contents(poly);
    }
  }

}


void BE_add_mono_mult(BinaryExpression * source, BinaryExpression * dest, uint_bitarray_t multiplier){
  //printf("old capacity: %d\n", dest->capacity);
  if(dest->capacity < source->length + dest->length){
    dest->capacity = source->length + dest->length;
    //printf("new capacity: %d\n", dest->capacity);
    dest->data = realloc(dest->data, sizeof(uint_bitarray_t)*dest->capacity);
  }
  for(int i = 0; i < source->length; i++){
    dest->data[i + dest->length] = (source->data[i] | multiplier);
  }
  dest->length += source->length;
  //printf("new length: %d\n", dest->length);
  //printf("new dest: ");BE_pprint(dest);printf("\n");
  BE_normalize(dest);
  //printf("normalised: ");BE_pprint(dest);printf("\n");
}

//this is aggressively inefficient
void BE_add_poly_mult(BinaryExpression * source, BinaryExpression * dest, BinaryExpression * multiplier){
  BinaryExpression * temp_source = source;
  BinaryExpression * temp_mult = multiplier;

  if(source->data == dest->data){
    temp_source = malloc(sizeof(BinaryExpression));
    temp_source->length = source->length;
    temp_source->capacity = source->capacity;
    temp_source->data = malloc(temp_source->capacity*sizeof(uint_bitarray_t));
    memcpy(temp_source->data, source->data, temp_source->capacity*sizeof(uint_bitarray_t));
  }

  if(multiplier->data == dest->data){
    temp_mult = malloc(sizeof(BinaryExpression));
    temp_mult->length = multiplier->length;
    temp_mult->capacity = multiplier->capacity;
    temp_mult->data = malloc(temp_mult->capacity*sizeof(uint_bitarray_t));
    memcpy(temp_mult->data, multiplier->data, temp_mult->capacity*sizeof(uint_bitarray_t));
  }
  
  for(int i = 0; i < temp_mult->length; i++){
    BE_add_mono_mult(temp_source, dest, temp_mult->data[i]);
  }

  if(temp_source != source){
    free(temp_source->data);
    free(temp_source);
  }
  if(temp_mult != multiplier){
    free(temp_mult->data);
    free(temp_mult);
  }
  
}

int BE_pprint(BinaryExpression * poly){
  int chars = 0;

  if(poly->length == 0){
    chars += printf("0 ");
  }
  
  for(int i = 0; i < poly->length; i++){
    if(poly->data[i] == ZERO){
      chars += printf("1 ");
    }else{

      for(int j = 0; j < 8*sizeof(uint_bitarray_t); j++){
        if((poly->data[i] >> j) & ONE){
          chars += printf("y%d ", j);
        }
      }
    }

    if(i != poly->length - 1){
      chars += printf("+ ");
    }
  }
  return chars;
}


void BE_print_summary(BinaryExpression * poly){
  if(poly->length == 1){
    if(poly->data[0] == ZERO){
      putc('1', stdout);
    }else{
      putc('*', stdout);
    }
  }else if(poly->length == 0){
    putc('0', stdout);
  }else{
    putc('*', stdout);
  }
}
