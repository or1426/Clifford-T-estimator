#ifndef BINARY_EXPRESSION_H
#define BINARY_EXPRESSION_H

#include "bitarray.h"
#include <stdlib.h>
#include <stdio.h>
/*
 * the idea here is simple
 * we store a polynomial formed of binary variables (y_i)_{i=1}^n
 * as a list of products each of which is added (xor'd) together
 * e.g. 1 ^ y_0 y_1 ^ y_7 y_8 y_43
 * each bitarray stores one product 
 */
struct BinaryExpression{
  uint_bitarray_t * data;
  int capacity;
  int length;
};

typedef struct BinaryExpression BinaryExpression;


void BE_shrink_to_contents(BinaryExpression * poly);
/*
 * normalize sorts the expressions into integer comparison order
 * and merges duplicates
 */
void BE_normalize(BinaryExpression * poly);

/*
 * dest += source*multiplier where multiplier is interpreted as a monomial
 */
void BE_add_mono_mult(BinaryExpression * source, BinaryExpression * dest, uint_bitarray_t multiplier);

/*
 * dest += source*multiplier where multiplier is another polynomial
 */
void BE_add_poly_mult(BinaryExpression * source, BinaryExpression * dest, BinaryExpression * multiplier);

int BE_pprint(BinaryExpression * poly);
void BE_print_summary(BinaryExpression * poly);
int BE_pprint_size(BinaryExpression * poly);
void BE_pprint_matrix(BinaryExpression ** M, int n, int k);
#endif 
