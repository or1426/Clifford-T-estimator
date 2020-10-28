#ifndef AARONSON_GOTTESMAN_H
#define AARONSON_GOTTESMAN_H

#include "QCircuit.h"

/*
 * We store a table of stabilisers (no destabilisers)
 * the table is a length k array of length 2n arrays
 * the first n elements are the powers of x the last n are the powers of z
 *  we assume always k <= n
 */
struct StabTable
{
    size_t n; //num qubits
    size_t k; //num stabilisers this only makes sense with  k <= n
    unsigned char ** table; 
    unsigned char * phases;
    QCircuit * circ; //remembers quantum operations done to the circuit
};

typedef struct StabTable StabTable;


/*
 * Init a new stabtable in the state |0>^k \otimes I^(n-k) / (2^(n-k))
 * Explicitly we have k stabilisers on n qubits with the jth being a z on the jth qubit
 * Assumptions: 
 * k <= n
 */
StabTable * StabTable_new(size_t n, size_t k);

/*
 * Frees all memory associated with the table
 */
int StabTable_free(StabTable * table);

/*
 * Applies a CX gate controlled on qubit a targetted on qubit b
 * it is assumed a and b are valid qubit numbers (i.e. < n)
 */
int StabTable_CX(StabTable * table, size_t a, size_t b);

/*
 * Applies a CZ gate controlled on qubit a targetted on qubit b
 * it is assumed a and b are valid qubit numbers (i.e. < n)
 */
int StabTable_CZ(StabTable * table, size_t a, size_t b);


/*
 * Applies an H gate targetted on qubit a
 * it is assumed a is a valid qubit number (i.e. a < n)
 */
int StabTable_H(StabTable * table, size_t a);

/*
 * Applies an S gate targetted on qubit a
 * it is assumed a is a valid qubit number (i.e. a < n)
 */
int StabTable_S(StabTable * table, size_t a);

#endif
