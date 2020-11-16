#ifndef AARONSON_GOTTESMAN_H
#define AARONSON_GOTTESMAN_H

#include "QCircuit.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

/*
 * We store a table of stabilisers (no destabilisers)
 * the table is a length k array of length 2n arrays
 * the first n elements are the powers of x the last n are the powers of z
 *  we assume always k <= n
 */
struct StabTable
{
    int n; //num qubits
    int k; //num stabilisers this only makes sense with  k <= n
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
StabTable * StabTable_new(int n, int k);

/*
 * Frees all memory associated with the table
 */
int StabTable_free(StabTable * table);

void StabTable_print(StabTable * table);
/*
 * Applies a CX gate controlled on qubit a targetted on qubit b
 * it is assumed a and b are valid qubit numbers (i.e. < n)
 */
int StabTable_CX(StabTable * table, int a, int b);

/* 
 * Applies a CZ gate controlled on qubit a targetted on qubit b
 * it is assumed a and b are valid qubit numbers (i.e. < n)
 */
int StabTable_CZ(StabTable * table, int a, int b);


/* 
 * Applies an H gate targetted on qubit a
 * it is assumed a is a valid qubit number (i.e. a < n)
 */
int StabTable_H(StabTable * table, int a);

/* 
 * Applies an S gate targetted on qubit a
 * it is assumed a is a valid qubit number (i.e. a < n)
 */
int StabTable_S(StabTable * table, int a);

/* 
 * Applies an X gate targetted on qubit a
 * it is assumed a is a valid qubit number (i.e. a < n)
 */
int StabTable_X(StabTable * table, int a);


/* 
 * creates a copy of the input with the same stabiliser table
 * does not copy the "circ" information
 */
StabTable * StabTable_copy(StabTable * input);


int StabTable_first_non_zero_in_col(StabTable * table, int col, int startpoint);
int StabTable_first_non_zero_in_row(StabTable * table, int row, int startpoint);
int StabTable_swap_rows(StabTable * table, int i, int j);
int StabTable_rowsum(StabTable * table, int h, int i);



/* Does not change the table at all
 * updates row to be row ^ table_i
 * returns the update to the phase of row (0 or 1)
 */
unsigned char StabTable_rowsum2(StabTable * table, unsigned char * row, unsigned char phase, int i);
/*
 * Apply constraints arising from the fact that we measure the first w qubits and project the last t onto T gates
 * In particular we kill stabilisers if qubits [0, w) get killed by taking the expectation value <0| P |0>
 * and we kill stabilisers if qubits in [w, table->n-t) aren't the identity
 * we do not remove any qubits from the table
 * we return the degeneract factor log_v, which is set to -1 if the probability was 0
 */
int StabTable_apply_constraints(StabTable * table, int w, int t);

/*
 * Our magic states are equatorial
 * so <T|Z|T> = 0
 * here we delete any stabilisers with a Z in the magic region
 * which we assume is the last t qubits 
 * we return the number of qubits which have identities on them in every generator after this deletion
 */

int StabTable_apply_T_constraints(StabTable * table, int t);

/*
 * create a QCircuit that brings the state represented by this tableau to the state|0><0|^k \otimes I^(n-k) / (2^(n-k))
 * Explicitly k stabilisers on n qubits with the jth being a z on the jth qubit
 * this procedure overwrites the information stored in the QCircuit circ of the table
 * 
 */
QCircuit * StabTable_simplifying_unitary(StabTable * table);

#endif
