#ifndef AARONSON_GOTTESMAN_H
#define AARONSON_GOTTESMAN_H

#include "QCircuit.h"
#include "bitarray.h"
#include "binary_expression.h"
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

struct EquivClassDecomp{
  size_t n_qubits;
  size_t n_classes;
  size_t * class_sizes;
  size_t ** classes;  
};

typedef struct EquivClassDecomp EquivClassDecomp;
void EquivClassDecomp_free(EquivClassDecomp * classes);
void StabTable_pprint_equivalence_classes(EquivClassDecomp * classes);

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
StabTable * StabTable_copy(StabTable * table);
void StabTable_pprint_table(StabTable * state, int t);

void StabTable_pprint_row(int n, int t, unsigned char phase, unsigned char * row);
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
  Put it in ZX form on qubit q
 */
int StabTable_ZX_form(StabTable * table, int starting_row, int q);

/*
  Attempt to make a cascading ZX form
 */
int StabTable_cascading_ZX_form(StabTable * table, int starting_qubit);
int StabTable_cascading_ZX_form2(StabTable * table, int starting_qubit);
int StabTable_shared_stab_ZX_form(StabTable * table, int starting_row, int q);
int StabTable_cascading_ZX_form_only_doubles(StabTable * table, int starting_qubit);
/*
 * Our magic states are equatorial
 * so <T|Z|T> = 0
 * here we delete any stabilisers with a Z in the magic region
 * which we assume is the last t qubits 
 * we return the number of qubits which have identities on them in every generator after this deletion
 */

int StabTable_apply_T_constraints(StabTable * table, int t);


/*
 * It is possible that we end up with a table with some qubits having identity for all stabilisers
 * These qubits do not change our result at all so we just delete them
 * updates the table and reuturnes the number of qubits deleted
 * if magic_qubit_numbers is non-null it is assumed to be a pointer to the first of a block of state->n ints
 * elements of this array will be moved and deleted in the same way that the magic qubits are
 * so if each element of the array is unique (for example the ints from 0 to state->n-1) then you can use it to keep track of which magic qubits we deleted
 */
int StabTable_delete_all_identity_qubits(StabTable * table, int * magic_qubit_numbers);
/*
 * create a QCircuit that brings the state represented by this tableau to the state|0><0|^k \otimes I^(n-k) / (2^(n-k))
 * Explicitly k stabilisers on n qubits with the jth being a z on the jth qubit
 * this procedure overwrites the information stored in the QCircuit circ of the table
 * 
 */
QCircuit * StabTable_simplifying_unitary(StabTable * table);

int delete_columns_with_at_most_one_nontrivial_stabilizer(StabTable * state);

int StabTable_row_reduction_upper_bound(StabTable * table);
int StabTable_row_reduction_Z_table(StabTable * table);
int StabTable_count_distinct_cols(StabTable * state);


int StabTable_y_tilde_inner_prod_no_phase(StabTable * state, uint_bitarray_t y, char ** M);


EquivClassDecomp StabTable_count_equivalence_classes(StabTable * state);

int commutativity_diagram(StabTable * state, int measured_qubits, int t, int verbose);

int StabTable_zero_inner_product(StabTable * state, int w, int t);
#endif

