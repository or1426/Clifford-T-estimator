#ifndef CH_H
#define CH_H

#include "bitarray.h" // for uint_bitarray_t definition
#include <complex.h>
typedef struct CHForm{
    unsigned int n;
    uint_bitarray_t * F;
    uint_bitarray_t * G;
    uint_bitarray_t * M;
    uint_bitarray_t g1;
    uint_bitarray_t g2;
    uint_bitarray_t s;
    uint_bitarray_t v;

    double complex w;
} CHForm;

/*
 * basic CH form functions
 */
void print_CHForm(CHForm * state);
int init_cb_CHForm(unsigned int n, CHForm * state);
CHForm copy_CHForm(CHForm * state);
int init_zero_CHForm(unsigned int n, CHForm * state);
int dealocate_state(CHForm * state);

/*
 * Clifford gates
 * C type gates can be applied left or right onto UC
 */

int SR(CHForm * state, unsigned int q);
int SL(CHForm * state, unsigned int q);

int CZR(CHForm * state, unsigned int q, unsigned int r);
int CZL(CHForm * state, unsigned int q, unsigned int r);

int CXR(CHForm * state, unsigned int q, unsigned int r);
int CXL(CHForm * state, unsigned int q, unsigned int r);

int SWAPR(CHForm * state, unsigned int q, unsigned int r);

/*
 * Non C type gates can only be applied from the left (they apply to the whole state not just UC)
 */
int XL(CHForm * state, unsigned int p);
/*
 * desupersitionise is pretty much proposition 4 of
 * Simulation of quantum circuits by low-rank stabilizer decompositions
 * it allows one to apply a gate written as a sum of two Paulis to a CH form
 * We use it to do H gates, projectors and magic sampling updates
 */
int desupersitionise(CHForm * state, uint_bitarray_t u,  unsigned int d);

int HL(CHForm * state, unsigned int q);
#endif
