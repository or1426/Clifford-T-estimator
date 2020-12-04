#include <Python.h>
#include <numpy/arrayobject.h>

#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <tgmath.h>
#include <stdio.h>
#include <time.h>
#include "AG.h"


//typedef unsigned __int128 uint_bitarray_t;
//#define popcount(x) popcount_unsigned__int128(x)
//#define parity(x) parity_unsigned__int128(x)
typedef uint_fast64_t uint_bitarray_t;
#define popcount(x) popcount_uint_fast64_t(x)
#define parity(x) parity_uint_fast64_t(x)

//#define popcount(x) popcount_generic(x)
//#define parity(x) parity_generic(x)

const uint_bitarray_t ONE = 1;

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



static unsigned int parity_generic(uint_bitarray_t x){
    unsigned int acc = 0;
    for(int i = 0; i < sizeof(x)*8; i++){
        acc ^= ((x>>i)&ONE);
    }
    return acc & ONE;
}
static unsigned int parity_uint_fast64_t(uint_fast64_t x){
    //uint_fast64_t mask = 0xFFFFFFFFFFFFFFFFu;
    return __builtin_parityll(x/*&mask*/);
}

static unsigned int parity_unsigned__int128(unsigned __int128 x){
    unsigned __int128 mask = ((unsigned __int128)0xFFFFFFFFFFFFFFFFu);
    return __builtin_parityll((unsigned long long)(x & (mask<<64))) ^ __builtin_parityll((unsigned long long)( (x&mask)  >>64));
}

static unsigned int popcount_generic(uint_bitarray_t x){
    unsigned int acc = 0;
    for(int i = 0; i < sizeof(x)*8; i++){
        acc += ((unsigned int)((x>>i) & ONE));
    }
    return acc;
}
static unsigned int popcount_uint_fast64_t(uint_fast64_t x){
    return __builtin_popcountll(x);
}

static unsigned int popcount_unsigned__int128(unsigned __int128 x){
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

static void printBits(uint_bitarray_t x, int n){
    uint_bitarray_t ONE = 1;
    for(int i = 0; i < n; i++){
        printf("%u", (unsigned int)((x>>i) & ONE));
    }
}
static void print_CHForm(CHForm * state){
    printf("(%lf, %lf)\n", creal(state->w), cimag(state->w));
    for(int i = 0; i < state->n; i++){
        printBits(state->F[i], state->n);printf(" ");
        printBits(state->G[i], state->n);printf(" ");
        printBits(state->M[i], state->n);printf(" ");
        printf("%u %u %u\n",(unsigned int)(((state->g1 >> i) & ONE) + 2*((state->g2 >> i) & ONE)),(unsigned int)((state->v >> i) & ONE),(unsigned int)((state->s >> i) & ONE));//,((state->v >> i) & ONE),);
    }

}

int init_cb_CHForm(unsigned int n, CHForm * state){
    state->n = n;
    if(n > 0){
        state->F = calloc(n, sizeof(uint_bitarray_t));
        state->G = calloc(n, sizeof(uint_bitarray_t));
        state->M = calloc(n, sizeof(uint_bitarray_t));
    }else{
        state->F = NULL;
        state->G = NULL;
        state->M = NULL;
    }
    for(int i = 0; i < state->n; i++){
        state->F[i] |= (ONE << i);
        state->G[i] |= (ONE << i);
    }
    state->g1 = 0;
    state->g2 = 0;
    state->s = 0;
    state->v = 0;
    state->w = 1. + 0.*I;

    return 0;
}

CHForm copy_CHForm(CHForm * state){
    CHForm copy;
    copy.n = state->n;
    if(state->n > 0){
        copy.F = calloc(state->n, sizeof(uint_bitarray_t));
        copy.G = calloc(state->n, sizeof(uint_bitarray_t));
        copy.M = calloc(state->n, sizeof(uint_bitarray_t));
    }else{
        copy.F = NULL;
        copy.G = NULL;
        copy.M = NULL;
    }
    for(int i = 0; i < copy.n; i++){
        copy.F[i] = state->F[i];
        copy.G[i] = state->G[i];
        copy.M[i] = state->M[i];
    }
    copy.g1 = state->g1;
    copy.g2 = state->g2;
    copy.s = state->s;
    copy.v = state->v;
    copy.w = state->w;

    return copy;
}

int init_zero_CHForm(unsigned int n, CHForm * state){
    state->n = n;
    if(n > 0){
        state->F = calloc(state->n, sizeof(uint_bitarray_t));
        state->G = calloc(state->n, sizeof(uint_bitarray_t));
        state->M = calloc(state->n, sizeof(uint_bitarray_t));
    }else{
        state->F = NULL;
        state->G = NULL;
        state->M = NULL;
    }
    state->g1 = 0;
    state->g2 = 0;
    state->s = 0;
    state->v = 0;
    state->w = 0. + 0.*I;

    return 0;
}

int dealocate_state(CHForm * state){
    if(state->F != NULL){
        free(state->F);
        state->F = NULL;
    }
    if(state->G != NULL){
        free(state->G);
        state->G = NULL;
    }
    if(state->M != NULL){
        free(state->M);
        state->M = NULL;
    }
    return 0;
}

/*
 * GATES
 */

int SR(CHForm * state, unsigned int q){
    for(int p = 0; p < state->n; p++){
        state->M[p] ^=  (state->F[p] & (ONE << q));
        state->g1 ^= ((state->F[p]>>q) & ONE) << p;
        state->g2 ^= ((state->F[p]>>q) & (state->g1 >>p) & ONE)<<p;
    }

    return 0;
}

int SL(CHForm * state, unsigned int q){
    state->M[q] ^=  state->G[q];
    state->g1 ^= (ONE <<q);
    state->g2 ^= (state->g1 & (ONE <<q));
    return 0;
}

int CZR(CHForm * state, unsigned int q, unsigned int r){
    for(int p = 0; p < state->n; p++){
        state->M[p] ^=  ((state->F[p] >> r) & ONE)<<q;
        state->M[p] ^=  ((state->F[p] >> q) & ONE)<<r;

        state->g2 ^= (((state->F[p] >> q) & ONE) & ((state->F[p] >> r) & ONE)) << p;
    }
    return 0;
}
int CZL(CHForm * state, unsigned int q, unsigned int r){
    state->M[q] ^= state->G[r];
    state->M[r] ^= state->G[q];
    return 0;
}


int CXR(CHForm * state, unsigned int q, unsigned int r){
    for(int p = 0; p < state->n; p++){
        state->G[p] ^= ((state->G[p]>>r) & ONE)<<q;
        state->F[p] ^= ((state->F[p]>>q) & ONE)<<r;
        state->M[p] ^= ((state->M[p]>>r) & ONE)<<q;
    }
    return 0;
}


int CXL(CHForm * state, unsigned int q, unsigned int r){

    state->g2 ^= ((state->g1>>r) & (state->g1>>q) & ONE) <<q;
    state->g1 ^= (((state->g1 >> r) & ONE)<<q);
    state->g2 ^= (((state->g2 >> r) & ONE)<<q);

    for(int p = 0; p < state->n; p++){
        state->g2 ^= ((state->M[q]>>p) & (state->F[r]>>p) & ONE) << q;
    }

    state->G[r] ^= state->G[q];
    state->F[q] ^= state->F[r];
    state->M[q] ^= state->M[r];

    return 0;
}
int SWAPR(CHForm * state, unsigned int q, unsigned int r){
    for(int p = 0; p < state->n; p++){
        state->G[p] ^= ((state->G[p]>>r) & ONE)<<q;
        state->F[p] ^= ((state->F[p]>>q) & ONE)<<r;
        state->M[p] ^= ((state->M[p]>>r) & ONE)<<q;

        state->G[p] ^= ((state->G[p]>>q) & ONE)<<r;
        state->F[p] ^= ((state->F[p]>>r) & ONE)<<q;
        state->M[p] ^= ((state->M[p]>>q) & ONE)<<r;

        state->G[p] ^= ((state->G[p]>>r) & ONE)<<q;
        state->F[p] ^= ((state->F[p]>>q) & ONE)<<r;
        state->M[p] ^= ((state->M[p]>>r) & ONE)<<q;
    }
    return 0;
}


int XL(CHForm * state, unsigned int p){
    //note that X is not C-type so we multiply an X gate onto the whole state not just U_C
    //for this reason XR doesn't make any sense
    //so we only have XL
    //If necessary we could think of an X acting from the right on U_C as an X acting from the left on U_H |s>

    int count = 0;
    for(int j =0; j < state->n; j++){
        //first hit |s> with UH^{-1} Z_j^{M_{p,j}} UH
        if((state->M[p] >> j) & ONE){
            if((state->v >> j) & ONE){
                state->s ^= (ONE << j);
            }else if((state->s >> j) & ONE){
                count += 1;
            }
        }
        //now hit |s> with UH^{-1} X_j^{F_{p,j}} UH
        if((state->F[p] >> j) & ONE){
            if((state->v >> j) & ONE){
                if((state->s >> j) & ONE){
                    count += 1;
                }
            }else{
                state->s ^= (ONE << j);
            }
        }
    }

    state->w *= (((count + ((state->g2>>p)&ONE)) %2) ? -1.:1.);
    state->w *= (((state->g1>>p) & ONE)  ? I:1.);
    return 0;
}


enum gate{
    cx,
    cz,
    s,
    h
};

int desupersitionise(CHForm * state, uint_bitarray_t u,  unsigned int d){

    uint_bitarray_t sNeqU = state->s ^ u;
    if(sNeqU == 0){
        return -1;
    }

    uint_bitarray_t v0 = (~state->v) & sNeqU;
    uint_bitarray_t v1 = (state->v) & sNeqU;
    int q = -1;
    if(v0 != 0){
        for(int i = 0; i< state->n; i++){
            if( (v0 >> i) & ONE){
                if(q < 0){
                    q = i;
                }else{
                    CXR(state, q, i);
                }
            }
        }
        for(int i = 0; i< state->n; i++){
            if( ((v1 >> i) & ONE)){
                CZR(state, q, i);
            }

        }

    }else{
        for(int i = 0; i < state->n; i++){
            if( (v1 >> i) & ONE){
                if(q < 0){
                    q = i;
                }else{
                    CXR(state, i, q);
                }
            }
        }
    }

    uint_bitarray_t y, z;
    if((state->s >> q) & ONE){
        y = u;
        y ^= (ONE << q);
        z = u;
    }else{
        y = state->s;
        z = state->s;
        z ^= (ONE << q);
    }
    unsigned int w = 0;
    unsigned int k = d;

    if((y>>q) & ONE){
        w = d;
        k = (4u-d) % 4u;
    }

    unsigned int a, b,c;

    double complex phase = sqrt(2.);
    for(unsigned int i = 0; i < w; i++){
        phase *= I;
    }

    if( (~(state->v >> q)) & ONE){ //v[q] == 0
        b = 1;
        if(k==0){
            a = 0;
            c = 0;
        }else if(k==1){
            a = 1;
            c = 0;
        }else if(k==2){
            a = 0;
            c = 1;
        }else /*(k==3)*/{
            a = 1;
            c = 1;
        }
    }else{//v[q] == 1
        if(k==0){
            a = 0;
            b = 0;
            c = 0;
        }else if(k==1){
            a = 1;
            b = 1;
            c = 1;
            phase *= (1 + I)/sqrt(2);
        }else if(k==2){
            a = 0;
            b = 0;
            c = 1;
        }else/*(k==3)*/{
            a = 1;
            b = 1;
            c = 0;
            phase *= (1 - I)/sqrt(2);
        }
    }

    state->s = y;
    state->s ^=  (state->s & (ONE << q)) ^ ((c & ONE) << q);
    state->v ^=  (state->v & (ONE << q)) ^ ((b & ONE) << q);
    state->w *= phase;
    if(a){
        SR(state, q);
    }

    return 0;
}

int HL(CHForm * state, unsigned int q){
    uint_bitarray_t t = state->s ^ (state->G[q] & state->v);
    uint_bitarray_t u = state->s ^ ((state->F[q] & (~state->v)) ^ (state->M[q] & state->v));
    uint_bitarray_t alpha = 0;
    uint_bitarray_t beta = 0;
    for(int i=0; i < state->n; i++){
        alpha ^= ((state->G[q] & (~state->v) & state->s)) >>i;
        beta ^= ((state->M[q] & (~state->v) & state->s) ^ (state->F[q] & state->v & (state->M[q] ^ state->s))) >>i;

    }
    alpha &= ONE;
    beta &= ONE;
    state->s = t;
    if(t == u){
        double complex a = 1;
        if(alpha){
            a *= -1;
        }
        double complex b = 1;
        if(beta ^ ((state->g2 >> q) & ONE) ){
            b *= -1;
        }
        if((state->g1 >> q) & ONE ){
            b *= I;
        }
        state->w *=  (a + b)/sqrt(2.);
    }else{
        unsigned int d = (((state->g1>>q) &ONE) + 2*((state->g2>>q) & ONE)  + 2*(alpha+beta)) % 4u;
        desupersitionise(state, u, d);
        if(alpha){
            state->w *= -1;
        }
        state->w /= sqrt(2);
    }

    return 0;
}


static CHForm * compute_CHForm(int n, PyArrayObject * F, PyArrayObject *G, PyArrayObject *M, PyArrayObject *g, PyArrayObject *v,PyArrayObject *s, Py_complex w)
{
    CHForm * state = calloc(1, sizeof(CHForm));

    init_zero_CHForm(n, state);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            state->F[i] |= (((uint_bitarray_t)F->data[i*F->strides[0] + j*F->strides[1]]) & ONE) << j;
            state->G[i] |= (((uint_bitarray_t)G->data[i*G->strides[0] + j*G->strides[1]]) & ONE) << j;
            state->M[i] |= (((uint_bitarray_t)M->data[i*M->strides[0] + j*M->strides[1]]) & ONE) << j;
        }

        state->g1 |= (((uint_bitarray_t)g->data[i*g->strides[0]]) & ONE) << i;
        state->g2 |= ((((uint_bitarray_t)g->data[i*g->strides[0]]) >> 1u) & ONE) << i;

        state->v |= (((uint_bitarray_t)v->data[i*v->strides[0]]) & ONE) << i;
        state->s |= (((uint_bitarray_t)s->data[i*s->strides[0]]) & ONE) << i;
    }
    state->w = w.real + I*w.imag;
    return state;
}


static CHForm * python_tuple_to_CHForm(PyObject * tuple){
    PyObject * py_n = PyTuple_GetItem(tuple, 0);
    PyArrayObject * py_F = (PyArrayObject *)PyTuple_GetItem(tuple, 1);
    PyArrayObject * py_G = (PyArrayObject *)PyTuple_GetItem(tuple, 2);
    PyArrayObject * py_M = (PyArrayObject *)PyTuple_GetItem(tuple, 3);
    PyArrayObject * py_g = (PyArrayObject *)PyTuple_GetItem(tuple, 4);
    PyArrayObject * py_v = (PyArrayObject *)PyTuple_GetItem(tuple, 5);
    PyArrayObject * py_s = (PyArrayObject *)PyTuple_GetItem(tuple, 6);
    PyObject * py_phase = PyTuple_GetItem(tuple, 7);

    int n = PyLong_AsLong(py_n);
    return compute_CHForm(n, py_F, py_G, py_M, py_g, py_v, py_s, PyComplex_AsCComplex(py_phase));
    
}

static PyObject * CHForm_to_python_tuple(CHForm * state){    
    const long int dimensions1[1] = {state->n};
    const long int dimensions2[2] = {state->n, state->n};

    PyArrayObject * F = (PyArrayObject*)PyArray_SimpleNew(2, dimensions2,  PyArray_UBYTE);
    PyArrayObject * G = (PyArrayObject*)PyArray_SimpleNew(2, dimensions2,  PyArray_UBYTE);
    PyArrayObject * M = (PyArrayObject*)PyArray_SimpleNew(2, dimensions2,  PyArray_UBYTE);
    PyArrayObject * g = (PyArrayObject*)PyArray_SimpleNew(1, dimensions1,  PyArray_UBYTE);
    PyArrayObject * v = (PyArrayObject*)PyArray_SimpleNew(1, dimensions1,  PyArray_UBYTE);
    PyArrayObject * s = (PyArrayObject*)PyArray_SimpleNew(1, dimensions1,  PyArray_UBYTE);

    for(int i = 0; i < state->n; i++){
        for(int j = 0; j < state->n; j++){
	    //printf("(%d, %d)\n",i,j);
            F->data[i*F->strides[0] + j*F->strides[1]] = (unsigned char)((state->F[i] >> j) & ONE);
            G->data[i*G->strides[0] + j*G->strides[1]] = (unsigned char)((state->G[i] >> j) & ONE);
            M->data[i*M->strides[0] + j*M->strides[1]] = (unsigned char)((state->M[i] >> j) & ONE);
        }
        g->data[i*g->strides[0]] = 2*((state->g2 >> i) & ONE) + ((state->g1 >> i) & ONE);
        v->data[i*v->strides[0]] = ((state->v >> i) & ONE);
        s->data[i*s->strides[0]] = ((state->s >> i) & ONE);
    }
    //printf("%lf + %lf\n", creal(state->w), cimag(state->w));
    PyObject * phase = (Py_complex*)PyComplex_FromDoubles(creal(state->w), cimag(state->w));
    return Py_BuildValue("iOOOOOOO", state->n, F, G, M, g, v, s, phase);
}

static CHForm * c_apply_gates_to_basis_state(int n, PyArrayObject * gates, PyArrayObject * controls, PyArrayObject * targets){
    CHForm * state = calloc(1, sizeof(CHForm));
    init_cb_CHForm(n, state);
    //printf("3\n");

    for(int i = 0; i < gates->dimensions[0]; i++){
        switch((char)gates->data[i*gates->strides[0]]) {
        case 'X':
            CXL(state, (unsigned int)controls->data[i*controls->strides[0]], (unsigned int)targets->data[i*targets->strides[0]]);
            break;
        case  'Z':
            CZL(state, (unsigned int)controls->data[i*controls->strides[0]], (unsigned int)targets->data[i*targets->strides[0]]);
            break;
        case 's':
            SL(state, (unsigned int)targets->data[i*targets->strides[0]]);
            break;
        case 'h':
            HL(state, (unsigned int)targets->data[i*targets->strides[0]]);
            break;
        }
    }

    return state;
}

static PyObject * apply_gates_to_basis_state(PyObject* self, PyObject* args){
    PyArrayObject * gates;
    PyArrayObject * controls;
    PyArrayObject * targets;
    int n;
    //printf("1\n");
    if (!PyArg_ParseTuple(args, "iO!O!O!", &n,
                          &PyArray_Type, &gates,
                          &PyArray_Type, &controls,
                          &PyArray_Type, &targets
            )){
        return NULL;
    }

    CHForm * state =  c_apply_gates_to_basis_state(n, gates, controls, targets);
    PyObject * tuple = CHForm_to_python_tuple(state);
    dealocate_state(state);
    return tuple;
}

static unsigned int sort_pauli_string(uint n, uint_bitarray_t * x, uint_bitarray_t * z, uint_bitarray_t mask)
{
    uint_bitarray_t t = 0;
    unsigned int sign = 0;
    for(int i = 0; i < n; i++){
        if((mask >> i) & ONE){
            t ^= z[i];
            sign ^= parity(t & x[i]);
        }
    }

    return sign;
}

/* Given a product of the form appearing in (55) of Bravyi et al
 * work out the minus sign you get if you pull all the Zs to the left hand side and all the xs to the right
 */

static unsigned int sort_pauli_string2(uint n, uint_bitarray_t * x, uint_bitarray_t * z, uint_bitarray_t mask)
{
    uint_bitarray_t t = 0;
    unsigned int sign = 0;
    for(int i = 0; i < n; i++){
        if((mask >> i) & ONE){
            sign ^= parity(t & x[i]);
            t ^= z[i];
        }
    }

    return sign;
}

static double complex measurement_overlap(CHForm * state, uint_bitarray_t x){
    //compute the inner product <x | state>
    //where the bitstring x determines a computational basis state
    uint_fast64_t u = 0;
    // u = x F
    for(int i =0; i < state->n; i++){
        for(int j=0; j <state->n;j++){
            u ^=  ((x>>j) & (state->F[j] >>i) & ONE) << i;
        }
    }
    if((u ^ state->s ) & (~state->v)){
        return 0;
    }
    unsigned int signbit = sort_pauli_string(state->n, state->F, state->M, x);
    signbit ^= parity(u & state->s & state->v);

    unsigned int g = 0;
    for(int i = 0; i < state->n; i++){
        if(( x >> i) &ONE ){
            g +=  ((state->g1 >> i) & ONE) + 2*((state->g2 >> i) & ONE);
        }
    }
    if(signbit & ONE){
        g += 2;
    }

    double complex phase = state->w;
    g %= 4;
    if(g == 1){
        phase *= I;
    }else if(g == 2){
        phase *= -1;
    }else if(g == 3){
        phase *= -1*I;
    }

    double sqrt2 = sqrt(2.);
    for(int i = 0; i < state->n; i++){
        if((state->v >> i) & ONE){
            phase /= sqrt2;
        }
    }

    return phase;
}

static void apply_z_projector(CHForm * state, int a, int q){
    //apply the projector |a><a| to the qubit q where a is the least significant bit of a
    unsigned int k = a ^ parity(state->G[q] & (~state->v) & state->s);
    uint_bitarray_t t = (state->G[q] & state->v) ^ state->s;

    if(t == state->s){
        if(k){
            state->w = 0;
        }else{
            desupersitionise(state, t, 2*k);
            state->w /= 2; // 2 since P = (I +- Z)/2
        }
    }
}
static void apply_z_projectors(CHForm * state, uint_bitarray_t a, uint_bitarray_t mask){
    //for each qubit i
    //if mask mask_i == 1
    //apply the projector |a_i><a_i|
    for(int i=0; i < state->n; i++){
        if((mask >> i) & ONE){
            unsigned int k = ((a>>i) ^ parity(state->G[i] & (~state->v) & state->s)) & ONE;
            uint_bitarray_t t = (state->G[i] & state->v) ^ state->s;
            if(t == state->s){
                if(k){
                    state->w = 0;
                }
            }else{
                desupersitionise(state, t, (2*k) % 4u);
                state->w /= 2; // 2 since P = (I +- Z)/2
            }
        }
    }
}

static CHForm * postselect_and_reduce(CHForm * state, uint_bitarray_t a, uint_bitarray_t mask){
    //handle the case where we're actually doing a full inner product separately
    if(popcount(mask) == state->n){
        state->w = measurement_overlap(state, a);
        if(state->n > 0){
            free(state->F);
            free(state->G);
            free(state->M);
            state->F = NULL;
            state->G = NULL;
            state->M = NULL;
        }
        state->n = 0;
        return state;
    }

    for(unsigned int i = 0; i < state->n; i++){
        if(((a & mask) >> i) & ONE){
            XL(state, i);
        }
    }
    apply_z_projectors(state, 0u, mask);

    //now we arrange it so there is at most one qubit with s_i = 1, v_i = 0
    //first try to find a qubit that isn't being deleted with s_i = 1, v_i = 0
    if(((~state->v) & state->s) != 0){
        //inside this block we know there are some qubits with s_i = 1, v_i = 0
        int control_qubit = -1;
        for(int i = 0; i < state->n; i++){
            if((((~mask) & state->s & (~state->v)) >> i) & ONE){
                control_qubit = i;
                break;
            }
        }

        //we want to insert 4 cnots to swap an s_i = 1 onto a qubit with v_i = mask_i = 0
        if(control_qubit < 0){
            //first find a mask qubit with s_i = 1, v_i = 0
            //and a non-mask qubit with s_i = 0, v_i = 0
            int mask_qubit = -1;
            int non_mask = -1;

            for(int i = 0; i < state->n; i++){
                if(((state->s & (~state->v)) >> i) & ONE){
                    mask_qubit = i;
                }
                if((((~mask) & (~state->s) & (~state->v)) >> i) & ONE){
                    non_mask = i;
                }
            }
            //insert this which is the identity
            //CX(mask_qubit, non_mask) CX(non_mask, mask_qubit) CX(non_mask, mask_qubit) CX(mask_qubit, non_mask)
            //multiply the left hand two onto U_C
            //and the right ones onto U_H |s>
            //where they will swap the which qubit has s = 0 and which has s = 1
            CXR(state, mask_qubit, non_mask);
            CXR(state, non_mask, mask_qubit);

            state->s |= (ONE << non_mask);
            state->s &= (~(ONE << mask_qubit));
            control_qubit = non_mask;
        }

        //so now we have a control qubit and (possibly) some others with s_i = 1, v_i = 0
        //we go through and switch the s_i's to zero and insert cnots controlled on our control
        for(int i = 0; i < state->n; i++){
            if(i != control_qubit){
                if((((~state->v) & state->s) >> i) & ONE){
                    CXR(state, control_qubit, i);
                }
            }
        }
        state->s ^= (state->s & (~state->v));
        state->s |= (ONE << control_qubit);
    }
    //at this point as many qubits are "free" as possible
    //i.e. there is at most one qubit with s_i = 1, v_i = 0
    //and this control qubit is /not/ one of our mask qubits
    //we also want to ensure that all of our mask qubits have s_i = v_i = 0
    //we know already at this point that if they have v_i = 0 they have s_i = 0
    //so we just swap those of them which have v_i = 1 with a qubit that has v_i = s_i = 0

    int swapCandidateIndex = 0;
    for(int i = 0; i < state->n; i++){
        if(((mask & state->v) >> i) & ONE){
            for(; ((mask | state->s | state->v)>>swapCandidateIndex) & ONE; swapCandidateIndex++){};
            SWAPR(state, i, swapCandidateIndex);
            state->s |= ((state->s >> i) & ONE) << swapCandidateIndex;
            state->v |= ((state->v >> i) & ONE) << swapCandidateIndex;

            state->s &= (~(ONE << i));
            state->v &= (~(ONE << i));
        }
    }

    //printf("confirmation!\n");
    //printBits(mask & state->s & state->v, state->n);printf("\n");

    //at this point all our mask qubits have s_i = v_i = 0
    //and there is at most one qubit with s_i=0, v_i = 1
    //now we ensure that for each mask qubit q we have G[q][q] == 1
    //and use that G is the inverse of F^T
    //to make each column we want to throw away "simple"
    //i.e. have a single 1 on the diagonal and zeros elsewhere
    uint_bitarray_t marked = 0;
    for(int q = 0; q < state->n; q++){
        if((mask >> q) & ONE){ //q is a masked qubit
            if(((state->G[q] >> q) & ONE) == 0){
                for(int i = 0; i < state->n; i++){
                    if(((state->G[q] & (~marked)) >> i) & ONE){
                        SWAPR(state,q,i);
                        break;
                    }
                }
            }
            for(int i=0; i < state->n; i++){
                if((i != q) && ((state->G[q] >> i) & ONE)){
                    CXR(state, i, q);
                }
            }
            marked |= (ONE<<q);
        }
    }

    //now we want to delete the masked qubits
    int shift = 0;
    int i = 0;
    //uint_bitarray_t i_bit_mask = 0u;
    //printBits(mask, state->n); printf("\n");
    /* printf("c mats\n"); */
    /* for(int q = 0; q < state->n; q++){ */
    /*     printBits(state->F[q], state->n); */
    /*     printf(" "); */
    /*     printBits(state->G[q], state->n); */
    /*     printf(" "); */
    /*     printBits(state->M[q], state->n); */
    /*     printf(" "); */
    /*     printf("\n"); */
    /* } */
    //delete rows
    for(; i+shift < state->n; i++){
        while(((mask>>(i+shift)) & ONE)){
            shift += 1;
        }

        if(i+shift < state->n){
            state->F[i] = state->F[i+shift];
            state->G[i] = state->G[i+shift];
            state->M[i] = state->M[i+shift];

            for(int j = 0; j < state->n; j++){
                state->F[j] = (state->F[j] ^ (state->F[j] & (ONE<<i))) | ((state->F[j] >> shift) & (ONE <<i));
                state->G[j] = (state->G[j] ^ (state->G[j] & (ONE<<i))) | ((state->G[j] >> shift) & (ONE <<i));
                state->M[j] = (state->M[j] ^ (state->M[j] & (ONE<<i))) | ((state->M[j] >> shift) & (ONE <<i));
            }

            state->v = (state->v ^ (state->v & (ONE<<i))) | ((state->v >> shift) & (ONE <<i));
            state->s = (state->s ^ (state->s & (ONE<<i))) | ((state->s >> shift) & (ONE <<i));
            state->g1 = (state->g1 ^ (state->g1 & (ONE<<i))) | ((state->g1 >> shift) & (ONE <<i));
            state->g2 = (state->g2 ^ (state->g2 & (ONE<<i))) | ((state->g2 >> shift) & (ONE <<i));
        }
    }

    state->n = state->n-shift;

    state->F = realloc(state->F, state->n * sizeof(uint_bitarray_t));
    state->G = realloc(state->G, state->n * sizeof(uint_bitarray_t));
    state->M = realloc(state->M, state->n * sizeof(uint_bitarray_t));

    uint_bitarray_t m = 0u;
    for(size_t i = 0; i < state->n; i++){
        m |=  (ONE << i);
    }
    for(size_t i = 0; i < state->n; i++){
        state->F[i] &= m;
        state->G[i] &= m;
        state->M[i] &= m;
    }
    state->g1 &= m;
    state->g2 &= m;
    state->v &= m;
    state->s &= m;

    return state;
}

static PyObject * apply_gates_to_basis_state_project_and_reduce(PyObject* self, PyObject* args){
    PyArrayObject * gates;
    PyArrayObject * controls;
    PyArrayObject * targets;
    PyArrayObject * a; // project |a_i><a_i| onto qubit i iff mask_i == 1
    PyArrayObject * mask;

    int n;
    //printf("1\n");
    if (!PyArg_ParseTuple(args, "iO!O!O!O!O!", &n,
                          &PyArray_Type, &gates,
                          &PyArray_Type, &controls,
                          &PyArray_Type, &targets,
                          &PyArray_Type, &a,
                          &PyArray_Type, &mask
            )){
        return NULL;
    }

    CHForm * state =  c_apply_gates_to_basis_state(n, gates, controls, targets);

    uint_bitarray_t bitA = 0;
    uint_bitarray_t bitMask = 0;


    for(int i = 0; i < n; i++){
        if((char)a->data[i*a->strides[0]]){
            bitA |= (ONE << i);
        }
        if((char)mask->data[i*mask->strides[0]]){
            bitMask |= (ONE << i);
        }
    }

    postselect_and_reduce(state, bitA, bitMask);

    PyObject * tuple = CHForm_to_python_tuple(state);
    dealocate_state(state);
    return tuple;
}

//equatorial matrices define equatorial states
//they are symmetric matrices with binary off diagonal elements
//and mod 4 diagonal elements
typedef struct equatorial_matrix{
    int n;
    uint_bitarray_t * mat;
    uint_bitarray_t d1;
    uint_bitarray_t d2;
} equatorial_matrix_t;

void init_zero_equatorial_matrix(equatorial_matrix_t * matrix, int n){
    matrix->n = n;
    matrix->mat = (uint_bitarray_t*)calloc(n, sizeof(uint_bitarray_t));
    matrix->d1 = 0u;
    matrix->d2 = 0u;
}

void init_random_equatorial_matrix(equatorial_matrix_t * matrix, int n){
    matrix->n = n;
    matrix->mat = (uint_bitarray_t*)calloc(n, sizeof(uint_bitarray_t));
    uint_bitarray_t mask = 0u;
    for(int i = 0; i < n; i++){
        mask |= (ONE<<i);
    }
    for(int i = 0; i < n; i++){
        matrix->mat[i] = (bitarray_rand()) & mask;
    }
    for(int i = 0; i < n; i++){
        for(int j = 0; j < i; j++){
            matrix->mat[i] &= ~(ONE << j);//(((matrix->mat[j]>>i) &ONE) <<j);
            matrix->mat[i] |= (((matrix->mat[j] >> i) & ONE) <<j);
        }
        matrix->mat[i] &= ~(ONE << i);
    }
    matrix->d1 = bitarray_rand() & mask;
    matrix->d2 = bitarray_rand() & mask;
}

void dealocate_equatorial_matrix(equatorial_matrix_t * matrix){
    matrix->n = 0;
    free(matrix->mat);
}

double complex equatorial_inner_product(CHForm* state, equatorial_matrix_t equatorial_state){

    if(state->n == 0){
        return conj(state->w);
    }
    //we store A+J in AJ
    uint_bitarray_t * AJ = calloc(state->n, sizeof(uint_bitarray_t));

    for(size_t i = 0; i < state->n; i++){
        for(size_t j = 0; j < i; j++){
            uint_bitarray_t bit = parity(state->M[i] & state->F[j]) & ONE;
            AJ[i] |= (bit << j);
            AJ[j] |= (bit << i);
        }
    }

    //add A to J
    for(size_t i = 0; i < state->n; i++){
        AJ[i] ^= equatorial_state.mat[i];
        AJ[i] &= ~(ONE<<i);
    }
    //now we need to sort out the diagonal
    uint_bitarray_t AJd1 = equatorial_state.d1;
    uint_bitarray_t AJd2 = equatorial_state.d2;

    AJd2 ^= (AJd1 & state->g1);
    AJd1 ^= state->g1;
    AJd2 ^= state->g2;

    uint_bitarray_t * GT = calloc(state->n, sizeof(uint_bitarray_t)); // store transpose of G
    for(size_t i = 0; i < state->n; i++){
        for(size_t j = 0; j < state->n; j++){
            GT[j] |= ((state->G[i] >> j) & ONE) << i;
        }
    }

    //now we want to compute (A G)^T = G^T A^T
    //this is because doing X Y^T is generally faster than doing XY
    //since we can do the row / row dot products with popcount(x & y)
    //we need to know the value of G^T A^T mod-4 so we can work out what the diagonal of G^T A G should be
    //so we store it in two binary matrices X and Y such that G^T A^T = 2X + Y

    uint_bitarray_t * X = calloc(state->n, sizeof(uint_bitarray_t));
    uint_bitarray_t * Y = calloc(state->n, sizeof(uint_bitarray_t));

    for(size_t i = 0; i < state->n; i++){
        for(size_t j = 0; j < state->n; j++){
            uint_bitarray_t pc = (popcount(GT[i] & AJ[j]) % 4u);
            X[i] |= ((pc>>1) & ONE) << j;
            Y[i] |= ((pc) & ONE) << j;
        }
    }

    //add the contribution fron G^T D
    for(size_t i = 0; i < state->n; i++){
        X[i] ^= (Y[i] & GT[i] & AJd1); // carry if both bits are 1
        Y[i] ^= (GT[i] & AJd1);
        X[i] ^= (GT[i] & AJd2);
    }

    //now we compute K = G^T (A G) = G^T (G^T A^T)^T
    //we store K as a full symmetric matric of bits
    //we store the even part of the diagonal in bitarray bitKd2;
    //since the diagonal is the only bit we need mod-4
    //in other words K = bitK + 2*diag(bitKd2)
    uint_bitarray_t * bitK = calloc(state->n, sizeof(uint_bitarray_t));
    uint_bitarray_t bitKd2 = 0;


    for(size_t i = 0; i < state->n; i++){
        for(size_t j = 0; j < i; j++){ //symmetric
            uint_bitarray_t pb = parity(GT[i] & Y[j]) & ONE;
            bitK[i] |=  pb << j;
            bitK[j] |=  pb << i;
        }
        //now we need to work out the diagonal
        //slightly more care is needed here as we care about the diagonal mod-4
        uint_bitarray_t pc = popcount(GT[i] & Y[i]);
        bitK[i] |= (pc & ONE) << i;
        bitKd2 |= (((pc>>1) & ONE) ^ (parity(GT[i] & X[i]) & ONE)) << i;
    }
    free(X);
    free(Y);

    unsigned int n = popcount(state->v);

    uint_bitarray_t sK = 0;
    unsigned int sKs = 0;
    for(size_t a = 0; a < state->n; a++){
        unsigned int pc = popcount(state->s & bitK[a]) % 4u;
        sK |= (pc & ONE) << a;
        sKs += pc * ((state->s >> a) & ONE);
    }

    sKs += 2*popcount(bitKd2 & state->s);

    //add 2*diag(s + sK) onto K
    bitKd2 ^= (state->s ^ sK);


    double complex prefactor = pow(0.5, (state->n+n)/2.);
    //printf("c sKs: %d, sKs2: %u\n", popcount(state->s & sK), sKs);
    unsigned int d = (sKs + 2 * popcount(state->s & state->v)) % 4u;
    if(d == 1){
        prefactor *= I;
    }else if(d == 2){
        prefactor *= -1.;
    }else if(d == 3){
        prefactor *= -1.*I;
    }

    uint_bitarray_t k = 0;
    uint_bitarray_t L = 0;

    uint_bitarray_t * M = calloc(n+1, sizeof(uint_bitarray_t));
    int fill_count_a = 0;

    for(int a = 0; (a<state->n); a++){
        if((state->v >> a) & ONE){
            k |= ((bitK[a] >> a) & ONE) << fill_count_a;
            L |= ((bitKd2 >> a) & ONE) << fill_count_a;
            fill_count_a += 1;
        }
    }
    fill_count_a = 0;
    int fill_count_b = 0;
    for(int a = 0; (a<state->n); a++){
        if((state->v >> a) & ONE){
            for(int b = 0; (b<a); b++){
                if((state->v >> b) & ONE){
                    M[fill_count_b] |= (((bitK[b] >> a) & ONE) ^ ((k >> fill_count_a) & (k >> fill_count_b) & ONE)) << fill_count_a;
                    fill_count_b += 1;
                }
            }
            fill_count_a += 1;
            fill_count_b = 0;
        }
    }
    M[n] = k;
    n +=1;

    //at this point we only need M and l
    //so free everything else
    free(bitK);
    free(AJ);
    free(GT);
    double re=0, im=0;
    int killed = 0;
    int exponent_of_2 = 0;
    bool exponent_of_minus_1 = false;
    bool last_element_asymetric = false;
    bool mu1_consts = false;
    bool mu2_consts = false;

    uint_fast64_t mask = 0;
    for(uint i = 0; i < n; i++){
        mask |= (ONE << i);
    }
    //printf("eb\n");
    while(true){
        uint r=0, c=0;
        bool found = false;
        for(uint i = 0; i < n && !found; i++){
            for(uint j = 0; j < i && !found; j++){
                if(((M[i] >> j) & ONE) != ((M[j] >> i) & ONE)){
                    r=i;
                    c=j;
                    found = true;
                }
            }
        }
        if(!found){
            //this is trivial apparently
            uint_bitarray_t diag = 0;
            for(uint i=0;i<n;i++){
                diag ^= ((M[i] >> i) & ONE) << i;
            }
            if(last_element_asymetric){
                if((diag & mask) == (L&mask)){
                    //printf("c1\n");
                    double signR = exponent_of_minus_1 ? (-1.) : 1.;
                    bool new_exponent_of_minus_1 = (exponent_of_minus_1 ^ mu2_consts);
                    double signI = new_exponent_of_minus_1 ? (-1.) : 1.;
                    re = pow(2., exponent_of_2 + n - killed)*signR;
                    im = pow(2., exponent_of_2 + n - killed)*signI;
                    break;
                }else{
                    re = 0.;
                    im = 0.;
                    break;
                }
            }else{
                if( ((diag & (~(ONE<<(n-1)))) & mask) == ((L & (~(ONE<<(n-1)))) & mask)){
                    if( ((diag & (ONE << (n-1)))&mask) == ((L & (ONE << (n-1)))&mask)){
                        double signR = exponent_of_minus_1 ? (-1.) : 1.;
                        re = signR * pow(2., exponent_of_2+n-killed);
                        im = 0;
                        break;
                    }else{
                        re = 0;
                        double signI = exponent_of_minus_1 ? (-1.) : 1.;
                        im = signI * pow(2., exponent_of_2+n-killed);
                        break;
                    }

                }else{
                    re = 0;
                    im = 0;
                    break;
                }
            }
        }else{
            if(r+1 == n){
                last_element_asymetric = true;
            }

            killed += 2;
            uint_fast64_t m1 = M[r];
            uint_fast64_t m2 = M[c];

            for(uint i=0; i<n;i++){
                m1 ^= (((M[i] >> r) & ONE) << i);
                m2 ^= (((M[i] >> c) & ONE) << i);
            }
            m1 &= (~(ONE << r));
            m1 &= (~(ONE << c));
            m2 &= (~(ONE << r));
            m2 &= (~(ONE << c));

            mu1_consts = ((L>>r) & ONE) ^ ((M[r]>>r) & ONE);
            mu2_consts = ((L>>c) & ONE) ^ ((M[c]>>c) & ONE);

            M[r] = 0;
            M[c] = 0;
            for(uint i=0; i<n;i++){
                M[i] &= ~(ONE << r);
                M[i] &= ~(ONE << c);
            }

            L &= (~(ONE << r));
            L &= (~(ONE << c));
            exponent_of_2 += 1;
            exponent_of_minus_1 ^= (mu1_consts & mu2_consts);

            for(uint i=0;i<n;i++){
                if((m1>>i) & ONE){
                    M[i] ^= m2;
                }
            }
            if(mu1_consts){
                L ^= m2;
            }
            if(mu2_consts){
                L ^= m1;
            }
        }
    }


    free(M);

    //printf("en\n");
    return conj(state->w) * prefactor * (re +im*I)/2;
}

double complex equatorial_inner_product2(CHForm* state, uint_bitarray_t * A, uint_bitarray_t * GT, equatorial_matrix_t equatorial_state){

    //we store A+J in AJ
    uint_bitarray_t * AJ = calloc(state->n, sizeof(uint_bitarray_t));

    //add A to J
    for(size_t i = 0; i < state->n; i++){
        AJ[i] |= (A[i] ^ equatorial_state.mat[i]);
    }
    //now we need to sort out the diagonal
    uint_bitarray_t AJd1 = equatorial_state.d1;
    uint_bitarray_t AJd2 = equatorial_state.d2;

    AJd2 ^= (AJd1 & state->g1);
    AJd1 ^= state->g1;
    AJd2 ^= state->g2;


    //now we want to compute (A G)^T = G^T A^T
    //this is because doing X Y^T is generally faster than doing XY
    //since we can do the row / row dot products with popcount(x & y)
    //we need to know the value of G^T A^T mod-4 so we can work out what the diagonal of G^T A G should be
    //so we store it in two binary matrices X and Y such that G^T A^T = 2X + Y

    uint_bitarray_t * X = calloc(state->n, sizeof(uint_bitarray_t));
    uint_bitarray_t * Y = calloc(state->n, sizeof(uint_bitarray_t));

    for(size_t i = 0; i < state->n; i++){
        for(size_t j = 0; j < state->n; j++){
            uint_bitarray_t pc = (popcount(GT[i] & AJ[j]) % 4u);
            X[i] |= ((pc>>1) & ONE) << j;
            Y[i] |= ((pc) & ONE) << j;
        }
    }

    //add the contribution fron G^T D
    for(size_t i = 0; i < state->n; i++){
        X[i] ^= (Y[i] & GT[i] & AJd1); // carry if both bits are 1
        Y[i] ^= (GT[i] & AJd1);
        X[i] ^= (GT[i] & AJd2);
    }


    //now we compute K = G^T (A G) = G^T (G^T A^T)^T
    //we store K as a full symmetric matric of bits
    //we store the even part of the diagonal in bitarray bitKd2;
    //since the diagonal is the only bit we need mod-4
    //in other words K = bitK + 2*diag(bitKd2)
    uint_bitarray_t * bitK = calloc(state->n, sizeof(uint_bitarray_t));
    uint_bitarray_t bitKd2 = 0;


    for(size_t i = 0; i < state->n; i++){
        for(size_t j = 0; j < i; j++){ //symmetric
            uint_bitarray_t pb = parity(GT[i] & Y[j]) & ONE;
            bitK[i] |=  pb << j;
            bitK[j] |=  pb << i;
        }
        //now we need to work out the diagonal
        //slightly more care is needed here as we care about the diagonal mod-4
        uint_bitarray_t pc = popcount(GT[i] & Y[i]);
        bitK[i] |= (pc & ONE) << i;
        bitKd2 |= (((pc>>1) & ONE) ^ (parity(GT[i] & X[i]) & ONE)) << i;
    }
    free(X);
    free(Y);

    unsigned int n = popcount(state->v);

    uint_bitarray_t sK = 0;
    unsigned int sKs = 0;
    for(size_t a = 0; a < state->n; a++){
        unsigned int pc = popcount(state->s & bitK[a]) % 4u;
        sK |= (pc & ONE) << a;
        sKs += pc * ((state->s >> a) & ONE);
    }

    sKs += 2*popcount(bitKd2 & state->s);

    //add 2*diag(s + sK) onto K
    bitKd2 ^= (state->s ^ sK);


    double complex prefactor = pow(0.5, (state->n+n)/2.);
    //printf("c sKs: %d, sKs2: %u\n", popcount(state->s & sK), sKs);
    unsigned int d = (sKs + 2 * popcount(state->s & state->v)) % 4u;
    if(d == 1){
        prefactor *= I;
    }else if(d == 2){
        prefactor *= -1.;
    }else if(d == 3){
        prefactor *= -1.*I;
    }

    uint_bitarray_t k = 0;
    uint_bitarray_t L = 0;

    uint_bitarray_t * M = calloc(n+1, sizeof(uint_bitarray_t));
    int fill_count_a = 0;

    for(int a = 0; (a<state->n); a++){
        if((state->v >> a) & ONE){
            k |= ((bitK[a] >> a) & ONE) << fill_count_a;
            L |= ((bitKd2 >> a) & ONE) << fill_count_a;
            fill_count_a += 1;
        }
    }
    fill_count_a = 0;
    int fill_count_b = 0;
    for(int a = 0; (a<state->n); a++){
        if((state->v >> a) & ONE){
            for(int b = 0; (b<a); b++){
                if((state->v >> b) & ONE){
                    M[fill_count_b] |= (((bitK[b] >> a) & ONE) ^ ((k >> fill_count_a) & (k >> fill_count_b) & ONE)) << fill_count_a;
                    fill_count_b += 1;
                }
            }
            fill_count_a += 1;
            fill_count_b = 0;
        }
    }
    M[n] = k;
    n +=1;

    //at this point we only need M and l
    //so free everything else
    free(bitK);
    free(AJ);
    //free(GT);
    double re=0, im=0;
    int killed = 0;
    int exponent_of_2 = 0;
    bool exponent_of_minus_1 = false;
    bool last_element_asymetric = false;
    bool mu1_consts = false;
    bool mu2_consts = false;

    uint_fast64_t mask = 0;
    for(uint i = 0; i < n; i++){
        mask |= (ONE << i);
    }
    //printf("eb\n");
    while(true){
        uint r=0, c=0;
        bool found = false;
        for(uint i = 0; i < n && !found; i++){
            for(uint j = 0; j < i && !found; j++){
                if(((M[i] >> j) & ONE) != ((M[j] >> i) & ONE)){
                    r=i;
                    c=j;
                    found = true;
                }
            }
        }
        if(!found){
            //this is trivial apparently
            uint_bitarray_t diag = 0;
            for(uint i=0;i<n;i++){
                diag ^= ((M[i] >> i) & ONE) << i;
            }
            if(last_element_asymetric){
                if((diag & mask) == (L&mask)){
                    //printf("c1\n");
                    double signR = exponent_of_minus_1 ? (-1.) : 1.;
                    bool new_exponent_of_minus_1 = (exponent_of_minus_1 ^ mu2_consts);
                    double signI = new_exponent_of_minus_1 ? (-1.) : 1.;
                    re = pow(2., exponent_of_2 + n - killed)*signR;
                    im = pow(2., exponent_of_2 + n - killed)*signI;
                    break;
                }else{
                    re = 0.;
                    im = 0.;
                    break;
                }
            }else{
                if( ((diag & (~(ONE<<(n-1)))) & mask) == ((L & (~(ONE<<(n-1)))) & mask)){
                    if( ((diag & (ONE << (n-1)))&mask) == ((L & (ONE << (n-1)))&mask)){
                        double signR = exponent_of_minus_1 ? (-1.) : 1.;
                        re = signR * pow(2., exponent_of_2+n-killed);
                        im = 0;
                        break;
                    }else{
                        re = 0;
                        double signI = exponent_of_minus_1 ? (-1.) : 1.;
                        im = signI * pow(2., exponent_of_2+n-killed);
                        break;
                    }

                }else{
                    re = 0;
                    im = 0;
                    break;
                }
            }
        }else{
            if(r+1 == n){
                last_element_asymetric = true;
            }

            killed += 2;
            uint_fast64_t m1 = M[r];
            uint_fast64_t m2 = M[c];

            for(uint i=0; i<n;i++){
                m1 ^= (((M[i] >> r) & ONE) << i);
                m2 ^= (((M[i] >> c) & ONE) << i);
            }
            m1 &= (~(ONE << r));
            m1 &= (~(ONE << c));
            m2 &= (~(ONE << r));
            m2 &= (~(ONE << c));

            mu1_consts = ((L>>r) & ONE) ^ ((M[r]>>r) & ONE);
            mu2_consts = ((L>>c) & ONE) ^ ((M[c]>>c) & ONE);

            M[r] = 0;
            M[c] = 0;
            for(uint i=0; i<n;i++){
                M[i] &= ~(ONE << r);
                M[i] &= ~(ONE << c);
            }

            L &= (~(ONE << r));
            L &= (~(ONE << c));
            exponent_of_2 += 1;
            exponent_of_minus_1 ^= (mu1_consts & mu2_consts);

            for(uint i=0;i<n;i++){
                if((m1>>i) & ONE){
                    M[i] ^= m2;
                }
            }
            if(mu1_consts){
                L ^= m2;
            }
            if(mu2_consts){
                L ^= m1;
            }
        }
    }



    free(M);

    //printf("en\n");
    return conj(state->w) * prefactor * (re +im*I)/2;
}


static void partial_equatorial_inner_product(CHForm* state, equatorial_matrix_t equatorial_state, uint_bitarray_t mask){
    int stateI = 0;
    int stateJ = 0;

    for(int i = 0; i < equatorial_state.n; i++){
        while((((mask >> stateI) & ONE) == 0) && (stateI < state->n)){
            stateI += 1;
        }
        //printf("%u, %u\n", (unsigned int)((equatorial_state.d1 >> i) & ONE) , (unsigned int)((equatorial_state.d2 >> i) & ONE));
        if(stateI < state->n){
            for(int k = 0; k < 4-((((equatorial_state.d1 >> i) & ONE) + 2*((equatorial_state.d2 >> i) & ONE))%4); k++){
                SL(state, stateI);
            }

            for(int j=0; j < i; j++){
                while((((mask >> stateJ) & ONE) == 0) && (stateJ < state->n)){
                    stateJ += 1;
                }
                if(stateJ < state->n){
                    if((equatorial_state.mat[i] >> j) & ONE){
                        CZL(state, stateI, stateJ);
                    }
                    stateJ += 1;
                }

            }
            stateI += 1;
        }
        stateJ = 0;
    }

    stateI = 0;

    for(int i = 0; i < equatorial_state.n; i++){
        while((((mask >> stateI) & ONE) == 0) && (stateI < state->n)){
            stateI += 1;
        }
        if(stateI < state->n){
            HL(state, stateI);
            stateI += 1;
        }

    }
    postselect_and_reduce(state, 0u, mask);
}

static PyObject * equatorial_inner_product_wrapper(PyObject* self, PyObject* args){
    PyArrayObject * gates;
    PyArrayObject * targets;
    PyArrayObject * controls;
    PyArrayObject * equatorial_state_matrix;
    int n = 0;

    if (!PyArg_ParseTuple(args, "iO!O!O!O!", &n,
                          &PyArray_Type, &gates,
                          &PyArray_Type, &controls,
                          &PyArray_Type, &targets,
                          &PyArray_Type, &equatorial_state_matrix
            )){
        return NULL;
    }

    CHForm * state =  c_apply_gates_to_basis_state(n, gates, controls, targets);
    equatorial_matrix_t A;
    init_zero_equatorial_matrix(&A, n);
    for(int i = 0; i < n; i++){
        for(int j = i+1; j < n; j++){
            A.mat[i] ^= (equatorial_state_matrix->data[i*equatorial_state_matrix->strides[0] + j*equatorial_state_matrix->strides[1]] & ONE) <<j;
            A.mat[j] ^= ((A.mat[i] >> j) & ONE) << i;
        }
        A.d1 ^= (equatorial_state_matrix->data[i*equatorial_state_matrix->strides[0] + i*equatorial_state_matrix->strides[1]] & ONE) << i;
        A.d2 ^= ((equatorial_state_matrix->data[i*equatorial_state_matrix->strides[0] + i*equatorial_state_matrix->strides[1]] >> 1) & ONE) << i;
    }
    double complex prod = equatorial_inner_product(state, A);
    dealocate_state(state);
    dealocate_equatorial_matrix(&A);
    PyObject * phase = (Py_complex*)PyComplex_FromDoubles(creal(prod), cimag(prod));
    return Py_BuildValue("O", phase);

}

static PyObject * equatorial_inner_product_wrapper2(PyObject* self, PyObject* args){
    PyArrayObject * F, *G, *M, *g, *v,*s;
    PyArrayObject * pyA;
    int n;
    Py_complex w;
    if (!PyArg_ParseTuple(args, "iO!O!O!O!O!O!DO!", &n,
                          &PyArray_Type, &F,
                          &PyArray_Type, &G,
                          &PyArray_Type, &M,
                          &PyArray_Type, &g,
                          &PyArray_Type, &v,
                          &PyArray_Type, &s,
                          &w, &PyArray_Type, &pyA
            )){
        return NULL;
    }


    CHForm * state = compute_CHForm(n, F, G, M, g, v, s, w);

    equatorial_matrix_t A;
    init_zero_equatorial_matrix(&A, n);
    for(int i = 0; i < n; i++){
        for(int j = i+1; j < n; j++){
            A.mat[i] ^= (pyA->data[i*pyA->strides[0] + j*pyA->strides[1]] & ONE) <<j;
            A.mat[j] ^= ((A.mat[i] >> j) & ONE) << i;
        }
        A.d1 ^= (pyA->data[i*pyA->strides[0] + i*pyA->strides[1]] & ONE) << i;
        A.d2 ^= ((pyA->data[i*pyA->strides[0] + i*pyA->strides[1]] >> 1) & ONE) << i;
    }

    double complex prod = equatorial_inner_product(state, A);
    dealocate_state(state);
    dealocate_equatorial_matrix(&A);
    PyObject * phase = (Py_complex*)PyComplex_FromDoubles(creal(prod), cimag(prod));
    return Py_BuildValue("O", phase);

}

static PyObject * measurement_overlap_wrapper(PyObject* self, PyObject* args){
    PyArrayObject * gates;
    PyArrayObject * targets;
    PyArrayObject * controls;
    PyArrayObject * a;
    int n = 0;

    if (!PyArg_ParseTuple(args, "iO!O!O!O!", &n,
                          &PyArray_Type, &gates,
                          &PyArray_Type, &controls,
                          &PyArray_Type, &targets,
                          &PyArray_Type, &a
            )){
        return NULL;
    }

    CHForm * state =  c_apply_gates_to_basis_state(n, gates, controls, targets);

    uint_bitarray_t bitA = 0;
    for(int i = 0; i < n; i++){
        if(a->data[i*a->strides[0]] & ONE){
            bitA |= (ONE << i);
        }
    }

    double complex w = measurement_overlap(state, bitA);
    dealocate_state(state);

    PyObject * phase = (Py_complex*)PyComplex_FromDoubles(creal(w), cimag(w));
    return Py_BuildValue("O", phase);

}

static PyObject * measurement_overlap_wrapper2(PyObject* self, PyObject* args){

    PyArrayObject * F, *G, *M, *g, *v,*s;
    PyArrayObject * a;
    int n;
    Py_complex w;
    if (!PyArg_ParseTuple(args, "iO!O!O!O!O!O!DO!", &n,
                          &PyArray_Type, &F,
                          &PyArray_Type, &G,
                          &PyArray_Type, &M,
                          &PyArray_Type, &g,
                          &PyArray_Type, &v,
                          &PyArray_Type, &s,
                          &w, &PyArray_Type, &a
            )){
        return NULL;
    }


    CHForm * state = compute_CHForm(n, F, G, M, g, v, s, w);

    uint_bitarray_t bitA = 0;
    for(int i = 0; i < n; i++){
        if(a->data[i*a->strides[0]] & ONE){
            bitA |= (ONE << i);
        }
    }

    double complex w1 = measurement_overlap(state, bitA);
    dealocate_state(state);

    PyObject * phase = (Py_complex*)PyComplex_FromDoubles(creal(w1), cimag(w1));
    return Py_BuildValue("O", phase);

}


static PyObject * partial_equatorial_inner_product_wrapper(PyObject* self, PyObject* args){
    PyArrayObject * gates;
    PyArrayObject * targets;
    PyArrayObject * controls;
    PyArrayObject * mask;
    PyArrayObject * equatorial_state_matrix;
    int n = 0;

    if (!PyArg_ParseTuple(args, "iO!O!O!O!O!", &n,
                          &PyArray_Type, &gates,
                          &PyArray_Type, &controls,
                          &PyArray_Type, &targets,
                          &PyArray_Type, &mask,
                          &PyArray_Type, &equatorial_state_matrix
            )){
        return NULL;
    }

    CHForm * state =  c_apply_gates_to_basis_state(n, gates, controls, targets);
    uint_bitarray_t bitMask = 0;
    for(int i = 0; i < n; i++){
        bitMask |= ((mask->data[i*mask->strides[0]] & ONE) << i);
    }

    int equatorial_matrix_size = equatorial_state_matrix->dimensions[0];
    equatorial_matrix_t A;

    init_zero_equatorial_matrix(&A, equatorial_matrix_size);

    for(int i = 0; i < equatorial_matrix_size; i++){
        for(int j = i+1; j < equatorial_matrix_size; j++){
            A.mat[i] ^= (equatorial_state_matrix->data[i*equatorial_state_matrix->strides[0] + j*equatorial_state_matrix->strides[1]] & ONE) <<j;
            A.mat[j] ^= ((A.mat[i] >> j) & ONE) << i;
        }
        A.d1 ^= (equatorial_state_matrix->data[i*equatorial_state_matrix->strides[0] + i*equatorial_state_matrix->strides[1]] & ONE) << i;
        A.d2 ^= ((equatorial_state_matrix->data[i*equatorial_state_matrix->strides[0] + i*equatorial_state_matrix->strides[1]] >> 1) & ONE) << i;
    }

    partial_equatorial_inner_product(state, A, bitMask);


    PyObject * tuple = CHForm_to_python_tuple(state);
    dealocate_state(state);
    dealocate_equatorial_matrix(&A);
    return tuple;
}

static PyObject * magic_sample_1(PyObject* self, PyObject* args){

    PyArrayObject * gates;
    PyArrayObject * controls;
    PyArrayObject * targets;
    PyArrayObject * a; // project |a_i><a_i| onto qubit i iff mask_i == 1
    PyArrayObject * mask;

    int n;
    int magic_samples;
    int equatorial_samples;
    unsigned int seed;
    //printf("1\n");
    if (!PyArg_ParseTuple(args, "iiiiO!O!O!O!O!", &n, &magic_samples, &equatorial_samples, &seed,
                          &PyArray_Type, &gates,
                          &PyArray_Type, &controls,
                          &PyArray_Type, &targets,
                          &PyArray_Type, &a,
                          &PyArray_Type, &mask
            )){
        return NULL;
    }

    srand(seed);

    //gatetize all the t gates
    int t = 0;
    for(int i = 0; i < gates->dimensions[0]; i++){
        if(((char)gates->data[i*gates->strides[0]]) == 't'){
            gates->data[i*gates->strides[0]] = 'X';
            controls->data[i*controls->strides[0]] = (unsigned char)targets->data[i*targets->strides[0]];
            targets->data[i*targets->strides[0]] = (unsigned char)(n + t);
            t += 1;
        }
    }
    //printf("c1 t = %d\n", t);
    //now we do the Clifford evolution "precomputation"

    CHForm * evolved_state = c_apply_gates_to_basis_state(n+t, gates, controls, targets);

    //now we project onto the measurement outcomes for the w qubits
    //and throw away those qubits
    //leaving us with an n+t-w qubit state
    uint_bitarray_t bitA = 0;
    uint_bitarray_t bitMask = 0;

    for(int i = 0; i < n; i++){
        if((char)a->data[i*a->strides[0]]){
            bitA |= (ONE << i);
        }
        if((char)mask->data[i*mask->strides[0]]){
            bitMask |= (ONE << i);
        }
    }

    postselect_and_reduce(evolved_state, bitA, bitMask);
    //printf("c1 after postselection\n");
    //print_CHForm(evolved_state);
    //at this point we want to generate a list of length equatorial_samples
    //containing n - w - qubit equatorial states
    //ie. (n-w) x (n-w) binary symmetric matrices

    equatorial_matrix_t * equatorial_matrices = calloc(equatorial_samples, sizeof(equatorial_matrix_t));
    for(int i = 0; i < equatorial_samples; i++){
        init_random_equatorial_matrix(&(equatorial_matrices[i]), evolved_state->n - t);
    }

    uint_bitarray_t magic_mask = 0u;
    int w = evolved_state->n - t;
    //printf("w = %d\n", w);
    for(int i = evolved_state->n - t; i < evolved_state->n; i++){
        magic_mask |= (ONE<<i);
    }

    uint_bitarray_t * ys = calloc(magic_samples, sizeof(uint_bitarray_t));
    for(int i = 0; i < magic_samples; i++){
        ys[i] = bitarray_rand() & magic_mask;
    }

    double complex * inner_prods = calloc(equatorial_samples, sizeof(double complex));
    double complex alpha = (1. - I*(sqrt(2.) - 1.))/2.;
    double beta = log2(4. - 2.*sqrt(2.));
    double complex alpha_phase = alpha / sqrt(1.-1./sqrt(2.));
    double complex alpha_c_phase = conj(alpha_phase);

    //uint_bitarray_t * A = calloc(evolved_state->n - t, sizeof(uint_bitarray_t));
    //uint_bitarray_t * GT = calloc(evolved_state->n - t, sizeof(uint_bitarray_t));
    for(int i = 0; i < magic_samples; i++){
        //sample a bitstring y of length t
        //printf("y: ");printBits(y, evolved_state->n);

        int hamming_weight = popcount(ys[i]);
        //printf("c1 copy 1\n");
        //print_CHForm(evolved_state);
        CHForm copy = copy_CHForm(evolved_state);

        for(int k = evolved_state->n - t; k < evolved_state->n; k++){
            if((ys[i] >> k ) & ONE){
                SL(&copy, k);
            }
            HL(&copy, k);
        }

        postselect_and_reduce(&copy, (uint_bitarray_t)0, magic_mask);

        /* for(size_t i = 0; i < copy.n; i++){ */
        /*     A[i] = 0u; */
        /*     GT[i] = 0u; */
        /* } */
        /* for(size_t i = 0; i < copy.n; i++){ */
        /*     for(size_t j = 0; j < i; j++){ */
        /*      uint_bitarray_t bit = parity(copy.M[i] & copy.F[j]) & ONE; */
        /*      A[i] |= (bit << j); */
        /*      A[j] |= (bit << i); */
        /*      GT[i] |= ((copy.G[j] >> i) & ONE) << j; */
        /*      GT[j] |= ((copy.G[i] >> j) & ONE) << i; */
        /*     } */
        /* } */



        //now copy is a state of n-w qubits

        double complex prefactor = powl(2., (t+ beta*t)/2.)*cpowl(alpha_c_phase, t-hamming_weight)*cpowl(alpha_phase, hamming_weight);

        for(int j = 0; j < equatorial_samples; j++){
            CHForm copy2 = copy_CHForm(&copy);
            double complex d = conj(equatorial_inner_product(&copy2, equatorial_matrices[j]))/(double)(magic_samples);

            inner_prods[j] += prefactor*d;
            dealocate_state(&copy2);
            //printf("1[%d,%d]: %d, (%lf, %lf), (%lf, %lf)\n",i, j, hamming_weight, creal(prefactor), cimag(prefactor), creal(d), cimag(d));
        }
        dealocate_state(&copy);

    }

    free(ys);
    //free(A);
    //free(GT);
    double acc = 0;
    for(int j = 0; j < equatorial_samples; j++){
        acc += powl(2,w)* creal(inner_prods[j]*conj(inner_prods[j]))/(double)equatorial_samples;
    }
    free(inner_prods);

    for(int j = 0; j < equatorial_samples; j++){
        dealocate_equatorial_matrix(&equatorial_matrices[j]);
    }
    free(equatorial_matrices);
    dealocate_state(evolved_state);
    free(evolved_state);

    return PyComplex_FromDoubles(creal(acc), cimag(acc));
}



static PyObject * magic_sample_2(PyObject* self, PyObject* args){
    PyArrayObject * gates;
    PyArrayObject * controls;
    PyArrayObject * targets;
    PyArrayObject * a; // project |a_i><a_i| onto qubit i iff mask_i == 1
    PyArrayObject * mask;

    int n;
    int magic_samples;
    int equatorial_samples;
    unsigned int seed;
    //printf("1\n");
    if (!PyArg_ParseTuple(args, "iiiiO!O!O!O!O!", &n, &magic_samples, &equatorial_samples, &seed,
                          &PyArray_Type, &gates,
                          &PyArray_Type, &controls,
                          &PyArray_Type, &targets,
                          &PyArray_Type, &a,
                          &PyArray_Type, &mask
            )){
        return NULL;
    }

    srand(seed);

    //gatetize all the t gates
    int t = 0;
    for(int i = 0; i < gates->dimensions[0]; i++){
        if(((char)gates->data[i*gates->strides[0]]) == 't'){
            gates->data[i*gates->strides[0]] = 'X';
            controls->data[i*controls->strides[0]] = targets->data[i*targets->strides[0]];
            targets->data[i*targets->strides[0]] = (unsigned char)(n + t);
            t += 1;
        }
    }
    //printf("c2 t = %d\n", t);
    //now we do the Clifford evolution "precomputation"

    CHForm * evolved_state = c_apply_gates_to_basis_state(n+t, gates, controls, targets);

    //now we project onto the measurement outcomes for the w qubits
    //and throw away those qubits
    //leaving us with an n+t-w qubit state
    uint_bitarray_t bitA = 0;
    uint_bitarray_t bitMask = 0;

    for(int i = 0; i < n; i++){
        if((a->data[i*a->strides[0]]) & ONE){
            bitA |= (ONE << i);
        }
        if((mask->data[i*mask->strides[0]]) & ONE){
            bitMask |= (ONE << i);
        }
    }
    postselect_and_reduce(evolved_state, bitA, bitMask);
    //printf("c2 after postselection\n");
    //print_CHForm(evolved_state);


    //at this point we want to generate a list of length equatorial_samples
    //containing n - w - qubit equatorial states
    //ie. (n-w) x (n-w) binary symmetric matrices

    equatorial_matrix_t * equatorial_matrices = calloc(equatorial_samples, sizeof(equatorial_matrix_t));
    for(int i = 0; i < equatorial_samples; i++){
        init_random_equatorial_matrix(&(equatorial_matrices[i]), evolved_state->n - t);
    }

    uint_bitarray_t magic_mask = 0;
    int w = evolved_state->n - t;

    for(int i = evolved_state->n - t; i < evolved_state->n; i++){
        magic_mask |= (ONE<<i);
    }

    uint_bitarray_t * ys = calloc(magic_samples, sizeof(uint_bitarray_t));
    for(int i = 0; i < magic_samples; i++){
        ys[i] = (bitarray_rand() & magic_mask) >> (evolved_state->n - t);
    }


    uint_bitarray_t equatorial_mask = 0;
    for(int i = 0; i < evolved_state->n - t; i++){
        equatorial_mask |= (ONE << i);
    }


    /* printf("magic 2:\n"); */
    /* printf("equatorial mask\n"); */
    /* printBits(equatorial_mask, evolved_state->n);printf("\n"); */
    /* printf("magic mask\n"); */
    /* printBits(magic_mask, evolved_state->n);printf("\n"); */
    /* printf("ys[0]\n"); */
    /* printBits(ys[0], evolved_state->n);printf("\n"); */
    /* printf("\n"); */
    /* printf("c2 equatorial matrix\n"); */
    /* for(int i = 0; i < equatorial_matrices[0].n; i++){ */
    /*     printBits(equatorial_matrices[0].mat[i], equatorial_matrices[0].n);printf("\n"); */
    /* } */
    /* printBits(equatorial_matrices[0].d1, equatorial_matrices[0].n);printf("\n"); */
    /* printBits(equatorial_matrices[0].d2, equatorial_matrices[0].n);printf("\n"); */


    double complex alpha = (1. - I*(sqrt(2.) - 1.))/2.;
    double beta = log2(4. - 2.*sqrt(2.));
    double complex acc = 0;
    double complex alpha_phase = alpha / sqrt(1.-1./sqrt(2.));
    double complex alpha_c_phase = conj(alpha_phase);

    for(int j = 0; j < equatorial_samples; j++){
        CHForm copy = copy_CHForm(evolved_state);
        partial_equatorial_inner_product(&copy, equatorial_matrices[j], equatorial_mask);

        //printf("c2 copy after equatorial partial product\n");
        //print_CHForm(&copy);
        //at this point we have a t qubit state and we take the inner product with each magic sample
        double complex overlaps = 0;
        for(int i = 0; i < magic_samples; i++){
            CHForm inner_copy = copy_CHForm(&copy);
            //uint_bitarray_t y = bitarray_rand() & magic_mask;
            int hamming_weight = popcount(ys[i]);
            double complex prefactor = powl(2., (t+ beta*t)/2)*cpowl(alpha_c_phase, t-hamming_weight)*cpowl(alpha_phase, hamming_weight);
            //printf("c2 prefactor(%lf, %lf)\n", creal(prefactor), cimag(prefactor));
            for(int k = 0; k < inner_copy.n; k++){
                if((ys[i] >> k) & ONE){
                    SL(&inner_copy, k);
                    //SL(&inner_copy, k);
                    //SL(&inner_copy, k);
                }
                HL(&inner_copy, k);
            }
            double complex d = measurement_overlap(&inner_copy, (uint_bitarray_t)0)/magic_samples;
            //printf("2[%d,%d]: %d, (%lf, %lf), (%lf, %lf)\n",i, j, hamming_weight, creal(prefactor), cimag(prefactor), creal(d), cimag(d));
            overlaps += prefactor*d;
            dealocate_state(&inner_copy);
        }
        acc += powl(2.,w)*creal(overlaps*conj(overlaps))/(double)equatorial_samples;
        dealocate_state(&copy);
    }

    for(int i = 0; i < equatorial_samples; i++){
        dealocate_equatorial_matrix(&equatorial_matrices[i]);
    }

    free(equatorial_matrices);
    dealocate_state(evolved_state);
    //printf("c2(%lf, %lf)\n", creal(acc), cimag(acc));
    return PyComplex_FromDoubles(creal(acc), cimag(acc));
}


static PyObject * main_simulation_algorithm(PyObject* self, PyObject* args){
    PyArrayObject * gates;
    PyArrayObject * controls;
    PyArrayObject * targets;
    PyArrayObject * a; // project |a_i><a_i| on qubit i on the first w qubits (a is length w array)

    int n;
    int measured_qubits;
    int magic_samples;
    int equatorial_samples;

    unsigned int seed;
    //printf("1\n");
    if (!PyArg_ParseTuple(args, "iiiiiO!O!O!O!", &n, &magic_samples, &equatorial_samples, &seed, &measured_qubits,
                          &PyArray_Type, &gates,
                          &PyArray_Type, &controls,
                          &PyArray_Type, &targets,
                          &PyArray_Type, &a
            )){
        return NULL;
    }

    srand(seed);

    //gatetize all the t gates
    int t = 0;
    for(int i = 0; i < gates->dimensions[0]; i++){
        if(((char)gates->data[i*gates->strides[0]]) == T){
            gates->data[i*gates->strides[0]] = CX;
            //controls->data[i*controls->strides[0]] = (unsigned int)targets->data[i*targets->strides[0]];
            unsigned int * ptr = (unsigned int *)PyArray_GETPTR1(controls, i);
            *ptr = *(unsigned int *)PyArray_GETPTR1(targets, i);
            ptr = (unsigned int *)PyArray_GETPTR1(targets, i);
            *ptr = (unsigned int)(n + t);
            t += 1;
        }
    }

    //now we do the stabiliser evolution
    //to compute W

    StabTable * state = StabTable_new(n+t, n+t);
    /* printf("0:\n"); */
    /* StabTable_print(state); */
    /* printf("\n"); */

    for(int i = 0; i < gates->dimensions[0]; i++){
        //printf("%d, %c, %d, %d\n", i, gates->data[i*gates->strides[0]],controls->data[i*controls->strides[0]], targets->data[i*targets->strides[0]]);
        switch((char)gates->data[i*gates->strides[0]]) {
        case CX:
            StabTable_CX(state, (*(unsigned int *)PyArray_GETPTR1(controls,i)), (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case CZ:
            StabTable_CZ(state, (*(unsigned int *)PyArray_GETPTR1(controls,i)), (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case S:
            StabTable_S(state, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case H:
            StabTable_H(state, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        }
    }

    //in the sequel we will assume that the CB measurement outcome we are interested in at the end is |0...0>
    //we "fix" this by applying a bunch of X gates to the measured qubits here
    for(int i = 0; i < measured_qubits; i++){
       if((unsigned char)a->data[i*a->strides[0]]){
           StabTable_X(state, i);
       }
    }


    int log_v = StabTable_apply_constraints(state, measured_qubits, t);
    if(log_v < 0){
        return PyComplex_FromDoubles(0., 0.);
    }
    
    //now delete the first (n) = table->n - t (since table->n = n + t at this point) qubits
    //at this point we should just be left with t qubits
    int q_to_delete = state->n - t;
    int new_size = t;
    for(int s = 0; s < state->k; s++){
        for(int q = q_to_delete; q < state->n; q++){
            state->table[s][q-q_to_delete] = state->table[s][q];
        }
        for(int q = q_to_delete; q < state->n; q++){
            state->table[s][q-2*q_to_delete+state->n] = state->table[s][q+state->n];
        }
        state->table[s] = realloc(state->table[s], sizeof(unsigned char) * 2 * new_size);
    }
    state->n = new_size;


    QCircuit * W = StabTable_simplifying_unitary(state);
    //printf("after computing W\n");

    state->circ = NULL;

    //now we have to work out what we need to do magic sampling with this W circuit
    //we need the CH-form for W|\tilde{0}>
    //and the AG tableau for W|0> (no tilde!)

    CHForm chState;
    init_cb_CHForm(t,&chState);
    //Hadamard everything because we want to know the evolution of |\tilde{0}> = |+>
    for(int i = 0; i < t; i++){
        HL(&chState, i);
    }
    //printf("before computing chState & agState\n");
    StabTable * agState = StabTable_new(t,t);

    for(int i = 0; i < W->length; i++){
        //printf("%d, %c, %d, %d\n", i, W->tape[i].tag, W->tape[i].control, W->tape[i].target);
        switch(W->tape[i].tag) {
        case CX:
            CXL(&chState, W->tape[i].control, W->tape[i].target);
            StabTable_CX(agState, W->tape[i].control, W->tape[i].target);
            //StabTable_CX(stateCopy, W->tape[i].control, W->tape[i].target);
            break;
        case CZ:
            CZL(&chState, W->tape[i].control, W->tape[i].target);
            StabTable_CZ(agState, W->tape[i].control, W->tape[i].target);
            //StabTable_CZ(stateCopy, W->tape[i].control, W->tape[i].target);
            break;
        case S:
            SL(&chState, W->tape[i].target);
            StabTable_S(agState, W->tape[i].target);
            //StabTable_S(stateCopy, W->tape[i].target);
            break;
        case H:
            HL(&chState, W->tape[i].target);
            StabTable_H(agState, W->tape[i].target);
            //StabTable_H(stateCopy, W->tape[i].target);
            break;
        }
    }
    /* printf("Action of W\n"); */
    /* StabTable_print(stateCopy); */
    /* StabTable_free(stateCopy); */
    /* return PyComplex_FromDoubles(0, 0); */
    //printf("after computing chState & agState\n");
    //now S_k^3  = e^{-i\pi/4} (1/sqrt(2)) (I + i Z_k)
    //W S_k^3 W^\dagger  = e^{-i\pi/4} (1/sqrt(2)) (I + i W Z_k W^\dagger)
    //W Z_k W^\dagger is exactly the k^th row of agState
    //Now we have to apply e^{-i\pi/4} (1/sqrt(2)) (I + i W Z_k W^\dagger) to w UC UH |s>
    //e^{-i\pi/4} (1/sqrt(2)) w (I + i W Z_k W^\dagger)UC UH |s> = e^{-i\pi/4} (1/sqrt(2)) w UC (UC^\dagger(I + i W Z_k W^\dagger)UC) UH |s>
    // = e^{-i\pi/4} (1/sqrt(2)) w UC (I + i UC^\dagger W Z_k W^\dagger UC) UH |s>

    //printf("before freeing W\n");
    //QCircuit_free(W);
    //printf("after freeing W\n");

    //at this point we want to generate a list of length equatorial_samples
    //containing t - r - qubit equatorial states
    //ie. (t-r) x (t-r) binary symmetric matrices
    equatorial_matrix_t * equatorial_matrices = calloc(equatorial_samples, sizeof(equatorial_matrix_t));
    for(int i = 0; i < equatorial_samples; i++){
        init_random_equatorial_matrix(&(equatorial_matrices[i]), state->n-state->k);
    }
    //printf("a\n");
    uint_bitarray_t magic_mask = 0;

    for(int i = 0; i < t; i++){
        magic_mask |= (ONE<<i);
    }
    //printf("b\n");
    uint_bitarray_t * ys = calloc(magic_samples, sizeof(uint_bitarray_t));
    for(int i = 0; i < magic_samples; i++){
        ys[i] = bitarray_rand() & magic_mask;
    }
    //printf("c\n");
    uint_bitarray_t equatorial_mask = 0;
    for(int i = 0; i < state->n-state->k; i++){
        equatorial_mask |= (ONE << i);
    }
    //printf("d\n");
    //we project onto 0 and throw away the first t-r qubits
    //leaving us with an r qubit state
    uint_bitarray_t bitA = 0;
    uint_bitarray_t bitMask = 0;
    for(int i = 0; i < state->k; i++){
        bitMask |= (ONE << i);
    }

    //printf("e\n");
    double complex * inner_prods = calloc(equatorial_samples, sizeof(double complex));
    double complex alpha = (1. - I*(sqrt(2.) - 1.))/2.;
    double beta = log2(4. - 2.*sqrt(2.));
    double complex alpha_phase = alpha / cabs(alpha);
    double complex alpha_c_phase = conj(alpha_phase);

    double sampling_time = 0.;
    double norm_est_time = 0.;
    clock_t start;
    clock_t end;


    for(int i = 0; i < magic_samples; i++){
        start = clock();
        //printf("%d\n", i);
        //generate our state
        CHForm copy = copy_CHForm(&chState);

        for(int bit = 0; bit < t; bit++){
            if((ys[i] >> bit) & ONE){
                //apply W S^3_bit W^\dagger to chState
                uint_bitarray_t * z_mat = calloc(t, sizeof(uint_bitarray_t));
                uint_bitarray_t * x_mat = calloc(t, sizeof(uint_bitarray_t));

                uint_bitarray_t mask = (uint_bitarray_t)0;
                uint_bitarray_t t_mask = (uint_bitarray_t)0;

                unsigned int g = 2*agState->phases[bit] ;
                for(int j = 0; j < t; j++){
                    if(agState->table[bit][j]){
                        x_mat[j] = copy.F[j];
                        z_mat[j] = copy.M[j];
                        mask |= (ONE << j);
                    }
                    if(agState->table[bit][j+agState->n]){
                        z_mat[j] ^= copy.G[j];
                    }
                    t_mask |= (ONE << j);
                    //if x and z are both 1 we get a factor of i because iXZ == Y
                    if((agState->table[bit][j] == 1) && (agState->table[bit][j+agState->n] == 1)){
                        g += 1;
                    }

                }
                g += popcount(copy.g1 & mask) + 2*popcount(copy.g2 & mask);
                g += 2*sort_pauli_string(t, x_mat, z_mat, mask);

                uint_bitarray_t u = 0u; // u_m is exponent of X_m
                uint_bitarray_t h = 0u; // h_m is exponent of Z_m

                for(int k = 0; k < t; k++){
                    if(agState->table[bit][k]){
                        u ^= (copy.F[k] & t_mask);
                        h ^= (copy.M[k] & t_mask);
                    }
                    if(agState->table[bit][k+agState->n]){
                        h ^= (copy.G[k] & t_mask);
                    }
                }
                //at this point the state we want is w U_c [I +  i^g \prod_m  Z_m^{h_m}X_m^{u_m}] U_H |s>
                //we need to  commute the product through U_H
                //and then desuperpositionise
                //we pull the same trick again
                // (\prod_m X_m^{u_m}Z_m^{h_m}  ) U_H = U_H (U_H^\dagger (\prod_m Z_m^{h_m}X_m^{u_m}) U_H)
                //n.b. Hadamard is self-adjoint
                //what happens - if V_k is zero then nothing happens
                //if V_k is 1 then we swap X and Z
                //if V_k is 1 and X and Z are both 1 we get a minus sign

                uint_bitarray_t u2 = (u & (~copy.v)) ^ (h & (copy.v));
                uint_bitarray_t h2 = (h & (~copy.v)) ^ (u & (copy.v));
                g += 2*parity(u & h & copy.v & t_mask);


                //at this point the state we want is w U_c [I +  i^g  U_H  \prod_m  Z_m^{h2_m} X_m^{u2_m}] |s>
                //we apply the paulis to |s>
                //every x flips a bit of s
                //every time a z hits a |1> we get a -1
                //xs happen first

                uint_bitarray_t y = (copy.s ^ u2) & t_mask;
                g += 2*parity(y & h2 & t_mask);
                g += 1; //an i appears in the expression for S^3 in terms of Z and I
                g %= 4;

                //at this point the state we want is w e^{-i\pi/4}/sqrt{2} U_c U_h( |s> + i^(g) |y>) //extra factors from the formula for S^3
                if((y & t_mask) == (copy.s & t_mask)){
                    double complex a = 1.;

                    if(g == 0){
                        a += 1.;
                    }
                    if(g == 1){
                        a += (1.)*I;
                    }
                    if(g == 2){
                        a += (-1.);
                    }
                    if(g == 3){
                        a += (-1.*I);
                    }

                    copy.w *= (a*(1 - 1.*I)/2.);
                }else{
                    desupersitionise(&copy, y,  g);
                    copy.w *= (1.-1.*I)/2.;
                }

                free(x_mat);
                free(z_mat);
            }
        }
        end = clock();
        sampling_time += ((double)(end - start)) / (double)CLOCKS_PER_SEC;
        start = clock();
        //at this point copy contains our magic sample
        //we want to project it and do fastnorm estimation
        //printf("before: %d %lu\n", i, ys[i]);
        //print_CHForm(&copy);
        //printf("\n");

        //now we project onto the measurement outcomes for the w qubits
        postselect_and_reduce(&copy, bitA, bitMask);
        //printf("after: %d %lu\n", i, ys[i]);
        //print_CHForm(&copy);
        //printf("\n");
        int hamming_weight = popcount(ys[i]);
        //printf("%d\n", hamming_weight);
        //powl(2., log_v+t+beta*t+(agState->k-w)/2.)*cpowl(alpha, t-hamming_weight)*cpowl(alpha_c, hamming_weight);
        //printf("%lf, %d, %d, %d, %d\n", beta, t,log_v, measured_qubits, hamming_weight);
        //printf("%lf + %lf I\n", creal(alpha_phase), cimag(alpha_phase));
        //printf("%lf + %lf I\n", creal(alpha_c_phase), cimag(alpha_c_phase));
        double complex prefactor = powl(2., ((beta + 1)*t + log_v - measured_qubits)/2.)*cpowl(alpha_phase, t-hamming_weight)*cpowl(alpha_c_phase, hamming_weight);
        //printf("%lf + %lf I\n", creal(prefactor), cimag(prefactor));
        //equatorial_samples = 10;
        for(int j = 0; j < equatorial_samples; j++){
            //CHForm copy2 = copy_CHForm(&copy);
            double complex d = conj(equatorial_inner_product(&copy, equatorial_matrices[j]))/(double)(magic_samples);
            //printf("%lf + %lf*I\n", creal(d), cimag(d));
            inner_prods[j] += prefactor*d;
            //dealocate_state(&copy2);
        }

        end = clock();
        norm_est_time += ((double)(end - start)) / (double)CLOCKS_PER_SEC;
        dealocate_state(&copy);
    }
    free(ys);
    //printf("sampling_time: %lf\nnorm_est_time: %lf\n", sampling_time, norm_est_time);
    double acc = 0;
    for(int j = 0; j < equatorial_samples; j++){
        acc += creal(inner_prods[j]*conj(inner_prods[j]))/(double)equatorial_samples;
    }
    free(inner_prods);

    for(int j = 0; j < equatorial_samples; j++){
        dealocate_equatorial_matrix(&equatorial_matrices[j]);
    }
    free(equatorial_matrices);
    StabTable_free(agState);
    dealocate_state(&chState);
    StabTable_free(state);
    //printf("c1(%lf, %lf)\n", creal(acc), cimag(acc));
    return PyComplex_FromDoubles(creal(acc), cimag(acc));
}

static PyObject * main_simulation_algorithm2(PyObject* self, PyObject* args){
    PyArrayObject * gates;
    PyArrayObject * controls;
    PyArrayObject * targets;
    PyArrayObject * a; // project |a_i><a_i| on qubit i on the first w qubits (a is length w array)

    int n;
    int measured_qubits;
    int magic_samples;
    int equatorial_samples;
    unsigned int seed;
    //printf("1\n");
    if (!PyArg_ParseTuple(args, "iiiiiO!O!O!O!", &n, &magic_samples, &equatorial_samples, &seed, &measured_qubits,
                          &PyArray_Type, &gates,
                          &PyArray_Type, &controls,
                          &PyArray_Type, &targets,
                          &PyArray_Type, &a
            )){
        return NULL;
    }

    //printf("%d, %d, %d\n", n, magic_samples, equatorial_samples);
    srand(seed);

    //gatetize all the t gates
    int t = 0;
    for(int i = 0; i < gates->dimensions[0]; i++){
        if(((char)gates->data[i*gates->strides[0]]) == T){
            gates->data[i*gates->strides[0]] = CX;
            controls->data[i*controls->strides[0]] = targets->data[i*targets->strides[0]];
            targets->data[i*targets->strides[0]] = (unsigned char)(n + t);
            t += 1;
        }
    }

    //now we do the stabiliser evolution
    //to compute W

    StabTable * state = StabTable_new(n+t, n+t);

    for(int i = 0; i < gates->dimensions[0]; i++){
        switch((char)gates->data[i*gates->strides[0]]) {
        case CX:
            StabTable_CX(state, (unsigned int)controls->data[i*controls->strides[0]], (unsigned int)targets->data[i*targets->strides[0]]);
            break;
        case CZ:
            StabTable_CZ(state, (unsigned int)controls->data[i*controls->strides[0]], (unsigned int)targets->data[i*targets->strides[0]]);
            break;
        case S:
            StabTable_S(state, (unsigned int)targets->data[i*targets->strides[0]]);
            break;
        case H:
            StabTable_H(state, (unsigned int)targets->data[i*targets->strides[0]]);
            break;
        }
    }

    //in the sequel we will assume that the CB measurement outcome we are interested in at the end is |0...0>
    //we "fix" this by applying a bunch of X gates to the measured qubits here
    for(int i = 0; i < measured_qubits; i++){
        if((unsigned char)a->data[i*a->strides[0]]){
            StabTable_X(state, i);
        }
    }

    int log_v = StabTable_apply_constraints(state, measured_qubits, t);
    //printf("log_v = %d\n", log_v);
    if(log_v < 0){
        return PyComplex_FromDoubles(0., 0.);
    }
    /* //at this point state is a stab table with fewer stabs than qubits */
    /* //representing a mixed state */
    /* //it is still on n+t qubits */
    /* //but the first n qubits are trivial (the stabs are all the identity there) */
    /* //we can just delete them */
    //now delete the first (n) = table->n - t (since table->n = n + t at this point) qubits
    //at this point we should just be left with t qubits
    int q_to_delete = state->n - t;
    int new_size = t;
    for(int s = 0; s < state->k; s++){
        for(int q = q_to_delete; q < state->n; q++){
            state->table[s][q-q_to_delete] = state->table[s][q];
        }
        for(int q = q_to_delete; q < state->n; q++){
            state->table[s][q-2*q_to_delete+state->n] = state->table[s][q+state->n];
        }
        state->table[s] = realloc(state->table[s], sizeof(unsigned char) * 2 * new_size);
    }
    state->n = new_size;
    //printf("before computing W\n");
    QCircuit * W = StabTable_simplifying_unitary(state);
    //printf("after computing W\n");

    state->circ = NULL;

    //now we have to work out what we need to do magic sampling with this W circuit
    //we need the CH-form for W|\tilde{0}>
    //and the AG tableau for W|0> (no tilde!)

    CHForm chState;
    init_cb_CHForm(t,&chState);
    //Hadamard everything because we want to know the evolution of |\tilde{0}> = |+>
    for(int i = 0; i < t; i++){
        HL(&chState, i);
    }
    //printf("before computing chState & agState\n");
    StabTable * agState = StabTable_new(t,t);

    for(int i = 0; i < W->length; i++){
        switch(W->tape[i].tag) {
        case CX:
            CXL(&chState, W->tape[i].control, W->tape[i].target);
            StabTable_CX(agState, W->tape[i].control, W->tape[i].target);
            break;
        case CZ:
            CZL(&chState, W->tape[i].control, W->tape[i].target);
            StabTable_CZ(agState, W->tape[i].control, W->tape[i].target);
            break;
        case S:
            SL(&chState, W->tape[i].target);
            StabTable_S(agState, W->tape[i].target);
            break;
        case H:
            HL(&chState, W->tape[i].target);
            StabTable_H(agState, W->tape[i].target);
            break;
        }
    }
    //printf("after computing chState & agState\n");
    //now S_k^3  = e^{i\pi/4} (1/sqrt(2)) (I + i Z_k)
    //W S_k^3 W^\dagger  = e^{i\pi/4} (1/sqrt(2)) (I + i W Z_k W^\dagger)
    //W Z_k W^\dagger is exactly the k^th row of agState
    //Now we have to apply e^{i\pi/4} (1/sqrt(2)) (I + i W Z_k W^\dagger) to w UC UH |s>
    //e^{i\pi/4} (1/sqrt(2)) w (I + i W Z_k W^\dagger)UC UH |s> = e^{i\pi/4} (1/sqrt(2)) w UC (UC^\dagger(I + i W Z_k W^\dagger)UC) UH |s>
    // = e^{i\pi/4} (1/sqrt(2)) w UC (I + i UC^\dagger W Z_k W^\dagger UC) UH |s>

    //printf("before freeing W\n");
    //QCircuit_free(W);
    //printf("after freeing W\n");

    //at this point we want to generate a list of length equatorial_samples
    //containing t - r - qubit equatorial states
    //ie. (t-r) x (t-r) binary symmetric matrices
    equatorial_matrix_t * equatorial_matrices = calloc(equatorial_samples, sizeof(equatorial_matrix_t));
    for(int i = 0; i < equatorial_samples; i++){
        init_random_equatorial_matrix(&(equatorial_matrices[i]), state->n-state->k);
    }
    //printf("a\n");
    uint_bitarray_t magic_mask = 0;

    for(int i = 0; i < t; i++){
        magic_mask |= (ONE<<i);
    }
    //printf("b\n");
    uint_bitarray_t * ys = calloc(magic_samples, sizeof(uint_bitarray_t));
    for(int i = 0; i < magic_samples; i++){
        ys[i] = bitarray_rand() & magic_mask;
    }
    //printf("c\n");
    uint_bitarray_t equatorial_mask = 0;
    for(int i = 0; i < state->n-state->k; i++){
        equatorial_mask |= (ONE << i);
    }
    //printf("d\n");
    //we project onto 0 and throw away the first t-r qubits
    //leaving us with an r qubit state
    uint_bitarray_t bitA = 0;
    uint_bitarray_t bitMask = 0;
    for(int i = 0; i < state->k; i++){
        bitMask |= (ONE << i);
    }

    //printf("e\n");
    double complex * inner_prods = calloc(equatorial_samples, sizeof(double complex));
    double complex alpha = (1. - I*(sqrt(2.) - 1.))/2.;
    double beta = log2(4. - 2.*sqrt(2.));
    double complex alpha_phase = alpha / cabs(alpha);
    double complex alpha_c_phase = conj(alpha_phase);

    double sampling_time = 0.;
    double norm_est_time = 0.;
    clock_t start;
    clock_t end;

    for(int i = 0; i < magic_samples; i++){
        start = clock();
        //printf("%d\n", i);
        //generate our state
        //CHForm copy = copy_CHForm(&chState);
        CHForm copy;
        init_cb_CHForm(t, &copy);

        for(int bit = 0; bit < t; bit++){
            HL(&copy, bit);
            if((ys[i] >> bit) & ONE){
                SL(&copy, bit);
                SL(&copy, bit);
                SL(&copy, bit);
            }
        }

        for(int j = 0; j < W->length; j++){
            switch(W->tape[j].tag) {
            case CX:
                CXL(&copy, W->tape[j].control, W->tape[j].target);
                break;
            case CZ:
                CZL(&copy, W->tape[j].control, W->tape[j].target);
                break;
            case S:
                SL(&copy, W->tape[j].target);
                break;
            case H:
                HL(&copy, W->tape[j].target);
                break;
            }
        }

        end = clock();
        sampling_time += ((double)(end - start)) / (double)CLOCKS_PER_SEC;
        start = clock();
        //at this point copy contains our magic sample
        //we want to project it and do fastnorm estimation

        //now we project onto the measurement outcomes for the w qubits
        postselect_and_reduce(&copy, bitA, bitMask);
        int hamming_weight = popcount(ys[i]);
        //powl(2., log_v+t+beta*t+(agState->k-w)/2.)*cpowl(alpha, t-hamming_weight)*cpowl(alpha_c, hamming_weight);
        //printf("%lf, %d, %d, %d, %d\n", beta, t,log_v, measured_qubits, hamming_weight);
        //printf("%lf + %lf I\n", creal(alpha_phase), cimag(alpha_phase));
        //printf("%lf + %lf I\n", creal(alpha_c_phase), cimag(alpha_c_phase));
        double complex prefactor = powl(2., ((beta + 1)*t + log_v - measured_qubits)/2.)*cpowl(alpha_phase, t-hamming_weight)*cpowl(alpha_c_phase, hamming_weight);
        //printf("%lf + %lf I\n", creal(prefactor), cimag(prefactor));
        for(int j = 0; j < equatorial_samples; j++){
            CHForm copy2 = copy_CHForm(&copy);
            double complex d = conj(equatorial_inner_product(&copy2, equatorial_matrices[j]))/(double)(magic_samples);
            inner_prods[j] += prefactor*d;
            dealocate_state(&copy2);
        }

        end = clock();
        norm_est_time += ((double)(end - start)) / (double)CLOCKS_PER_SEC;
        dealocate_state(&copy);
    }
    free(ys);
    //printf("sampling_time2: %lf\nnorm_est_time2: %lf\n", sampling_time, norm_est_time);
    double complex acc = 0;
    for(int j = 0; j < equatorial_samples; j++){
        acc += creal(inner_prods[j]*conj(inner_prods[j]))/(double)equatorial_samples;
    }
    free(inner_prods);

    for(int j = 0; j < equatorial_samples; j++){
        dealocate_equatorial_matrix(&equatorial_matrices[j]);
    }
    free(equatorial_matrices);
    StabTable_free(agState);
    dealocate_state(&chState);
    StabTable_free(state);
    //printf("c1(%lf, %lf)\n", creal(acc), cimag(acc));
    return PyComplex_FromDoubles(creal(acc), cimag(acc));
}



static PyObject * v_r_info(PyObject* self, PyObject* args){
    PyArrayObject * gates;
    PyArrayObject * controls;
    PyArrayObject * targets;
    PyArrayObject * a; // project |a_i><a_i| on qubit i on the first w qubits (a is length w array)

    int n;
    int measured_qubits;
    //printf("1\n");
    if (!PyArg_ParseTuple(args, "iiO!O!O!O!", &n, &measured_qubits,
                          &PyArray_Type, &gates,
                          &PyArray_Type, &controls,
                          &PyArray_Type, &targets,
                          &PyArray_Type, &a
            )){
        return NULL;
    }


    //gatetize all the t gates
    int t = 0;
    for(int i = 0; i < gates->dimensions[0]; i++){
        if(((char)gates->data[i*gates->strides[0]]) == T){
            gates->data[i*gates->strides[0]] = CX;
            //controls->data[i*controls->strides[0]] = (unsigned int)targets->data[i*targets->strides[0]];
            unsigned int * ptr = (unsigned int *)PyArray_GETPTR1(controls, i);
            *ptr = *(unsigned int *)PyArray_GETPTR1(targets, i);
            ptr = (unsigned int *)PyArray_GETPTR1(targets, i);
            *ptr = (unsigned int)(n + t);
            //printf("%d, %d, %u, %u\n", t, n, (*(unsigned int *)PyArray_GETPTR1(controls,i)), (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            t += 1;
        }
    }

    //now we do the stabiliser evolution
    //to compute W

    StabTable * state = StabTable_new(n+t, n+t);
    /* printf("0:\n"); */
    /* StabTable_print(state); */
    /* printf("\n"); */

    for(int i = 0; i < gates->dimensions[0]; i++){
        //printf("%d, %c, %d, %d\n", i, gates->data[i*gates->strides[0]],controls->data[i*controls->strides[0]], targets->data[i*targets->strides[0]]);
        switch((char)gates->data[i*gates->strides[0]]) {
        case CX:
            StabTable_CX(state, (*(unsigned int *)PyArray_GETPTR1(controls,i)), (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case CZ:
            StabTable_CZ(state, (*(unsigned int *)PyArray_GETPTR1(controls,i)), (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case S:
            StabTable_S(state, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case H:
            StabTable_H(state, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        }
    }

    //StabTable_print(state);
    //in the sequel we will assume that the CB measurement outcome we are interested in at the end is |0...0>
    //we "fix" this by applying a bunch of X gates to the measured qubits here
    //for(int i = 0; i < measured_qubits; i++){
    //  if((unsigned char)a->data[i*a->strides[0]]){
    //       StabTable_X(state, i);
    //   }
    //}

    int log_v = StabTable_apply_constraints(state, measured_qubits, t);
    return Py_BuildValue("ii", log_v, t-state->k);

    //StabTable_free(state);

    //Py_RETURN_NONE;
}


static PyObject * lhs_rank_info(PyObject* self, PyObject* args){
    PyArrayObject * gates;
    PyArrayObject * controls;
    PyArrayObject * targets;
    PyArrayObject * a; // project |a_i><a_i| on qubit i on the first w qubits (a is length w array)

    int n;
    int measured_qubits;

    if (!PyArg_ParseTuple(args, "iiO!O!O!O!", &n,  &measured_qubits,
                          &PyArray_Type, &gates,
                          &PyArray_Type, &controls,
                          &PyArray_Type, &targets,
                          &PyArray_Type, &a
            )){
        return NULL;
    }

    //gatetize all the t gates
    int t = 0;
    for(int i = 0; i < gates->dimensions[0]; i++){
        if(((char)gates->data[i*gates->strides[0]]) == T){
            gates->data[i*gates->strides[0]] = CX;
            //controls->data[i*controls->strides[0]] = (unsigned int)targets->data[i*targets->strides[0]];
            unsigned int * ptr = (unsigned int *)PyArray_GETPTR1(controls, i);
            *ptr = *(unsigned int *)PyArray_GETPTR1(targets, i);
            ptr = (unsigned int *)PyArray_GETPTR1(targets, i);
            *ptr = (unsigned int)(n + t);
            t += 1;
        }
    }

    //now we do the stabiliser evolution
    //to compute W

    StabTable * state = StabTable_new(n+t, n+t);
    /* printf("0:\n"); */
    /* StabTable_print(state); */
    /* printf("\n"); */

    for(int i = 0; i < gates->dimensions[0]; i++){
        //printf("%d, %c, %d, %d\n", i, gates->data[i*gates->strides[0]],controls->data[i*controls->strides[0]], targets->data[i*targets->strides[0]]);
        switch((char)gates->data[i*gates->strides[0]]) {
        case CX:
            StabTable_CX(state, (*(unsigned int *)PyArray_GETPTR1(controls,i)), (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case CZ:
            StabTable_CZ(state, (*(unsigned int *)PyArray_GETPTR1(controls,i)), (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case S:
            StabTable_S(state, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case H:
            StabTable_H(state, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        }
    }

    //in the sequel we will assume that the CB measurement outcome we are interested in at the end is |0...0>
    //we "fix" this by applying a bunch of X gates to the measured qubits here
    //for(int i = 0; i < measured_qubits; i++){
    //   if((unsigned char)a->data[i*a->strides[0]]){
    //       StabTable_X(state, i);
    //   }
    //}

    //printf("\n");StabTable_print(state);printf("\n");

    int log_v = StabTable_apply_constraints(state, measured_qubits, t);
    //printf("\n");StabTable_print(state);printf("\n");
    int first_k = state->k;


    if(log_v < 0){
        return PyComplex_FromDoubles(0., 0.);
    }

    //StabTable * before_T_copy = StabTable_copy(state);
    //printf("state->k = %d\n", state->k);
    int identity_qubits = StabTable_apply_T_constraints(state, t);
    //printf("state->k = %d\n", state->k);
    //at this point the variable state contains the long G guys

    StabTable * longG = StabTable_copy(state);
    //now we evolve these back in time with V

    for(int i = gates->dimensions[0]-1; i >= 0 ; i--){
        //printf("%d, %c, %d, %d\n", i, gates->data[i*gates->strides[0]],controls->data[i*controls->strides[0]], targets->data[i*targets->strides[0]]);
        switch((char)gates->data[i*gates->strides[0]]) {
        case CX:
            StabTable_CX(longG, (*(unsigned int *)PyArray_GETPTR1(controls,i)), (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case CZ:
            StabTable_CZ(longG, (*(unsigned int *)PyArray_GETPTR1(controls,i)), (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case S:
            StabTable_S(longG, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            StabTable_S(longG, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            StabTable_S(longG, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case H:
            StabTable_H(longG, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        }
    }

    //StabTable_print(longG);
    longG->n = longG->n - t;
    for(int i = 0; i < longG->k; i++){
        for(int j = 0; j < longG->n; j++){
            longG->table[i][j+longG->n] = longG->table[i][j+longG->n + t];
        }
        longG->table[i] = realloc(longG->table[i], 2*(longG->n)*sizeof(unsigned char));
    }
    //StabTable_print(longG);
    //now we have to do row echelon form on longG to work out the rank
    int h = 0;
    int k = longG->n;
    int rank = 0;
    while(h < longG->k && k < 2*longG->n){
        int poss_pivot = StabTable_first_non_zero_in_col(longG, k, h);
        if(poss_pivot < 0){
            k += 1;
        }else{
            int pivot = poss_pivot; //now known to be non-negative
            if(pivot != h){
                //swap rows h and pivot of the table
                StabTable_swap_rows(longG,h,pivot);
            }
            for(int j = 0; j < longG->k; j++){
                if((j != h) && (longG->table[j][k] != 0)){
                    StabTable_rowsum(longG,j,h);
                }
            }
            h += 1;
            k += 1;
            rank += 1;

        }
    }

    //printf("idenits = %d\n", identity_qubits);
    //printf("Rank = %d\n", rank);
    //printf("lonG->k = %d\n", longG->k);    
    //StabTable_print(longG);printf("\n");

    //StabTable_free(longG);
    return Py_BuildValue("iiii", first_k, longG->n, longG->k, rank);

}

static PyObject * calculate_algorithm(PyObject* self, PyObject* args){
    PyArrayObject * gates;
    PyArrayObject * controls;
    PyArrayObject * targets;
    PyArrayObject * a; // project |a_i><a_i| on qubit i on the first w qubits (a is length w array)

    int n;
    int measured_qubits;

    if (!PyArg_ParseTuple(args, "iiO!O!O!O!", &n, &measured_qubits,
                          &PyArray_Type, &gates,
                          &PyArray_Type, &controls,
                          &PyArray_Type, &targets,
                          &PyArray_Type, &a
            )){
        return NULL;
    }
    //clock_t start = clock();
    
    //gatetize all the t gates
    int t = 0;
    for(int i = 0; i < gates->dimensions[0]; i++){
	/* switch((*(unsigned char *)PyArray_GETPTR1(gates,i))) { */
        /* case CX: */
        /*     printf("CX(%d, %d), ", (*(unsigned int *)PyArray_GETPTR1(targets,i)),(*(unsigned int *)PyArray_GETPTR1(controls,i))); */
        /*     break; */
        /* case CZ: */
	/*     printf("CZ(%d, %d), ", (*(unsigned int *)PyArray_GETPTR1(targets,i)),(*(unsigned int *)PyArray_GETPTR1(controls,i))); */
        /*     break; */
        /* case S: */
	/*     printf("S(%d), ", (*(unsigned int *)PyArray_GETPTR1(targets,i))); */
        /*     break; */
        /* case H: */
	/*     printf("H(%d), ", (*(unsigned int *)PyArray_GETPTR1(targets,i))); */
        /*     break; */
	/* case T: */
	/*     printf("T(%d), ", (*(unsigned int *)PyArray_GETPTR1(targets,i))); */
        /*     break; */
        /* } */

        if(((char)gates->data[i*gates->strides[0]]) == T){
            gates->data[i*gates->strides[0]] = CX;
            unsigned int * ptr = (unsigned int *)PyArray_GETPTR1(controls, i);
            *ptr = *(unsigned int *)PyArray_GETPTR1(targets, i);
            ptr = (unsigned int *)PyArray_GETPTR1(targets, i);
            *ptr = (unsigned int)(n + t);
            t += 1;
        }
    }
    //printf("\n");
    //now we do the stabiliser evolution
    //to compute W
    //printf("t = %d\n", t);
    //printf("w = %d\n", measured_qubits);
    StabTable * state = StabTable_new(n+t, n+t);
    /* printf("0:\n"); */
    /* StabTable_print(state); */
    /* printf("\n"); */

    for(int i = 0; i < gates->dimensions[0]; i++){
        //printf("%d, %c, %d, %d\n", i, gates->data[i*gates->strides[0]],controls->data[i*controls->strides[0]], targets->data[i*targets->strides[0]]);
        switch((*(unsigned char *)PyArray_GETPTR1(gates,i))) {
        case CX:
            StabTable_CX(state, (*(unsigned int *)PyArray_GETPTR1(controls,i)), (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case CZ:
            StabTable_CZ(state, (*(unsigned int *)PyArray_GETPTR1(controls,i)), (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case S:
            StabTable_S(state, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case H:
            StabTable_H(state, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        }
    }

    //in the sequel we will assume that the CB measurement outcome we are interested in at the end is |0...0>
    //we "fix" this by applying a bunch of X gates to the measured qubits here
    for(int i = 0; i < measured_qubits; i++){
       if((*(unsigned char *)PyArray_GETPTR1(a,i)) == 1){
	   //printf("flipping %d\n",i);
           StabTable_X(state, i);
       }
    }

    //printf("Entering constraints code\n");
    //StabTable_print(state);printf("\n");
    int log_v = StabTable_apply_constraints(state, measured_qubits, t);
    //printf("constraints code returned log_v = %d\n",log_v);
    //StabTable_print(state);printf("\n");
    
    if(log_v < 0){
	return Py_BuildValue("d", 0); //PyComplex_FromDoubles(0., 0.);
    }
    

    //now delete the first (n) = table->n - t (since table->n = n + t at this point) qubits
    // at this point we should just be left with t qubits
    int q_to_delete = state->n - t;
    int new_size = t;
    for(int s = 0; s < state->k; s++){
        for(int q = q_to_delete; q < state->n; q++){
            state->table[s][q-q_to_delete] = state->table[s][q];
        }
        for(int q = q_to_delete; q < state->n; q++){
            state->table[s][q-2*q_to_delete+state->n] = state->table[s][q+state->n];
        }
        state->table[s] = realloc(state->table[s], sizeof(unsigned char) * 2 * new_size);
    }
    state->n = new_size;
    int idents = StabTable_apply_T_constraints(state,t);
    int qubits_deleted = 0;
    for(int q = 0; q < state->n; q++){
	int non_identity_paulis = 0;
	for(int s = 0; (s < state->k) && (non_identity_paulis == 0); s++){
	    if((state->table[s][q] == 1) || (state->table[s][q+state->n] == 1)){
		non_identity_paulis += 1;
	    }
	}
	if(non_identity_paulis == 0){
	    //every stabiliser is identity on this qubit
	    //so we can just delete this qubit
	    qubits_deleted += 1;
	}else{
	    if(qubits_deleted > 0){
		for(int s = 0; s < state->k; s++){
		    state->table[s][q-qubits_deleted] = state->table[s][q];
		    state->table[s][q+state->n-qubits_deleted] = state->table[s][q+state->n];
		}
	    }
	}
    }
    //now move all the Z guys left to fill the gap we just made
    if(qubits_deleted > 0){
	for(int s = 0; s < state->k; s++){
	    for(int q = 0; q < state->n; q++){
		state->table[s][q+state->n-qubits_deleted] = state->table[s][q+state->n];
	    }
	    state->table[s] = (unsigned char *)realloc(state->table[s], 2*(state->n - qubits_deleted)*sizeof(unsigned char));
	}
	state->n = state->n - qubits_deleted;
    }

    //printf("final state:\n");
    //StabTable_print(state);
    //we explicitly compute the sum appearing in 10
    uint_bitarray_t full_mask = 0u;
    for(int i = 0; i < state->k; i++){
        full_mask |= (ONE << i);
    }
    double complex acc = 0.;
    //printf("final k = %d\n", state->k);
    //printBits(full_mask,5);printf("\n");
    for(uint_bitarray_t mask = 0u; mask <= full_mask; mask++){
	//printf("mask = ");printBits(mask,5);printf("\n");
        unsigned char * row = calloc(2*state->n, sizeof(unsigned char));
        unsigned char phase = 0;
        //we do the element of the sum corresponding to the bits of mask being 1
        for(int j = 0; j < state->k; j++){
            if((mask >> j) & ONE){
		//printf("%d, ", j);
                phase = StabTable_rowsum2(state, row, phase, j);
            }
        }
	//printf("\n");
        //so now (row, phase) indicate a particular length t string of pauli matrices
        //and we want to compute <T|^t (row,phase |T>^t
        int ICount = 0;
        int XCount = 0;
        int YCount = 0;
        int ZCount = 0;

        for(int j = 0; j < state->n; j++){
            if((row[j] == 0) && (row[j+state->n] == 0)){
                ICount += 1;
            }
	    if((row[j] == 1) && (row[j+state->n] == 0)){
                XCount += 1;
            }
	    if((row[j] == 0) && (row[j+state->n] == 1)){
                ZCount += 1;
            }
	    if((row[j] == 1) && (row[j+state->n] == 1)){
                YCount += 1;
            }
        }
	
        double val = powl(1./2., (XCount + YCount)/2.);
	//printf("val=%lf\n", creal(val));
	//printf("I = %d, X = %d, Y = %d, Z = %d, total = %d\n", ICount, XCount, YCount, ZCount, ICount+ XCount+ YCount+ ZCount);
        if(ZCount == 0){	    
            if(((phase + YCount) % 2) == 0){
                acc += val;
            }else{
                acc -= val;
            }
        }
        free(row);
    }
    if(full_mask == 0u){
	acc = 1;
    }
    //printf("%lf\n",creal(acc));
    acc *= powl(2., log_v - measured_qubits);
    //printf("%lf\n",creal(acc));
    //end = clock();
    //double calc_time = ((double)(end - start)) / (double)CLOCKS_PER_SEC;
    StabTable_free(state);
    
    
    return Py_BuildValue("d", acc);
}

static PyObject * StabTable_to_python_tuple(StabTable * table){

    const long int dimensions1[1] = {table->k};
    const long int dimensions2[2] = {table->k, 2*table->n};

    PyArrayObject * py_table = (PyArrayObject*)PyArray_SimpleNew(2, dimensions2,  PyArray_UBYTE);
    PyArrayObject * py_phases = (PyArrayObject*)PyArray_SimpleNew(1, dimensions1,  PyArray_UBYTE);
    for(int s = 0; s < table->k; s++){
        for(int q = 0; q < 2*table->n; q++){
	    py_table->data[s*py_table->strides[0] + q*py_table->strides[1]] = (unsigned char)table->table[s][q];
        }
        py_phases->data[s*py_phases->strides[0]] = table->phases[s];
    }
    return Py_BuildValue("iiOO", table->n, table->k, (PyObject*)py_table, (PyObject*)py_phases);
}

static StabTable * python_tuple_to_StabTable(PyObject * tuple){
    PyObject * py_n = PyTuple_GetItem(tuple, 0);
    PyObject * py_k = PyTuple_GetItem(tuple, 1);
    PyObject * py_table = PyTuple_GetItem(tuple, 2);
    PyObject * py_phases = PyTuple_GetItem(tuple, 3);

    int n = PyLong_AsLong(py_n);
    int k = PyLong_AsLong(py_k);
    
    StabTable * table = StabTable_new(n,  k);
    for(int s = 0; s < table->k; s++){
	for(int q = 0; q < 2*table->n; q++){
	    table->table[s][q] = *((unsigned char*)PyArray_GETPTR2(py_table, s, q));
	}
	table->phases[s]= *((unsigned char*)PyArray_GETPTR1(py_phases, s));
    }

    return table;
}


static PyObject * compress_algorithm(PyObject* self, PyObject* args){
    PyArrayObject * gates;
    PyArrayObject * controls;
    PyArrayObject * targets;
    PyArrayObject * a; // project |a_i><a_i| on qubit i on the first w qubits (a is length w array)

    int n;
    int measured_qubits;

    if (!PyArg_ParseTuple(args, "iiO!O!O!O!", &n, &measured_qubits,
                          &PyArray_Type, &gates,
                          &PyArray_Type, &controls,
                          &PyArray_Type, &targets,
                          &PyArray_Type, &a
            )){
        return NULL;
    }

    //gatetize all the t gates
    int t = 0;
    for(int i = 0; i < gates->dimensions[0]; i++){
        if(((char)gates->data[i*gates->strides[0]]) == T){
            gates->data[i*gates->strides[0]] = CX;
            //controls->data[i*controls->strides[0]] = (unsigned int)targets->data[i*targets->strides[0]];
            unsigned int * ptr = (unsigned int *)PyArray_GETPTR1(controls, i);
            *ptr = *(unsigned int *)PyArray_GETPTR1(targets, i);
            ptr = (unsigned int *)PyArray_GETPTR1(targets, i);
            *ptr = (unsigned int)(n + t);
            t += 1;
        }
    }

    //now we do the stabiliser evolution
    //to compute W

    StabTable * state = StabTable_new(n+t, n+t);

    for(int i = 0; i < gates->dimensions[0]; i++){
        //printf("%d, %c, %d, %d\n", i, gates->data[i*gates->strides[0]],controls->data[i*controls->strides[0]], targets->data[i*targets->strides[0]]);
        switch((char)gates->data[i*gates->strides[0]]) {
        case CX:
            StabTable_CX(state, (*(unsigned int *)PyArray_GETPTR1(controls,i)), (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case CZ:
            StabTable_CZ(state, (*(unsigned int *)PyArray_GETPTR1(controls,i)), (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case S:
            StabTable_S(state, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        case H:
            StabTable_H(state, (*(unsigned int *)PyArray_GETPTR1(targets,i)));
            break;
        }
    }

    //in the sequel we will assume that the CB measurement outcome we are interested in at the end is |0...0>
    //we "fix" this by applying a bunch of X gates to the measured qubits here
    for(int i = 0; i < measured_qubits; i++){
       if((unsigned char)a->data[i*a->strides[0]]){
           StabTable_X(state, i);
       }
    }

    int log_v = StabTable_apply_constraints(state, measured_qubits, t);
    if(log_v < 0){
	StabTable_free(state);
        return PyComplex_FromDoubles(0., 0.);
    }
    //int idents = StabTable_apply_T_constraints(state,t);

    //now delete the first (n) = table->n - t (since table->n = n + t at this point) qubits
    //at this point we should just be left with t qubits
    int q_to_delete = state->n - t;
    int new_size = t;
    for(int s = 0; s < state->k; s++){
        for(int q = q_to_delete; q < state->n; q++){
            state->table[s][q-q_to_delete] = state->table[s][q];
        }
        for(int q = q_to_delete; q < state->n; q++){
            state->table[s][q-2*q_to_delete+state->n] = state->table[s][q+state->n];
        }
        state->table[s] = realloc(state->table[s], sizeof(unsigned char) * 2 * new_size);
    }
    state->n = new_size;

    //now some of the stabilisers are all identity on some T qubits due to the imposition of the T constrainsts
    int qubits_deleted = 0;
    for(int q = 0; q < state->n; q++){
	int non_identity_paulis = 0;
	for(int s = 0; (s < state->k) && (non_identity_paulis == 0); s++){
	    if((state->table[s][q] == 1) || (state->table[s][q+state->n] == 1)){
		non_identity_paulis += 1;
	    }
	}
	if(non_identity_paulis == 0){
	    //every stabiliser is identity on this qubit
	    //so we can just delete this qubit
	    qubits_deleted += 1;
	}else{
	    if(qubits_deleted > 0){
		for(int s = 0; s < state->k; s++){
		    state->table[s][q-qubits_deleted] = state->table[s][q];
		    state->table[s][q+state->n-qubits_deleted] = state->table[s][q+state->n];
		}
	    }
	}
    }
    //now move all the Z guys left to fill the gap we just made
    if(qubits_deleted > 0){
	for(int s = 0; s < state->k; s++){
	    for(int q = 0; q < state->n; q++){
		state->table[s][q+state->n-qubits_deleted] = state->table[s][q+state->n];
	    }
	    state->table[s] = (unsigned char *)realloc(state->table[s], 2*(state->n - qubits_deleted)*sizeof(unsigned char));
	}
	state->n = state->n - qubits_deleted;
    }

    QCircuit * W = StabTable_simplifying_unitary(state);
    //printf("after computing W\n");

    state->circ = NULL;

    //now we have to work out what we need to do magic sampling with this W circuit
    //we need the CH-form for W|\tilde{0}>
    //and the AG tableau for W|0> (no tilde!)

    CHForm chState;
    init_cb_CHForm(t,&chState);
    //Hadamard everything because we want to know the evolution of |\tilde{0}> = |+>
    for(int i = 0; i < t; i++){
        HL(&chState, i);
    }
    //printf("before computing chState & agState\n");
    StabTable * agState = StabTable_new(t,t);

    for(int i = 0; i < W->length; i++){
        switch(W->tape[i].tag) {
        case CX:
            CXL(&chState, W->tape[i].control, W->tape[i].target);
            StabTable_CX(agState, W->tape[i].control, W->tape[i].target);
            break;
        case CZ:
            CZL(&chState, W->tape[i].control, W->tape[i].target);
            StabTable_CZ(agState, W->tape[i].control, W->tape[i].target);
            break;
        case S:
            SL(&chState, W->tape[i].target);
            StabTable_S(agState, W->tape[i].target);
            break;
        case H:
            HL(&chState, W->tape[i].target);
            StabTable_H(agState, W->tape[i].target);
            break;
        }
    }
    //printf("a\n");
    QCircuit_free(W);
    //printf("b\n");
    t = state->n;
    //printf("c\n");
    int r = t - state->k;
    //printf("d\n");
    StabTable_free(state);
    //printf("e\n");
    //relevant info to return
    //t - qubits_deleted == state->n == chState->n == agState->n
    //r
    //log_v
    //chState and agState    
    //printf("f\n");
    PyObject * pyChState = CHForm_to_python_tuple(&chState);
    //printf("g\n");
    PyObject * pyAGState = StabTable_to_python_tuple(agState);
    //printf("h\n");
    dealocate_state(&chState);
    //printf("i\n");
    StabTable_free(agState);
    //printf("j\n");
    
    return Py_BuildValue("iiiOO", t, r, log_v, pyChState, pyAGState);
}


static PyObject * estimate_algorithm(PyObject* self, PyObject* args){
    int magic_samples, equatorial_samples, r, log_v, measured_qubits, seed;
    PyObject * CHTuple;
    PyObject * AGTuple;
	
    if (!PyArg_ParseTuple(args, "iiiiiiO!O!", &magic_samples, &equatorial_samples, &measured_qubits, &log_v, &r, &seed,
			  &PyTuple_Type, &CHTuple,
			  &PyTuple_Type, &AGTuple
            )){
        return NULL;
    }
    srand(seed);

    CHForm * chState = python_tuple_to_CHForm(CHTuple);
    StabTable * agState = python_tuple_to_StabTable(AGTuple);


    //at this point we want to generate a list of length equatorial_samples
    //containing r - qubit equatorial states
    //ie. r x r binary symmetric matrices
    equatorial_matrix_t * equatorial_matrices = calloc(equatorial_samples, sizeof(equatorial_matrix_t));
    for(int i = 0; i < equatorial_samples; i++){
        init_random_equatorial_matrix(&(equatorial_matrices[i]), r);
    }
    //printf("a\n");
    uint_bitarray_t magic_mask = 0;

    for(int i = 0; i < agState->n; i++){
        magic_mask |= (ONE<<i);
    }
    //printf("b\n");
    uint_bitarray_t * ys = calloc(magic_samples, sizeof(uint_bitarray_t));
    for(int i = 0; i < magic_samples; i++){
        ys[i] = bitarray_rand() & magic_mask;
    }
    //printf("c\n");
    uint_bitarray_t equatorial_mask = 0;
    for(int i = 0; i < r; i++){
        equatorial_mask |= (ONE << i);
    }
    //printf("d\n");
    //we project onto 0 and throw away the first t-r qubits
    //leaving us with an r qubit state
    uint_bitarray_t bitA = 0;
    uint_bitarray_t bitMask = 0;
    for(int i = 0; i < chState->n-r; i++){
        bitMask |= (ONE << i);
    }

    //printf("e\n");
    double complex * inner_prods = calloc(equatorial_samples, sizeof(double complex));
    double complex alpha = (1. - I*(sqrt(2.) - 1.))/2.;
    double beta = log2(4. - 2.*sqrt(2.));
    double complex alpha_phase = alpha / cabs(alpha);
    double complex alpha_c_phase = conj(alpha_phase);


    for(int i = 0; i < magic_samples; i++){
        //printf("%d\n", i);
        //generate our state
        CHForm copy = copy_CHForm(chState);

        for(int bit = 0; bit < chState->n; bit++){
            if((ys[i] >> bit) & ONE){
                //apply W S^3_bit W^\dagger to chState
                uint_bitarray_t * z_mat = calloc(chState->n, sizeof(uint_bitarray_t));
                uint_bitarray_t * x_mat = calloc(chState->n, sizeof(uint_bitarray_t));

                uint_bitarray_t mask = (uint_bitarray_t)0;
                uint_bitarray_t t_mask = (uint_bitarray_t)0;

                unsigned int g = 2*agState->phases[bit] ;
                for(int j = 0; j < chState->n; j++){
                    if(agState->table[bit][j]){
                        x_mat[j] = copy.F[j];
                        z_mat[j] = copy.M[j];
                        mask |= (ONE << j);
                    }
                    if(agState->table[bit][j+agState->n]){
                        z_mat[j] ^= copy.G[j];
                    }
                    t_mask |= (ONE << j);
                    //if x and z are both 1 we get a factor of i because iXZ == Y
                    if((agState->table[bit][j] == 1) && (agState->table[bit][j+agState->n] == 1)){
                        g += 1;
                    }

                }
                g += popcount(copy.g1 & mask) + 2*popcount(copy.g2 & mask);
                g += 2*sort_pauli_string(chState->n, x_mat, z_mat, mask);

                uint_bitarray_t u = 0u; // u_m is exponent of X_m
                uint_bitarray_t h = 0u; // h_m is exponent of Z_m

                for(int k = 0; k < chState->n; k++){
                    if(agState->table[bit][k]){
                        u ^= (copy.F[k] & t_mask);
                        h ^= (copy.M[k] & t_mask);
                    }
                    if(agState->table[bit][k+agState->n]){
                        h ^= (copy.G[k] & t_mask);
                    }
                }
                //at this point the state we want is w U_c [I +  i^g \prod_m  Z_m^{h_m}X_m^{u_m}] U_H |s>
                //we need to  commute the product through U_H
                //and then desuperpositionise
                //we pull the same trick again
                // (\prod_m X_m^{u_m}Z_m^{h_m}  ) U_H = U_H (U_H^\dagger (\prod_m Z_m^{h_m}X_m^{u_m}) U_H)
                //n.b. Hadamard is self-adjoint
                //what happens - if V_k is zero then nothing happens
                //if V_k is 1 then we swap X and Z
                //if V_k is 1 and X and Z are both 1 we get a minus sign

                uint_bitarray_t u2 = (u & (~copy.v)) ^ (h & (copy.v));
                uint_bitarray_t h2 = (h & (~copy.v)) ^ (u & (copy.v));
                g += 2*parity(u & h & copy.v & t_mask);


                //at this point the state we want is w U_c [I +  i^g  U_H  \prod_m  Z_m^{h2_m} X_m^{u2_m}] |s>
                //we apply the paulis to |s>
                //every x flips a bit of s
                //every time a z hits a |1> we get a -1
                //xs happen first

                uint_bitarray_t y = (copy.s ^ u2) & t_mask;
                g += 2*parity(y & h2 & t_mask);
                g += 1; //an i appears in the expression for S^3 in terms of Z and I
                g %= 4;

                //at this point the state we want is w e^{-i\pi/4}/sqrt{2} U_c U_h( |s> + i^(g) |y>) //extra factors from the formula for S^3
                if((y & t_mask) == (copy.s & t_mask)){
                    double complex a = 1.;

                    if(g == 0){
                        a += 1.;
                    }
                    if(g == 1){
                        a += (1.)*I;
                    }
                    if(g == 2){
                        a += (-1.);
                    }
                    if(g == 3){
                        a += (-1.*I);
                    }

                    copy.w *= (a*(1 - 1.*I)/2.);
                }else{
                    desupersitionise(&copy, y,  g);
                    copy.w *= (1.-1.*I)/2.;
                }

                free(x_mat);
                free(z_mat);
            }
        }

        //at this point copy contains our magic sample
        //we want to project it and do fastnorm estimation

        //now we project onto the measurement outcomes for the w qubits
        postselect_and_reduce(&copy, bitA, bitMask);
        int hamming_weight = popcount(ys[i]);
        double complex prefactor = powl(2., ((beta + 1)*chState->n + log_v - measured_qubits)/2.)*cpowl(alpha_phase, chState->n-hamming_weight)*cpowl(alpha_c_phase, hamming_weight);
        for(int j = 0; j < equatorial_samples; j++){
            //CHForm copy2 = copy_CHForm(&copy);
            double complex d = conj(equatorial_inner_product(&copy, equatorial_matrices[j]))/(double)(magic_samples);
            
            inner_prods[j] += prefactor*d;
        }
	dealocate_state(&copy);
    }
    free(ys);

    double acc = 0;
    for(int j = 0; j < equatorial_samples; j++){
        acc += creal(inner_prods[j]*conj(inner_prods[j]))/(double)equatorial_samples;
    }
    free(inner_prods);

    for(int j = 0; j < equatorial_samples; j++){
        dealocate_equatorial_matrix(&equatorial_matrices[j]);
    }
    
    free(equatorial_matrices);
    StabTable_free(agState);
    dealocate_state(chState);
    free(chState);
    return PyComplex_FromDoubles(creal(acc), cimag(acc));       					   
}
static PyMethodDef myMethods[] = {
    { "apply_gates_to_basis_state", apply_gates_to_basis_state, METH_VARARGS, "Applies a bunch of gates to an initial computational-basis state"},
    { "apply_gates_to_basis_state_project_and_reduce", apply_gates_to_basis_state_project_and_reduce, METH_VARARGS, "Applies a bunch of gates to an initial computational-basis state, then applies a bunch of z projectors and removes the (now product state) qubits we projected"},
    { "equatorial_inner_product_wrapper", equatorial_inner_product_wrapper, METH_VARARGS, "perform equatorial inner product"},
    { "equatorial_inner_product_wrapper2", equatorial_inner_product_wrapper2, METH_VARARGS, "perform equatorial inner product"},
    { "partial_equatorial_inner_product_wrapper", partial_equatorial_inner_product_wrapper, METH_VARARGS, "perform partial equatorial inner product"},
    { "magic_sample_1", magic_sample_1, METH_VARARGS, "do the sampling algorithm with magic sampling first"},
    { "magic_sample_2", magic_sample_2, METH_VARARGS, "do the sampling algorithm with fastnorm first"},
    { "measurement_overlap_wrapper", measurement_overlap_wrapper, METH_VARARGS, "compute a computational basis measurement outcome overlap"},
    { "measurement_overlap_wrapper2", measurement_overlap_wrapper2, METH_VARARGS, "compute a computational basis measurement outcome overlap"},
    { "main_simulation_algorithm", main_simulation_algorithm, METH_VARARGS, "compute a computational basis measurement outcome overlap"},
    { "main_simulation_algorithm2", main_simulation_algorithm2, METH_VARARGS, "compute a computational basis measurement outcome overlap dumb sampling method"},
    { "v_r_info", v_r_info, METH_VARARGS, "stuff"},
    { "lhs_rank_info", lhs_rank_info, METH_VARARGS, "stuff"},
    { "calculate_algorithm", calculate_algorithm, METH_VARARGS, "stuff"},
    { "compress_algorithm", compress_algorithm, METH_VARARGS, "Run the compress algorithm precomputation"},
    { "estimate_algorithm", estimate_algorithm, METH_VARARGS, "Run the estimate algorithm"},
    { NULL, NULL, 0, NULL }
};


// Our Module Definition struct
static struct PyModuleDef cPSCS = {
    PyModuleDef_HEAD_INIT,
    "cPSCS",
    "Interface with c PSCS code",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_cPSCS(void)
{
    import_array();
    return PyModule_Create(&cPSCS);
}
