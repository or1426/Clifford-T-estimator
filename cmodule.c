#include <Python.h>
#include <numpy/arrayobject.h>

#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

typedef unsigned __int128 uint_bitarray_t;
//typedef uint_fast64_t uint_bitarray_t;

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



static unsigned int popcount_mod_2(uint_bitarray_t x){
    unsigned int acc = 0;
    for(int i = 0; i < sizeof(x)*8; i++){
        acc ^= (x>>i);
    }
    return acc & ONE;
}

static unsigned int popcount(uint_bitarray_t x){
    unsigned int acc = 0;
    for(int i = 0; i < sizeof(x)*8; i++){
        acc += (x>>i) & ONE;
    }
    return acc;
}


uint_bitarray_t bitarray_rand(){
    uint_bitarray_t val = 0;
    for(int i = 0; i < sizeof(uint_bitarray_t); i++){
	val ^= (((uint_bitarray_t)(rand() % 256)) << i);
    }
    return val;
}

static void printBits(uint_bitarray_t x, int n){
    uint_bitarray_t ONE = 1;
    for(int i = 0; i < n; i++){
        printf("%ld", (x>>i) & ONE );
    }
}

int init_cb_CHForm(unsigned int n, CHForm * state){
    state->n = n;
    state->F = calloc(n, sizeof(uint_bitarray_t));
    state->G = calloc(n, sizeof(uint_bitarray_t));
    state->M = calloc(n, sizeof(uint_bitarray_t));

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
    
    copy.F = calloc(state->n, sizeof(uint_bitarray_t));
    copy.G = calloc(state->n, sizeof(uint_bitarray_t));
    copy.M = calloc(state->n, sizeof(uint_bitarray_t));

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
    state->F = calloc(n, sizeof(uint_bitarray_t));
    state->G = calloc(n, sizeof(uint_bitarray_t));
    state->M = calloc(n, sizeof(uint_bitarray_t));
    state->g1 = 0;
    state->g2 = 0;
    state->s = 0;
    state->v = 0;
    state->w = 0. + 0.*I;

    return 0;
}

int dealocate_state(CHForm * state){
    free(state->F);
    free(state->G);
    free(state->M);
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


int XL(CHForm * state, unsigned int q){
    //note that X is not C-type so we multiply an X gate onto the whole state not just U_C
    //for this reason XR doesn't make any sense
    //so we only have XL
    //If necessary we could think of an X acting from the right on U_C as an X acting from the left on U_H |s>

    unsigned int alpha = popcount_mod_2(state->G[q] & (~state->v) & state->s);
    state->s ^= state->G[q] & state->v;
    if(alpha == 1){
        state->w *= -1;
    }

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


static CHForm * compute_CHForm(PyObject* self, PyObject* args)
{
    PyArrayObject * F, *G, *M, *g, *v,*s;
    int n;
    Py_complex w;
    if (!PyArg_ParseTuple(args, "iO!O!O!O!O!O!D", &n,
                          &PyArray_Type, &F,
                          &PyArray_Type, &G,
                          &PyArray_Type, &M,
                          &PyArray_Type, &g,
                          &PyArray_Type, &v,
                          &PyArray_Type, &s,
                          &w
            )){
        return NULL;
    }


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
            sign ^= popcount_mod_2(t & x[i]);
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
    unsigned int signbit = sort_pauli_string(state->n, state->F, state->M, ~x);
    signbit ^= popcount_mod_2(u & state->s & state->v);

    unsigned int g = 0;
    for(int i = 0; i < state->n; i++){
        if(( x >> i) &ONE ){
            g +=  (state->g1 >> i) + 2*(state->g2 >> i);
        }
    }
    if(signbit & ONE){
        g += 2;
    }

    double complex phase = state->w;

    if(g == 1){
        phase *= I;
    }else if(g == 2){
        phase *= -1;
    }else if(g == 3){
        phase *= -1*I;
    }

    double sqrt2 = sqrt(2.);
    for(int i = 0; i < state->n; i++){
        if(state->v >> i){
            phase /= sqrt2;
        }
    }

    return phase;
}

static void apply_z_projector(CHForm * state, int a, int q){
    //apply the projector |a><a| to the qubit q where a is the least significant bit of a
    unsigned int k = a ^ popcount_mod_2(state->G[q] & (~state->v) & state->s);
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
            unsigned int k = ((a>>i) ^ popcount_mod_2(state->G[i] & (~state->v) & state->s)) & ONE;
            uint_bitarray_t t = (state->G[i] & state->v) ^ state->s;
            //printf("ct: ");printBits(t, 128);printf("\n");
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
    for(unsigned int i = 0; i < state->n; i++){
        if(((a & mask) >> i) & ONE){
            XL(state, i);
        }
    }

    apply_z_projectors(state, 0u, mask);
    //printf("c after z proj, v = ");printBits(state->v,state->n);printf("\n");
    //printf("c after z proj, s = ");printBits(state->s,state->n);printf("\n");
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




    state->n = state->n-shift;
    state->F = realloc(state->F, state->n * sizeof(uint_bitarray_t));
    state->G = realloc(state->G, state->n * sizeof(uint_bitarray_t));
    state->M = realloc(state->M, state->n * sizeof(uint_bitarray_t));

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
    //printBits(bitMask,n);printf("\n");

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
    matrix->d1 = 0;
    matrix->d2 = 0;
}

void init_random_equatorial_matrix(equatorial_matrix_t * matrix, int n){
    matrix->n = n;
    matrix->mat = (uint_bitarray_t*)calloc(n, sizeof(uint_bitarray_t));
    for(int i = 0; i < n; i++){
	matrix->mat[i] = bitarray_rand();
    }
    for(int i = 0; i < n; i++){
	for(int j = 0; j < n; j++){
	    matrix->mat[i] ^= (((matrix->mat[j]>>i) &ONE) <<j);
	}
    }
    
    matrix->d1 = bitarray_rand();
    matrix->d2 = bitarray_rand();
}

void dealocate_equatorial_matrix(equatorial_matrix_t * matrix){
    matrix->n = 0;
    free(matrix->mat);
}

double complex equatorial_inner_product(CHForm* state, equatorial_matrix_t equatorial_state){
    
    //we store A+J in AJ
    unsigned char * AJ = calloc(state->n*state->n, sizeof(unsigned char));
							 
    for(int i = 0; i < state->n; i++){
	for(int j = 0; j < i; j++){
	    AJ[i + state->n*j] = ((popcount_mod_2(state->M[i] & state->F[j])) + ((equatorial_state.mat[i] >>j) & ONE)) ;
	    AJ[j + state->n*i] = AJ[i + state->n*j];
	}
	AJ[i+state->n*i] = (((state->g1>>i)&ONE) +
			    ((equatorial_state.d1>>i)&ONE)
			    + 2*(((state->g2>>i)&ONE) + ((equatorial_state.d2>>i)&ONE)));
    }

    
    unsigned char * K = calloc(state->n*state->n, sizeof(unsigned char));

    //now we want to compute K = G^T AJ G
    for(int i = 0; i < state->n; i++){
	for(int j = 0; j < state->n; j++){
	    for(int a = 0; a < state->n; a++){
		K[i+j*state->n] += AJ[i+a*state->n] * ((state->G[a]>>j)&ONE);
	    }
	}
    }

    for(int i = 0; i < state->n; i++){
	for(int j = 0; j < state->n; j++){
	    AJ[i+j*state->n] = 0; //we don't need A+J anymore so just use it as tempory "scratch space"
	    for(int a = 0; a < state->n; a++){
		AJ[i+j*state->n] += ((state->G[a]>>i)&ONE) * K[a+j*state->n];
	    }
	}
    }

    free(K);
    K = AJ;
    
    int n = popcount(state->v);
    //printBits(state->v, state->n);printf("\n");
    //printf("n = %d\n", n);
    unsigned char * B = calloc(n*n, sizeof(unsigned char));
    int fill_count_a = 0;
    int fill_count_b = 0;

    uint_bitarray_t diag = state->s; // diag = diag(s + sK)
    uint_bitarray_t sK = 0;
    unsigned int sKs = 0;
    for(int a = 0; a < state->n; a++){
	for(int p = 0; p < state->n; p++){
	    sK ^= ((state->s >> p) & K[a+p*state->n] & ONE) << a;
	    sKs += (state->s>>a)*(state->s>>p)*K[a+p*state->n];
	}		
    }
    
    diag ^= sK;

    double complex prefactor = pow(0.5, (state->n+n)/2.);
    //printf("c sKs: %d, sKs2: %u\n", popcount(state->s & sK), sKs);
    unsigned int d = (sKs + 2 * popcount(state->s & state->v)) % 4;
    if(d == 1){
	prefactor *= I;
    }else if(d == 2){
	prefactor *= -1.;
    }else if(d == 3){
	prefactor *= -1.*I;
    }
    for(int a = 0; fill_count_a < n; a++){
	if((state->v >> a) & ONE){
	    B[fill_count_a + n*fill_count_a] = 2*((diag >> a) & ONE);
	    for(int b = 0; fill_count_b < n; b++){
		if((state->v >> b) & ONE){
		    B[fill_count_a + n*fill_count_b] += K[a+state->n*b];
		    fill_count_b += 1;
		}
	    }
	    fill_count_b = 0;
	    fill_count_a += 1;
	}
    }


    //k + 2l = diag(B)
    uint_bitarray_t k = 0;
    uint_bitarray_t L = 0;
    for(int a = 0; a < n; a++){
	k ^= (B[a + n*a] & ONE) << a;
	L ^= ((B[a + n*a] >> 1) & ONE) << a;
    }
    
    uint_bitarray_t * M = calloc(n+1, sizeof(uint_bitarray_t));
    for(int a = 0; a < n; a++){
	for(int b=a+1; b < n; b++){
	    M[a] ^= ((B[a+n*b] ^ ((k>>a)&(k>>b))) & ONE) << b;
	}
    }
    

    

    M[n] = k;
    n +=1;

    //at this point we only need M and l
    //so free everything else
    free(B);

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
	    uint_fast64_t diag = 0;
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

    return conj(state->w) * prefactor * (re +im*I)/2;
}

static PyObject * magic_sample(PyObject* self, PyObject* args){
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
    if (!PyArg_ParseTuple(args, "iiiuO!O!O!O!O!", &n, &magic_samples, &equatorial_samples, &seed,
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

    //now we do the Clifford evolution "recomputation"

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


    //at this point we want to generate a list of length equatorial_samples
    //containing n - w - qubit equatorial states
    //ie. (n-w) x (n-w) binary symmetric matrices

    equatorial_matrix_t * equatorial_matrices = calloc(equatorial_samples, sizeof(equatorial_matrix_t));
    for(int i = 0; i < equatorial_samples; i++){
	init_random_equatorial_matrix(&(equatorial_matrices[i]), evolved_state->n - t);
    }

    uint_bitarray_t magic_mask = 0;
    for(int i = evolved_state->n - t; i < evolved_state->n; i++){
	magic_mask |= (ONE<<i);
    }
    
    //int * weights = calloc(magic_samples, sizeof(int));
    double complex * inner_prods = calloc(equatorial_samples, sizeof(double complex));
    double complex alpha = (1. - I*(sqrt(2.) - 1.))/2.;
    
    
    for(int i = 0; i < magic_samples; i++){
	//sample a bitstring y of length t
	uint_bitarray_t y = bitarray_rand() & magic_mask;
	int hamming_weight = popcount(y);
	CHForm copy = copy_CHForm(evolved_state);
	postselect_and_reduce(&copy, y, magic_mask);
	//now copy is a state of n-w qubits

	double complex prefactor = pow(alpha, t-hamming_weight)*pow(conj(alpha), hamming_weight);
	
	for(int j = 0; j < equatorial_samples; j++){
	    inner_prods[j] += prefactor*equatorial_inner_product(&copy, equatorial_matrices[i]);
	}
	
    }
    free(equatorial_matrices);
    dealocate_state(evolved_state);
    double acc = 0;
    for(int j = 0; j < equatorial_samples; j++){
	acc += creal(inner_prods[j]*conj(inner_prods[j]));
    }
    free(inner_prods);
    return Py_BuildValue("d", acc);
	

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

static PyMethodDef myMethods[] = {
    { "apply_gates_to_basis_state", apply_gates_to_basis_state, METH_VARARGS, "Applies a bunch of gates to an initial computational-basis state"},
    { "apply_gates_to_basis_state_project_and_reduce", apply_gates_to_basis_state_project_and_reduce, METH_VARARGS, "Applies a bunch of gates to an initial computational-basis state, then applies a bunch of z projectors and removes the (now product state) qubits we projected"},
    { "equatorial_inner_product_wrapper", equatorial_inner_product_wrapper, METH_VARARGS, "perform equatorial inner product"},
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
