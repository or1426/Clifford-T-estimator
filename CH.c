#include "CH.h"
#include "stdio.h"
#include <math.h>
#include <tgmath.h>
#include <stdlib.h>

void print_CHForm(CHForm * state){
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
 * Clifford gates
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

//1
