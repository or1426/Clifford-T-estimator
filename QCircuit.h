#ifndef QCIRCUIT_H
#define QCIRCUIT_H

enum gate_tag{
    CX,
    CZ,
    H,
    S,
    T
};

typedef enum gate_tag gate_tag_t;

struct Gate{
    gate_tag_t tag;
    size_t target;
    size_t control; //ignored if not CX or CZ
};

typedef struct Gate Gate;

struct QCircuit{
    size_t length; // stores the (index of the last char in the tape + 1)
    size_t capacity; //stores the current capacity of the tape
    Gate * tape; //every time we do a CX, CZ, H or S we write it down on the tape
};
typedef struct QCircuit QCircuit;


QCircuit * QCircuit_new();

int QCircuit_free(QCircuit * circ);

/*
 * Appends a char to the end of the tape
 * Doubling the length of the tape if necessary
 */
int QCircuit_append(QCircuit * circ, char c);

/*
 * Return a new QCircuit which inverts circ
 */
QCircuit * QCircuit_daggered(QCircuit * circ);

#endif
