#include "QCircuit.h"

QCircuit* QCircuit_new(){
    QCircuit * circ = malloc(sizeof(QCircuit));
    
    circ->length = 0;
    circ->capacity = 128;
    circ->tape = calloc(circ->capacity, sizeof(Gate));

    return circ;
}

int QCircuit_free(QCircuit * circ){
    free(circ->tape);
    free(circ);
    circ = NULL;
    return 0;
}

/*
 * Appends a char to the end of the tape
 * Doubling the length of the tape if necessary
 */
int QCircuit_append(QCircuit * circ, Gate g){
    if(circ->length == circ->capacity)
    {
	circ->capacity *= 2;
	circ->tape = realloc(circ->tape, circ->capacity*sizeof(Gate));
    }
    
    circ->tape[circ->length] = g;
    circ->length += 1;
    return 0;
}

QCircuit * QCircuit_daggered(QCircuit * circ){
    QCircuit * new = QCircuit_new();
    new->capacity = circ->length;
    new->tape = realloc(new->tape, new->capacity*sizeof(Gate));

    for(int i = 0; i <circ->length; i++){
	if(circ->tape[circ->length-i-1].tag == S){
	    QCircuit_append(new, (Gate){.tag=S, .target=circ->tape[circ->length-i-1].target, .control = 0});
	    QCircuit_append(new, (Gate){.tag=S, .target=circ->tape[circ->length-i-1].target, .control = 0});
	    QCircuit_append(new, (Gate){.tag=S, .target=circ->tape[circ->length-i-1].target, .control = 0});
	}else if(circ->tape[circ->length-i-1].tag == T){
	    QCircuit_append(new, (Gate){.tag=T, .target=circ->tape[circ->length-i-1].target, .control = 0});
	    QCircuit_append(new, (Gate){.tag=T, .target=circ->tape[circ->length-i-1].target, .control = 0});
	    QCircuit_append(new, (Gate){.tag=T, .target=circ->tape[circ->length-i-1].target, .control = 0});
	    QCircuit_append(new, (Gate){.tag=T, .target=circ->tape[circ->length-i-1].target, .control = 0});
	    QCircuit_append(new, (Gate){.tag=T, .target=circ->tape[circ->length-i-1].target, .control = 0});
	    QCircuit_append(new, (Gate){.tag=T, .target=circ->tape[circ->length-i-1].target, .control = 0});
	    QCircuit_append(new, (Gate){.tag=T, .target=circ->tape[circ->length-i-1].target, .control = 0});
	}else{ //other gates are their own inverse
	    QCircuit_append(new, (Gate){.tag=circ->tape[circ->length-i-1].tag, .target=circ->tape[circ->length-i-1].target, .control=circ->tape[circ->length-i-1].control});
	}	
    }
    return new;
}
