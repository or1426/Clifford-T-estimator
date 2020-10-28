#include "AG.h"

/*
 * Init a new stabtable in the state |0>^k \otimes I^(n-k) / (2^(n-k))
 * Explicitly we have k stabilisers on n qubits with the jth being a z on the jth qubit
 * Assumptions: 
 * k <= n
 */
StabTable * StabTable_new(size_t n, size_t k){
    StabTable * table = malloc(sizeof(StabTable));
    table->n = n;
    table->k = k;
    table->table = calloc(k, sizeof(unsigned char *));
    for(size_t i = 0; i < table->k; i++)
    {
	table->table[i] = calloc(2*table->n, sizeof(unsigned char));
	table->table[i][n+i] = 1;
    }
    table->phase = calloc(table->n, sizeof(unsigned char));
    table->tape = QCircuit_new();
    
    return table;
}

/*
 * Frees all memory associated with the table
 */
int StabTable_free(StabTable * table){
    for(size_t i = 0; i < table->k; i++)
    {
	free(table->table[i]);
    }
    free(table->table);
    QCircuit_free(table->tape);
    free(table);
    table = NULL;
    return 0;    
}


int StabTable_CX(StabTable * table, size_t a, size_t b){
    QCircuit_append(table->circ, (GATE){.tag=CX, .target=b, .control=a});
    for(size_t i = 0; i < table->k; i++)
    {
	table->r[i] ^= (table->table[i][a] & table->table[i][b+table->n]) ^ (table->table[i][b] ^ table->table[i][a+table->n] ^ 1);
	table->table[i][b] ^= table->table[i][a];
	table->table[i][a+table->n] ^= table->table[i][b+table->n];						
    }
    return 0;
}


int StabTable_H(StabTable * table, size_t a){
    QCircuit_append(table->circ, (GATE){.tag=H, .target=a, .control=0});
    unsigned char scratch = 0;
    for(size_t i = 0; i < table->k; i++)
    {
	table->r[i] ^= (table->table[i][a] & table->table[i][a+table->n]);
	scratch = table->table[i][a];
	table->table[i][a] = table->table[i][a+table->n];
	table->table[i][a+table->n] = scratch;	
    }
    return 0;
}

int StabTable_CZ(StabTable * table, size_t a, size_t b){
    //TODO: Write an actual CZ implementation 
    StabTable_H(table, b);
    StabTable_CX(table, a, b);
    StabTable_H(table, b);
    return 0;
}


int StabTable_S(StabTable * table, size_t a){
    QCircuit_append(table->circ, (GATE){.tag=S, .target=a, .control=0});
    for(size_t i = 0; i < table->k; i++)
    {
	table->r[i] ^= (table->table[i][a] & table->table[i][a+table->n]);
	table->table[i][a+table->n] ^= table->table[i][a];
    }
    return 0;
}

unsigned char g(unsigned char x1, unsigned char z1, unsigned char x2, unsigned char z2){
    
    if(x1 == 0 and z1 == 0){
	return 0u;
    }
    if(x1 == 1 and z1 == 0){
	return z2*(2*x2-1) % 4u;
    }
    if(x1 == 0 and z1 == 1){
	return z2*(2*x2-1) % 4u;
    }
    if(x1 == 1 and z1 == 1){
	return (z2 - x2) % 4u;
    }
    
    //we should never reach here - maybe printing a warning would be sensible
    return 0;
}

int StabTable_rowsum(StabTable * table, size_t h, size_t i){
    unsigned char sum = 2*(table->phase[h] + table->phase[i]);
    for(size_t j = 0; j < table->k; j++){
	sum += g(table->table[i][j], table->table[i][j+table->n], table->table[h][j], table->table[h][j+table->n]);
    }
    sum %= 4;
    if(sum == 0){
	table->phase[h] = 1;
    }
    if(sum == 2){
	table->phase[h] = 0;
    }
    for(size_t j = 0; j < 2*state->n; j++){
	table->table[h][j] ^= table->table[i][j];
    }
    return 0;
}

int StabTable_first_non_zero_in_col(StabTable * table, size_t col, size_t startpoint){
    for(int i = startpoint; i < table->k; i++){
	if(table->table[col][i] != 0){
	    return i;
	}
    }
    return -1;
}

int StabTable_first_non_zero_in_row(StabTable * table, size_t row, size_t startpoint){
    for(int i = startpoint; i < 2*table->n; i++){
	if(table->table[i][row] != 0){
	    return i;
	}
    }
    return -1;
}

int StabTable_swap_rows(StabTable * table, size_t i, size_t j){
    unsigned char * row_scratch;
    row_scratch = table->table[i];
    table->table[i] = table->table[j];
    table->table[j] = row_scratch;
    unsigned char phase_scratch;
    phase_scratch = table->phase[i];
    table->table[i] = table->table[j];
    table->table[j] = phase_scratch;
    return 0;		
}

//create a QCircuit that creates this table from the initial tableau representing  the state|0>^k \otimes I^(n-k) / (2^(n-k))                 
//Explicitly k stabilisers on n qubits with the jth being a z on the jth qubit
//this procedure overwrites the information stored in the QCircuit circ of the table
QCircuit * generating_unitary(StabTable * table){
    //clear the circuit
    QCircuit_free(table->circ);
    table->circ = QCircuit_new();

    //first we do "augmented gaussian elimination" on the circuit
    //augmented here means that if we don't find a pivot in our column
    //we will hadamard that qubit and try again
    //in other words bring in a pivot from the z part if necessary

    int h = 0;
    int k = 0;
    while(h < table->k && k < table->n){
	int poss_pivot = StabTable_first_non_zero_in_col(table, k, h);
	if(poss_pivot < 0){
	    StabTable_H(table, k);
	    poss_pivot = StabTable_first_non_zero_in_col(table, k, h);
	}
	if(poss_pivot < 0){
	    k += 1;
	}else{
	    size_t pivot = (size_t)poss_pivot; //now known to be non-negative
	    if(pivot != h){
		//swap rows h and pivot of the table
		StabTable_swap_rows(h,pivot);				   
	    }
	    for(size_t j = 0; j < table->k; j++){
		if((j != h) && (table->table[j][k] != 0)){
		    StabTable_rowsum(j,h);
		}
	    }
	    h += 1;
	    k += 1;
	}
    }

    //so now we have a reduced row echelon form with the X part of the table having full rank

    //we swap columns (using CX) to make the X part into a kxk identity followed by a "junk" block

    for(size_t r = 0; r < table->k; r++){
	if(table->table[r][r] == 0){
	    int col = StabTable_first_non_zero_in_row(StabTable, r, 0);

	    StabTable_CX(table, r, col);
	    StabTable_CX(table, col, r);
	    StabTable_CX(table, r, col);
	}
    }

    //now we use CX to clear out the "junk" block at the end of the X table
    for(size_t r = 0; r < table->k; r++){
	for(size_t col = table->k; col < table->n; col++){
	    if(table->table[r][col] != 0){
		StabTable_CX(table, r, col);
	    }
	}
    }

    //now we clear the leading diagonal of the z block
    for(size_t r = 0; r < table->k; r++){
	if(table->table[r][r+table->n] != 0){
	    StabTable_S(table, r);
	}
    }

    //clear out the last k x (n-k) block of the z matrix
    for(size_t col = table->k; col < table->n; col++){
	for(size_t r = 0; r < table->k; r++){
	    if(table->table[r][col+table->n] != 0){
		StabTable_CZ(r, col);
	    }
	}
    }

    //clear out the first k x k block of the Z matrix, using that it is symmetric
    for(size_t col = 0; col < table->k; col++){
	for(size_t r = 0; r < col; r++){
	    if(table->table[r][col+table->n] != 0){
		StabTable_CZ(table, r, col);
	    }	    
	}
    }
    //fix the phases
    for(size_t r = 0; r < table->k; r++){
	if(table->phase[r] != 0){
	    t.S(r);
	    t.S(r);
	}
    }
    
    //swap the identity matrix to the z part
    for(size_t r = 0; r < table->k; r++){
	StabTable_H(table, r);
    }

    return QCircuit_daggered(table->circ);
}


/*
 * Apply constraints arising from the fact that we measure the first w qubits and project the last t onto T gates
 * In particular we kill stabilisers if qubits [0, w) get killed by taking the expectation value <0| P |0>
 * and we kill stabilisers if qubits in [0, n-t) aren't the identity
 */
int apply_constraints(StabTable * table, size_t w, size_t t){

    size_t k = 0;
    int log_v = 0; //this keeps track of duplicated sums this increases by 1 if a measurement was non-constraining
    unsigned char * deleted_stabs = calloc(table->k, sizeof(unsigned char)); //put a 1 here for qubits we will delete
    
    for(size_t q=0; q < w; q++){ //iterate over all the measured qubits
	int x_and_z_stab= -1; //store the index of the first stab we come to with both x and z = 1 on this qubit 
	int x_and_not_z_stab = -1; //store the index of the first stab we come to with x=1, z=0
	int z_and_not_x_stab = -1; //store the index of the first stab we come to with z=1, x=0
	
	for(int s=0; s < table->k && (((x_and_z_stab < 0) + (x_and_not_z_stab < 0) + (z_and_not_x_stab < 0)) < 2); s++){//iterate over all stabilisers and find interesting stabilisers
	    if(table->table[s][q] == 1 && table->table[s][q+table->n] == 1){
		x_and_z_stab = s;
	    }
	    if(table->table[s][q] == 1 && table->table[s][q+table->n] == 0){
		x_and_not_z_stab = s;
	    }
	    if(table->table[s][q] == 0 && table->table[s][q+table->n] == 1){
		z_and_not_x_stab = s;
	    }	    
	}

	//there are several cases here
	//either a single z, a single x, a single y or we can generate the whole Pauli group on this qubit

	//case 1) we generate the whole group
	//put things in standard form (first stab is x then z)
	
	if(((x_and_z_stab >= 0) + (x_and_not_z_stab >= 0) + (z_and_not_x_stab >= 0)) >= 2){ //we have at least two of the set
	    if(x_and_not_z_stab < 0){//we don't have a generator for x alone, but we can make one
		StabTable_rowsum(table, x_and_z_stab, z_and_not_x_stab);
		//now we have a z and an x but not both
		x_and_not_z_stab = x_and_z_stab;
		x_and_z_stab = -1;
	    }else if(z_and_not_x_stab < 0){//we odn't have a generator for z alone, but we can make one
		StabTable_rowsum(table, x_and_z_stab, x_and_not_z_stab);
		//now we have a z and an x but not both
		z_and_not_x_stab = x_and_z_stab;
		x_and_z_stab = -1;
	    }
	}

	//now the only possibilities are that we have an x_and_z, an x a z or an x and a z
	//if we have an z move it to the "top"
	if(z_and_not_x_stab >= 0){
	    StabTable_swap_rows(table, k, z_and_not_x_stab);
	    z_and_not_x_stab = k;
	    k += 1;
	}
	//if we have an x move it to the "top"
	if(x_and_not_z_stab >= 0){
	    StabTable_swap_rows(table, k, x_and_not_z_stab);
	    x_and_not_z_stab = k;
	    k += 1;
	}
	//if we have a xz move it to the "top"
	if(x_and_z_stab >= 0){
	    StabTable_swap_rows(table, k, x_and_z_stab);
	    x_and_z_stab = k;
	    k += 1;
	}

	//kill anything left on this qubit
	for(size_t s = k; s < table->k; s++){
	    if((table->table[s][q] == 1) && (table->table[s][q+table->n]) && (x_and_z_stab >= 0)){
		StabTable_rowsum(table, s, x_and_z_stab);
	    }

	    if(table->table[s][q] == 1){
		StabTable_rowsum(table, s, x_and_not_z_stab);
	    }
	    
	    if(table->table[s][q+table->n] == 1){
		StabTable_rowsum(table, s, z_and_not_x_stab);
	    }
	}

	//Case 1 - there is a generator that does not commute with Z_q
	//due to our manipulations there can be at most one generator that does not commute with Z_q
	//and this is generator k-1 (either x_and_z or just x)
	//k-1 is either an x or an xz==y, this generator does not commute with Z_q
	if(x_and_not_z_stab >= 0){
	    deleted_stabs[x_and_not_z_stab] = 1;
	}
	if(x_and_z_stab >= 0){ 
	    deleted_stabs[x_and__z_stab] = 1;
	}

	//Case 2 - Z_q commutes with every generator
	if((x_and_not_z_stab < 0) && (x_and_z_stab < 0)){
	    // in this case the (k-1)^th stabiliser is of the form I \otimes Z_q \otimes something
	    //where "something" is in the row image of the rows below this one
	    //which are all of the form I \otimes I_q \otimes (other somethings)
	    //due to previous simplifications
	    //we want to find a bunch of row additions that make the (k-1)^th stabiliser into I \otimes Z_q \otimes I
	    //we basically use gaussian elimination here on the last table->k - k generators

	    for(size_t j = q+1; j < state->n; j++){
		if((table->table[k-1][j] == 1) && (table->table[k-1][j+table->n] == 1)){
		    int resource_row = -1;
		    for(size_t row = k; row < table->k; row++){			
			if((table->table[row][j] == 1) && (table->table[row][j+table->n] == 1)){
			    if(resource_row < 0){
				resource_row = row;
				StabTable_rowsum(table, k-1, resource_row);
			    }else{
				StabTable_rowsum(table, row, resource_row);
			    }			    
			}					
		    }
		}

		if(table->table[k-1][j] == 1){
		    int resource_row = -1;
		    for(size_t row = k; row < table->k; row++){			
			if(table->table[row][j] == 1){
			    if(resource_row < 0){
				resource_row = row;
				StabTable_rowsum(table, k-1, resource_row);
			    }else{
				StabTable_rowsum(table, row, resource_row);
			    }			    
			}					
		    }
		}

		if(table->table[k-1][j+table->n] == 1){
		    int resource_row = -1;
		    for(size_t row = k; row < table->k; row++){			
			if(table->table[row][j+table->n] == 1){
			    if(resource_row < 0){
				resource_row = row;
				StabTable_rowsum(table, k-1, resource_row);
			    }else{
				StabTable_rowsum(table, row, resource_row);
			    }			    
			}					
		    }
		}
	    }

	    //so now stabiliser k-1 is of the form (\pm 1) * I \otimes Z_q \otimes I
	    //we are interested in the phase \pm 1
	    //two cases - if its +1 the measurement is non-constraining on this qubit and we double the degeneracy factor
	    //if its -1 then our set is inconsistent - we'll have both +Z_q and -Z_q
	    //practically this means that the probability is zero
	    //in any case we delete this stabiliser

	    deleted_stabs[k-1] = 1;
	    if(table->phase[k-1] == 0){
		log_v += 1;
	    }else{
		log_v = -1; // value to signal inconsistency
		break; // we break out of the for loop
	    }	    	    
	}
    }

    /*relevant info to return*/
    //log_v
    //deleted_stabs
    //table
    
}
