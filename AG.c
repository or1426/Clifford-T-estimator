#include "AG.h"

/*
 * Init a new stabtable in the state |0>^k \otimes I^(n-k) / (2^(n-k))
 * Explicitly we have k stabilisers on n qubits with the jth being a z on the jth qubit
 * Assumptions:
 * k <= n
 */
StabTable * StabTable_new(int n, int k){
    StabTable * table = malloc(sizeof(StabTable));
    table->n = n;
    table->k = k;
    table->table = calloc(k, sizeof(unsigned char *));
    for(int i = 0; i < table->k; i++)
    {
        table->table[i] = calloc(2*table->n, sizeof(unsigned char));
        table->table[i][n+i] = 1;
    }
    table->phases = calloc(table->n, sizeof(unsigned char));
    table->circ = QCircuit_new();

    return table;
}

/*
 * Frees all memory associated with the table
 */
int StabTable_free(StabTable * table){
    for(int i = 0; i < table->k; i++)
    {
        free(table->table[i]);
    }
    free(table->table);
    QCircuit_free(table->circ);
    free(table);
    table = NULL;
    return 0;
}


int StabTable_CX(StabTable * table, int a, int b){
    QCircuit_append(table->circ, (Gate){.tag=CX, .target=b, .control=a});
    for(int i = 0; i < table->k; i++)
    {
        table->phases[i] ^= (table->table[i][a] & table->table[i][b+table->n]) ^ (table->table[i][b] ^ table->table[i][a+table->n] ^ 1);
        table->table[i][b] ^= table->table[i][a];
        table->table[i][a+table->n] ^= table->table[i][b+table->n];
    }
    return 0;
}

void StabTable_print(StabTable * table){
    for(int s = 0; s < table->k; s++){
	printf("%02d | ", s);
	for(int q = 0; q < table->n; q++){
	    printf("%d ", table->table[s][q]);
	}
	printf("|");
	for(int q = table->n; q < 2*table->n; q++){
	    printf("%d ", table->table[s][q]);
	}
	printf("| %d\n", table->phases[s]);
    }
}

int StabTable_H(StabTable * table, int a){
    QCircuit_append(table->circ, (Gate){.tag=H, .target=a, .control=0});
    unsigned char scratch = 0;
    for(int i = 0; i < table->k; i++)
    {
        table->phases[i] ^= (table->table[i][a] & table->table[i][a+table->n]);
        scratch = table->table[i][a];
        table->table[i][a] = table->table[i][a+table->n];
        table->table[i][a+table->n] = scratch;
    }
    return 0;
}

int StabTable_CZ(StabTable * table, int a, int b){
    //TODO: Write an actual CZ implementation
    StabTable_H(table, b);
    StabTable_CX(table, a, b);
    StabTable_H(table, b);
    return 0;
}


int StabTable_S(StabTable * table, int a){
    QCircuit_append(table->circ, (Gate){.tag=S, .target=a, .control=0});
    for(int i = 0; i < table->k; i++)
    {
        table->phases[i] ^= (table->table[i][a] & table->table[i][a+table->n]);
        table->table[i][a+table->n] ^= table->table[i][a];
    }
    return 0;
}

int StabTable_X(StabTable * table, int a){
    //X = HZH = H S^2 H
    //TODO: write an actial X implementation
    StabTable_H(table, a);
    StabTable_S(table, a);
    StabTable_S(table, a);
    StabTable_H(table, a);
    return 0;
}


unsigned char g(unsigned char x1, unsigned char z1, unsigned char x2, unsigned char z2){

    if(x1 == 0 && z1 == 0){
        return 0u;
    }
    if(x1 == 1 && z1 == 0){
        return z2*(2*x2-1) % 4u;
    }
    if(x1 == 0 && z1 == 1){
        return z2*(2*x2-1) % 4u;
    }
    if(x1 == 1 && z1 == 1){
        return (z2 - x2) % 4u;
    }

    //we should never reach here - maybe printing a warning would be sensible
    return 0;
}

int StabTable_rowsum(StabTable * table, int h, int i){
    unsigned char sum = 2*(table->phases[h] + table->phases[i]);
    for(int j = 0; j < table->n; j++){
        sum += g(table->table[i][j], table->table[i][j+table->n], table->table[h][j], table->table[h][j+table->n]);
    }
    sum %= 4;
    if(sum == 0){
        table->phases[h] = 1;
    }
    if(sum == 2){
        table->phases[h] = 0;
    }
    for(int j = 0; j < 2*table->n; j++){
        table->table[h][j] ^= table->table[i][j];
    }
    return 0;
}

int StabTable_first_non_zero_in_col(StabTable * table, int col, int startpoint){
    for(int i = startpoint; i < table->k; i++){
        if(table->table[col][i] != 0){
            return i;
        }
    }
    return -1;
}

int StabTable_first_non_zero_in_row(StabTable * table, int row, int startpoint){
    for(int i = startpoint; i < 2*table->n; i++){
        if(table->table[i][row] != 0){
            return i;
        }
    }
    return -1;
}

int StabTable_swap_rows(StabTable * table, int i, int j){
    if( !(i < table->k) ||  (i < 0)){
	printf("StabTable_swap_rows called with i=%d for table->k=%d\n", i, table->k);
    }
    if( !(j < table->k) ||  (j < 0)){
	printf("StabTable_swap_rows called with j=%d for table->k=%d\n", j, table->k);
    }
    if(i != j){
	unsigned char * row_scratch;
	row_scratch = table->table[i];
	table->table[i] = table->table[j];
	table->table[j] = row_scratch;
	unsigned char phase_scratch;
	phase_scratch = table->phases[i];
	table->phases[i] = table->phases[j];
	table->phases[j] = phase_scratch;
    }
    return 0;
}

//creates a copy of the input with the same stabiliser table
//does not copy the "circ" information
StabTable * StabTable_copy(StabTable * input){
    StabTable * copy = StabTable_new(input->n, input->k);
    for(int s = 0; s < input->k; s++){
	for(int q = 0; q < input->n; q++){
	    copy->table[s][q] = input->table[s][q];
	    copy->table[s][q+copy->n] = input->table[s][q+copy->n];
	}
	copy->phases[s] = input->phases[s];	    
    }
    return copy;
}

//create a QCircuit that creates this table from the initial tableau representing  the state|0>^k \otimes I^(n-k) / (2^(n-k))
//Explicitly k stabilisers on n qubits with the jth being a z on the jth qubit
//this procedure overwrites the information stored in the QCircuit circ of the table
QCircuit * StabTable_simplifying_unitary(StabTable * table){
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
            int pivot = (int)poss_pivot; //now known to be non-negative
            if(pivot != h){
                //swap rows h and pivot of the table
                StabTable_swap_rows(table,h,pivot);
            }
            for(int j = 0; j < table->k; j++){
                if((j != h) && (table->table[j][k] != 0)){
                    StabTable_rowsum(table,j,h);
                }
            }
            h += 1;
            k += 1;
        }
    }

    //so now we have a reduced row echelon form with the X part of the table having full rank

    //we swap columns (using CX) to make the X part into a kxk identity followed by a "junk" block

    for(int r = 0; r < table->k; r++){
        if(table->table[r][r] == 0){
            int col = StabTable_first_non_zero_in_row(table, r, 0);

            StabTable_CX(table, r, col);
            StabTable_CX(table, col, r);
            StabTable_CX(table, r, col);
        }
    }

    //now we use CX to clear out the "junk" block at the end of the X table
    for(int r = 0; r < table->k; r++){
        for(int col = table->k; col < table->n; col++){
            if(table->table[r][col] != 0){
                StabTable_CX(table, r, col);
            }
        }
    }

    //now we clear the leading diagonal of the z block
    for(int r = 0; r < table->k; r++){
        if(table->table[r][r+table->n] != 0){
            StabTable_S(table, r);
        }
    }

    //clear out the last k x (n-k) block of the z matrix
    for(int col = table->k; col < table->n; col++){
        for(int r = 0; r < table->k; r++){
            if(table->table[r][col+table->n] != 0){
                StabTable_CZ(table,r, col);
            }
        }
    }

    //clear out the first k x k block of the Z matrix, using that it is symmetric
    for(int col = 0; col < table->k; col++){
        for(int r = 0; r < col; r++){
            if(table->table[r][col+table->n] != 0){
                StabTable_CZ(table, r, col);
            }
        }
    }
    //fix the phases
    for(int r = 0; r < table->k; r++){
        if(table->phases[r] != 0){
            StabTable_S(table,r);
	    StabTable_S(table,r);
        }
    }

    //swap the identity matrix to the z part
    for(int r = 0; r < table->k; r++){
        StabTable_H(table, r);
    }

    return table->circ;
}


/*
 * Apply constraints arising from the fact that we measure the first w qubits and project the last t onto T gates
 * In particular we kill stabilisers if qubits [0, w) get killed by taking the expectation value <0| P |0>
 * and we kill stabilisers if qubits in [w, table->n-t) aren't the identity
 * we do not remove any qubits from the table
 */
int StabTable_apply_constraints(StabTable * table, int w, int t){
    int k = 0;
    StabTable_print(table);
    printf("{%d, %d}\n", table->k,table->n);
    for(int q=0; q < table->n - t; q++){ //iterate over all the non-t qubits
	printf("q = %d\n", q);
        int x_and_z_stab= -1; //store the index of the first stab we come to with both x and z = 1 on this qubit
        int x_and_not_z_stab = -1; //store the index of the first stab we come to with x=1, z=0
        int z_and_not_x_stab = -1; //store the index of the first stab we come to with z=1, x=0

        for(int s=k; s < table->k && (((x_and_z_stab < 0) + (x_and_not_z_stab < 0) + (z_and_not_x_stab < 0)) > 1); s++){//iterate over all stabilisers and find interesting stabilisers
            if((table->table[s][q] == 1) && (table->table[s][q+table->n] == 1)){
                x_and_z_stab = s;
            }
            if((table->table[s][q] == 1) && (table->table[s][q+table->n] == 0)){
                x_and_not_z_stab = s;
            }
	    if((table->table[s][q] == 0) && (table->table[s][q+table->n] == 1)){
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
            }else if(z_and_not_x_stab < 0){//we don't have a generator for z alone, but we can make one
                StabTable_rowsum(table, x_and_z_stab, x_and_not_z_stab);
                //now we have a z and an x but not both
                z_and_not_x_stab = x_and_z_stab;
                x_and_z_stab = -1;
            }
        }
	

	if((x_and_z_stab >= 0) && (x_and_not_z_stab >= 0) && (z_and_not_x_stab >= 0)){ //we have all 3
	    //ignore the y one
	    x_and_z_stab = -1;
	}

	printf("%d (%d, %d, %d)\n", q, x_and_not_z_stab, z_and_not_x_stab, x_and_z_stab);
	StabTable_print(table);
	printf("\n");
	
        //now the only possibilities are that we have an x_and_z, an x a z or an x and a z
        //if we have an z move it to the "top"
        if(z_and_not_x_stab >= 0){
	    printf("Swapping z to top\n");
	    StabTable_print(table);printf("\n");
            StabTable_swap_rows(table, k, z_and_not_x_stab);
	    StabTable_print(table);
	    if(k == x_and_not_z_stab){
		x_and_not_z_stab = z_and_not_x_stab;
	    }
            z_and_not_x_stab = k;
            k += 1;
        }
        //if we have an x move it to the "top"
        if(x_and_not_z_stab >= 0){
	    printf("Swapping x to top\n");
	    StabTable_print(table);printf("\n");
            StabTable_swap_rows(table, k, x_and_not_z_stab);
	    StabTable_print(table);
	    if(k == z_and_not_x_stab){
		z_and_not_x_stab = x_and_not_z_stab;
	    }
            x_and_not_z_stab = k;
            k += 1;
        }
        //if we have a xz move it to the "top"
        if(x_and_z_stab >= 0){	    
            StabTable_swap_rows(table, k, x_and_z_stab);
            x_and_z_stab = k;
            k += 1;
        }
	printf("%d (%d, %d, %d)\n", q, x_and_not_z_stab, z_and_not_x_stab, x_and_z_stab);
	StabTable_print(table);
	printf("\n");
        //kill anything left on this qubit
        for(int s = 0; s < table->k; s++){
            if((s != x_and_z_stab) && (s != x_and_not_z_stab) && (s != z_and_not_x_stab)){
                if((table->table[s][q] == 1) && (table->table[s][q+table->n] == 1) && (x_and_z_stab >= 0)){
                    StabTable_rowsum(table, s, x_and_z_stab);
                }

                if(table->table[s][q] == 1 && x_and_not_z_stab >= 0){
                    StabTable_rowsum(table, s, x_and_not_z_stab);
                }

                if(table->table[s][q+table->n] == 1 && z_and_not_x_stab >= 0){
                    StabTable_rowsum(table, s, z_and_not_x_stab);
                }
            }
        }

	StabTable_print(table);
	printf("\n");
	
        //Case 2 - we're dealing with a measured qubit and Z_q commutes with every generator
        if((q < w) && (x_and_not_z_stab < 0) && (x_and_z_stab < 0)){
            // in this case the (k-1)^th stabiliser is of the form I \otimes Z_q \otimes something
            //where "something" is in the row image of the rows below this one
            //which are all of the form I \otimes I_q \otimes (other somethings)
            //due to previous simplifications
            //we want to find a bunch of row additions that make the (k-1)^th stabiliser into I \otimes Z_q \otimes I
            //we basically use gaussian elimination here on the last table->k - k generators

            for(int j = q+1; j < table->n; j++){
                if((table->table[k-1][j] == 1) && (table->table[k-1][j+table->n] == 1)){
                    int resource_row = -1;
                    for(int row = k+1; row < table->k; row++){
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
                    for(int row = k; row < table->k; row++){
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
                    for(int row = k; row < table->k; row++){
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
        }
    }
    StabTable_print(table);

    int log_v = 0; //this keeps track of duplicated sums this increases by 1 if a measurement was non-constraining
    unsigned char * deleted_stabs = calloc(table->k, sizeof(unsigned char)); //put a 1 here for qubits we will delete
    
    //so now our stabiliser table has the following format
    //on each of the first table->n - t qubits
    //there is either 1 or two stabilisers which are not the identity on that qubit
    //if there are two then they are X and Z
    //if there is 1 then its X, Y or Z
    //we have an additional restriction which is that on the measured qubits if all the stabs commute with Z_q
    //then one of our generators is exactly (\pm 1) Z_q, i.e. this generator is the identity on everything else
    
    //now we deal with the measured qubits
    for(int q = 0; q < w; q++){
	//these are going to be the (up to) two stabs that are non-trivial on this qubit
	int s1 = -1;
	int s2 = -1;

	for(int s = 0; (s < table->k) && (s2 < 0); s++){
	    if((table->table[s][q] != 0) || (table->table[s][q+table->n] != 0)){
		if(s1 < 0){
		    s1 = s;
		}else{
		    s2 = s;
		}
	    }
	    //if we have two stabs lets make sure s1 is the x guy
	    if(s1 >= 0 && s2 >= 0){
		if(table->table[s1][q] == 0){
		    StabTable_swap_rows(table, s1,s2);
		}
	    }

	    //in this case we just delete the first stab
	    deleted_stabs[s1] = 1;
	    //we have two cases
	    //either there are gens that do not commute with Z_q or they all commute
	    //if there is a non-commuting one things are fine - the measurement on this qubit has 50/50 outcomes
	    printf("s1 = %d\n", s1);
	    if(table->table[s1][q] == 0){ 
		//the only stab is Z (on this qubit)
		//we ensured that if the stab has a Z on this qubit then the stab is equal to (\pm 1) * Z_q
		if(table->phases[s1] == 0){
		    if(log_v >= 0){
			log_v += 1;
		    }
		}else{
		    log_v = -1;
		}
	    }
	}
    }

    //now we deal with the non-measured qubits
    for(int q = w; q < table->n - t; q++){
	//these are going to be the (up to) two stabs that are non-trivial on this qubit
	int s1 = -1;
	int s2 = -1;

	for(int s = 0; (s < table->k) && (s2 < 0); s++){
	    if((table->table[s][q] != 0) || (table->table[s][q+table->n] != 0)){
		if(s1 < 0){
		    s1 = s;
		}else{
		    s2 = s;
		}
	    }

	    if(s1 >= 0){
		deleted_stabs[s1] = 1;
	    }
	    if(s2 >= 0){
		deleted_stabs[s2] = 1;
	    }	    
	}	
    }

    //so now we go through the list of stabilisers
    //and delete the ones with deleted_stabs indicator set to 1
    int old_max_index = table->k;
    int new_index = 0;
    for(int old_index = 0; old_index < old_max_index; old_index++){
	if(deleted_stabs[old_index] == 0){ //not deleted so we just copy it across
	    table->table[new_index] = table->table[old_index];
	    table->phases[new_index] = table->phases[old_index];
	    new_index += 1;
	}else{ //this stab is deleted 
	    free(table->table[old_index]);
	    table->table[old_index] = NULL;
	}
    }
    table->k = new_index;
    realloc(table->table, table->k*sizeof(unsigned char *));
    
    /*relevant info to return*/
    //log_v
    //table


    free(deleted_stabs);
    return log_v;

}
