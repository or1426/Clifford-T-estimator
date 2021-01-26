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
    free(table->phases);
    QCircuit_free(table->circ);
    free(table);
    table = NULL;
    return 0;
}


int StabTable_CX(StabTable * table, int a, int b){
    QCircuit_append(table->circ, (Gate){.tag=CX, .target=b, .control=a});
    for(int i = 0; i < table->k; i++)
    {
        table->phases[i] ^= (table->table[i][a] & table->table[i][b+table->n]) & (table->table[i][b] ^ table->table[i][a+table->n] ^ 1);
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

    if((x1 == 0) && (z1 == 0)){
        return 0u;
    }
    if((x1 == 1) && (z1 == 0)){
        return (z2*(2*x2-1)) % 4u;
    }
    if((x1 == 0) && (z1 == 1)){
        return (x2*(1-2*z2)) % 4u;
    }
    if((x1 == 1) && (z1 == 1)){
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
    sum %= 4u;
    
    if(sum == 0){
        table->phases[h] = 0;
    }
    if(sum == 2){
        table->phases[h] = 1;
    }
    for(int j = 0; j < 2*table->n; j++){
        table->table[h][j] += table->table[i][j];
	table->table[h][j] %= 2;	
    }
    return 0;
}

unsigned char StabTable_rowsum2(StabTable * table, unsigned char * row, unsigned char phase, int i){
    unsigned char sum = 2*(phase + table->phases[i]);
    for(int j = 0; j < table->n; j++){
        sum += g(table->table[i][j], table->table[i][j+table->n], row[j], row[j+table->n]);
    }
    sum %= 4u;    
    for(int j = 0; j < 2*table->n; j++){
        row[j] += table->table[i][j];
	row[j] %= 2;
    }
    if(sum == 0){
        return 0;
    }
    if(sum == 2){
        return 1;
    }
    return 0;
}

int StabTable_delete_all_identity_qubits(StabTable * state, int * magic_qubit_numbers){
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
		if(magic_qubit_numbers != NULL){
		    magic_qubit_numbers[q-qubits_deleted] = magic_qubit_numbers[q];
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
	if(magic_qubit_numbers != NULL){
	    magic_qubit_numbers = realloc(magic_qubit_numbers, state->n);					  
	}

    }
    return qubits_deleted;
}

int StabTable_first_non_zero_in_col(StabTable * table, int col, int startpoint){
    for(int i = startpoint; i < table->k; i++){
        if(table->table[i][col] != 0){
            return i;
        }
    }
    return -1;
}

int StabTable_first_non_zero_in_row(StabTable * table, int row, int startpoint){
    for(int i = startpoint; i < 2*table->n; i++){
        if(table->table[row][i] != 0){
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
    //printf("before phase fix\n");
    //StabTable_print(table);
    //fix the phases
    for(int r = 0; r < table->k; r++){
        if(table->phases[r] != 0){
            StabTable_S(table,r);
            StabTable_S(table,r);
        }
    }
    //printf("after phase fix\n");
    //StabTable_print(table);
    //swap the identity matrix to the z part
    for(int r = 0; r < table->k; r++){
        StabTable_H(table, r);
    }

    //printf("after swap\n");
    //StabTable_print(table);
    return table->circ;
}


int StabTable_find_x_on_q(StabTable * input, int q, int a){
    for(int row = a; row < input->k-1; row++){
        if((input->table[row][q]  == 1) && (input->table[row][q+input->n] == 0)){
            return row;
        }
    }
    return -1;
}
int StabTable_find_z_on_q(StabTable * input, int q, int a){
    for(int row = a; row < input->k-1; row++){
        if((input->table[row][q]  == 0) && (input->table[row][q+input->n] == 1)){
            return row;
        }
    }
    return -1;
}
int StabTable_find_y_on_q(StabTable * input, int q, int a){
    for(int row = a; row < input->k-1; row++){
        if((input->table[row][q]  == 1) && (input->table[row][q+input->n] == 1)){
            return row;
        }
    }
    return -1;
}


/*
 * we test that if you ignore the first j+1 qubits whether that last (n-j-1) part of the last stabiliser in the table can be generated (up to phase)
 * by the other stabilisers
 * for use in the apply_constraints code
 */
int StabTable_independence_test(StabTable * table, int q){
    //we basically do gaussian elimination

    int a = 0;
    int b = q+1;
    while(a < table->k-1 && b < table->n){
        int x = StabTable_find_x_on_q(table, b, a);
        int y = StabTable_find_y_on_q(table, b, a);
        int z = StabTable_find_z_on_q(table, b, a);

        if((y >= 0) && (x >= 0)){
            StabTable_rowsum(table, y, x);
            z = y;
            y = -1;
        }
        if((y >= 0) && (z >= 0)){
            StabTable_rowsum(table, y, z);
            x = y;
            y = -1;
        }

        if(x >= 0){
            if(x != a){
                StabTable_swap_rows(table,a,x);
            }
            if(z == a){
                z = x;
            }
            for(int j = 0; j < table->k; j++){
                if((j != a) && (table->table[j][b] != 0)){
                    StabTable_rowsum(table,j,a);
                }
            }
            a += 1;
        }
        if(y >= 0){
            if(y != a){
                StabTable_swap_rows(table,a,y);
            }
            for(int j = 0; j < table->k; j++){
                if((j != a) && (table->table[j][b] != 0) && (table->table[j][b+table->n] != 0)){
                    StabTable_rowsum(table,j,a);
                }
            }
            a += 1;
        }
        if(z >= 0){
            if(z != a){
                StabTable_swap_rows(table,a,z);
            }
            for(int j = 0; j < table->k; j++){
                if((j != a) && (table->table[j][b+table->n] != 0)){
                    StabTable_rowsum(table,j,a);
                }
            }
            a += 1;
        }
        b += 1;
    }

    //int independent = 0;
    for(int p = q+1; p < table->n; p++){
        if((table->table[table->k-1][p] == 1) || (table->table[table->k-1][p+table->n] == 1)){
            return 1;
        }
    }

    return 0;
}

/*
 * Apply constraints arising from the fact that we measure the first w qubits and project the last t onto T gates
 * In particular we kill stabilisers if qubits [0, w) get killed by taking the expectation value <0| P |0>
 * and we kill stabilisers if qubits in [w, table->n-t) aren't the identity
 * we do not remove any qubits from the table
 */
int StabTable_apply_constraints(StabTable * table, int w, int t){
    int log_v = 0;

    //printf("Applying constraints\n");
    //StabTable_print(table);printf("\n");
    //first apply region a constraints (measurement)
    for(int q=0; q < w; q++){ //iterate over all the measured qubits	
        //printf("q=%d\n",q);
        //StabTable_print(table);
        int x_and_z_stab= -1; //store the index of the first stab we come to with both x and z = 1 on this qubit
        int x_and_not_z_stab = -1; //store the index of the first stab we come to with x=1, z=0
        int z_and_not_x_stab = -1; //store the index of the first stab we come to with z=1, x=0

        for(int s=0; s < table->k && (((x_and_z_stab < 0) + (x_and_not_z_stab < 0) + (z_and_not_x_stab < 0)) > 1); s++){//iterate over all stabilisers and find interesting stabilisers
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
        
	//StabTable_print(table);
	//printf("\n");
        if((x_and_z_stab >= 0) && (x_and_not_z_stab >= 0) && (z_and_not_x_stab >= 0)){ //we have all 3
            //ignore the y one if we have all 3
            x_and_z_stab = -1;
        }

        //now the only possibilities are that we have an x_and_z, an x a z or an x and a z
        //kill everything else on this qubit
        for(int s = 0; s < table->k; s++){
            if((s != x_and_z_stab) && (s != x_and_not_z_stab) && (s != z_and_not_x_stab)){
                if((table->table[s][q] == 1) && (table->table[s][q+table->n] == 1) && (x_and_z_stab >= 0)){
                    StabTable_rowsum(table, s, x_and_z_stab);
                }

                if((table->table[s][q] == 1)/* && (x_and_not_z_stab >= 0)*/){
                    StabTable_rowsum(table, s, x_and_not_z_stab);
                }

		if((table->table[s][q+table->n] == 1)/* && (z_and_not_x_stab >= 0)*/){
                    StabTable_rowsum(table, s, z_and_not_x_stab);
                }
            }
        }
	//printf("x=%d,y=%d,z=%d\n", x_and_not_z_stab, x_and_z_stab, z_and_not_x_stab);
	//printf("ZXForm(%d)\n",q);
        //StabTable_print(table);

        //case 1 - there is a generator which does not commute with Z_q
        if((x_and_z_stab >= 0) || (x_and_not_z_stab >= 0)){
	    //printf("non-commuting guy\n");
            //we can't have both >= 0
            int non_commuting_generator = (x_and_z_stab >= 0) ? x_and_z_stab : x_and_not_z_stab;
            if(non_commuting_generator < 0){
                printf("error - non_commuting_generator = %d\n", non_commuting_generator);
            }

            //we swap the non_commuting_generator to the end and delete it
            StabTable_swap_rows(table, non_commuting_generator, table->k-1);
            free(table->table[table->k-1]);
            table->k = table->k - 1;
            table->table = realloc(table->table, table->k * sizeof(unsigned char *));
            table->phases = realloc(table->phases, table->k * sizeof(unsigned char));

        }else{ //case 2 - all generators commute with Z_q
            //our generating set contains either Z_q or -Z_q
            //we need to work out which one it is
            //printf("All gens commute with Z_%d\n", q);
            //swap our Z_q guy to the end because it seems annoying
            StabTable_swap_rows(table, z_and_not_x_stab, table->k-1);
            int independent = StabTable_independence_test(table, q);
            //printf("independent = %d\n", independent);
            //StabTable_print(table);

            if(!independent){
                if(table->phases[table->k-1] == 0){
                    // +Z_q
                    log_v += 1;
                    free(table->table[table->k-1]);
                    table->k = table->k - 1;
                    table->table = realloc(table->table, table->k * sizeof(unsigned char *));
                    table->phases = realloc(table->phases, table->k * sizeof(unsigned char));
                }else{
		    
                    /* printf("q=%d\n",q); */
                    /* for(int rs = 0; rs < table->k; rs++){ */
                    /*  if(table->phases[rs]){ */
                    /*      printf("-"); */
                    /*  }else{ */
                    /*      printf("+"); */
                    /*  } */
                    /*     for(int qubit=0; qubit<table->n;qubit++){ */
                    /*         if((table->table[rs][qubit]) == 0 && (table->table[rs][qubit+table->n] == 0)){ */
                    /*             printf("I"); */
                    /*         }if((table->table[rs][qubit]) == 1 && (table->table[rs][qubit+table->n] == 0)){ */
                    /*             printf("X"); */
                    /*         }if((table->table[rs][qubit]) == 0 && (table->table[rs][qubit+table->n] == 1)){ */
                    /*             printf("Z"); */
                    /*         }if((table->table[rs][qubit]) == 1 && (table->table[rs][qubit+table->n] == 1)){ */
                    /*             printf("Y"); */
                    /*         } */

                    /*     } */
                    /*     printf("\n"); */
                    //}
                    // -Z_q
                    //our chosen measurement outcome is impossible
                    return -1;
                }
            }else{
		printf("independent\n");
	    }
        }
        //end of the two cases
    }


    //we now have n-w stabilisers on n qubits
    //time to impose region b constraints
    //printf("unmeasured qubits constraints\n");
    //StabTable_print(table);printf("\n");
    //first apply region a constraints (measurement)
    for(int q=w; q < table->n - t ; q++){ //iterate over all the non-magic qubits
	//printf("q=%d\n",q);
        //StabTable_print(table);	
	//printf("non measured, non magic qubit %d\n", q);
        int x_and_z_stab= -1; //store the index of the first stab we come to with both x and z = 1 on this qubit
        int x_and_not_z_stab = -1; //store the index of the first stab we come to with x=1, z=0
        int z_and_not_x_stab = -1; //store the index of the first stab we come to with z=1, x=0

        for(int s=0; s < table->k && (((x_and_z_stab < 0) + (x_and_not_z_stab < 0) + (z_and_not_x_stab < 0)) > 1); s++){//iterate over all stabilisers and find interesting stabilisers
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
            //ignore the y one if we have all 3
            x_and_z_stab = -1;
        }

        //now the only possibilities are that we have an x_and_z, an x a z or an x and a z
        //kill everything else on this qubit
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
	//printf("x=%d,y=%d,z=%d\n", x_and_not_z_stab, x_and_z_stab, z_and_not_x_stab);
	//printf("ZXForm(%d)\n",q);
        //StabTable_print(table);

        //now we just delete the non-identity guys on this qubit

        int num_to_delete = 0;
        if(x_and_z_stab >= 0){
            //if we have a Y stab we don't have either of the others
            StabTable_swap_rows(table, x_and_z_stab, table->k-1);
            num_to_delete += 1;
        }else{
            if(x_and_not_z_stab >= 0){
                StabTable_swap_rows(table, x_and_not_z_stab, table->k-1);
                if(table->k - 1 == z_and_not_x_stab){
                    z_and_not_x_stab = x_and_not_z_stab;
                }
                num_to_delete += 1;
            }
            if(z_and_not_x_stab >= 0){
                StabTable_swap_rows(table, z_and_not_x_stab, table->k-1-num_to_delete);
                num_to_delete += 1;
            }
        }
	//printf("num_to_delete = %d\n", num_to_delete);

        for(int deleteStabs = 0; deleteStabs < num_to_delete; deleteStabs++){
            free(table->table[table->k-1-deleteStabs]);
        }
        table->k = table->k - num_to_delete;
        table->table = realloc(table->table, sizeof(unsigned char *) * table->k);
        table->phases = realloc(table->phases, sizeof(unsigned char) * table->k);
    }

    return log_v;

}

/*
 * Our magic states are equatorial
 * so <T|Z|T> = 0
 * here we delete any stabilisers with a Z in the magic region
 * which we assume is the last t qubits
 * we return the number of qubits which have identities on them in every generator after this deletion
 */
int StabTable_apply_T_constraints(StabTable * table, int t){
    int starting_rows = table->k;
    int deleted_rows = 0;
    for(int reps = 0; reps < starting_rows; reps++){	
        for(int q=table->n-t; q < table->n; q++){ //iterate over all the magic qubits
            int x_and_z_stab= -1; //store the index of the first stab we come to with both x and z = 1 on this qubit
            int x_and_not_z_stab = -1; //store the index of the first stab we come to with x=1, z=0
            int z_and_not_x_stab = -1; //store the index of the first stab we come to with z=1, x=0

            for(int s=0; s < table->k && (((x_and_z_stab < 0) + (x_and_not_z_stab < 0) + (z_and_not_x_stab < 0)) > 1); s++){//iterate over all stabilisers and find interesting stabilisers
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
                //ignore the y one if we have all 3
                x_and_z_stab = -1;
            }


            if((z_and_not_x_stab >= 0) && (x_and_not_z_stab < 0)){
                //kill all other z stuff on this qubit
                for(int s = 0; s < table->k; s++){
                    if((s != z_and_not_x_stab) && (table->table[s][q+table->n] == 1)){
                        StabTable_rowsum(table,s, z_and_not_x_stab);
                    }
                }
                //now delete the z guy
                StabTable_swap_rows(table, z_and_not_x_stab, table->k-1);
                free(table->table[table->k-1]);
                table->k = table->k-1;
                table->table = (unsigned char **)realloc(table->table, table->k*sizeof(unsigned char *));
                table->phases = (unsigned char *)realloc(table->phases, table->k*sizeof(unsigned char));
		deleted_rows += 1;
            }
        }
    }
    return deleted_rows;
}
