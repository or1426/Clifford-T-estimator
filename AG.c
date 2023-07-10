#include "AG.h"
#include "string.h"
#include "binary_expression.h"
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
  StabTable * StabTable_copy(StabTable * table){
  StabTable * new = malloc(sizeof(StabTable));
  new->n = table->n;
  new->k = table->k;
  new->table = calloc(new->k, sizeof(unsigned char *));
  new->phases = calloc(table->k, sizeof(unsigned char));
  for(int s = 0; s < table->k; s++){
  new->phases[s] = table->phases[s];
  new->table[s] = calloc(2*table->n, sizeof(unsigned char));
  for(int q = 0; q < 2*table->n; q++){
  new->table[s][q] = table->table[s][q];
  }
  }
  return new;
  }
*/
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

int StabTable_CX_no_circ(StabTable * table, int a, int b){
  for(int i = 0; i < table->k; i++)
    {
      table->phases[i] ^= (table->table[i][a] & table->table[i][b+table->n]) & (table->table[i][b] ^ table->table[i][a+table->n] ^ 1);
      table->table[i][b] ^= table->table[i][a];
      table->table[i][a+table->n] ^= table->table[i][b+table->n];
    }
  return 0;
}

int StabTable_swap_no_circ(StabTable * table, int a, int b){
  unsigned char scratch;
  for(int i = 0; i < table->k; i++)
    {
      scratch = table->table[i][a];
      table->table[i][a] = table->table[i][b];
      table->table[i][b] = scratch;

      scratch = table->table[i][a+table->n];
      table->table[i][a+table->n] = table->table[i][b+table->n];
      table->table[i][b+table->n] = scratch;
    }
  return 0;
}


void StabTable_print(StabTable * table){
  for(int s = 0; s < table->k; s++){
    printf("%02d | ", s);
    for(int q = 0; q < table->n; q++){
      printf("%d ", table->table[s][q]);
    }
    printf("| ");
    for(int q = table->n; q < 2*table->n; q++){
      printf("%d ", table->table[s][q]);
    }
    printf("| %d\n", table->phases[s]);
  }
}

void StabTable_pprint_row(int n, int t, unsigned char phase, unsigned char * row){
  if(t >= 0){
    char str[n+3];
    str[n+2] = '\0';
    if(phase){
      str[0] = '-';
    }else{
      str[0] = '+';
    }

    int j = 1;
    for(int i = 0; i < n; i++){
      if(!row[i] && !row[i+n]){
        str[j] = 'I';
      }
      else if(row[i] && !row[i+n]){
        str[j] = 'X';
      }
      else if(row[i] && row[i+n]){
        str[j] = 'Y';
      }
      else if(!row[i] && row[i+n]){
        str[j] = 'Z';
      }

      if(i == n-t-1){
        str[j+1] = ' ';
        j += 1;
      }

      j += 1;
    }
    printf("%s\n", str);
  }else{
    char str[n+2];
    str[n+1] = '\0';
    if(phase){
      str[0] = '-';
    }else{
      str[0] = '+';
    }

    int j = 1;
    for(int i = 0; i < n; i++){
      if(!row[i] && !row[i+n]){
        str[j] = 'I';
      }
      else if(row[i] && !row[i+n]){
        str[j] = 'X';
      }
      else if(row[i] && row[i+n]){
        str[j] = 'Y';
      }
      else if(!row[i] && row[i+n]){
        str[j] = 'Z';
      }

      j += 1;
    }
    printf("%s\n", str);
  }


}

void StabTable_pprint_row_sections(int n, unsigned char phase, unsigned char * row, int n_sections, int * section_indices){
  char str[n+2+n_sections];

  str[n+1+n_sections] = '\0';
  if(phase){
    str[0] = '-';
  }else{
    str[0] = '+';
  }

  int j = 1;
  int sections_so_far = 0;

  for(int i = 0; i < n; i++){
    if((sections_so_far < n_sections) && (i == section_indices[sections_so_far])){
      sections_so_far += 1;
      str[j] = ' ';
      j +=1;
    }

    if(!row[i] && !row[i+n]){
      str[j] = 'I';
    }
    else if(row[i] && !row[i+n]){
      str[j] = 'X';
    }
    else if(row[i] && row[i+n]){
      str[j] = 'Y';
    }
    else if(!row[i] && row[i+n]){
      str[j] = 'Z';
    }

    j += 1;

  }
  printf("%s\n", str);
}


void StabTable_pprint_table(StabTable * state, int t){
  for(int row=0; row<state->k; row++){
    StabTable_pprint_row(state->n,t, state->phases[row], state->table[row]);
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

int StabTable_H_no_circ(StabTable * table, int a){
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

  //printf("qubits = %u, qubits deleted = %d\n", state->n, qubits_deleted);

  if(qubits_deleted > 0){
    for(int s = 0; s < state->k; s++){
      for(int q = 0; q < state->n; q++){
        state->table[s][q+state->n-qubits_deleted] = state->table[s][q+state->n];
      }
      state->table[s] = (unsigned char *)realloc(state->table[s], 2*(state->n - qubits_deleted)*sizeof(unsigned char));
    }
    state->n = state->n - qubits_deleted;
  }
  //printf("qubits = %u, qubits deleted = %d\n", state->n, qubits_deleted);
  //StabTable_pprint_table(state, -1);
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
        StabTable_rowsum(table,h,pivot);
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

int StabTable_ZX_form(StabTable * table, int starting_row, int q)
{
  int x_and_z_stab= -1; //store the index of the first stab we come to with both x and z = 1 on this qubit
  int x_and_not_z_stab = -1; //store the index of the first stab we come to with x=1, z=0
  int z_and_not_x_stab = -1; //store the index of the first stab we come to with z=1, x=0

  for(int s=starting_row; s < table->k && (((x_and_z_stab < 0) + (x_and_not_z_stab < 0) + (z_and_not_x_stab < 0)) > 1); s++){//iterate over all stabilisers and find interesting stabilisers
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

      if((table->table[s][q] == 1) && (x_and_not_z_stab >= 0)){
        StabTable_rowsum(table, s, x_and_not_z_stab);
      }

      if((table->table[s][q+table->n] == 1) && (z_and_not_x_stab >= 0)){
        StabTable_rowsum(table, s, z_and_not_x_stab);
      }
    }
  }

  int row_count = 0;
  if(x_and_z_stab >= 0){
    StabTable_swap_rows(table, starting_row, x_and_z_stab);
    x_and_z_stab = starting_row;
    row_count+=1;
  }else{
    if(z_and_not_x_stab >= 0){
      StabTable_swap_rows(table, starting_row, z_and_not_x_stab);

      if(starting_row == x_and_not_z_stab){
        x_and_not_z_stab = z_and_not_x_stab;
      }
      z_and_not_x_stab = starting_row;
      row_count+=1;
    }
    if(x_and_not_z_stab >= 0){
      StabTable_swap_rows(table, starting_row  + row_count, x_and_not_z_stab);
      x_and_not_z_stab = starting_row  + row_count;
      row_count+=1;
    }
  }

  //if there is anything left on this qubit we failed to ZX-form it
  //this can only happen becaus of the "cascading" structure
  //i.e. we can't use a stabilizer as a leading stabilizer for this qubit because it is already a leading stabilizer for some other qubit
  int sign = 1;
  for(int s = 0; s < table->k; s++){
    if((s != x_and_z_stab) && (s != x_and_not_z_stab) && (s != z_and_not_x_stab)){
      if((table->table[s][q] == 1) || (table->table[s][q+table->n] == 1)){
        sign = -1;
        break;
      }
    }
  }

  return sign*row_count;
}


int StabTable_cascading_ZX_form(StabTable * table, int starting_qubit)
{
  int starting_row = 0;

  int one_stabiliser_qubit;
  int two_stabiliser_qubit;

  //printf("ZX formed qubits: ");

  //we keep looping until we don't find any qubits we can ZX form anymore
  while((one_stabiliser_qubit >= 0) || (two_stabiliser_qubit >= 0)){
    one_stabiliser_qubit = -1;
    two_stabiliser_qubit = -1;
    for(int q = starting_qubit; q < table->n; q++){
      int stabiliser_count = StabTable_ZX_form(table, starting_row, q);

      if(stabiliser_count == 1){

        starting_row += 1;
        one_stabiliser_qubit = q;
        //printf("%d ", q);
        break;
      }
      if(stabiliser_count == 2 && two_stabiliser_qubit == -1){
        two_stabiliser_qubit = q;
      }
    }

    //if we didn't find a one stabiliser guy we're willing to accept a two stabiliser guy
    if((one_stabiliser_qubit == -1) && (two_stabiliser_qubit >= 0)){
      int stabiliser_count = StabTable_ZX_form(table, starting_row, two_stabiliser_qubit);
      if(stabiliser_count != 2){
        printf("count was %d, but we really thought it was gonna be 2!\n", stabiliser_count);
      }
      //printf("%d ", two_stabiliser_qubit);
      starting_row += 2;
    }
  }
  //printf("\n");
  return 0;
}


int StabTable_cascading_ZX_form2(StabTable * table, int starting_qubit)
{
  int starting_row = 0;

  int junk_count = 0;
  int one_stab_qubit_count = 0;
  int two_stab_qubit_count = 0;

  for(int q = 0; q < table->n; q++){
    if(starting_row == table->k -1){
      break;
    }
    int stabiliser_count = StabTable_ZX_form(table, starting_row, q);
    if(stabiliser_count == 2){
      starting_row += 2;
      two_stab_qubit_count += 1;
      if(two_stab_qubit_count - 1 != q){
        StabTable_CX(table, two_stab_qubit_count-1, q);
        StabTable_CX(table, q, two_stab_qubit_count-1);
        StabTable_CX(table, two_stab_qubit_count-1, q);
      }
    }
  }
  for(int q = two_stab_qubit_count; q < table->n; q++){
    if(starting_row == table->k -1){
      break;
    }
    int stabiliser_count = StabTable_ZX_form(table, starting_row, q);
    if(stabiliser_count == 1){
      starting_row += 1;
      one_stab_qubit_count += 1;
      if(two_stab_qubit_count + one_stab_qubit_count - 1 != q){
        StabTable_CX(table, two_stab_qubit_count + one_stab_qubit_count - 1, q);
        StabTable_CX(table, q, two_stab_qubit_count + one_stab_qubit_count - 1);
        StabTable_CX(table, two_stab_qubit_count + one_stab_qubit_count - 1, q);
      }
    }
  }
  //fprintf(stderr, "%d %d %d %d %d\n", table->n, table->k, one_stab_qubit_count, two_stab_qubit_count, starting_row);
  int n_sections = 2;
  int  section_indices[2] = {two_stab_qubit_count, one_stab_qubit_count+two_stab_qubit_count};
  //for(int s = 0; s < table->k; s++){
  //  StabTable_pprint_row_sections(table->n, table->phases[s], table->table[s], n_sections, section_indices);
  //}
  //free(section_indices);
  return 0;
}


int delete_columns_with_at_most_one_nontrivial_stabilizer(StabTable * state){
  int deleted_qubits = 0;
  for(int q = 0; q < state->n; q++){
    char first_stab_on_this_qubit = 0; // 0 identity 1 is x 2 is z 3 is y
    char all_stabs_match_on_this_qubit = 1;
    for(int s = 0; s < state->k; s++){
      if(first_stab_on_this_qubit == 0){
        if(state->table[s][q] == 1){ // x
          first_stab_on_this_qubit += 1;
        }
        if(state->table[s][q+state->n] == 1){ // z
          first_stab_on_this_qubit += 2;
        }
      }else{
        if((first_stab_on_this_qubit == 1) && (state->table[s][q+state->n] == 1)){
          all_stabs_match_on_this_qubit = 0;
        }
        if((first_stab_on_this_qubit == 2) && (state->table[s][q] == 1)){
          all_stabs_match_on_this_qubit = 0;
        }
        if((first_stab_on_this_qubit == 3) && (state->table[s][q] != state->table[s][q+state->n])){
          all_stabs_match_on_this_qubit = 0;
        }
        if(!all_stabs_match_on_this_qubit){
          break;
        }
      }
    }

    if(all_stabs_match_on_this_qubit){
      for(int s = 0; s < state->k; s++){
        state->table[s][q] = 0;
        state->table[s][q+state->n] = 0;
      }
    }
  }
  //printf("zeroing columns we ignore\n");
  //StabTable_pprint_table(state, -1);

  return StabTable_delete_all_identity_qubits(state, NULL);
}


int StabTable_cascading_ZX_form_only_doubles(StabTable * table, int starting_qubit)
{
  printf("doubles ZX form: table->n = %d, table->k = %d\n", table->n, table->k);
  //printf("at start of cascading ZX form state->n = %d\n", table->n);
  int starting_row = 0;

  //printf("ZX formed qubits: ");
  int q = starting_qubit;

  int zx_formed_count = 0;

  //we keep going as long as we havent run out of qubits and stabilizers
  while((q < table->n) && (starting_row< table->k)){

    printf("Start row = %d, trying to ZX form qubit %d: ", starting_row, q);
    int stabilizer_count = StabTable_shared_stab_ZX_form(table, starting_row, q);
    if(stabilizer_count >= 0){
      printf("stab_count =  %d\n", stabilizer_count);
      starting_row += stabilizer_count;
      zx_formed_count += 1;
    }else{
      printf("failed\n");
    }
    q += 1;
  }

  //printf("end of cascading ZX form state->n = %d\n", table->n);
  //printf("\n");

  return zx_formed_count;
}


int StabTable_shared_stab_ZX_form(StabTable * table, int starting_row, int q)
{
  int x_and_z_stab= -1; //store the index of the first stab we come to with both x and z = 1 on this qubit
  int x_and_not_z_stab = -1; //store the index of the first stab we come to with x=1, z=0
  int z_and_not_x_stab = -1; //store the index of the first stab we come to with z=1, x=0

  for(int s=starting_row; s < table->k && (((x_and_z_stab < 0) + (x_and_not_z_stab < 0) + (z_and_not_x_stab < 0)) > 1); s++){//iterate over all stabilisers and find interesting stabilisers
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

      if((table->table[s][q] == 1) && (x_and_not_z_stab >= 0)){
        StabTable_rowsum(table, s, x_and_not_z_stab);
      }

      if((table->table[s][q+table->n] == 1) && (z_and_not_x_stab >= 0)){
        StabTable_rowsum(table, s, z_and_not_x_stab);
      }
    }
  }

  //if there is anything left on this qubit we failed to ZX-form it
  //this can only happen becaus of the "cascading" structure
  //i.e. we can't use a stabilizer as a leading stabilizer for this qubit because it is already a leading stabilizer for some other qubit
  int x_stabs = 0;
  int z_stabs = 0;
  int y_stabs = 0;
  for(int s = 0; s < table->k; s++){
    if((table->table[s][q] == 1) && (table->table[s][q+table->n] == 0)){
      x_stabs +=1;
    }
    if((table->table[s][q] == 1) && (table->table[s][q+table->n] == 1)){
      y_stabs +=1;
    }
    if((table->table[s][q] == 0) && (table->table[s][q+table->n] == 1)){
      z_stabs +=1;
    }
  }

  if((x_stabs <= 1) &&  (z_stabs <= 1) && (y_stabs == 0)){ // if we have no ys and at most one x and one z
    //if we have at most two stabilizers we consider that we have "ZX formed" this qubit
    int row_count = 0;
    if(x_and_z_stab >= 0){
      StabTable_swap_rows(table, starting_row, x_and_z_stab);
      row_count+=1;
    }else{
      if(z_and_not_x_stab >= 0){
        StabTable_swap_rows(table, starting_row, z_and_not_x_stab);

        if(starting_row == x_and_not_z_stab){
          x_and_not_z_stab = z_and_not_x_stab;
        }

        row_count+=1;
      }
      if(x_and_not_z_stab >= 0){
        StabTable_swap_rows(table, starting_row  + row_count, x_and_not_z_stab);
        row_count+=1;
      }
    }
    return row_count;

  }
  return -1;
}

void xor_row(unsigned char * target, unsigned char * source, int n){
  for(int i = 0; i < n; i++){
    target[i] ^= source[i];
  }
}

void or_row(unsigned char * target, unsigned char * source, int n){
  for(int i = 0; i < n; i++){
    target[i] |= source[i];
  }
}

int StabTable_row_reduction_Z_table(StabTable * table){
  int rank = 0;
  for(int q = 0; q < table->n; q++){
    int found_pivot = -1;
    for(int s = rank; s < table->k; s++){
      if(table->table[s][q+table->n]){
        found_pivot = s;
        break;
      }
    }
    if(found_pivot != -1){
      if(found_pivot != rank){
        StabTable_swap_rows(table, found_pivot, rank);
      }

      for(int s = 0; s < table->k; s++){
        if((s != rank) && table->table[s][q+table->n]){
          StabTable_rowsum(table, s, rank);
        }
      }

      rank += 1;
    }
  }
  return rank;
}

//returns a lower bound on the rank of the matrix diag(y) . X + Z
int StabTable_row_reduction_upper_bound(StabTable * table) {

  unsigned char ** X = calloc(table->n, sizeof(unsigned char *));
  unsigned char ** Z = calloc(table->n, sizeof(unsigned char *));

  for(int i = 0; i < table->n; i++) {
    X[i] = calloc(table->k, sizeof(unsigned char));
    Z[i] = calloc(table->k, sizeof(unsigned char));

    for(int j = 0; j < table->k; j++) {
      X[i][j] = table->table[j][i];
      Z[i][j] = table->table[j][i + table->n];
    }
  }


  // we want to row reduction on the matrix  M(y) = diag(y) . X + Z where y consists of boolean unknowns
  // we want to upper bound the size of the kernel of M(y)
  // which means lower bounding the size of the image
  // rank-nullity theorem go brrr

  int k = 0;
  for(int s = 0; s < table->k; s++) {
    int leading_1_row = -1;

    for(int q = k; q < table->n; q++) {
      if((Z[q][s] == 1) && (X[q][s] == 0)) { // looking for a leading 1 in this column
        leading_1_row = q;
        break;
      }
    }

    if(leading_1_row >= 0) {
      if(leading_1_row != k){
        unsigned char * scratch = X[k];
        X[k] = X[leading_1_row];
        X[leading_1_row] = scratch;

        scratch = Z[k];
        Z[k] = Z[leading_1_row];
        Z[leading_1_row] = scratch;
      }

      for(int q = k+1; q < table->n; q++){
        if((X[q][s] == 0) && (Z[q][s] == 1)){
          xor_row(Z[q], Z[k], table->k);
          or_row(X[q], X[k], table->k);

        }
        if((X[q][s] == 1) && (Z[q][s] == 0)){
          or_row(X[q], X[k], table->k);
          or_row(X[q], Z[k], table->k);
          X[q][s] = 0;
        }
        if((X[q][s] == 1) && (Z[q][s] == 1)){
          or_row(X[q], X[k], table->k);
          or_row(X[q], Z[k], table->k);
          X[q][s] = 0;
          Z[q][s] = 0;
        }
      }
      k += 1;
    }
  }

  /* for(int i = 0; i < table->n; i++){ */
  /*   for(int j = 0; j < table->k; j++){ */
  /*     printf("%d", X[i][j]); */
  /*   } */
  /*   printf(" | "); */
  /*   for(int j = 0; j < table->k; j++){ */
  /*     printf("%d", Z[i][j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  for(int i = 0; i < table->n; i++) {
    free(X[i]);
    free(Z[i]);
  }
  free(X);
  free(Z);
  return k;
}

void identify_entanglement_classes(StabTable * table){
  int class_size = 2;
  int found_class = 0;

  int * support = malloc(class_size * sizeof(int));


  //iterate over stabilizers and find one with only two non-identity qubits
  do{
    for(int q = 0; q < class_size; ++q){
      support[q] = -1;
    }

    int initial_class_member = -1; // the generator we find with support size less than the class size
    for(int s = 0; s < table->k; ++s){
      int support_size = 0;
      int count = 0;
      for(int q = 0; q < table->n; ++q){
        if(table->table[s][q] || table->table[s][table->n + q]){
          count += 1;
        }
      }

      if(support_size <= class_size){
        initial_class_member = s;
        break;
      }

    }

    if(initial_class_member >= 0){

    }

  }while(found_class);
}


/* void StabTable_set_to_data(StabTable * state, uint_bitarray_t z_offdiag_data, uint_bitarray_t z_diag_data, uint_bitarray_t z_junk_data, uint_bitarray_t x_junk_data, uint_bitarray_t hadamard_data){ */
/*   //set z data */
/*   for(int q = 0; q < state->k; q++){ */
/*     state->table[q][q] = ((z_diag_data >> q) & ONE); */
/*     for(int s = 0; s < q; s++){ */
/*       state->table[s][q] = ((z_offdiag_data >> (s+q*state->k)) & ONE); */
/*       state->table[q][s] = ((z_offdiag_data >> (s+q*state->k)) & ONE); */
/*     } */
/*   } */
/*   for(int q = state->k; q < state->n; q++){ */
/*     for(int s = 0; s < state->k; s++){ */
/*       state->table[s][q] = ((z_junk_data >> (s + (q-state->k)*state->k)) & ONE); */
/*     } */
/*   } */

/*   //now the x data we do with cnots and hadamards */
/*   for(int q = state->k; q < state->n; q++){ */
/*     for(int s = 0; s < state->k; s++){ */
/*       if(x_junk_data >> ((s + (q-state->k)*state->k) & ONE)){ */
/*      StabTable_CX_no_circ(state, s, q); */
/*       }       */
/*     } */
/*   } */
/*   for(int q = 0; q < state->n; q++){ */
/*     if(hadamard_data >> q){ */
/*       StabTable_H_no_circ(state, q); */
/*     } */
/*   }   */
/* } */

/* int iterate_over_pure_stab_states(size_t n){ */
/*   //need n(n-1)/2 bits to make the antisymetric part of Z */
/*   if(n*(n-1)/2 > sizeof(uint_bitarray_t)) { */
/*     return -1; */
/*   } */

/*   StabTable * state = StabTable_new(n, k); */
/*   for(uint_bitarray_t z_offdiag_data = 0; z_offdiag_data != (ONE << (n*(n-1)/2)); z_offdiag_data++){ */
/*     for(uint_bitarray_t z_diag_data = 0; z_diag_data != (ONE << n); z_diag_data++){ */
/*       for(uint_bitarray_t hadamard_data = 0; hadamard_data != (ONE << n); hadamard_data++){ */
/*      //set up the state */



/*       } */
/*     } */
/*   } */
/* } */


/* int iterate_over_stab_states(size_t n, size_t k){ */
/*   //need n(n-1)/2 bits to make the antisymetric part, and k(n-k) to make each left over part */
/*   if(k*(k-1) > 2*sizeof(uint_bitarray_t) || k*(n-k) > sizeof(uint_bitarray_t)) { */
/*     return -1; */
/*   } */


/*   StabTable * state = StabTable_new(n, k); */
/*   for(uint_bitarray_t z_offdiag_data = 0; z_offdiag_data != (ONE << (k*(k-1)/2)); z_offdiag_data++){ */
/*     for(uint_bitarray_t z_diag_data = 0; z_diag_data != (ONE << k); z_diag_data++){ */
/*       for(uint_bitarray_t z_junk_data = 0; z_junk_data != (ONE << k*(n-k)); z_junk_data++){ */
/*      for(uint_bitarray_t x_junk_data = 0; x_junk_data != (ONE << k*(n-k)); x_junk_data++){ */
/*        for(uint_bitarray_t hadamard_data = 0; hadamard_data != (ONE << n); hadamard_data++){      */
/*          StabTable_set_to_data(state, z_offdiag_data, z_diag_data, z_junk_data, x_junk_data, hadamard_data); */
/*        } */
/*      }        */
/*       } */
/*     } */
/*   } */

/* } */

int StabTable_count_distinct_cols(StabTable * state){
  int * column_classes = malloc(state->n * sizeof(int));
  for(int q = 0; q < state->n; q++){
    column_classes[q] = -1;
  }


  int unclassed_qubits = state->n;
  int num_classes = 0;
  while(unclassed_qubits > 0){
    int first_unclassified_qubit = -1;
    for(int q = 0; q < state->n; q++){
      if(column_classes[q] == -1){
        first_unclassified_qubit = q;
        break;
      }
    }
    if(first_unclassified_qubit < 0){
      break; // we didn't find any unclassified qubits
    } else {
      column_classes[first_unclassified_qubit] = num_classes;

      for(int q = first_unclassified_qubit + 1; q < state->n; q++){
        if(column_classes[q] == -1){
          int matching = 1;
          for(int s = 0; s < state->k; s++){
            if((state->table[s][first_unclassified_qubit] != state->table[s][q]) ||
               (state->table[s][first_unclassified_qubit+state->n] != state->table[s][q+state->n])){
              matching = 0;
              break;
            }
          }
          if(matching) {
            column_classes[q] = num_classes;
            unclassed_qubits -= 1;
          }
        }
      }

      num_classes += 1;
    }
  }
  /* for(int i = 0; i < state->n; i++){ */
  /*   printf("%d, ", column_classes[i]); */
  /* } */
  /* printf("\n"); */
  free(column_classes);
  return num_classes;
}


/* struct EquivClassDecomp{ */
/*   size_t n_qubits; */
/*   size_t n_classes; */
/*   size_t * class_sizes; */
/*   size_t ** classes;   */
/* }; */

void EquivClassDecomp_init(EquivClassDecomp * classes, size_t n_qubits){
  classes->n_qubits = n_qubits;
  classes->n_classes = 0;
  classes->class_sizes = NULL;
  classes->classes = NULL;
}

void EquivClassDecomp_free(EquivClassDecomp * classes){
  for(size_t i = 0; i < classes->n_classes; i++){
    free(classes->classes[i]);
  }
  free(classes->class_sizes);
  classes->class_sizes = NULL;
  classes->classes = NULL;
  classes->n_qubits = 0;
  classes->n_classes = 0;
}


void EquivClassDecomp_add_class(EquivClassDecomp * classes, size_t q){
  classes->n_classes += 1;
  classes->class_sizes = realloc(classes->class_sizes, classes->n_classes*sizeof(size_t));
  classes->class_sizes[classes->n_classes-1] = 1;
  classes->classes = realloc(classes->classes, classes->n_classes*sizeof(size_t*));
  classes->classes[classes->n_classes-1] = malloc(sizeof(size_t));
  classes->classes[classes->n_classes-1][0] = q;
}

void EquivClassDecomp_add_to_class(EquivClassDecomp * classes, size_t idx, size_t q){
  classes->class_sizes[idx] += 1;
  classes->classes[idx] = realloc(classes->classes[idx], classes->class_sizes[idx]*sizeof(size_t));
  classes->classes[idx][classes->class_sizes[idx]-1] = q;
}

size_t EquivClassDecomp_find_class_from_representative(EquivClassDecomp * classes, size_t q){
  for(int i = 0; i < classes->n_classes; i++){
    for(int j = 0; j < classes->class_sizes[i]; j++){
      if(classes->classes[i][j] == q){
        return i;
      }
    }
  }

  return classes->n_classes;
}

//bring our table to a form where every generator has either a leading x or leading z Pauli
int * StabTable_row_reduce(StabTable * state){
  int * pivots = malloc(state->k * sizeof(int));
  for(size_t i = 0; i < state->k; i++){
    pivots[i] = -1;
  }
  //fprintf(stderr, "%d %d\n", state->n, state->k);

  //size_t row = 0;
  for(int i = 0; i < state->k; i++){
    //fprintf(stderr, "%d\n", i);
    char found_pivot = 0;
    for(int q = 0; q < 2*state->n; q++){
      if(state->table[i][q]){
        char q_in_pivots = 0;
        for(int idx = 0; idx < i; idx++){
          if(pivots[idx] == q){
            q_in_pivots = 1;
            break;
          }
        }
        if(!q_in_pivots){
          found_pivot = 1;
          pivots[i] = q;
          for(int s = 0; s < state->k; s++){
            if(state->table[s][q] && s != i){
              StabTable_rowsum(state, s, i);
            }
          }
          break;
        }

      }
    }
    if(!found_pivot){
      fprintf(stderr, "failed to find pivot for row %d\n", i);
    }
  }

  return pivots;
}

char StabTable_check_row_in_table_image(StabTable * table, int * pivots, unsigned char * row, unsigned char phase){
  /* //printf("Resources:\n"); */
  /* //StabTable_pprint_table(table, -1); */
  /* //printf("target row: "); */
  /* if(phase == 0){ */
  /*   printf("+"); */
  /* }else{ */
  /*   printf("-"); */
  /* } */
  /* for(int i = 0; i < table->n; i++){ */
  /*   if(row[i] == 0 && row[i+table->n] == 0){ */
  /*     printf("I"); */
  /*   } */
  /*   if(row[i] == 1 && row[i+table->n] == 0){ */
  /*     printf("X"); */
  /*   } */
  /*   if(row[i] == 0 && row[i+table->n] == 1){ */
  /*     printf("Z"); */
  /*   } */
  /*   if(row[i] == 1 && row[i+table->n] == 1){ */
  /*     printf("Y"); */
  /*   } */
  /* }printf("\n"); */

  for(int i = 0; i < 2*table->n; i++){
    if(row[i]){
      for(int idx = 0; idx < table->k; idx++){
        if(pivots[idx] == i){
          phase = StabTable_rowsum2(table, row, phase, idx);
        }
      }
    }
  }
  /* printf("Result: "); */
  /* if(phase == 0 ){ */
  /*   printf("+"); */
  /* }else{ */
  /*   printf("-"); */
  /* } */
  /* for(int i = 0; i < table->n; i++){ */
  /*   if(row[i] == 0 && row[i+table->n] == 0){ */
  /*     printf("I"); */
  /*   } */
  /*   if(row[i] == 1 && row[i+table->n] == 0){ */
  /*     printf("X"); */
  /*   } */
  /*   if(row[i] == 0 && row[i+table->n] == 1){ */
  /*     printf("Z"); */
  /*   } */
  /*   if(row[i] == 1 && row[i+table->n] == 1){ */
  /*     printf("Y"); */
  /*   } */
  /* }printf("\n"); */


  //if(phase != 0){
  //  return 0;
  //}

  for(int i = 0; i < 2*table->n; i++){
    if(row[i]){
      return 0;
    }
  }
  return 1;
}

void StabTable_pprint_equivalence_classes(EquivClassDecomp * classes){
  if(classes != NULL){
    for(int class_idx = 0; class_idx < classes->n_classes; class_idx++){
      printf("%d: (%d", class_idx, classes->classes[class_idx][0]);
      for(int q = 1; q < classes->class_sizes[class_idx]; q++){
        printf(", %d", classes->classes[class_idx][q]);
      }
      printf(")\n");
    }
  }

}


EquivClassDecomp StabTable_count_equivalence_classes(StabTable * state){

  EquivClassDecomp decomp;
  EquivClassDecomp_init(&decomp, state->n);
  int * pivots = StabTable_row_reduce(state);
  //StabTable_pprint_table(state, -1);
  StabTable * scratch_table = StabTable_new(state->n, state->k);
  for(int s = 0; s < state->k; s++){
    scratch_table->phases[s] = state->phases[s];
    for(int q = 0; q < state->n; q++){
      scratch_table->table[s][q] = state->table[s][q];
      scratch_table->table[s][q+state->n] = state->table[s][q+state->n];
    }
  }

  unsigned char * scratch_row = malloc(2*state->n);
  for(int q1 = 0; q1 < state->n; q1++){
    char found_class = 0;
    for(int class_idx = 0; class_idx < decomp.n_classes; class_idx++){
      int q2 = decomp.classes[class_idx][0];
      //printf("before swapping %d, %d\n", q1,q2);
      //StabTable_pprint_table(scratch_table, -1);
      //printf("after swapping %d, %d\n", q1,q2);
      StabTable_swap_no_circ(scratch_table, q1, q2);
      //StabTable_pprint_table(scratch_table, -1);
      //printf("\n");
      char everything_in_image = 1;
      for(int s = 0; s < state->k; s++){
        memcpy(scratch_row, scratch_table->table[s], 2*state->n);
        char is_in_image = StabTable_check_row_in_table_image(state, pivots, scratch_row, scratch_table->phases[s]);
        if(!is_in_image){
          everything_in_image = 0;
          break;
        }
      }
      StabTable_swap_no_circ(scratch_table, q2, q1);
      //printf("Invariant: %d\n", everything_in_image);
      if(everything_in_image){ // table invariant under our swap
        found_class = 1;
        EquivClassDecomp_add_to_class(&decomp, class_idx, q1);
        break;
      }
    }
    if(!found_class){
      EquivClassDecomp_add_class(&decomp, q1);
    }
  }

  //int count = decomp.n_classes;
  //StabTable_pprint_equivalence_classes(&decomp);
  //StabTable_pprint_equivalence_classes(&decomp);
  //EquivClassDecomp_free(&decomp);

  free(scratch_row);
  free(pivots);
  StabTable_free(scratch_table);
  return decomp;

}

void commutativity_diagram_add_row(int ** M, int n, int source, int dest, char multiply_source_by_variable){
  for(int i = 0; i < n; i++){
    if(M[source][i] < 0 || ((M[source][i] > 0) && (M[dest][i] > 0) && multiply_source_by_variable)){
      M[dest][i] = -2;
    }else if((M[source][i] > 0) && (M[dest][i] == 0) && multiply_source_by_variable){
      M[dest][i] = -1;
    }else if((M[source][i] > 0) && (M[dest][i] == -1) && multiply_source_by_variable){
      M[dest][i] = 0;
    }
    else if(M[source][i] == 1 && M[dest][i] >= 0 && !multiply_source_by_variable){
      M[dest][i] += 1;
      M[dest][i] %= 2;
    }
  }
}

int commutativity_diagram(StabTable * state, int measured_qubits, int t, int verbose){

  int ** M = (int **)calloc(measured_qubits + t, sizeof(int *));
  for(int i = 0; i < state->k; i++){
    M[i] = calloc(state->n, sizeof(int));
    for(int j = 0; j < state->n; j++){
      M[i][j] = state->table[i][j];
      if(i >= measured_qubits){
        M[i][j] = -M[i][j];
      }
    }
    if(i >= measured_qubits){
      M[i][i + (state->n-t) - measured_qubits] = 1;
    }
  }
  if(verbose){

    BinaryExpression ** M_print = calloc(state->k, sizeof(BinaryExpression *));
    for(int i = 0; i < state->k; i++){
      M_print[i] = calloc(state->n, sizeof(BinaryExpression));
      for(int j = 0; j < state->n; j++){
        if(M[i][j] == 0){
          M_print[i][j].data = NULL;
          M_print[i][j].capacity = 0;
          M_print[i][j].length = 0;
        }else{
          M_print[i][j].data = calloc(1, sizeof(uint_bitarray_t));
          M_print[i][j].capacity = 1;
          M_print[i][j].length = 1;
          if(M[i][j] == 1){
            M_print[i][j].data[0] = ZERO;
          }else if(M[i][j] == -1 && (i >= measured_qubits)){
            M_print[i][j].data[0] = (ONE << (i-measured_qubits));
          }
        }
      }
    }
    printf("Alg 3 before Gaussian eliminating\n");
    BE_pprint_matrix(M_print, state->n, state->k);
    for(int i = 0; i <state->k; i++){
      for(int j = 0; j <state->n; j++){
	free(M_print[i][j].data);
      }
      free(M_print[i]);
    }
  }

  int we_made_changes;
  do{
    we_made_changes = 0;
    //lets use column ops on the bottom left block
    int col2 = 0;
    for(int row2 = measured_qubits; row2 < measured_qubits+t; row2++){
      int pivot = -1;

      for(int i = col2; i < state->n -t; i++){
        if(M[row2][i] == -1){
          pivot = i;
          break;
        }
      }
      if(pivot >= 0){
        if(pivot != col2){
          for(int i = 0; i < state->k; i++){
            int scratch = M[i][pivot];
            M[i][pivot] = M[i][col2];
            M[i][col2] = scratch;
          }
        }
        for(int j = 0; j < state->n; j++){
          if(M[row2][j] == -1 && j != col2){
            for(int i = 0; i < state->k; i++){
              M[i][j] ^= M[i][col2];
            }
          }
        }
        col2 += 1;
      }
    }
    //printf("%d\n", col2);
    //any columns in the bottom left block which are subsets of y values from the bottom right block we use to delete ys in the bottom right
    /*
      for(int i = 0; i < col2; i++){
      int index_count = 0;
      for(int j = measured_qubits; j < state->k; j++){
      if(M[j][i] == -1){
      nonzero_indices[index_count] = j;
      index_count += 1;
      }
      }

      for(int j = state->n-t; j < state->n; j++){
      int subset = 1;
      for(int k = 0; k < index_count; k++){
      if(M[nonzero_indices[k]][j] != -1){
      subset = 0;
      break;
      }
      }
      if(subset && (index_count > 0)){

      we_made_changes = 1;
      for(int k = 0; k < state->k; k++){
      M[k][j] ^= M[k][i];
      }

      }
      }
      }
    */
    //free(nonzero_indices);
    int row = 0;

    //use row operations to simplify top right block
    for(int col = state->n - t; col < state->n; col++){
      int pivot = -1;
      for(int i = row; i < measured_qubits; i++){
        if(M[i][col] > 0){
          pivot = i;
          break;
        }
      }
      if(pivot >= 0){
        if(pivot != row){
          int * scratch = M[pivot];
          M[pivot] = M[row];
          M[row] = scratch;
        }
        for(int i = 0; i < measured_qubits; i++){
          if((M[i][col] != 0) && i != row){
            for(int j = 0; j < state->n; j++){
              M[i][j] ^= M[row][j];
            }
          }
        }
        row += 1;
      }
    }


    //any rows in the top right block which are subsets of y values from the bottom right block we use to delete ys in the bottom right
    int * nonzero_indices = malloc(t*sizeof(int));
    for(int i = 0; i < row; i++){
      int index_count = 0;
      for(int j = state->n-t; j < state->n; j++){
        if(M[i][j] == 1){
          nonzero_indices[index_count] = j;
          index_count += 1;
        }
      }

      for(int j = measured_qubits; j < state->k; j++){
        int subset = 1;
        for(int k = 0; k < index_count; k++){
          if(M[j][nonzero_indices[k]] != -1){
            subset = 0;
            break;
          }
        }
        if(subset && (index_count > 0)){
          we_made_changes = 1;
          for(int k = 0; k < state->n; k++){
            M[j][k] ^= (-M[i][k]);
          }
        }
      }
    }
    free(nonzero_indices);

  }while(we_made_changes);
  /*
    for(int j = 0; j < state->k; j++){
    if(j == measured_qubits){
    putc('\n', stdout);
    }
    for(int i = 0; i < state->n; i++){
    if(i == state->n - t){
    putc(' ', stdout);
    }
    if(M[j][i] == 0){
    putc('0', stdout);
    }
    if(M[j][i] > 0){
    putc('1', stdout);
    }
    if(M[j][i] < 0 ){
    putc('*', stdout);
    }

    //printf("%d ", M[j][i]);
    }
    putc('\n', stdout);
    }*/

  BinaryExpression ** M2 = calloc(state->k, sizeof(BinaryExpression *));
  for(int i = 0; i < state->k; i++){
    M2[i] = calloc(state->n, sizeof(BinaryExpression));
    for(int j = 0; j < state->n; j++){
      if(M[i][j] == 0){
        M2[i][j].data = NULL;
        M2[i][j].capacity = 0;
        M2[i][j].length = 0;
      }else{
        M2[i][j].data = calloc(1, sizeof(uint_bitarray_t));
        M2[i][j].capacity = 1;
        M2[i][j].length = 1;
        if(M[i][j] == 1){
          M2[i][j].data[0] = ZERO;
        }else if(M[i][j] == -1 && (i >= measured_qubits)){
          M2[i][j].data[0] = (ONE << (i-measured_qubits));
        }
      }
    }
  }
  int rows = 0;
  for(int i = state->n-t; i < state->n; i++){
    int pivot = -1;
    for(int j = rows; j < state->k; j++){
      //fprintf(stderr, "%d, %d\n", i, j);
      if(M2[j][i].length == 1 && M2[j][i].data[0] == ZERO){
        pivot = j;
        break;
      }
    }
    if(pivot >= 0){
      if(pivot != rows){
        BinaryExpression * scratch = M2[pivot];
        M2[pivot] = M2[rows];
        M2[rows] = scratch;
      }
      for(int j = 0; j < state->k; j++){
        if(M2[j][i].length != 0 && j != rows){
          for(int k = 0; k < state->n; k++){
            BE_add_poly_mult(&M2[rows][k], &M2[j][k], &M2[j][i]);
          }
        }
      }
      rows += 1;
    }
  }
  
  /*
    printf("++++++++++++++++\n");
    for(int i = 0; i < state->k; i++){
    if(i == measured_qubits){
    putc('\n', stdout);
    }
    for(int j = 0; j < state->n; j++){
    if(j == state->n - t){
    putc(' ', stdout);
    }
    BE_print_summary(&M2[i][j]);
    }
    putc('\n', stdout);
    }

    printf("rows = %d\n", rows);
  */
  int cols = 0;
  //first row-reduce without messing with pivots who have complicated expressions in their columns
  for(int i = 0; i < state->n-t; i++){
    int pivot = -1;
    for(int j = rows; j < state->k; j++){
      if((M2[j][i].length == 1) &&  (M2[j][i].data[0] == 0)){
        /*
          int found_complicated_expression = 0;
          for(int k = j+1; k < state->k; k++){
          if((M2[k][i].length > 1) || (M2[k][i].length == 1 && M2[k][i].data[0] != ZERO)){
          found_complicated_expression = 1;
          break;
          }
          }
        */
        //if(!found_complicated_expression){
        pivot = j;
        break;
        //}
      }
    }
    if(pivot >= 0){
      //printf("found pivot: rows = %d, cols = %d, p = %d, i = %d len = %d:   ", rows,cols, pivot, i, M2[pivot][i].length);
      //BE_pprint(&M2[pivot][i]);printf(",   ");
      if(pivot != rows){
        BinaryExpression * scratch = M2[pivot];
        M2[pivot] = M2[rows];
        M2[rows] = scratch;
      }
      if(i != cols){
        BinaryExpression scratch;
        for(int j = 0; j < state->k; j++){
          scratch = M2[j][i];
          M2[j][i] = M2[j][cols];
          M2[j][cols] = scratch;
        }
      }
      //BE_pprint(&M2[rows][cols]);printf("\n");
      for(int j = 0; j < state->k; j++){
        if(j != rows && M2[j][cols].length != 0){
          //printf("adding row %d to %d to zero elem %d\n", rows, j, cols);
          //for(int k = 0; k < state->n; k++){
          //  BE_pprint(&M2[rows][k]); printf(" ");
          //}
          //printf("\n");
          //for(int k = 0; k < state->n; k++){
          //  BE_pprint(&M2[j][k]); printf(" ");
          //}
          //printf("\n");

          for(int k = 0; k < state->n; k++){
            BE_add_poly_mult(&M2[rows][k], &M2[j][k], &M2[j][cols]);
          }
          //if(M2[j][cols].length != 0){
          //  printf("hello\n");
          //  for(int k = 0; k < state->n; k++){
          //    BE_pprint(&M2[j][k]); printf(" ");
          //  }
          //}
        }
      }
      rows += 1;
      cols += 1;
    }

  }
  //printf("%d, %d\n",rows,cols);
  /*
  //now give up and try to row-reduce everything else
  for(int i = cols; i < state->n-t; i++){
  int pivot = -1;
  for(int j = rows; j < state->k; j++){
  if((M2[j][i].length == 1) &&  (M2[j][i].data[0] == 0)){
  printf("pivot: i = %d j = %d\n", i, j);
  pivot = j;
  break;
  }
  }
  if(pivot >= 0){
  if(pivot != rows){
  BinaryExpression * scratch = M2[pivot];
  M2[pivot] = M2[rows];
  M2[rows] = scratch;
  }
  if(i != cols){
  BinaryExpression scratch;
  for(int j = 0; j < state->k; j++){
  scratch = M2[j][i];
  M2[j][i] = M2[j][cols];
  M2[j][cols] = scratch;
  }
  }

  for(int j = 0; j < state->k; j++){
  if(j != rows && M2[j][cols].length != 0){
  for(int k = 0; k < state->n; k++){
  BE_add_poly_mult(&M2[rows][k], &M2[j][k], &M2[j][cols]);
  }
  }
  }
  rows += 1;
  cols += 1;
  }

  }
  */
  /*
    printf("\n\n");
    printf("-----------------------\n");
    printf("%d; %d\n",rows,cols);
    for(int i = 0; i < state->k; i++){
    if(i == measured_qubits){
    putc('\n', stdout);
    }
    for(int j = 0; j < state->n; j++){
    if(j == state->n - t){
    putc(' ', stdout);
    }
    BE_print_summary(&M2[i][j]);
    }
    putc('\n', stdout);
    }
    printf("-----------------------\n");
  */
  if(verbose){
    printf("Alg 3 after Gaussian eliminating\n");
    BE_pprint_matrix(M2, state->n, state->k);
  }
  for(int i = 0; i < state->k; i++){
    free(M[i]);
    for(int j = 0; j < state->n; j++){
      //BE_print_summary(&M2[i][j]);
      free(M2[i][j].data);
    }
    free(M2[i]);
  }
  free(M2);
  free(M);

  /*
    BinaryExpression one;
    one.data = malloc(sizeof(uint_bitarray_t));
    one.length = 1;
    one.capacity = 1;
    one.data[0] = 0;
    BinaryExpression y1;
    y1.data = malloc(sizeof(uint_bitarray_t));
    y1.length = 1;
    y1.capacity = 1;
    y1.data[0] = ONE;
    BinaryExpression y2;
    y2.data = malloc(sizeof(uint_bitarray_t));
    y2.length = 1;
    y2.capacity = 1;
    y2.data[0] = (ONE << 1);

    BinaryExpression poly;
    poly.data = malloc(2*sizeof(uint_bitarray_t));
    poly.length = 2;
    poly.capacity = 2;
    poly.data[0] = 0;
    poly.data[1] = ONE | (ONE << 1 );
    BE_pprint(&poly);printf("\n");

    BE_add_poly_mult(&one, &poly, &poly);printf("\n");
    BE_pprint(&poly);printf("\n");


    printf("\n\n%d\n\n", rows);
  */
  return rows;
}

int commutativity_diagram_old(StabTable * state, int measured_qubits, int t){
  //we con
  //qubits = 3
  //measured_qubits = 2
  //t = 3
  //seed = 620
  //
  //    | Z0...Z3   | Z4...Z6
  //----+-----------+-----------
  //P0  | 101       | 100
  //P1  | 010       | 100
  //----+-----------+-----------
  //Q0  | 000       | 100
  //Q1  | ***       | 010
  //Q2  | **0       | *01

  int ** M = (int **)calloc(measured_qubits + t, sizeof(int *));
  for(int i = 0; i < state->k; i++){
    M[i] = calloc(state->n, sizeof(int));
    for(int j = 0; j < state->n; j++){
      M[i][j] = state->table[i][j];
      if(i >= measured_qubits){
        M[i][j] = -M[i][j];
      }
    }
    if(i >= measured_qubits){
      M[i][i + t - measured_qubits] = 1;
    }
  }

  int ** M2 = (int **)calloc(measured_qubits + t, sizeof(int *));
  for(int i = 0; i < state->k; i++){
    M2[i] = calloc(state->n, sizeof(int));
    for(int j = 0; j < state->n; j++){
      M2[i][j] = M[i][j];
    }
  }

  /*
    for(int j = 0; j < state->k; j++){
    if(j == measured_qubits){
    putc('\n', stdout);
    }
    for(int i = 0; i < state->n; i++){
    if(i == state->n - t){
    putc(' ', stdout);
    }
    if(M[j][i] == 0){
    putc('0', stdout);
    }
    if(M[j][i] > 0){
    putc('1', stdout);
    }
    if(M[j][i] < 0 ){
    putc('*', stdout);
    }

    //printf("%d ", M[j][i]);
    }
    putc('\n', stdout);
    }

    printf("--------------\n");
  */

  int row = 0;

  //first Gaussian elimination
  for(int col = state->n - t; col < state->n; col++){
    int pivot = -1;
    for(int i = row; i < measured_qubits; i++){
      if(M[i][col] > 0){
        pivot = i;
        break;
      }
    }
    if(pivot >= 0){
      if(pivot != row){
        int * scratch = M[pivot];
        M[pivot] = M[row];
        M[row] = scratch;
      }
      for(int i = 0; i < measured_qubits; i++){
        if((M[i][col] != 0) && i != row){
          for(int j = 0; j < state->n; j++){
            M[i][j] ^= M[row][j];
          }
        }
      }
      row += 1;
    }
  }

  //lets use column ops on the bottom left block
  int col2 = 0;
  for(int row2 = measured_qubits; row2 < measured_qubits+t; row2++){
    int pivot = -1;

    for(int i = col2; i < state->n -t; i++){
      if(M[row2][i] == -1){
        pivot = i;
        break;
      }
    }
    if(pivot >= 0){
      if(pivot != col2){
        for(int i = 0; i < state->k; i++){
          int scratch = M[i][pivot];
          M[i][pivot] = M[i][col2];
          M[i][col2] = scratch;
        }
      }
      for(int j = 0; j < state->n; j++){
        if(M[row2][j] == -1 && j != col2){
          for(int i = 0; i < state->k; i++){
            M[i][j] ^= M[i][col2];
          }
        }
      }
      col2 += 1;
    }
  }
  /*
    printf("@@@@@@@@@@@@@@@@@\n");
    for(int j = 0; j < state->k; j++){
    if(j == measured_qubits){
    putc('\n', stdout);
    }
    for(int i = 0; i < state->n; i++){
    if(i == state->n - t){
    putc(' ', stdout);
    }
    if(M[j][i] == 0){
    putc('0', stdout);
    }
    if(M[j][i] > 0){
    putc('1', stdout);
    }
    if(M[j][i] < 0 ){
    putc('*', stdout);
    }

    //printf("%d ", M[j][i]);
    }
    putc('\n', stdout);
    }
    printf("@@@@@@@@@@@@@@@@@@@\n");
  */

  //second Gaussian elimination
  row = 0;

  int * banned_positions = malloc(t*sizeof(int));

  for(int col = state->n - t; col < state->n; col++){
    int pivot = -1;
    int banned_positions_count = 0;
    for(int i = 0; i < t; i++){
      banned_positions[i] = -1;
    }

    for(int i = row; i < t; i++){
      if(M[i+measured_qubits][col] < 0 && (M[i+measured_qubits][state->n-t+i] == 1)){
        banned_positions[banned_positions_count] = state->n-t+i;
        banned_positions_count += 1;
      }
    }



    for(int i = row; i < state->k; i++){
      if(M[i][col] > 0){
        char hits_banned_position = 0;
        for(int j = 0; j < banned_positions_count; j++){
          if(M[i][banned_positions[j]] != 0){
            hits_banned_position = 1;
            break;
          }
        }


        if(!hits_banned_position){
          pivot = i;
          break;
        }
      }
    }
    if(pivot >= 0){
      if(pivot != row){
        int * scratch = M[pivot];
        M[pivot] = M[row];
        M[row] = scratch;
      }
      for(int i = 0; i < state->k; i++){
        if((M[i][col] != 0) && i != row){
          commutativity_diagram_add_row(M, state->n, row, i, M[i][col] < 0);
          M[i][col] = 0;
        }
      }
      row += 1;
    }else{
      /*
        printf("+++++++++++++++++\n");
        printf("row=%d col=%d\n", row, col-(state->n - t));
        for(int j = 0; j < state->k; j++){
        if(j == measured_qubits){
        putc('\n', stdout);
        }
        for(int i = 0; i < state->n; i++){
        if(i == state->n - t){
        putc(' ', stdout);
        }
        if(M[j][i] == 0){
        putc('0', stdout);
        }
        if(M[j][i] > 0){
        putc('1', stdout);
        }
        if(M[j][i] < 0 ){
        putc('*', stdout);
        }

        //printf("%d ", M[j][i]);
        }
        putc('\n', stdout);
        }
        printf("+++++++++++++++++\n");
      */


    }
  }

  //second Gaussian elimination
  for(int col = 0; col < state->n - t; col++){
    int pivot = -1;
    for(int i = row; i < state->k; i++){
      if(M[i][col] > 0){
        pivot = i;
        break;
      }
    }
    if(pivot >= 0){
      if(pivot != row){
        int * scratch = M[pivot];
        M[pivot] = M[row];
        M[row] = scratch;
      }
      for(int i = 0; i < state->k; i++){
        if((M[i][col] != 0) && i != row){
          commutativity_diagram_add_row(M, state->n, row, i, M[i][col] < 0);
          M[i][col] = 0;
        }
      }
      row += 1;
    }
  }


  /*
    if(row < t){
    for(int j = 0; j < state->k; j++){
    if(j == measured_qubits){
    putc('\n', stdout);
    }
    for(int i = 0; i < state->n; i++){
    if(i == state->n - t){
    putc(' ', stdout);
    }
    if(M2[j][i] == 0){
    putc('0', stdout);
    }
    if(M2[j][i] > 0){
    putc('1', stdout);
    }
    if(M2[j][i] < 0 ){
    putc('*', stdout);
    }

    //printf("%d ", M[j][i]);
    }
    putc('\n', stdout);
    }

    printf("--------------\n");

    printf("%d %d %d\n", state->n,t,row);
    for(int j = 0; j < state->k; j++){
    if(j == measured_qubits){
    putc('\n', stdout);
    }
    for(int i = 0; i < state->n; i++){
    if(i == state->n - t){
    putc(' ', stdout);
    }
    if(M[j][i] == 0){
    putc('0', stdout);
    }
    if(M[j][i] > 0){
    putc('1', stdout);
    }
    if(M[j][i] < 0 ){
    putc('*', stdout);
    }

    //printf("%d ", M[j][i]);
    }
    putc('\n', stdout);
    }

    }
  */
  for(int i = 0; i < state->k; i++){
    free(M[i]);
    free(M2[i]);

  }
  free(M);
  free(M2);
  free(banned_positions);
  return row;
}



int StabTable_zero_inner_product(StabTable * state, int w, int t){
  int s = 0;
  for(int q = 0; q < w; q++){
    int pivot = -1;
    for(int k = s; k < state->k; k++){
      if(state->table[k][q] == 1){
	pivot = k;
	break;
      }
    }
    if(pivot >= 0){
      if(s != pivot){
	StabTable_swap_rows(state, s, pivot);
      }
      for(int k = 0; k < state->k; k++){
	if((k != s) && (state->table[k][q] == 1)){
	  StabTable_rowsum(state, k, s);
	}
      }
      s += 1;
    }  
  }

  for(int q = state->n-t; q < state->n; q++){
    int pivot = -1;
    for(int k = s; k < state->k; k++){
      if(state->table[k][q] == 1){
	pivot = k;
	break;
      }
    }
    if(pivot >= 0){
      if(s != pivot){
	StabTable_swap_rows(state, s, pivot);
      }
      for(int k = 0; k < state->k; k++){
	if((k != s) && (state->table[k][q] == 1)){
	  StabTable_rowsum(state, k, s);
	}
      }
      s += 1;
    }  
  }
  int x_region_rank = s;
  
  return x_region_rank;
}

int StabTable_y_tilde_inner_prod_no_phase(StabTable * state, uint_bitarray_t y, char ** M){
  /* for(int q = 0; q < t; q++){ */
  /*   if((y>>q) & ONE){ */
  /*     StabTable_S(state, q); */
  /*   } */
  /*   StabTable_H(state, q); */
  /* } */


  //now do the inner product with 0
  for(int q = 0; q < state->n; q++){
    for(int k = 0; k < state->k; k++){
      M[q][k] = state->table[k][q+state->n] ^ (((y >> q) & ONE) & state->table[k][q]);
    }
  }

  //now we find the rank of M
  int row = 0;
  for(int col = 0; col < state->k; col++){
    int found_pivot = -1;
    for(int r = row; r < state->n; r++){
      if(M[r][col]){
        found_pivot = r;
        break;
      }
    }
    if(found_pivot >= 0){
      if(found_pivot != row){
        char * scratch = M[row];
        M[row] = M[found_pivot];
        M[found_pivot] = scratch;
      }
      for(int r = 0; r < state->n; r++){
        if((r != row) && M[r][col]){
          for(int c = 0; c < state->k; c++){
            M[r][c] ^= M[row][c];
          }
        }
      }
      row += 1;
    }
  }
  return state->k - row; // this is the nullity
}

int StabTable_y_tilde_inner_prod(StabTable * state, uint_bitarray_t y){
  /*
    for(int q = 0; q < state->n; q++){
    if((y >> q) & ONE){
    StabTable_H(state, q);
    StabTable_S(state, q);
    }else{
    StabTable_H(state, q);
    }
    }

    //https://arxiv.org/pdf/1210.6646.pdf

    int row = 0;
    for(int q = 0; q < state->n; q++){
    int pivot_row = -1;
    for(int i = row; i < state->k; i++){
    if(state->table[i][q]){
    pivot_row = i;
    break;
    }
    }
    if(pivot_row >= 0){
    if(pivot_row != row){
    StabTable_swap_rows(state, pivot_row, row);
    }
    for(int i = 0; i < state->k; i++){
    if((i != row) && state->table[i][q]){
    StabTable_rowsum(state, i, row);
    }
    }
    row += 1;
    }
    }
    int xy_rows = row;
    int orthogonality = 0;
    for(int q = 0; q < state->n; q++){
    int pivot_row = -1;
    for(int i = row; i < state->k; i++){
    if(state->table[i][q+state->n]){
    pivot_row = i;
    break;
    }
    }
    if(pivot_row >= 0){
    if(pivot_row != row){
    StabTable_swap_rows(state, pivot_row, row);
    }
    for(int i = 0; i < state->k; i++){
    if((i != row) && state->table[i][q]){
    StabTable_rowsum(state, i, row);
    }
    }

    if(table->phases[row] == 1){
    orthogonality = 1;
    break
    }
    row += 1;
    }
    }
    //now we seek the inner product tr(|0><0| state)
    for(int q = 0; q < state->n; q++){
    if((y >> q) & ONE){
    StabTable_S(state, q);
    StabTable_S(state, q);
    StabTable_S(state, q);
    StabTable_H(state, q);
    }else{
    StabTable_H(state, q);
    }
    }

    if(orthogonality){
    return -1;
    }
    return xy_rows;
  */
  return 0;

}
