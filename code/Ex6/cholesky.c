#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include "omp.h"
#include "cholesky.h"
#include <math.h>


int NumVezes;
//Parallel For
void cholesky_blocked_par_for(const int ts, const int nt, double* Ah[nt][nt])
{
   
   for (int k = 0; k < nt; k++) {

      // Diagonal Block factorization
      potrf (Ah[k][k], ts, ts);
      // Triangular systems
      #pragma omp parallel for
      for (int i = k + 1; i < nt; i++) {
         trsm (Ah[k][k], Ah[k][i], ts, ts);
      }

      // Update trailing matrix
      for (int i = k + 1; i < nt; i++) {
         #pragma omp parallel for
         for (int j = k + 1; j < i; j++) {
            gemm (Ah[k][i], Ah[k][j], Ah[j][i], ts, ts);
         }
         syrk (Ah[k][i], Ah[i][i], ts, ts);
      }

   }
}

void cholesky_blocked_par_task_without_depend(const int ts, const int nt, double* Ah[nt][nt])
{
   #pragma omp parallel
   #pragma omp single
   for (int k = 0; k < nt; k++) {

      // Diagonal Block factorization
      potrf (Ah[k][k], ts, ts);

      // Triangular systems
      for (int i = k + 1; i < nt; i++) {
         #pragma omp task
         trsm (Ah[k][k], Ah[k][i], ts, ts);
      }
      #pragma omp taskwait
      // Update trailing matrix
      for (int i = k + 1; i < nt; i++) {
         
         for (int j = k + 1; j < i; j++) {
            #pragma omp task
            gemm (Ah[k][i], Ah[k][j], Ah[j][i], ts, ts);
         }
         #pragma omp task
         syrk (Ah[k][i], Ah[i][i], ts, ts);
      }
      #pragma omp taskwait
   }
}

void cholesky_blocked_par_task_with_depend(const int ts, const int nt, double* Ah[nt][nt])
{
   
   #pragma omp parallel
   #pragma omp single
   for (int k = 0; k < nt; k++) {

      // Diagonal Block factorization
      #pragma omp task depend(out: Ah[k][k])
      potrf (Ah[k][k], ts, ts);

      // Triangular systems
      for (int i = k + 1; i < nt; i++) {
         #pragma omp task depend(in: Ah[k][k]) depend(out: Ah[k][i]) // TASK 1
         trsm (Ah[k][k], Ah[k][i], ts, ts);
      }
      
      // Update trailing matrix
      for (int i = k + 1; i < nt; i++) {
         for (int j = k + 1; j < i; j++) {
            #pragma omp task depend(in: Ah[k][i], Ah[k][j]) depend(inout: Ah[j][i]) // TASK 2
            gemm (Ah[k][i], Ah[k][j], Ah[j][i], ts, ts);
         }
         #pragma omp task depend(in: Ah[k][i]) depend(inout: Ah[i][i]) // TASK 3
         syrk (Ah[k][i], Ah[i][i], ts, ts);
      }
   }
}

//Sequential
void cholesky_blocked(const int ts, const int nt, double* Ah[nt][nt])
{
   for (int k = 0; k < nt; k++) {
      // Diagonal Block factorization
      potrf (Ah[k][k], ts, ts);

      // Triangular systems
      for (int i = k + 1; i < nt; i++) {
         trsm (Ah[k][k], Ah[k][i], ts, ts);
      }

      // Update trailing matrix
      for (int i = k + 1; i < nt; i++) {
         for (int j = k + 1; j < i; j++) {
            gemm (Ah[k][i], Ah[k][j], Ah[j][i], ts, ts);
         }
         syrk (Ah[k][i], Ah[i][i], ts, ts);
      }
   }
}

void print_matrix(int n, double * const matrix){
   printf("\t{\n\t");
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			printf("%.5f, ",matrix[i*n + j]);
		}
		printf("\n\t");
	}
	printf("};\n");
}

/*
    NOVAS FUNÇÕES
*/
double media(double myarr[], int numSize)
{
    double sum = 0.0;
    int i = 0;

    for(i = 0; i < numSize; i++)
        sum += myarr[i];

    return sum/numSize;
}

double variancia(double myarr[], int numSize)
{
    double sum = 0.0;
    double dev = 0.0;
    double med = media(myarr, numSize);
    int i = 0;

    for(i = 0; i<numSize; i++)
    {
        dev = myarr[i] - med;
        sum += (dev * dev);
    }

    return sum / numSize;
}

double desvio_padrao(double myarr[], int numSize )
{
    double v = variancia(myarr, numSize);
    return sqrt(v);
}

double findMax(double myarr[])
{
    int i;
    double max;
    max = myarr[0];
    // min = myarr[0];

    /*
     * Find maximum and minimum in all array elements.
     */
    for(i=1; i<NumVezes; i++)
    {
        /* If current element is greater than max */
        if(myarr[i] > max)
        {
            max = myarr[i];
        }
    }

    return max;
}

double findMin(double myarr[])
{
    int i;
    double min;
    min = myarr[0];

    /*
     * Find maximum and minimum in all array elements.
     */
    for(i=1; i<NumVezes; i++)
    {
        /* If current element is smaller than min */
        if(myarr[i] < min)
        {
            min = myarr[i];
        } 
    }

    return min;
}

int main(int argc, char* argv[])
{

   if ( argc < 5) {
      printf( "cholesky matrix_size block_size num_threads num_executions\n" );
      exit( -1 );
   }
   NumVezes = atoi(argv[4]);
   double times_seq[NumVezes];
   double gflops_seq[NumVezes];
   double times_for[NumVezes];
   double gflops_for[NumVezes];
   double times_without_depend[NumVezes];
   double gflops_without_depend[NumVezes];
   double times_with_depend[NumVezes];
   double gflops_with_depend[NumVezes];


   const int  n = atoi(argv[1]); // matrix size
   const int ts = atoi(argv[2]); // tile size
   int num_threads = atoi(argv[3]); // number of threads to use
   omp_set_num_threads(num_threads);
   // Allocate matrix
   double * const matrix = (double *) malloc(n * n * sizeof(double));
   assert(matrix != NULL);

   // Init matrix
   initialize_matrix(n, ts, matrix);

   // Allocate matrix
   double * const original_matrix = (double *) malloc(n * n * sizeof(double));
   assert(original_matrix != NULL);

   // Allocate matrix
   double * const expected_matrix = (double *) malloc(n * n * sizeof(double));
   assert(expected_matrix != NULL);

   const int nt = n / ts;

   // Allocate blocked matrix
   double *Ah[nt][nt];

   for (int i = 0; i < nt; i++) {
      for (int j = 0; j < nt; j++) {
         Ah[i][j] = malloc(ts * ts * sizeof(double));
         assert(Ah[i][j] != NULL);
      }
   }

   

   /*************************************************************************************************************
    * NOTE FOR STUDENTS: 
    * COPY the following code (between multiline comments, up to "End Parallel For") to invoke your versio
    * AND make the following changes:
    *  1. change "cholesky_blocked_par_for" to the name of your method
    *  2. change "par_for_time" and "par_for_gflops" for names that reflect your implementation (e.g. par_task_time)
    *     2.1. Don't forget to change the "par_for_time" variable that is used to calculate the gflops
    *  3. add two lines that print the time for your code (below in "Print Result" section)
    *************************************************************************************************************/


   for(int t = 0; t < NumVezes; t++) { 

   for (int i = 0; i < n * n; i++ ) {
      original_matrix[i] = matrix[i];
   }
   // Sequential
   convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);
   // warming up libraries
   cholesky_blocked(ts, nt, (double* (*)[nt]) Ah);
   convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);
   // done warming up
   float t1 = get_time();
   //run sequential version
   cholesky_blocked(ts, nt, (double* (*)[nt]) Ah);
   float t2 = get_time() - t1;
   //calculate timing metrics
   const float seq_time = t2;
   float seq_gflops = (((1.0 / 3.0) * n * n * n) / ((seq_time) * 1.0e+9));

   times_seq[t] = seq_time;
   gflops_seq[t] = seq_gflops;
   //saving matrix to the expected result matrix
   convert_to_linear(ts, nt, n, Ah, (double (*)[n]) matrix);
   for (int i = 0; i < n * n; i++ ) {
      expected_matrix[i] = matrix[i];
   }
   // End Sequential   








   /*****************************************************************************************************
    * Parallel For
    *****************************************************************************************************/
   //resetting matrix
   for (int i = 0; i < n * n; i++ ) {
      matrix[i] = original_matrix[i];
   }
   //require to work with blocks
   convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);
   t1 = get_time();
   //run parallel version using parallel fors
   cholesky_blocked_par_for(ts, nt, (double* (*)[nt]) Ah);
   t2 = get_time() - t1;
   //calculate timing metrics
   float par_for_time = t2;
   float par_for_gflops = (((1.0 / 3.0) * n * n * n) / ((par_for_time) * 1.0e+9));
   times_for[t] = par_for_time;
   gflops_for[t] = par_for_gflops;
   //asserting result, comparing the output to the expect matrix
   convert_to_linear(ts, nt, n, Ah, (double (*)[n]) matrix);
   assert_matrix(n,matrix,expected_matrix);

   /*****************************************************************************************************
    * End Parallel For
    *****************************************************************************************************/



   /*****************************************************************************************************
    * Parallel Task Without Dependency
    *****************************************************************************************************/
   //resetting matrix
   for (int i = 0; i < n * n; i++ ) {
      matrix[i] = original_matrix[i];
   }
   //require to work with blocks
   convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);
   t1 = get_time();
   //run parallel version using tasks without dependencies
   cholesky_blocked_par_task_without_depend(ts, nt, (double* (*)[nt]) Ah);
   t2 = get_time() - t1;
   //calculate timing metrics
   float par_task_without_dependency_time = t2;
   float par_task_without_dependency_gflops = (((1.0 / 3.0) * n * n * n) / ((par_task_without_dependency_time) * 1.0e+9));
   times_without_depend[t] = par_task_without_dependency_time;
   gflops_without_depend[t] = par_task_without_dependency_gflops;
   //asserting result, comparing the output to the expect matrix
   convert_to_linear(ts, nt, n, Ah, (double (*)[n]) matrix);
   assert_matrix(n,matrix,expected_matrix);

   /*****************************************************************************************************
    * End Parallel Task Without Dependency
    *****************************************************************************************************/





   /*****************************************************************************************************
    * Parallel Task With Dependency
    *****************************************************************************************************/
   //resetting matrix
   for (int i = 0; i < n * n; i++ ) {
      matrix[i] = original_matrix[i];
   }
   //require to work with blocks
   convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);
   t1 = get_time();
   //run parallel version using tasks with dependencies
   cholesky_blocked_par_task_with_depend(ts, nt, (double* (*)[nt]) Ah);
   t2 = get_time() - t1;
   //calculate timing metrics
   float par_task_with_dependency_time = t2;
   float par_task_with_dependency_gflops = (((1.0 / 3.0) * n * n * n) / ((par_task_with_dependency_time) * 1.0e+9));
   times_with_depend[t] = par_task_with_dependency_time;
   gflops_with_depend[t] = par_task_with_dependency_gflops;

   //asserting result, comparing the output to the expect matrix
   convert_to_linear(ts, nt, n, Ah, (double (*)[n]) matrix);
   assert_matrix(n,matrix,expected_matrix);

   /*****************************************************************************************************
    * End Parallel Task With Dependency
    *****************************************************************************************************/
   }

   // Print result
   /*
   printf( "====================== CHOLESKY RESULTS ======================\n" );
   printf( "  matrix size:                                     %dx%d\n", n, n);
   printf( "  block size:                                      %dx%d\n", ts, ts);
   printf( "  number of threads:                               %d\n", num_threads);
   printf( "  seq_time (s):                                    %f\n", seq_time);
   printf( "  seq_performance (gflops):                        %f\n", seq_gflops);
   printf( "  par_for_time (s):                                %f\n", par_for_time);
   printf( "  par_for_performance (gflops):                    %f\n", par_for_gflops);
   printf( "  par_task_without_dependency_time (s):            %f\n", par_task_without_dependency_time);
   printf( "  par_task_without_dependency_performance (gflops):%f\n", par_task_without_dependency_gflops);
   printf( "  par_task_with_dependency_time (s):               %f\n", par_task_with_dependency_time);
   printf( "  par_task_with_dependency_performance (gflops):   %f\n", par_task_with_dependency_gflops);
   printf( "==============================================================\n" );
   */
   printf( "====================== CHOLESKY RESULTS ======================\n" );
   printf( "  Tamanho da Matriz:                          %dx%d\n", n, n);
   printf( "  Tamanho do Bloco:                           %dx%d\n", ts, ts);
   printf( "  Número de Threads:                          %d\n", num_threads);
   printf( "  Número de execuções:                        %d\n", NumVezes);
   printf( "========================== SEQUENCIAL ========================\n" );
   printf( "  Tempo Máximo:                               %f\n", findMax(times_seq));
   printf( "  Tempo Mínimo:                               %f\n", findMin(times_seq));
   printf( "  Média dos tempos:                           %f\n", media(times_seq, NumVezes));
   printf( "  Desvio Padrão dos tempos:                   %f\n", desvio_padrao(times_seq, NumVezes));
   printf( "  Média gflops:                               %f\n", media(gflops_seq, NumVezes));
   printf( "==============================================================\n" );
   printf( "========================== Parallel For ========================\n" );
   printf( "  Tempo Máximo:                               %f\n", findMax(times_for));
   printf( "  Tempo Mínimo:                               %f\n", findMin(times_for));
   printf( "  Média dos tempos:                           %f\n", media(times_for, NumVezes));
   printf( "  Desvio Padrão dos tempos:                   %f\n", desvio_padrao(times_for, NumVezes));
   printf( "  Média gflops:                               %f\n", media(gflops_for, NumVezes));
   printf( "==============================================================\n" );
   printf( "======================= Task Without Dependency ====================\n" );
   printf( "  Tempo Máximo:                               %f\n", findMax(times_without_depend));
   printf( "  Tempo Mínimo:                               %f\n", findMin(times_without_depend));
   printf( "  Média dos tempos:                           %f\n", media(times_without_depend, NumVezes));
   printf( "  Desvio Padrão dos tempos:                   %f\n", desvio_padrao(times_without_depend, NumVezes));
   printf( "  Média gflops:                               %f\n", media(gflops_without_depend, NumVezes));
   printf( "==============================================================\n" );
   printf( "======================= Task With Dependency ====================\n" );
   printf( "  Tempo Máximo:                               %f\n", findMax(times_with_depend));
   printf( "  Tempo Mínimo:                               %f\n", findMin(times_with_depend));
   printf( "  Média dos tempos:                           %f\n", media(times_with_depend, NumVezes));
   printf( "  Desvio Padrão dos tempos:                   %f\n", desvio_padrao(times_with_depend, NumVezes));
   printf( "  Média gflops:                               %f\n", media(gflops_with_depend, NumVezes));
   printf( "==============================================================\n" );

   free(original_matrix);
   free(expected_matrix);
   // Free blocked matrix
   for (int i = 0; i < nt; i++) {
      for (int j = 0; j < nt; j++) {
         assert(Ah[i][j] != NULL);
         free(Ah[i][j]);
      }
   }
   // Free matrix
   free(matrix);

   return 0;
}

