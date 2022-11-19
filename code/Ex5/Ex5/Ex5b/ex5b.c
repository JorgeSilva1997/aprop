#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>

/**
 * Matrix multiplication: A[L,M]* B[M,N]
 **/
#define N 1026
#define BS 32
#define MIN_RAND 0
#define MAX_RAND 100

int NumVezes;
int M[N][N];
int original_M[N][N];
int correct_M[N][N];

void fill(int* matrix, int height,int width);
void print(int* matrix,int height,int width);
void copy_to_M();
void setup_correct_M();
void assert(int M[N][N],int expected[N][N]);

/**
 * The sequential version
*/
void seq()
{
    int i, j;
    for(i = 1; i < N - 1; i++)
        for(j = 1; j < N - 1; j++)
           M[i][j] = (M[i][j-1] + M[i-1][j] + M[i][j+1] + M[i+1][j])/4.0;
}

/**
 * The parallel version
*/
void par()
{
    //TODO - parallelize me
    #pragma omp parallel
    #pragma omp single 
    {
        int i, j, ti, tj;
        int num_bloc = (N - 2) / BS;
        for(ti = 0; ti < num_bloc; ti++)
            for(tj = 0; tj < num_bloc; tj++)
            {   
                int init_i = 1 + ti * BS;
                int init_j = 1 + tj * BS;
                
                #pragma omp task depend(in:M[init_i - BS][init_j], M[init_i][init_j - BS]) depend(out:M[init_i][init_j])    
                for(i = init_i; i < 1 + (ti + 1) *BS; i++)
                    for(j = init_j; j < 1 + (tj + 1) *BS; j++)
                        M[i][j] = (M[i][j-1] + M[i-1][j] + M[i][j+1] + M[i+1][j])/4.0;
            }
    }                             
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




int main(int argc, char *argv[])
{   

    if ( argc < 2) {
      printf( "Insira o numero de vezes para o loop\n" );
      exit( -1 );
    }

    if((N - 2) % BS != 0){
        fprintf(stderr,"[ERROR] N is not multiple of BS: %d %% %d = %d\n", N,BS,N%BS);
        exit(1);
    }
    ;
    srand(time(NULL));
    //Fill A and B with random ints
    fill((int *)original_M,N,N);
    copy_to_M();
    //Run sequential version to compare time 
    //and also to have the correct result
    double begin = omp_get_wtime();
    seq();
    double end = omp_get_wtime();
    double sequential_time = end - begin;
     //save correct result in global variable 'correct_M'
    setup_correct_M();

    NumVezes = atoi(argv[1]); 
    double times[NumVezes]; 
    for(int i = 0; i < NumVezes; i++) {
        // reset matrix
        copy_to_M();
        // //invoke your parallel code here. it should be very similar to the previous code
        //e.g.:
        begin = omp_get_wtime();
        par();
        end = omp_get_wtime();
        double parallel_time = end - begin;
        times[i] = parallel_time;
        //compare your result invoking the following code (just uncomment the code):
        assert(M,correct_M);
    }

    /*
        Prints
    */
    printf("- ================ Pretty Print ================ -\n");
    printf("\tNº de vezes executado =   %d\n", NumVezes);
    printf("\tTempo máximo =            %f\n", findMax(times));
    printf("\tTempo mínimo =            %f\n", findMin(times));
    printf("\tMédia dos tempos =        %f\n", media(times, NumVezes));
    printf("\tDesvio Padrao =           %f\n", desvio_padrao(times, NumVezes));
    printf("\tTempo Execução Sequencial=%f\n", sequential_time);
    printf("- ============================================== -\n");
}

void copy_to_M(){
    for (int l = 0; l < N; l++)
    {
        for (int n = 0; n < N; n++)
        {
            M[l][n] = original_M[l][n] ;
        }
    }
}

void fill(int* matrix, int height,int width){
    for (int l = 0; l < height; l++)
    {
        for (int n = 0; n < width; n++)
        {
            *((matrix+l*width) + n) = MIN_RAND + rand()%(MAX_RAND-MIN_RAND+1);
        }
    }
}

void print(int* matrix,int height,int width){
    
    for (int l = 0; l < height; l++)
    {
        printf("[");
        for (int n = 0; n < width; n++)
        {
            printf(" %5d",*((matrix+l*width) + n));
        }
        printf(" ]\n");
    }
}

void assert(int C[N][N],int expected[N][N]){
    for (int l = 0; l < N; l++)
    {
        for (int n = 0; n < N; n++)
        {
            if(C[l][n] != expected[l][n]){
                printf("Wrong value at position [%d,%d], expected %d, but got %d instead\n",l,n,expected[l][n],C[l][n]);
                exit(-1);
            }
        }
        
    }
}

void setup_correct_M(){
    
    for (int l = 0; l < N; l++)
    {
        for (int n = 0; n < N; n++)
        {
            correct_M[l][n] = M[l][n];
        }
    }
}
