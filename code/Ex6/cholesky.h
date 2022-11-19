#include <math.h>
#include <mkl/mkl.h>
#include <sys/time.h>
#include <sys/times.h>


double threshold = 0.1;

void dgemm_ (const char *transa, const char *transb, int *l, int *n, int *m, double *alpha,
             const void *a, int *lda, void *b, int *ldb, double *beta, void *c, int *ldc);
void dtrsm_ (char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha,
             double *a, int *lda, double *b, int *ldb);
void dtrmm_ (char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha,
             double *a, int *lda, double *b, int *ldb);
void dsyrk_ (char *uplo, char *trans, int *n, int *k, double *alpha, double *a, int *lda,
             double *beta, double *c, int *ldc);


void add_to_diag_hierarchical (double ** matrix, const int ts, const int nt, const float alpha)
{
	for (int i = 0; i < nt * ts; i++)
		matrix[(i/ts) * nt + (i/ts)][(i%ts) * ts + (i%ts)] += alpha;
}

void add_to_diag(double * matrix, const int n, const double alpha)
{
	for (int i = 0; i < n; i++)
		matrix[ i + i * n ] += alpha;
}

float get_time()
{
	static double gtod_ref_time_sec = 0.0;

	struct timeval tv;
	gettimeofday(&tv, NULL);

	// If this is the first invocation of through dclock(), then initialize the
	// "reference time" global variable to the seconds field of the tv struct.
	if (gtod_ref_time_sec == 0.0)
		gtod_ref_time_sec = (double) tv.tv_sec;

	// Normalize the seconds field of the tv struct so that it is relative to the
	// "reference time" that was recorded during the first invocation of dclock().
	const double norm_sec = (double) tv.tv_sec - gtod_ref_time_sec;

	// Compute the number of seconds since the reference time.
	const double t = norm_sec + tv.tv_usec * 1.0e-6;

	return (float) t;
}

void initialize_matrix(const int n, const int ts, double *matrix)
{
	int ISEED[4] = {0,0,0,1};
	int intONE=1;

	for (int i = 0; i < n*n; i+=n) {
		dlarnv_(&intONE, &ISEED[0], &n, &matrix[i]);
	}

	for (int i=0; i<n; i++) {
		for (int j=0; j<n; j++) {
			matrix[j*n + i] = matrix[j*n + i] + matrix[i*n + j];
			matrix[i*n + j] = matrix[j*n + i];
		}
	}

	add_to_diag(matrix, n, (double) n);
}

static void gather_block(const int N, const int ts, double *Alin, double *A)
{
	for (int i = 0; i < ts; i++)
		for (int j = 0; j < ts; j++) {
			A[i*ts + j] = Alin[i*N + j];
		}
}

static void scatter_block(const int N, const int ts, double *A, double *Alin)
{
	for (int i = 0; i < ts; i++)
		for (int j = 0; j < ts; j++) {
			Alin[i*N + j] = A[i*ts + j];
		}
}

static void convert_to_blocks(const int ts, const int DIM, const int N, double Alin[N][N], double *A[DIM][DIM])
{
	for (int i = 0; i < DIM; i++)
		for (int j = 0; j < DIM; j++) {
			gather_block ( N, ts, &Alin[i*ts][j*ts], A[i][j]);
		}
}

static void convert_to_linear(const int ts, const int DIM, const int N, double *A[DIM][DIM], double Alin[N][N])
{
	for (int i = 0; i < DIM; i++)
		for (int j = 0; j < DIM; j++) {
			scatter_block ( N, ts, A[i][j], (double *) &Alin[i*ts][j*ts]);
		}
}

static double * malloc_block (const int ts)
{
	double * const block = (double *) malloc(ts * ts * sizeof(double));
    assert(block != NULL);

	return block;
}




static void potrf(double * const A, int ts, int ld)
{
   static int INFO;
   static const char L = 'L';
   dpotrf_(&L, &ts, A, &ld, &INFO);
}

static void trsm(double *A, double *B, int ts, int ld)
{
   static char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
   static double DONE = 1.0;
   dtrsm_(&RI, &LO, &TR, &NU, &ts, &ts, &DONE, A, &ld, B, &ld );
}

static void syrk(double *A, double *B, int ts, int ld)
{
   static char LO = 'L', NT = 'N';
   static double DONE = 1.0, DMONE = -1.0;
   dsyrk_(&LO, &NT, &ts, &ts, &DMONE, A, &ld, &DONE, B, &ld );
}

static void gemm(double *A, double *B, double *C, int ts, int ld)
{
   static const char TR = 'T', NT = 'N';
   static double DONE = 1.0, DMONE = -1.0;
   dgemm_(&NT, &TR, &ts, &ts, &ts, &DMONE, A, &ld, B, &ld, &DONE, C, &ld);
}



static void assert_matrix(int n, double * const matrix, double * const expected){
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
         double diff = fabs(matrix[i*n + j]- expected[i*n + j]);
         if (diff > threshold){
				printf("Wrong value at position [%d,%d], expected %.5f, but got %.5f instead (diff. of %.5f)\n",i,j,expected[i*n + j],matrix[i*n + j],diff);
                exit(-1);
			}
		}
	}
}
