// File:  C_H2MM.h
// 
// fwd_back_photonbyphoton
#ifdef _WIN32
#include <windows.h>
#endif
typedef struct
{
	size_t nphot;
	size_t *delta;
	size_t *det;
} phstream;

struct temp
{
	long len_burst;
	unsigned long long *times;
	long *detectors;
	struct temp *next;
};
typedef struct temp temps;

typedef struct
{
	size_t nstate;
	size_t ndet;
	size_t nphot;
	size_t niter;
	size_t conv;
	double *prior;
	double *trans;
	double *obs;
	double loglik;
} h2mm_mod;

typedef struct
{
	phstream *phot;
	size_t max_phot;
	size_t *cur_burst;
	size_t num_burst;
	size_t sk; // number of states, indexing chosen to match
	size_t sj; // square of the number of states
	size_t si; // cube of the number of states
	size_t sT; // fourth power of number of states
	double *Rho;
	double *A;
	h2mm_mod *current;
	h2mm_mod *new;
#ifdef linux
	pthread_mutex_t *h2mm_lock;
#endif
} fback_vals;

// C_H2MM
typedef struct
{
	size_t max_iter;
	size_t num_cores;
	double max_time;
	double min_conv;
} lm;


// rho_calc
typedef struct // pwrs is a structure that contains pointers to the A (transmat) and Rho arrays, plus information on dimensions and which values are to be calculated
{
	size_t max_pow;
	size_t sT;
	size_t si;
	size_t sj; // note that this is also the stride for the delta t in the A array
	size_t sk; // note that this is also the number of states
	size_t tv;
	size_t tq;
	size_t td;
	size_t *pow_list;
	double *A;
	double *Rho;
} pwrs;

// viterbi
typedef struct
{
	size_t nphot;
	size_t nstate;
	double loglik;
	size_t *path;
	double *scale;
} ph_path;


typedef struct
{
	size_t si;
	size_t sT;
	double *A;
	phstream *phot;
	ph_path *path;
	h2mm_mod *model;
} vit_vals;


// C_H2MM

pwrs* get_deltas(unsigned long num_burst, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, phstream *b);

h2mm_mod* C_H2MM(unsigned long num_bursts, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, h2mm_mod *in_model, lm *limits);

// fwd_back_photonbyphoton
#ifdef linux
void* fwd_back_PhotonByPhoton(void* burst);
#elif _WIN32
DWORD WINAPI fwd_back_PhotonByPhoton(void* burst);
#endif

void h2mm_normalize(h2mm_mod *model_params);

h2mm_mod* compute_ess_dhmm(size_t num_bursts, phstream *b, pwrs *powers, h2mm_mod *in, lm *limits);

// rho_calc

void* rhoulate(void *vals);

void* rho_all(size_t nstate, double* transmat, pwrs *powers);

// viterbi
#ifdef linux
void* viterbi_burst(void* in_vals);
#elif _WIN32
DWORD WINAPI viterbi_burst(void* in_vals);
#endif
int viterbi(unsigned long num_bursts, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, h2mm_mod *model, ph_path *path_array);

// C_H2MM_interface

temps* burst_read(char *fname, size_t *n);

h2mm_mod* h2mm_read(char *fname);
