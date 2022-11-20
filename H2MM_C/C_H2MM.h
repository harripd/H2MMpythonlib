// File:  C_H2MM.h
// Author: Paul David Harris
// Purpose: Header files for H2MM and H2MM Viterbi algorithm
// Date Created: 13 Feb 2021
// Date Modified: 31 Oct 2022

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#elif _WIN32
#include <windows.h>
#endif

// typedefs for fwd_back_photonbyphoton.c

// Stucture for a single burst
typedef struct
{
	unsigned long nphot; // size of burst (# of photons
	unsigned long *delta; // pointer to array of deltas, size fo nphot
	unsigned long *det; // pointer to array of photon stream indexes, size of nphot
} phstream;

// Structure for building linked list fo bursts, before calculating deltas
struct temp
{
	unsigned long len_burst; // size of burst
	unsigned long long *times; // absolute arrival time
	long *detectors; // detector index
	struct temp *next; // pointer to next burst
};
typedef struct temp temps;

// Structure for h2mm model
typedef struct
{
	unsigned long nstate; // number of states in the h2mm model
	unsigned long ndet; // number of photon streams in the model
	unsigned long nphot; // number of photons in data set, filled out by C_H2MM function
	unsigned long niter; // number of iterations to converged, filled out by compute_ess_dhmm function
	unsigned long conv; // converged status, filled out by compute_ess_dhmm function- 1 is converged, 0 is not converged, others indicate errors
	double *prior; // prior array (1D), size: nstate
	double *trans; // trans array (2D), size: nstate x nstate
	double *obs; // obs array (2D), size: (ndet x nstate)
	double loglik; // loglike, updated by fwd_back_PhotonByPhoton threads
} h2mm_mod;

typedef struct
{
	unsigned long cur_burst; // next burst to work on
	unsigned long num_burst; // total number of bursts in set
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_t *burst_mutex; // mutex for checking on cur_burst
#elif _WIN32
	HANDLE burst_mutex; // mutex for checking on cur_burst
#endif
} brst_mutex;

// structure contains all inputs for fwd_back_PhotonByPhoton
typedef struct
{
	phstream *phot; // array of phstream structures (burst data)
	h2mm_mod *current; // the h2mm_mod from last iteration
	h2mm_mod *new; // the new h2mm_mod being generated in current iteration
	brst_mutex *burst_lock;
	unsigned long max_phot; // the size of the larges burst, used for allocating various arrays
	unsigned long sk; // number of states, indexing chosen to match
	unsigned long sj; // square of the number of states
	unsigned long si; // cube of the number of states
	unsigned long sT; // fourth power of number of states
	double *Rho; // Rho array, an sT x sk x sk x sk x sk array
	double *A; // contains powers of transition matrix, a sT x sk x sK array
	// internal arrays, just to avoid having to calloc / free over and over
	double* alpha; // no need to zero
	double* beta; // no need to zero
	double* b; // no need to zero
	double** gamma; // no need to zero
	double* xi_temp; // no need to zero
	double* xi_summed; // zero after each iteration 
	double* obs_temp; // zero after each iteration 
	double* prior; // zero after each iteration
	double loglik; // zero after each iteration
	unsigned long llerror; // If a NAN or other was encountered in the calculation
} fback_vals;



// C_H2MM.c structures

// the "limits" for calculations
typedef struct
{
	unsigned long max_iter; // number of iterations before algorithm stops optimization
	unsigned long num_cores; // number of cores to use (threads to spin up)
	double max_time; // maximum time in seconds before stoping optimization
	double min_conv; // minimum difference between loglik of iterations for algorithm to consider the model converged
} lm;

// limits for the h2mm_model parameters
typedef struct
{
	h2mm_mod *mins; // minimum values for h2mm_mod parameters
	h2mm_mod *maxs; // maximum values for h2mm_mod parameters
} h2mm_minmax;
// rho_calc.c structures 

// structure for the Rho and A array
typedef struct // pwrs is a structure that contains pointers to the A (transmat) and Rho arrays, plus information on dimensions and which values are to be calculated
{
	unsigned long max_pow; // maximum power, the maximum delta in the data set
	unsigned long sT; // stride for Rho of 0th index, cooresponds to the delta t of
	unsigned long si;
	unsigned long sj; // note that this is also the stride for the delta t in the A array
	unsigned long sk; // note that this is also the number of states
	unsigned long tv; // for future imporovments if Rho calculation is parallelized, idenfifies one of the two powers to use to calculate the next
	unsigned long tq; // same as tv, but the other power
	unsigned long td; // for future improvements if Rho calculation is parallelized, idenfifies destination power, tv and tq should match 
	double *A; // the A array, contains the powers of the tranistion probability matrix
	double *Rho; // the Rho array, as defined in H2MM
} pwrs;

typedef struct // pwrs is a structure that contains pointers to the A (transmat) and Rho arrays, plus information on dimensions and which values are to be calculated
{
	unsigned long max_pow; // maximum power, the maximum delta in the data set
	unsigned long sj; // note that this is also the stride for the delta t in the A array
	unsigned long sk; // note that this is also the number of states
	double *A; // the A array, contains the powers of the tranistion probability matrix
} trpow;

// viterbi.c structures 

// structure identifies a burst, and its viterbi values
typedef struct
{
	unsigned long nphot; // number of photons in burst
	unsigned long nstate; // number of states in model
	double loglik; // loglikelihood of the path being the actual path
	unsigned long *path; // most likely state of each photon
	double *scale; // likelihood of assignment of state of photon
} ph_path;

// structure contains all inputs for viterbi_burst thread

typedef struct
{
	unsigned long si; // the number of states, also stride of 1st dimension of A
	unsigned long sT; // stride of 0th dimension of A, gives power of A
	unsigned long max_phot; // size of larges burst, used for allocating arrays
	double *A; // A array, the powers of the transition probability matrix
	phstream *phot; // pointer to burst array, part of input
	ph_path *path; // pointer to viterbi path found by viterbi algorithm
	h2mm_mod *model; // h2mm model used in calculating the path
	brst_mutex *burst_lock; // mutex for updating cur_burst
} vit_vals;

// structure for threading loglik calculations
typedef struct
{
	double *ll;
	unsigned long **state;
	h2mm_mod *model;
	trpow *A;
	phstream *b;
	brst_mutex *burst_lock;
} pll_vals;


// Function definitions

// loop_functions.c definitions
// Outer level functions for optimizating models against data

int h2mm_optimize(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, h2mm_mod *in_model, h2mm_mod *out_model, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, void (*print_func)(unsigned long,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call);

int h2mm_optimize_array(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, h2mm_mod *in_model, h2mm_mod **out_models, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, void (*print_func)(unsigned long,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call);

int h2mm_optimize_gamma(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, h2mm_mod *in_model, h2mm_mod *out_model, double ***gamma, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, void (*print_func)(unsigned long,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call);

int h2mm_optimize_gamma_array(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, h2mm_mod *in_model, h2mm_mod **out_models, double ***gamma, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, void (*print_func)(unsigned long,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call);


// model_array.c functions
// Outer level functions for calcualating loglik and gamma of arrays of models

int calc_multi(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, unsigned long num_models, h2mm_mod *models, lm *limits);

int calc_multi_gamma(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, unsigned long num_models, h2mm_mod *models, double ****gamma, lm *limits);


// viterbi.c functions
// functions for performing viterbi analysis

#if defined(__linux__) || defined(__APPLE__)
void* viterbi_burst(void* in_vals);
#elif _WIN32
DWORD WINAPI viterbi_burst(void* in_vals);
#endif

int viterbi(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, h2mm_mod *model, ph_path *path_array, unsigned long num_cores);

// burst_threads.c functions
// functions for coordinating threads of h2mm optimizations

unsigned long get_next_burst(brst_mutex *burst);

#if defined(__linux__) || defined(__APPLE__)
void* fwd_bck_no_gamma(void* burst);
#elif _WIN32
DWORD WINAPI fwd_bck_no_gamma(void* burst);
#endif

#if defined(__linux__) || defined(__APPLE__)
void* fwd_bck_gamma(void* burst);
#elif _WIN32
DWORD WINAPI fwd_bck_gamma(void* burst);
#endif

#if defined(__linux__) || defined(__APPLE__)
void* fwd_only(void* burst);
#elif _WIN32
DWORD WINAPI fwd_only(void* burst);
#endif


// fwd_back.c functions
// core parts of the h2mm algorithm calculation

void fwd_calc(fback_vals* D, unsigned long cur_burst, unsigned long recursion_size, unsigned long recursion_stride);

void bck_calc(fback_vals* D, unsigned long cur_burst, unsigned long recursion_size, unsigned long recursion_stride, double* gamma);

void thread_update_h2mm_loglik(fback_vals* D);

void thread_update_h2mm_arrays(fback_vals* D);


// rho_calc.c functions
// functions for pre-calculating powers of transition matrix and Rho

trpow* transpow(unsigned long nstate, unsigned long maxdif, double* trans);

void* rhoulate(void *vals);

void* rho_all(unsigned long nstate, double* transmat, pwrs *powers);


// model_limits_funcs.c functions
// models for bouding models in optimizations

int h2mm_check_converged(h2mm_mod * new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limits);

int limit_check_only(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limit, void *lims);

int limit_revert(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limit, void *lims);

int limit_revert_old(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limit, void *lims);

int limit_minmax(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limit, void *lims);


// state_path.c functions
// functions for generating Markov paths

void cumsum(unsigned long len, double* arr, double* dest);

unsigned long randchoice(unsigned long len, double* arr);

int statepath(h2mm_mod* model, unsigned long lent, unsigned long* path, unsigned int seed);

int sparsestatepath(h2mm_mod* model, unsigned long lent, unsigned long long* times, unsigned long* path, unsigned int seed);

int phpathgen(h2mm_mod* model, unsigned long lent, unsigned long* path, unsigned long* traj, unsigned int seed);


// utils.c functions
// miscilaneous small functions for basic tasks

unsigned long get_max_delta(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, phstream *b);

void baseprint(unsigned long niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func);

void h2mm_normalize(h2mm_mod *model_params);

unsigned long get_max_det(unsigned long num_burst, phstream *bursts);

unsigned long check_det(unsigned long num_burst, phstream *bursts, h2mm_mod *in_model);

int duplicate_toempty_model(h2mm_mod *source, h2mm_mod *dest);

int duplicate_toempty_models(unsigned long num_model, h2mm_mod **source, h2mm_mod **dest);

h2mm_mod* h2mm_model_calc_log(h2mm_mod *source);

int copy_model(h2mm_mod *source, h2mm_mod *dest);

int copy_model_vals(h2mm_mod *source, h2mm_mod *dest);

h2mm_mod* allocate_models(const unsigned long n, const unsigned long nstate, const unsigned long ndet, const unsigned long nphot);

int free_model(h2mm_mod *model);

int free_models(const unsigned long n, h2mm_mod *model);

int zero_model(h2mm_mod *model);

unsigned long get_max_phot(unsigned long num_burst, phstream *bursts);

pwrs* allocate_powers(h2mm_mod *in_model, unsigned long max_delta);

int free_powers(pwrs *power);

int free_trpow(trpow *power);

int allocate_path(unsigned long nphot, unsigned long nstate, ph_path* path);

int free_path_arrs(ph_path* path);

ph_path* allocate_paths(unsigned long num_burst, unsigned long* len_burst, unsigned long nstate);

int free_paths(unsigned long num_burst, ph_path* paths);

// pathloglik.c functions

int pathloglik(unsigned long num_burst, unsigned long *len_burst, unsigned long **deltas, unsigned long ** dets, unsigned long **states, h2mm_mod *model, double *loglik, unsigned long num_cores);

#if defined(__linux__) || defined(__APPLE__)
void* path_ll(void* in);
#elif _WIN32
DWORD WINAPI path_ll(void* in);
#endif

// C_H2MM_txtread.c functions

temps* burst_read(char *fname, unsigned long *n);

h2mm_mod* h2mm_read(char *fname);

h2mm_mod* h2mm_read(char* fname);

int print_model(h2mm_mod* model);
