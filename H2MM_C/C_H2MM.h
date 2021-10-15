// File:  C_H2MM.h
// Author: Paul David Harris
// Purpose: Header files for H2MM and H2MM Viterbi algorithm
// Date Created: 13 Feb 2021
// Data Modified: 18 Sept 2021

#ifdef _WIN32
#include <windows.h>
#endif

// typedefs for fwd_back_photonbyphoton.c

// Stucture for a single burst
typedef struct
{
	size_t nphot; // size of burst (# of photons
	size_t *delta; // pointer to array of deltas, size fo nphot
	size_t *det; // pointer to array of photon stream indexes, size of nphot
} phstream;

// Structure for building linked list fo bursts, before calculating deltas
struct temp
{
	long len_burst; // size of burst
	unsigned long long *times; // absolute arrival time
	long *detectors; // detector index
	struct temp *next; // pointer to next burst
};
typedef struct temp temps;

// Structure for h2mm model
typedef struct
{
	size_t nstate; // number of states in the h2mm model
	size_t ndet; // number of photon streams in the model
	size_t nphot; // number of photons in data set, filled out by C_H2MM function
	size_t niter; // number of iterations to converged, filled out by compute_ess_dhmm function
	size_t conv; // converged status, filled out by compute_ess_dhmm function- 1 is converged, 0 is not converged, others indicate errors
	double *prior; // prior array (1D), size: nstate
	double *trans; // trans array (2D), size: nstate x nstate
	double *obs; // obs array (2D), size: (ndet x nstate)
	double loglik; // loglike, updated by fwd_back_PhotonByPhoton threads
} h2mm_mod;


// structure contains all inputs for fwd_back_PhotonByPhoton
typedef struct
{
	phstream *phot; // array of phstream structures (burst data)
	size_t max_phot; // the size of the larges burst, used for allocating various arrays
	size_t *cur_burst; // ponter to same size_t, lets each thread of fwd_back_PhotonByPhoton know which burst to compute next
	size_t num_burst; // number of bursts in data set, the size of the phot array
	size_t sk; // number of states, indexing chosen to match
	size_t sj; // square of the number of states
	size_t si; // cube of the number of states
	size_t sT; // fourth power of number of states
	double *Rho; // Rho array, an sT x sk x sk x sk x sk array
	double *A; // contains powers of transition matrix, a sT x sk x sK array
	h2mm_mod *current; // the h2mm_mod from last iteration
	h2mm_mod *new; // the new h2mm_mod being generated in current iteration
#ifdef linux
	pthread_mutex_t *h2mm_lock; // mutex for checking on cur_burst
#endif
} fback_vals;

// C_H2MM.c structures

// the "limits" for calculations
typedef struct
{
	size_t max_iter; // number of iterations before algorithm stops optimization
	size_t num_cores; // number of cores to use (threads to spin up)
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
	size_t max_pow; // maximum power, the maximum delta in the data set
	size_t sT; // stride for Rho of 0th index, cooresponds to the delta t of
	size_t si;
	size_t sj; // note that this is also the stride for the delta t in the A array
	size_t sk; // note that this is also the number of states
	size_t tv; // for future imporovments if Rho calculation is parallelized, idenfifies one of the two powers to use to calculate the next
	size_t tq; // same as tv, but the other power
	size_t td; // for future improvements if Rho calculation is parallelized, idenfifies destination power, tv and tq should match 
	size_t *pow_list; // for future improvemtns if Rho calculation is parallelized, identifies which powers are needed to be calculated
	double *A; // the A array, contains the powers of the tranistion probability matrix
	double *Rho; // the Rho array, as defined in H2MM
} pwrs;

typedef struct // pwrs is a structure that contains pointers to the A (transmat) and Rho arrays, plus information on dimensions and which values are to be calculated
{
	size_t max_pow; // maximum power, the maximum delta in the data set
	size_t sj; // note that this is also the stride for the delta t in the A array
	size_t sk; // note that this is also the number of states
	double *A; // the A array, contains the powers of the tranistion probability matrix
} trpow;

// viterbi.c structures 

// structure identifies a burst, and its viterbi values
typedef struct
{
	size_t nphot; // number of photons in burst
	size_t nstate; // number of states in model
	double loglik; // loglikelihood of the path being the actual path
	size_t *path; // most likely state of each photon
	double *scale; // likelihood of assignment of state of photon
} ph_path;

// structure contains all inputs for viterbi_burst thread
typedef struct
{
	size_t si; // the number of states, also stride of 1st dimension of A
	size_t sT; // stride of 0th dimension of A, gives power of A
	size_t *cur_burst; // pointer to common record of next burst to be calculated
	size_t max_phot; // size of larges burst, used for allocating arrays
	size_t num_burst; // number of bursts in data set
	double *A; // A array, the powers of the transition probability matrix
	phstream *phot; // pointer to burst array, part of input
	ph_path *path; // pointer to viterbi path found by viterbi algorithm
	h2mm_mod *model; // h2mm model used in calculating the path
#ifdef linux
	pthread_mutex_t *vit_lock; // mutex for updating cur_burst
#endif
} vit_vals;


// Function definitions

// C_H2MM.c function signatures
pwrs* get_deltas(unsigned long num_burst, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, phstream *b); // builds burst arrays, and finds deltas between abolute arrival times

void baseprint(size_t niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func); // function to be used as a function pointer for printing to console

h2mm_mod* C_H2MM(unsigned long num_bursts, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, h2mm_mod *in_model, lm *limits, void (*model_limits_func)(h2mm_mod*,h2mm_mod*,h2mm_mod*,void*), void *model_limits, void (*print_func)(size_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call); // The algorithm called by the wrappers/interface files

// fwd_back_photonbyphoton_par.c function signatures

// function called by each thread in H2MM, each thread calculates succesive bursts
#ifdef linux
void* fwd_back_PhotonByPhoton(void* burst);
#elif _WIN32
DWORD WINAPI fwd_back_PhotonByPhoton(void* burst);
#endif

void h2mm_normalize(h2mm_mod *model_params); // funciton to normalize new H2MM model calculated by fw_back_PhotonByPhoton threads

h2mm_mod* compute_ess_dhmm(size_t num_bursts, phstream *b, pwrs *powers, h2mm_mod *in, lm *limits, void (*model_limits_func)(h2mm_mod*,h2mm_mod*,h2mm_mod*,void*), void *model_limits, void (*print_func)(size_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*), void *print_call); // called to optimize H2MM model, 

int compute_multi(unsigned long num_bursts, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, h2mm_mod *mod_array, lm *limits); // calculate the loglik of an array of h2mm_mod based on single dataset

// rho_calc.c function signatures

trpow* transpow(h2mm_mod* model, size_t maxdif); // calculate the power of trans matrix

void* rhoulate(void *vals); // calculates a power of Rho and A

void* rho_all(size_t nstate, double* transmat, pwrs *powers); // calculated new Rho and A matrixes

// model_limits_funcs.c functions signatures

void limit_revert(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, void *lims); // reverts any values that are out of range to their previous values

void limit_revert_old(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, void *lims); // reverts any values that are out of range to their values from the "old" model ie the model before the one whose loglik was just calculated 

void limit_minmax(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, void *lims); // replaces values that are out of range to the minimum or maximum value

// viterbi.c function signatures 

// function called by each thread in viterbi, each thread calculates succesive bursts
#ifdef linux
void* viterbi_burst(void* in_vals);
#elif _WIN32
DWORD WINAPI viterbi_burst(void* in_vals);
#endif

// main function for calculating the viterbi path 

int viterbi(unsigned long num_bursts, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, h2mm_mod *model, ph_path *path_array, unsigned long num_cores);

// C_H2MM_interface

temps* burst_read(char *fname, size_t *n); // read burst data from a .txt file

h2mm_mod* h2mm_read(char *fname); // read h2mm model from .txt file

// simulation functions

void cumsum(unsigned long len, double* arr, double* dest); // calculate the cumulative sum of an array

unsigned long randchoice(unsigned long len, double* arr); // select a random set index within range len, based on array arr

int statepath(h2mm_mod* model, unsigned long lent, unsigned long* path, unsigned int seed); // generate random statepath based on model, with equally spaced times

int sparsestatepath(h2mm_mod* model, unsigned long lent, unsigned long long* times, unsigned long* path, unsigned int seed); // generate random statepath of sparsely spaced times

int phpathgen(h2mm_mod* model, unsigned long lent, unsigned long* path, unsigned long* traj, unsigned int seed); // generate random set of detectors based on given statepath
