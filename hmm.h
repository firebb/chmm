float logadd(float, float);
float sum(float *, int);
float forward(int *, int, int, int, float *, float *, float *);
float forward_backward(int *, int, int, int, float *, float *, float *, float *);
void viterbi(int *, int, int, int, float *, float *, float *);
void baum_welch(int *data, int nseq, int iterations, int length, int nstates, int nobvs,
        float *, float * , float *, float*);
void init_count();
void update_prob(int, int, float *, float *, float *, float*);
void usage();
void freeall();

#define IDX(i,j,d) (((i)*(d))+(j))
#define IDXT(i,j,d) (((j)*(d))+(i))
#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)
