/* forward algoritm: return observation likelihood */
double forward(int *data, int len, int backward, int nstates, int nobvs,
        double *prior, double * trans, double *obvs)
{
    /* construct trellis */
    double alpha[len][nstates];
    double beta[len][nstates];

    double loglik;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < nstates; j++) {
                alpha[i][j] = - INFINITY;
                beta[i][j] = - INFINITY;
            }
        }

        /* forward pass */
        #pragma omp for
        for (int i = 0; i < nstates; i++) {
            alpha[0][i] = prior[i] + obvs[IDXT(i,data[0],nobvs)];
        }
        
        for (int i = 1; i < len; i++) {
            #pragma omp for 
            for (int j = 0; j < nstates; j++) {
                for (int k = 0; k < nstates; k++) {
                    double p = alpha[i-1][k] + trans[IDX(k,j,nstates)] + obvs[IDXT(j,data[i],nobvs)];
                    alpha[i][j] = logadd(alpha[i][j], p);
                }
            }
        }
    }
    loglik = -INFINITY;
    for (int i = 0; i < nstates; i++) {
        loglik = logadd(loglik, alpha[len-1][i]);
    }

    return loglik;
}
