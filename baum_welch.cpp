float *gmm = NULL;             /* gamma */
float *xi = NULL;              /* xi */
float *pi = NULL;              /* pi */

/* forward backward algoritm: return observation likelihood */
float forward_backward(int *data, int len, int nstates, int nobvs, float *prior, float *trans, float *transT, float *obvs)
{
    /* construct trellis */
    float *alpha = (float *)aligned_alloc(32, len * nstates * sizeof(float));
    float *beta = (float *)aligned_alloc(32, len * nstates * sizeof(float));

    float loglik;
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < nstates; j++) {
            alpha[i*nstates + j] = - INFINITY;
            beta[i*nstates + j] = - INFINITY;
        }
    }

    /* forward pass */
    for (int i = 0; i < nstates; i++) {
        alpha[i] = prior[i] + obvs[IDXT(i,data[0],nstates)];
    }
    
    __m256 result_AVX;
    __m256 alpha_AVX; 
    __m256 trans_AVX, obvs_AVX;
    __m256 all_Inf = _mm256_set1_ps(-INFINITY);

    for (int i = 1; i < len; i++) {
#pragma omp parallel for
       // for (int j = 0; j < nstates; j++) {
       //     for (int k = 0; k < nstates; k++) {
       //         float p = alpha[(i-1) * nstates + k] + trans[IDX(k,j,nstates)] + obvs[IDXT(j,data[i],nstates)];
       //         alpha[i * nstates + j] = logadd(alpha[i * nstates + j], p);
       //     }
       // }
       
        for (int j = 0; j < nstates; j+=8) {
            result_AVX = _mm256_set1_ps(-INFINITY);
            obvs_AVX = _mm256_load_ps(obvs + data[i] * nstates + j);
            for (int k = 0; k < nstates; k++) {
                alpha_AVX = _mm256_set1_ps(alpha[(i-1) * nstates + k]);
                trans_AVX = _mm256_load_ps(trans + k*nstates + j);
                // calculate p
                alpha_AVX = _mm256_add_ps(alpha_AVX, trans_AVX);
                alpha_AVX = _mm256_add_ps(alpha_AVX, obvs_AVX);
                result_AVX = logadd(result_AVX, alpha_AVX);
            }
            _mm256_store_ps(alpha + i*nstates + j, result_AVX);
        }
    }
    loglik = -INFINITY;
    int thread_num = omp_get_max_threads();
#pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        float local_loglik = -INFINITY;
#pragma omp for
        for (int i = 0; i < nstates; i++) {
            local_loglik = logadd(local_loglik, alpha[(len-1) * nstates + i]);
        }
#pragma omp critical
        {
            loglik = logadd(local_loglik, loglik);
        }
    }

    /* backward pass & update counts */
    for (int i = 0; i < nstates; i++) {
        beta[(len-1) * nstates + i] = 0;         /* 0 = log (1.0) */
    }

    __m256 loglik_AVX = _mm256_set1_ps(loglik);
    __m256 gmm_AVX, xi_AVX;
    __m256 beta_AVX;
    __m256 p_AVX;
    for (int i = 1; i < len; i++) {
       // gmm_AVX = _mm256_set1_ps(-INFINITY);
        #pragma omp parallel for
        for (int j = 0; j < nstates; j++) {

            float e = alpha[(len-i) * nstates + j] + beta[(len-i) * nstates + j] - loglik;
            gmm[IDX(j,data[len-i],nobvs)] = logadd(gmm[IDX(j,data[len-i],nobvs)], e);

            for (int k = 0; k < nstates; k++) {
                float p = beta[(len-i) * nstates + k] + trans[IDX(j,k,nstates)] + obvs[IDXT(k,data[len-i],nstates)];
                beta[(len-1-i) * nstates + j] = logadd(beta[(len-1-i) * nstates + j], p);

                e = alpha[(len-1-i) * nstates + j] + beta[(len-i) * nstates + k]
                    + trans[IDX(j,k,nstates)] + obvs[IDXT(k,data[len-i],nstates)] - loglik;
                xi[IDX(j,k,nstates)] = logadd(xi[IDX(j,k,nstates)], e);
            }
        }
       // for (int j = 0; j < nstates; j+=8) {
       //     alpha_AVX = _mm256_load_ps(alpha + (len-i) * nstates + j);
       //     beta_AVX = _mm256_load_ps(beta + (len-i) * nstates + j);
       //     alpha_AVX = _mm256_add_ps(alpha_AVX, beta_AVX);
       //     alpha_AVX = _mm256_sub_ps(alpha_AVX, loglik_AVX);
       //     gmm_AVX = logadd(gmm_AVX, alpha_AVX);
       //     xi_AVX = _mm256_set1_ps(-INFINITY);
       //     beta_AVX = _mm256_set1_ps(-INFINITY);
       //     

       //     alpha_AVX = _mm256_load_ps(alpha + (len-1-i) * nstates + j);
       //     for (int k = 0; k < nstates; k++) {
       //         __m256 p_AVX = _mm256_set1_ps(beta[(len-i) * nstates + k]);
       //         //trans_AVX = _mm256_load_ps()
       //         obvs_AVX = _mm256_set1_ps(obvs + data[len-i] * nstates + k);
       //         p_AVX = _mm256_add_ps(p_AVX, trans_AVX);
       //         p_AVX = _mm256_add_ps(p_AVX, obvs_AVX);
       //         beta_AVX = logadd(beta_AVX, p_AVX); 

       //         __m256 e_AVX = _mm256_set1_ps(beta[(len-i) * nstates + k]);
       //         // Use old transittion
       //         // Use old obvs
       //         e_AVX = _mm256_add_ps(e_AVX, alpha_AVX);
       //         e_AVX = _mm256_add_ps(e_AVX, trans_AVX);
       //         e_AVX = _mm256_add_ps(e_AVX, obvs_AVX);
       //         e_AVX = _mm256_sub_ps(e_AVX, loglik_AVX);
       //         xi_AVX = logadd(xi_AVX, e_AVX);
       //     }
       //     // Store beta and xi
       //     _mm256_store_ps(beta + (len-1-i) * nstates + j, beta_AVX);
       //     //_mm256_store_ps(xi + );
       // }
        //Store back to gmm
        //_mm256_store_ps(gmm + data[len-i]);
    }
    float p = -INFINITY;

#pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        float local_p = -INFINITY;
#pragma omp for
        for (int i = 0; i < nstates; i++) {
            local_p = logadd(local_p, prior[i] + beta[i] + obvs[IDXT(i,data[0],nstates)]);
        }
#pragma omp critical
        {
            p = logadd(local_p, p);
        }
    }

#pragma omp parallel for
    for (int i = 0; i < nstates; i++) {
        float e = alpha[i] + beta[i] - loglik;
        gmm[IDX(i,data[0],nobvs)] = logadd(gmm[IDX(i,data[0],nobvs)], e);

        pi[i] = logadd(pi[i], e);
    }

#ifdef DEBUG
    /* verify if forward prob == backward prob */
    if (fabs(p - loglik) > 1e-5) {
        fprintf(stderr, "Error: forward and backward incompatible: %lf, %lf\n", loglik, p);
    }
#endif

    return loglik;
}

void baum_welch(int *data, int nseq, int iterations, int length, int nstates, int nobvs, float *prior, float *trans, float *transT, float *obvs)
{
    float *loglik = (float *) malloc(sizeof(float) * nseq);
    if (loglik == NULL) handle_error("malloc");
    for (int i = 0; i < iterations; i++) {
        double startTime = CycleTimer::currentSeconds();
        init_count();
        for (int j = 0; j < nseq; j++) {
            loglik[j] = forward_backward(data + length * j, length, nstates, nobvs, prior, trans, transT, obvs);
        }
        float p = sum(loglik, nseq);

        update_prob(nstates, nobvs, prior, trans, obvs);

        printf("iteration %d log-likelihood: %.4lf\n", i + 1, p);
        printf("updated parameters:\n");
        //printf("# initial state probability\n");
        //for (int j = 0; j < nstates; j++) {
        //    printf(" %.4f", exp(prior[j]));
        //}
        //printf("\n");
        //printf("# state transition probability\n");
        //for (int j = 0; j < nstates; j++) {
        //    for (int k = 0; k < nstates; k++) {
        //        printf(" %.4f", exp(trans[IDX(j,k,nstates)]));
        //    }
        //    printf("\n");
        //}
        //printf("# state output probility\n");
        //for (int j = 0; j < nstates; j++) {
        //    for (int k = 0; k < nobvs; k++) {
        //        printf(" %.4f", exp(obvs[IDX(j,k,nobvs)]));
        //    }
        //    printf("\n");
        //}
        //printf("\n");
        double endTime = CycleTimer::currentSeconds();
        printf("Time taken %.4f milliseconds\n",  (endTime - startTime) * 1000);
    }
    free(loglik);
}

void update_prob(int nstates, int nobvs, float *prior, float *trans, float *obvs) {
    float pisum = - INFINITY;
    int thread_num = omp_get_max_threads();
    float gmmsum[nstates];
    float xisum[nstates];
    //size_t i, j;

    for (int i = 0; i < nstates; i++) {
        gmmsum[i] = - INFINITY;
        xisum[i] = - INFINITY;

        pisum  = logadd(pisum, pi[i]);
    }

//#pragma omp parallel 
//    {
//        int tid = omp_get_thread_num();
//        float local_pisum = -INFINITY;
//        for (int i = tid; i < nstates; i+=thread_num) {
//            local_pisum = logadd(local_pisum, pi[i]);
//        }
//#pragma omp critical
//        {
//            pisum = logadd(local_pisum, pisum);
//        }
//    }

    for (int i = 0; i < nstates; i++) {
        prior[i] = pi[i] - pisum;
    }

    #pragma omp parallel for
    for (int i = 0; i < nstates; i++) {
        for (int j = 0; j < nstates; j++) {
            xisum[i] = logadd(xisum[i], xi[IDX(i,j,nstates)]);
        }
        for (int j = 0; j < nobvs; j++) {
            gmmsum[i] = logadd(gmmsum[i], gmm[IDX(i,j,nobvs)]);
        }
    }

    /* May need to blocking!!!*/
    for (int i = 0; i < nstates; i++) {
        for (int j = 0; j < nstates; j++) {
            trans[IDX(i,j,nstates)] = xi[IDX(i,j,nstates)] - xisum[i];
        }
        for (int j = 0; j < nobvs; j++) {
            obvs[IDXT(i,j,nstates)] = gmm[IDX(i,j,nobvs)] - gmmsum[i];
        }
    }
}
