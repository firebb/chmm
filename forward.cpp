/* forward algoritm: return observation likelihood */
float forward(int *data, int len, int nstates, int nobvs,
        float *prior, float * trans, float *obvs)
{
    /* construct trellis */
    // float alpha[len][nstates];
    // float beta[len][nstates];
    float *alpha = (float *)aligned_alloc(32, len * nstates * sizeof(float));
    float *beta = (float *)aligned_alloc(32, len * nstates * sizeof(float));

    float loglik;

    double startTime = CycleTimer::currentSeconds();
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < nstates; j++) {
                alpha[i * nstates + j] = - INFINITY;
                beta[i * nstates + j] = - INFINITY;
            }
        }

        /* forward pass */
        #pragma omp for
        for (int i = 0; i < nstates; i++) {
            alpha[i] = prior[i] + obvs[IDXT(i,data[0],nstates)];
        }

        __m256 result_AVX;
        __m256 alpha_AVX; 
        __m256 trans_AVX, obvs_AVX;
        __m256 max_AVX, min_AVX; 
        __m256 all_one = _mm256_set1_ps(1);
        __m256 all_Inf = _mm256_set1_ps(-INFINITY);
        __m256 mask;

        for (int i = 1; i < len; i++) {
            #pragma omp for
            //for (int j = 0; j < nstates; j++) {
            //    for (int k = 0; k < nstates; k++) {
            //        float p = alpha[(i-1) * nstates + k] + trans[IDXT(k,j,nstates)] + obvs[IDX(j,data[i],nobvs)];
            //        
            //        if(k==0)
            //        printf("before alpha:%f\n", alpha[i*nstates +j]);
            //        alpha[i * nstates + j] = logadd(alpha[i * nstates + j], p);
            //        if(k==0)
            //        printf("After alpha:%f\n", alpha[i*nstates +j]);
            //    }
            //}
            for (int j = 0; j < nstates; j+=8) {
                result_AVX = _mm256_set1_ps(-INFINITY);
                obvs_AVX = _mm256_load_ps(obvs + data[i] * nstates + j);
                for (int k = 0; k < nstates; k++) {
                    alpha_AVX = _mm256_set1_ps(alpha[(i-1) * nstates + k]);
                    trans_AVX = _mm256_load_ps(trans + k*nstates + j);
                  //  float *alpha_k = (float *)&alpha_AVX;
                  //  if(k==0 && i==1 && j==0)
                  //     printf("alpha-start:%f, alpha: %f\n", alpha[(i-1) * nstates + k], alpha_k[0]); 
                    // calculate p
                    alpha_AVX = _mm256_add_ps(alpha_AVX, trans_AVX);
                    alpha_AVX = _mm256_add_ps(alpha_AVX, obvs_AVX);

                    max_AVX = _mm256_max_ps(alpha_AVX, result_AVX);
                    min_AVX = _mm256_min_ps(alpha_AVX, result_AVX);
                   // alpha_k = (float *)&alpha_AVX;
                   // float *min = (float *)&min_AVX;
                   // float *max = (float *)&max_AVX;
                   // float *trans = (float *)&trans_AVX;
                   // float *obvs = (float *)&obvs_AVX;
                   // if(k==0&&i==1&&j==0)
                   //     printf("trans: %f, obvs: %f, alpha: %f, min: %f, max: %f\n", trans[0], obvs[0], alpha_k[0], min[0], max[0]);                    // max = max + log(exp(min-max))
                    min_AVX = _mm256_sub_ps(min_AVX, max_AVX);
                  //  if(k==0&&i==1&&j==0)
                  //      printf("min1:%f\n",min[0]);
                    min_AVX = exp256_ps(min_AVX);
                  //  if(k==0&&i==1&&j==0)
                  //      printf("min2:%f\n",min[0]);
                    //mask = _mm256_cmp_ps(min_AVX, all_zero, _CMP_EQ_UQ);
                    min_AVX = _mm256_add_ps(min_AVX, all_one);
                    min_AVX = log256_ps(min_AVX);
                    //min_AVX = _mm256_blendv_ps(min_AVX, all_Inf, mask);
                  //  if(k==0&&i==1&&j==0)
                  //      printf("min3:%f\n",min[0]);

                    result_AVX = _mm256_add_ps(max_AVX,  min_AVX);
                  //  float *result = (float *)&result_AVX;
                  //  if(k==0&&i==1&&j==0)
                  //      printf("result:%f\n",result[0]);
                }
                _mm256_store_ps(alpha + i*nstates + j, result_AVX);
                //_mm256_store_ps(alpha + i*nstates + j, min_AVX);
                //printf("alpha:%f\n", alpha[i*nstates +j]);
            }
        }
    }

    loglik = -INFINITY;
    for (int i = 0; i < nstates; i++) {
        loglik = logadd(loglik, alpha[(len-1) * nstates + i]);
    }
    double endTime = CycleTimer::currentSeconds();
    printf("Time taken %.4f milliseconds\n",  (endTime - startTime) * 1000);

    free(alpha);
    free(beta);
    return loglik;
}
