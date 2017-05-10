/*
 * Copyright (c) 2009, Chuan Liu <chuan@cs.jhu.edu>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */


#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <immintrin.h>
#include "CycleTimer.h"
#include "hmm.h"
#include "forward.cpp"
#include "viterbi.cpp"
#include "baum_welch.cpp"


/* p7_LOGSUM_SCALE defines the precision of the calculation; the
 * default of 1000.0 means rounding differences to the nearest 0.001
 * nat. p7_LOGSUM_TBL defines the size of the lookup table; the
 * default of 16000 means entries are calculated for differences of 0
 * to 16.000 nats (when p7_LOGSUM_SCALE is 1000.0).  e^{-p7_LOGSUM_TBL /
 * p7_LOGSUM_SCALE} should be on the order of the machine FLT_EPSILON,
 * typically 1.2e-7.
 *
 * reference : 
 * https://svn.janelia.org/eddylab/eddys/src/hmmer/tags/3.1b2/src/logsum.c
 */
#define p7_LOGSUM_SCALE 1000.f
#define p7_LOGSUM_TBL   16000
static float flogsum_lookup[p7_LOGSUM_TBL]; /* p7_LOGSUM_TBL=16000: (A-B) = 0..16 nats, steps of 0.001 */

int nstates = 0;                /* number of states */
int nobvs = 0;                  /* number of observations */
int nseq = 0;                   /* number of data sequences  */
int length = 0;                 /* data sequencel length */
float *prior = NULL;           /* initial state probabilities */
float *trans = NULL;           /* state transition probabilities */
float *transT = NULL;           /* state transition probabilities */
float *obvs = NULL;            /* output probabilities */
float *obvsT = NULL;            /* output probabilities */
int *data = NULL;

void p7_FLogsumInit(void)
{
    for (int i = 0; i < p7_LOGSUM_TBL; i++) 
        flogsum_lookup[i] = log(1. + exp((double) -i / p7_LOGSUM_SCALE));
}

int main(int argc, char *argv[])
{
    char *configfile = NULL;
    FILE *fin, *bin;

    char *linebuf = NULL;
    size_t buflen = 0;

    int iterations = 3;
    int mode = 3;
    int threadnum;

    int c;
    float d;
    float *loglik;
    float p;
    int i, j, k;
    opterr = 0;

    p7_FLogsumInit();

    while ((c = getopt(argc, argv, "c:n:hp:t:")) != -1) {
        switch (c) {
            case 'c':
                configfile = optarg;
                break;
            case 'h':
                usage();
                exit(EXIT_SUCCESS);
            case 'n':
                iterations = atoi(optarg);
                break;
            case 'p':
                mode = atoi(optarg);
                if (mode != 1 && mode != 2 && mode != 3) {
                    fprintf(stderr, "illegal mode: %d\n", mode);
                    exit(EXIT_FAILURE);
                }
                break;
            case 't':
                threadnum = atoi(optarg);
                omp_set_num_threads(threadnum);
                break;
            case '?':
                fprintf(stderr, "illegal options\n");
                exit(EXIT_FAILURE);
            default:
                abort();
        }
    }

    if (configfile == NULL) {
        fin = stdin;
    } else {
        fin = fopen(configfile, "r");
        if (fin == NULL) {
            handle_error("fopen");
        }
    }

    i = 0;
    while ((c = getline(&linebuf, &buflen, fin)) != -1) {
        if (c <= 1 || linebuf[0] == '#')
            continue;

        if (i == 0) {
            if (sscanf(linebuf, "%d", &nstates) != 1) {
                fprintf(stderr, "config file format error: %d\n", i);
                freeall();
                exit(EXIT_FAILURE);
            }

            prior = (float *) aligned_alloc(32, sizeof(float) * nstates);
            if (prior == NULL) handle_error("aligned_alloc");

            trans = (float *) aligned_alloc(32, sizeof(float) * nstates * nstates);
            if (trans == NULL) handle_error("aligned_alloc");

            transT = (float *) aligned_alloc(32, sizeof(float) * nstates * nstates);
            if (transT == NULL) handle_error("aligned_alloc");

            xi = (float *) aligned_alloc(32, sizeof(float) * nstates * nstates);
            if (xi == NULL) handle_error("aligned_alloc");

            pi = (float *) aligned_alloc(32, sizeof(float) * nstates);
            if (pi == NULL) handle_error("aligned_alloc");

        } else if (i == 1) {
            if (sscanf(linebuf, "%d", &nobvs) != 1) {
                fprintf(stderr, "config file format error: %d\n", i);
                freeall();
                exit(EXIT_FAILURE);
            }

            obvs = (float *) aligned_alloc(32, sizeof(float) * nstates * nobvs);
            if (obvs == NULL) handle_error("aligned_alloc");

            obvsT = (float *) aligned_alloc(32, sizeof(float) * nstates * nobvs);
            if (obvsT == NULL) handle_error("aligned_alloc");

            gmm = (float *) aligned_alloc(32, sizeof(float) * nstates * nobvs);
            if (gmm == NULL) handle_error("aligned_alloc");

        } else if (i == 2) {
            /* read initial state probabilities */ 
            bin = fmemopen(linebuf, buflen, "r");
            if (bin == NULL) handle_error("fmemopen");
            for (j = 0; j < nstates; j++) {
                if (fscanf(bin, "%f", &d) != 1) {
                    fprintf(stderr, "config file format error: %d\n", i);
                    freeall();
                    exit(EXIT_FAILURE);
                }
                prior[j] = logf(d);
            }
            fclose(bin);

        } else if (i <= 2 + nstates) {
            /* read state transition  probabilities */ 
            bin = fmemopen(linebuf, buflen, "r");
            if (bin == NULL) handle_error("fmemopen");
            for (j = 0; j < nstates; j++) {
                if (fscanf(bin, "%f", &d) != 1) {
                    fprintf(stderr, "config file format error: %d\n", i);
                    freeall();
                    exit(EXIT_FAILURE);
                }
                trans[IDX((i - 3),j, nstates)] = logf(d);
                transT[IDXT((i - 3),j, nstates)] = logf(d);
            }
            fclose(bin);
        } else if (i <= 2 + nstates * 2) {
            /* read output probabilities */
            bin = fmemopen(linebuf, buflen, "r");
            if (bin == NULL) handle_error("fmemopen");
            for (j = 0; j < nobvs; j++) {
                if (fscanf(bin, "%f", &d) != 1) {
                    fprintf(stderr, "config file format error: %d\n", i);
                    freeall();
                    exit(EXIT_FAILURE);
                }
                obvs[IDX((i - 3 - nstates),j,nobvs)] = logf(d);
                obvsT[IDXT((i - 3 - nstates),j,nstates)] = logf(d);
            }
            fclose(bin);
        } else if (i == 3 + nstates * 2) {
            if (sscanf(linebuf, "%d %d", &nseq, &length) != 2) {
                fprintf(stderr, "config file format error: %d\n", i);
                freeall();
                exit(EXIT_FAILURE);
            }
            data = (int *) aligned_alloc (32, sizeof(int) * nseq * length);
            if (data == NULL) handle_error("aligned_alloc");
        } else if (i <= 3 + nstates * 2 + nseq) {
            /* read data */
            bin = fmemopen(linebuf, buflen, "r");
            if (bin == NULL) handle_error("fmemopen");
            for (j = 0; j < length; j++) {
                if (fscanf(bin, "%d", &k) != 1 || k < 0 || k >= nobvs) {
                    fprintf(stderr, "config file format error: %d\n", i);
                    freeall();
                    exit(EXIT_FAILURE);
                }
                data[(i - 4 - nstates * 2) * length + j] = k;
            }
            fclose(bin);
        }

        i++;
    }
    fclose(fin);
    if (linebuf) free(linebuf);

    if (i < 4 + nstates * 2 + nseq) {
        fprintf(stderr, "configuration incomplete.\n");
        freeall();
        exit(EXIT_FAILURE);
    }

    if (mode == 3) {
        baum_welch(data, nseq, iterations, length, nstates, nobvs, prior, transT, obvs);
    } else if (mode == 2) {
        for (i = 0; i < nseq; i++) {
            viterbi(data + length * i, length, nstates, nobvs, prior, trans, obvsT);
        }
    } else if (mode == 1) {
        loglik = (float *) malloc(sizeof(float) * nseq);
        if (loglik == NULL) handle_error("malloc");
        for (i = 0; i < nseq; i++) {
            loglik[i] = forward(data + length * i, length, nstates, nobvs, prior, transT, obvs);
        }
        p = sum(loglik, nseq);
        for (i = 0; i < nseq; i++)
            printf("%.4f\n", loglik[i]);
        printf("total: %.4f\n", p);
        free(loglik);
    }

    freeall();
    return 0;
}

/* compute sum of the array using Kahan summation algorithm */
float sum(float *data, int size)
{
    float sum = data[0];
    int i;
    float y, t;
    float c = 0.0;
    for (i = 1; i < size; i++) {
        y = data[i] - c;
        t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

/* initilize counts */
void init_count() {
    size_t i;
    for (i = 0; i < nstates * nobvs; i++)
        gmm[i] = - INFINITY;

    for (i = 0; i < nstates * nstates; i++)
        xi[i] = - INFINITY;

    for (i = 0; i < nstates; i++)
        pi[i] = - INFINITY;
}

float logadd(float x, float y) {
    if (y <= x)
        return (y == -INFINITY || (x-y) >= 15.7f) ? x : x + flogsum_lookup[(int)((x-y)*p7_LOGSUM_SCALE)];
    else
        return (x == -INFINITY || (y-x) >= 15.7f) ? y : y + flogsum_lookup[(int)((y-x)*p7_LOGSUM_SCALE)];
}

void usage() {
    fprintf(stdout, "hmm [-hnt] [-c config] [-p(1|2|3)]\n");
    fprintf(stdout, "usage:\n");
    fprintf(stdout, "  -h   help\n");
    fprintf(stdout, "  -c   configuration file\n");
    fprintf(stdout, "  -tN  use N threads\n");
    fprintf(stdout, "  -p1  compute the probability of the observation sequence\n");
    fprintf(stdout, "  -p2  compute the most probable sequence (Viterbi)\n");
    fprintf(stdout, "  -p3  train hidden Markov mode parameters (Baum-Welch)\n");
    fprintf(stdout, "  -n   number of iterations\n");
}

/* free all memory */
void freeall() {
    if (trans) free(trans);
    if (obvs) free(obvs);
    if (prior) free(prior);
    if (data) free(data);
    if (gmm) free(gmm);
    if (xi) free(xi);
    if (pi) free(pi);
}
