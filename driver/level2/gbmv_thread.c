/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "common.h"

#if   !defined(CONJ) && !defined(XCONJ)
#define MYAXPY	AXPYU_K
#define MYDOT	DOTU_K
#elif  defined(CONJ) && !defined(XCONJ)
#define MYAXPY	AXPYC_K
#define MYDOT	DOTC_K
#elif !defined(CONJ) &&  defined(XCONJ)
#define MYAXPY	AXPYU_K
#define MYDOT	DOTC_K
#else
#define MYAXPY	AXPYC_K
#define MYDOT	DOTU_K
#endif

static int gbmv_kernel(blas_arg_t* args, BLASLONG* range_m, BLASLONG* range_n, FLOAT* dummy1, FLOAT* buffer, BLASLONG pos) {

    FLOAT* a, * x, * y;
    BLASLONG lda, incx;
    BLASLONG n_from, n_to;
    BLASLONG i, offset_l, offset_u, uu, ll, ku, kl;

#ifdef TRANSA
#ifndef COMPLEX
    FLOAT result;
#else
    OPENBLAS_COMPLEX_FLOAT result;
#endif
#endif

    a = (FLOAT*)args->a;
    x = (FLOAT*)args->b;
    y = (FLOAT*)args->c;

    lda = args->lda;
    incx = args->ldb;
    ku = args->ldc;
    kl = args->ldd;

    n_from = 0;
    n_to = args->n;

    if (range_m) y += *range_m * COMPSIZE;

    if (range_n) {
        n_from = *(range_n + 0);
        n_to = *(range_n + 1);

        a += n_from * lda * COMPSIZE;
    }

    n_to = MIN(n_to, args->m + ku);

#ifdef TRANSA
    if (incx != 1) {
        COPY_K(args->m, x, incx, buffer, 1);
        x = buffer;
    }
#endif

    SCAL_K(
#ifndef TRANSA
        args->m,
#else
        args->n,
#endif
        0, 0, ZERO,
#ifdef COMPLEX
        ZERO,
#endif
        y, 1, NULL, 0, NULL, 0);

    offset_u = ku - n_from;
    offset_l = ku - n_from + args->m;
#ifdef TRANSA
const unsigned short trans = 1;
#else
const unsigned short trans = 0;
#endif

#ifdef DOUBLE
const unsigned short doubl = 1;
#else
const unsigned short doubl = 0;
#endif

#ifdef SKYLAKEX
const unsigned short skylakex = 1;
#else
const unsigned short skylakex = 0;
#endif

#ifdef C910V
const unsigned short c910v = 1;
#else
const unsigned short c910v = 0; 
#endif

#ifdef RISCV64_ZVL256B
const unsigned short riscv64_zvl256b = 1;
#else
const unsigned short riscv64_zvl256b = 0; 
#endif

{	
#if !defined(COMPLEX) && (defined(SKYLAKEX) || defined(C910V) || defined(RISCV64_ZVL256B))
}
if ((skylakex && ((doubl && (trans && (ku+kl+1)<16 || !trans && (ku+kl+1)<8)) || (!doubl && (ku+kl+1)<16 ))) || ((c910v) && ((trans )|| (!trans && ((!doubl && (ku+kl+1)%16 != 0 && (ku+kl+1)%16 != 14) || (doubl && (ku+kl+1)%8 != 0))))) || (riscv64_zvl256b && ((doubl && (ku+kl+1) <= 8) || (!doubl && (ku+kl+1) <= 16))))
{
    BLASLONG length, tail, iterations;
    x = (FLOAT*)args->b;

#ifdef RISCV64_ZVL256B

#ifndef DOUBLE
#define BLOCK_SIZE 32
#else
#define BLOCK_SIZE 16
#endif

#else
	
#ifndef DOUBLE
#define BLOCK_SIZE 16
#else
#define BLOCK_SIZE 8
#endif	

#endif	


#ifndef TRANSA

    const BLASLONG M = args->m;
    const BLASLONG N = n_to;

    BLASLONG j = n_from;

    for (; j < ku; j++)
    {
        length = MIN(j + kl + 1, M);

        AXPYU_K(length, 0, 0, 1 * x[j],
            a + ku - j, 1, y, 1, NULL, 0);

        a += lda;
    }

    tail = MIN(N, M - kl);

    GBMV_N(j, tail, kl, ku, a, lda, x, y, 1.0);

    iterations = MAX(0, (tail - j - 1) / BLOCK_SIZE);
    j += BLOCK_SIZE * iterations;
    a += lda * BLOCK_SIZE * (iterations);

    for (; j < MIN(N, MIN(args->n + ku, M + ku)); j++)
    {
        length = MIN(ku + kl + 1, M - (j - ku));
        AXPYU_K(length, 0, 0, 1 * x[j],
            a, 1, y + j - ku, 1, NULL, 0);

        a += lda;
    }

#else

    const BLASLONG N = args->m;
    const BLASLONG M = n_to;

    i = n_from;
    for (; i < ku; i++)
    {
        length = MIN(i + kl + 1, N);

        y[i] += 1 * DOTU_K(length, a + ku - i, 1, x, 1);

        a += lda;
    }

    tail = MIN(M, N - kl);

    GBMV_T(i, tail, kl, ku, a, lda, x, y, 1.0);

    iterations = MAX(0, (tail - i - 1) / BLOCK_SIZE);
    i += BLOCK_SIZE * iterations;
    a += lda * BLOCK_SIZE * (iterations);

    for (; i < MIN(M, MIN(args->m + ku, N + ku)); i++)
    {
        length = MIN(kl + ku + 1, N - (i - ku));

        y[i] += 1 * DOTU_K(length, a, 1, x + i - ku, 1);

        a += lda;
    }

#endif
}
else {	//NOT MODIFIED CODE
#endif

#ifndef TRANSA
    x += n_from * incx * COMPSIZE;
    y -= offset_u * COMPSIZE;
#else
    x -= offset_u * COMPSIZE;
    y += n_from * COMPSIZE;
#endif
    for (i = n_from; i < n_to; i++) {

        uu = MAX(offset_u, 0);
        ll = MIN(offset_l, ku + kl + 1);

#ifndef TRANSA
        MYAXPY(ll - uu, 0, 0,
            *(x + 0),
#ifdef COMPLEX
#ifndef XCONJ
            * (x + 1),
#else
            - *(x + 1),
#endif
#endif
            a + uu * COMPSIZE, 1, y + uu * COMPSIZE, 1, NULL, 0);

        x += incx * COMPSIZE;
#else
        result = MYDOT(ll - uu, a + uu * COMPSIZE, 1, x + uu * COMPSIZE, 1);

#ifndef COMPLEX
        * y = result;
#else
        * (y + 0) += CREAL(result);
#ifndef XCONJ
        * (y + 1) += CIMAG(result);
#else
        * (y + 1) -= CIMAG(result);
#endif
#endif

        x += COMPSIZE;
#endif

        y += COMPSIZE;

        offset_u--;
        offset_l--;

        a += lda * COMPSIZE;
    }
}

    return 0;
}

#ifndef COMPLEX
int CNAME(BLASLONG m, BLASLONG n, BLASLONG ku, BLASLONG kl, FLOAT  alpha, FLOAT* a, BLASLONG lda, FLOAT* x, BLASLONG incx, FLOAT* y, BLASLONG incy, FLOAT* buffer, int nthreads) {
#else
int CNAME(BLASLONG m, BLASLONG n, BLASLONG ku, BLASLONG kl, FLOAT * alpha, FLOAT * a, BLASLONG lda, FLOAT * x, BLASLONG incx, FLOAT * y, BLASLONG incy, FLOAT * buffer, int nthreads) {
#endif

    blas_arg_t args;
    blas_queue_t queue[MAX_CPU_NUMBER];
    BLASLONG range_m[MAX_CPU_NUMBER + 1];
    BLASLONG range_n[MAX_CPU_NUMBER + 1];

    BLASLONG width, i, num_cpu;

#ifdef SMP
#ifndef COMPLEX
#ifdef XDOUBLE
    int mode = BLAS_XDOUBLE | BLAS_REAL;
#elif defined(DOUBLE)
    int mode = BLAS_DOUBLE | BLAS_REAL;
#else
    int mode = BLAS_SINGLE | BLAS_REAL;
#endif
#else
#ifdef XDOUBLE
    int mode = BLAS_XDOUBLE | BLAS_COMPLEX;
#elif defined(DOUBLE)
    int mode = BLAS_DOUBLE | BLAS_COMPLEX;
#else
    int mode = BLAS_SINGLE | BLAS_COMPLEX;
#endif
#endif
#endif

    args.m = m;
    args.n = n;

    args.a = (void*)a;
    args.b = (void*)x;
    args.c = (void*)buffer;

    args.lda = lda;
    args.ldb = incx;
    args.ldc = ku;
    args.ldd = kl;

    num_cpu = 0;

    range_n[0] = 0;
    i = n;

    while (i > 0) {

        width = blas_quickdivide(i + nthreads - num_cpu - 1, nthreads - num_cpu);

        if (width < 4) width = 4;
        if (i < width) width = i;

        range_n[num_cpu + 1] = range_n[num_cpu] + width;

#ifndef TRANSA
        range_m[num_cpu] = num_cpu * ((m + 15) & ~15);
        if (range_m[num_cpu] > m * num_cpu) range_m[num_cpu] = m * num_cpu;
#else
        range_m[num_cpu] = num_cpu * ((n + 15) & ~15);
        if (range_m[num_cpu] > n * num_cpu) range_m[num_cpu] = n * num_cpu;
#endif

        queue[num_cpu].mode = mode;
        queue[num_cpu].routine = gbmv_kernel;
        queue[num_cpu].args = &args;
        queue[num_cpu].range_m = &range_m[num_cpu];
        queue[num_cpu].range_n = &range_n[num_cpu];
        queue[num_cpu].sa = NULL;
        queue[num_cpu].sb = NULL;
        queue[num_cpu].next = &queue[num_cpu + 1];

        num_cpu++;
        i -= width;
    }

    if (num_cpu) {
        queue[0].sa = NULL;
#ifndef TRANSA
        queue[0].sb = buffer + num_cpu * (((m + 255) & ~255) + 16) * COMPSIZE;
#else
        queue[0].sb = buffer + num_cpu * (((n + 255) & ~255) + 16) * COMPSIZE;
#endif

        queue[num_cpu - 1].next = NULL;

        exec_blas(num_cpu, queue);
    }

    for (i = 1; i < num_cpu; i++) {
        AXPYU_K(
#ifndef TRANSA
            m,
#else
            n,
#endif
            0, 0,
#ifndef COMPLEX
            ONE,
#else
            ONE, ZERO,
#endif
            buffer + range_m[i] * COMPSIZE, 1, buffer, 1, NULL, 0);
    }

    AXPYU_K(
#ifndef TRANSA
        m,
#else
        n,
#endif
        0, 0,
#ifndef COMPLEX
        alpha,
#else
        alpha[0], alpha[1],
#endif
        buffer, 1, y, incy, NULL, 0);

    return 0;
}
