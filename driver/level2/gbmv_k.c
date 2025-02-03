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
#include <ctype.h>
#include "common.h"



#ifndef TRANS
#define M m
#define N n
#else
#define N m
#define M n
#endif

void CNAME(BLASLONG m, BLASLONG n, BLASLONG ku, BLASLONG kl, FLOAT alpha,
    FLOAT* a, BLASLONG lda,
    FLOAT* x, BLASLONG incx, FLOAT* y, BLASLONG incy, void* buffer) {

    BLASLONG i, j, length, tail, iterations;
    FLOAT* X = x;
    FLOAT* Y = y;
    FLOAT* gemvbuffer = (FLOAT*)buffer;
    FLOAT* bufferY = gemvbuffer;
    FLOAT* bufferX = gemvbuffer;

    if (incy != 1) {
        Y = bufferY;
        bufferX = (FLOAT*)(((BLASLONG)bufferY + M * sizeof(FLOAT) + 4095) & ~4095);
        // gemvbuffer = bufferX;
        COPY_K(M, y, incy, Y, 1);
    }

    if (incx != 1) {
        X = bufferX;
        // gemvbuffer = (FLOAT *)(((BLASLONG)bufferX + N * sizeof(FLOAT) + 4095) & ~4095);
        COPY_K(N, x, incx, X, 1);
    }
#ifdef TRANS
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
#if defined(SKYLAKEX) || defined(C910V) || defined(RISCV64_ZVL256B)
}
if ((skylakex && ((doubl && (trans && (ku+kl+1)<16 || !trans && (ku+kl+1)<8)) || (!doubl && (ku+kl+1)<16 ))) || ((c910v) && ((trans )|| (!trans && ((!doubl && (ku+kl+1)%16 != 0 && (ku+kl+1)%16 != 14) || (doubl && (ku+kl+1)%8 != 0))))) || (riscv64_zvl256b && ((doubl && (ku+kl+1) <= 8) || (!doubl && (ku+kl+1) <= 16))))
{

/*#ifdef RISCV64_ZVL256B

#ifndef DOUBLE
#define BLOCK_SIZE 16
#else
#define BLOCK_SIZE 8
#endif

#else*/
	
#ifndef DOUBLE
#define BLOCK_SIZE 16
#else
#define BLOCK_SIZE 8
#endif	

//#endif	

#ifndef TRANS

    j = 0;

    for (; j < ku; j++)
    {
        length = MIN(j + kl + 1, M);

        AXPYU_K(length, 0, 0, alpha * X[j],
            a + ku - j, 1, Y, 1, NULL, 0);

        a += lda;
    }

    tail = MIN(N, M - kl);

    GBMV_N(j, tail, kl, ku, a, lda, X, Y, alpha);

    iterations = (tail - j - 1) / BLOCK_SIZE;
    j += BLOCK_SIZE * iterations;
    a += lda * BLOCK_SIZE * (iterations);

    for (; j < MIN(N, M + ku); j++)
    {
        length = MIN(ku + kl + 1, M - (j - ku));

        AXPYU_K(length, 0, 0, alpha * X[j],
            a, 1, Y + j - ku, 1, NULL, 0);

        a += lda;
    }

#else

    i = 0;

    for (; i < ku; i++)
    {
        length = MIN(i + kl + 1, N);

        Y[i] += alpha * DOTU_K(length, a + ku - i, 1, X, 1);

        a += lda;
    }

    tail = MIN(M, N - kl);

    GBMV_T(i, tail, kl, ku, a, lda, X, Y, alpha);

    iterations = (tail - i - 1) / BLOCK_SIZE;
    i += BLOCK_SIZE * iterations;
    a += lda * BLOCK_SIZE * (iterations);

    for (; i < MIN(M, N + ku); i++)
    {
        length = MIN(kl + ku + 1, N - (i - ku));

        Y[i] += alpha * DOTU_K(length, a, 1, X + i - ku, 1);

        a += lda;
    }

#endif	
}
else {
#endif
	
    BLASLONG start, end, offset_u = ku, offset_l = ku + m;

    for (i = 0; i < MIN(n, m + ku); i++) {

        start = MAX(offset_u, 0);
        end = MIN(offset_l, ku + kl + 1);

        length = end - start;

#ifndef TRANS
        AXPYU_K(length, 0, 0,
            alpha * X[i],
            a + start, 1, Y + start - offset_u, 1, NULL, 0);
#else
        Y[i] += alpha * DOTU_K(length, a + start, 1, X + start - offset_u, 1);
#endif

        offset_u--;
        offset_l--;

        a += lda;
    }
}

    if (incy != 1) {
        COPY_K(M, Y, 1, y, incy);
    }

    return;
}