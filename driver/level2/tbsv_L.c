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

// const static FLOAT dp1 = 1.;

int CNAME(BLASLONG n, BLASLONG k, FLOAT *a, BLASLONG lda, FLOAT *b, BLASLONG incb, void *buffer){
  BLASLONG i;
  FLOAT *B = b;
  BLASLONG length;

  if (incb != 1) {
    B = buffer;
    COPY_K(n, b, incb, buffer, 1);
  }

#if defined(SKYLAKEX) || defined(C910V) || defined(RISCV64_ZVL256B)
#ifdef DOUBLE
  if (k > 64) {
#else
  if (k > 128) {
#endif
#endif
    for (i = 0; i < n; i++) {

#ifdef TRANSA
      length  = i;
      if (length > k) length = k;

      if (length > 0) {
        B[i] -= DOTU_K(length, a + k - length, 1, B + i - length, 1);
      }
#endif

#ifndef UNIT
#ifdef TRANSA
      B[i] /= a[k];
#else
      B[i] /= a[0];
#endif
#endif

#ifndef TRANSA
      length  = n - i - 1;
      if (length > k) length = k;

      if (length > 0) {
        AXPYU_K(length, 0, 0,
         -B[i],
         a + 1, 1, B + i + 1, 1, NULL, 0);
      }
#endif

      a += lda;
    }
#if defined(SKYLAKEX) || defined(C910V) || defined(RISCV64_ZVL256B)
  } else {
#ifdef TRANSA
    for (BLASLONG i = 0; i < k; i++) {
      for (BLASLONG j = k - i; j < k; j++) {
        B[i] -= a[j] * B[j - k + i];
      }
#ifndef UNIT
      B[i] /= a[k];
#endif
      a += lda;
    }
    for (BLASLONG i = k; i < n; i++) {
      FLOAT dot = TBSV_T(k, a, B + i - k);
      B[i] -= dot;
#ifndef UNIT
      B[i] /= a[k];
#endif
      a += lda;
    }
#endif

#ifndef TRANSA
    for (BLASLONG i = 0; i <= n - k - 1; i++) {
#ifndef UNIT
      B[i] /= a[0];
#endif
      TBSV_N(k, a + 1, B + i + 1, -B[i]);
      a += lda;
    }
    for (BLASLONG i = n - k; i < n; i++) {
#ifndef UNIT
      B[i] /= a[0];
#endif
      for (BLASLONG j = i + 1; j < n; j++) {
        B[j] -= a[j - i] * B[i];
      }
      a += lda;
    }
#endif
  }
#endif
  if (incb != 1) {
    COPY_K(n, buffer, 1, b, incb);
  }

  return 0;
}

