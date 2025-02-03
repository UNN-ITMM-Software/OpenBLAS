#include "common.h"
#include <stdio.h>

#if defined (C910V)

#if !defined(DOUBLE)

#define VSETVL(n) RISCV_RVV(vsetvl_e32m4)(n)
#define VSETVL_MAX RISCV_RVV(vsetvlmax_e32m1)()
#define FLOAT_V_T vfloat32m4_t
#define FLOAT_V_T_M1 vfloat32m1_t
#define VLEV_FLOAT RISCV_RVV(vle32_v_f32m4)
#define VLSEV_FLOAT RISCV_RVV(vlse32_v_f32m4)
#define VFMACCVV_FLOAT RISCV_RVV(vfmacc_vv_f32m4)
#define VFMVVF_FLOAT RISCV_RVV(vfmv_v_f_f32m4)
#define VFMVVF_FLOAT_M1 RISCV_RVV(vfmv_v_f_f32m1)
#define VFDOTVV_FLOAT RISCV_RVV(vfdot_vv_f32m4)
#define VFMACCVF_FLOAT RISCV_RVV(vfmacc_vf_f32m4)
#define VFMACCVV_FLOAT RISCV_RVV(vfmacc_vv_f32m4)
#define VFMULVV_FLOAT RISCV_RVV(vfmul_vv_f32m4)
#define VSEV_FLOAT RISCV_RVV(vse32_v_f32m4)
#define VFADDVV_FLOAT RISCV_RVV(vfadd_vv_f32m4)

#else

#define VSETVL(n) RISCV_RVV(vsetvl_e64m4)(n)
#define VSETVL_MAX RISCV_RVV(vsetvlmax_e64m1)()
#define FLOAT_V_T vfloat64m4_t
#define FLOAT_V_T_M1 vfloat64m1_t
#define VLEV_FLOAT RISCV_RVV(vle64_v_f64m4)
#define VLSEV_FLOAT RISCV_RVV(vlse64_v_f64m4)
#define VFMACCVV_FLOAT RISCV_RVV(vfmacc_vv_f64m4)
#define VFMVVF_FLOAT RISCV_RVV(vfmv_v_f_f64m4)
#define VFMVVF_FLOAT_M1 RISCV_RVV(vfmv_v_f_f64m1)
#define VFDOTVV_FLOAT RISCV_RVV(vfdot_vv_f64m4)
#define VFMACCVF_FLOAT RISCV_RVV(vfmacc_vf_f64m4)
#define VFMULVV_FLOAT RISCV_RVV(vfmul_vv_f64m4)
#define VSEV_FLOAT RISCV_RVV(vse64_v_f64m4)
#define VFADDVV_FLOAT RISCV_RVV(vfadd_vv_f64m4)

#endif

#else
	
#if !defined(DOUBLE)

#define VSETVL(n) RISCV_RVV(vsetvl_e32m2)(n)
#define VSETVL_MAX RISCV_RVV(vsetvlmax_e32m2)()
#define FLOAT_V_T vfloat32m2_t
#define FLOAT_V_T_M1 vfloat32m1_t
#define VLEV_FLOAT RISCV_RVV(vle32_v_f32m2)
#define VLSEV_FLOAT RISCV_RVV(vlse32_v_f32m2)
#define VFMACCVV_FLOAT RISCV_RVV(vfmacc_vv_f32m2)
#define VFMVVF_FLOAT RISCV_RVV(vfmv_v_f_f32m2)
#define VFMVVF_FLOAT_M1 RISCV_RVV(vfmv_v_f_f32m2)
#define VFDOTVV_FLOAT RISCV_RVV(vfdot_vv_f32m2)
#define VFMACCVF_FLOAT RISCV_RVV(vfmacc_vf_f32m2)
#define VFMULVV_FLOAT RISCV_RVV(vfmul_vv_f32m2)
#define VSEV_FLOAT RISCV_RVV(vse32_v_f32m2)
#define VFADDVV_FLOAT RISCV_RVV(vfadd_vv_f32m2)

#else

#define VSETVL(n) RISCV_RVV(vsetvl_e64m2)(n)
#define VSETVL_MAX RISCV_RVV(vsetvlmax_e64m2)()
#define FLOAT_V_T vfloat64m2_t
#define FLOAT_V_T_M1 vfloat64m1_t
#define VLEV_FLOAT RISCV_RVV(vle64_v_f64m2)
#define VLSEV_FLOAT RISCV_RVV(vlse64_v_f64m2)
#define VFMACCVV_FLOAT RISCV_RVV(vfmacc_vv_f64m2)
#define VFMVVF_FLOAT RISCV_RVV(vfmv_v_f_f64m2)
#define VFMVVF_FLOAT_M1 RISCV_RVV(vfmv_v_f_f64m2)
#define VFDOTVV_FLOAT RISCV_RVV(vfdot_vv_f64m2)
#define VFMACCVF_FLOAT RISCV_RVV(vfmacc_vf_f64m2)
#define VFMULVV_FLOAT RISCV_RVV(vfmul_vv_f64m2)
#define VSEV_FLOAT RISCV_RVV(vse64_v_f64m2)
#define VFADDVV_FLOAT RISCV_RVV(vfadd_vv_f64m2)
#endif

#endif


int CNAME(BLASLONG istart, BLASLONG iend, BLASLONG n, BLASLONG k, FLOAT* a, BLASLONG lda, FLOAT* B, FLOAT* Y)
{  
	BLASLONG i, j, block, tail, length;
	BLASLONG stride_a;
	unsigned int blockLen;
	FLOAT_V_T diag, summ, x, y, z;
	
	stride_a = lda * sizeof(FLOAT);
	blockLen = VSETVL(n);
	
  	if (istart > iend) {
		
		block = (istart - iend) / blockLen;
		tail = istart - block * blockLen;
		a += (istart - blockLen) * lda;

		for (i = istart - blockLen; i >= tail; i -= blockLen) {

			length = k;

            if (length > 0) {
				x = VLSEV_FLOAT(a, stride_a, blockLen);
				y = VLEV_FLOAT(B + i - k, blockLen);
				summ = VFMULVV_FLOAT(x, y, blockLen);	

				for (j = 1; j < k; j++)	{
					x = VLSEV_FLOAT(a + j, stride_a, blockLen);
					y = VLEV_FLOAT(B + i - k + j, blockLen);
					summ = VFMACCVV_FLOAT(summ, x, y, blockLen);
				}
			
				x = VLEV_FLOAT(B + i, blockLen);
	#ifndef UNIT
				y = VLSEV_FLOAT(a + k, stride_a, blockLen);
				summ = VFMACCVV_FLOAT(summ, x, y, blockLen);
	#else	
				summ = VFADDVV_FLOAT(x, summ, blockLen);	
	#endif			
				VSEV_FLOAT(B + i, summ, blockLen);
			}
#ifndef UNIT			
			else {

				x = VLEV_FLOAT(B + i, blockLen);
				y = VLSEV_FLOAT(a + k, stride_a, blockLen);
				summ = VFMULVV_FLOAT(x, y, blockLen);
				VSEV_FLOAT(B + i, summ, blockLen);
			}
#endif				
			
			a -= lda * blockLen;
		}

		a += lda * (blockLen - 1);
		
		for (i = tail - 1; i >= iend; i--) {
#ifndef UNIT
			B[i] *= a[k];
#endif
			length = i;
			if (length > k) length = k;

			if (length > 0) {
				B[i] += DOTU_K(length, a + k - length, 1, B + i - length, 1);
			}

			a -= lda;
		} 
	}  
	else {
		block = (iend - istart) / blockLen;
		tail = istart + block * blockLen;
		
		for (i = istart; i < tail; i += blockLen) {

			length = k;
			summ = VLEV_FLOAT(Y + i, blockLen);

			for (j = 0; j < k; j++)	{
				x = VLSEV_FLOAT(a + j, stride_a, blockLen);
				y = VLEV_FLOAT(B + i - k + j, blockLen);
				summ = VFMACCVV_FLOAT(summ, x, y, blockLen);
			}
			x = VLEV_FLOAT(B + i, blockLen);
#ifndef UNIT
			y = VLSEV_FLOAT(a + k, stride_a, blockLen);
			summ = VFMACCVV_FLOAT(summ, x, y, blockLen);
#else	
			summ = VFADDVV_FLOAT(x, summ, blockLen);	
#endif		
			VSEV_FLOAT(Y + i, summ, blockLen);
			a += lda * blockLen;
		}

		for (i = tail; i < iend; i++) {
#ifndef UNIT
			Y[i] += a[k] * B[i];
#else
			Y[i] += B[i];
#endif
			length = i;
			if (length > k) length = k;

			if (length > 0) {
				Y[i] += DOTU_K(length, a + k - length, 1, B + i - length, 1);
			}

			a += lda;
		} 
	}

  return 0;
}