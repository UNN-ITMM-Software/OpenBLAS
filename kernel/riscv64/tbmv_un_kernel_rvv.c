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
#define VFADD_FLOAT RISCV_RVV(vfadd_vv_f32m4)

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
#define VFADD_FLOAT RISCV_RVV(vfadd_vv_f64m4)

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
  	BLASLONG i, j, tail, length, nBlocks;
	BLASLONG stride_a;
	unsigned int blockLen;
	FLOAT_V_T diag, b_old, y, z;
	
	stride_a = lda * sizeof(FLOAT);
	blockLen = VSETVL(n);
	
	nBlocks = (iend - istart) / blockLen;
	tail = istart + nBlocks * blockLen;
		
	for (i = istart; i < tail; i += blockLen) {
		length = k;
		b_old = VLEV_FLOAT(B + i, blockLen);

	#ifndef UNIT
		y = VLSEV_FLOAT(a + k, stride_a, blockLen);
        z = VFMULVV_FLOAT(b_old, y, blockLen);
		VSEV_FLOAT(Y + i, z, blockLen);
	#else
		#ifdef SMP
		VSEV_FLOAT(Y + i, b_old, blockLen);
		#endif	
	#endif
	
		for (j = 0; j < k; j++)	{
			diag = VLSEV_FLOAT(a + j, stride_a, blockLen);
			z = VLEV_FLOAT(Y + i + j - length, blockLen);
			z = VFMACCVV_FLOAT(z, diag, b_old, blockLen);
			VSEV_FLOAT(Y + i + j - length, z, blockLen);
		}

		a += lda * blockLen;
	}

	for (i = tail; i < iend; i++) {
		length = k;
		if (length > 0) {
		  AXPYU_K(length, 0, 0,
			 B[i],
			 a + k - length, 1, Y + i - length, 1, NULL, 0);
		}

#ifdef SMP
       Y[i] = B[i]; 
#endif
	#ifndef UNIT
	   Y[i] *= a[k];
	#endif
	
		a += lda;
	}
		
	return 0;
}