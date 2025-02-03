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



int CNAME(BLASLONG istart, BLASLONG iend, BLASLONG n, BLASLONG k, FLOAT* a, BLASLONG lda, FLOAT* B, FLOAT* Y) {
	BLASLONG i, j, block, tail, length;
	BLASLONG stride_a;
	unsigned int blockLen;
	FLOAT_V_T diag, b_old, y, z;
	
	stride_a = lda * sizeof(FLOAT);

	blockLen = VSETVL(n);

	block = (istart - iend + 1) / blockLen;
	tail = istart - iend  + 1 - block * blockLen;
	
	for (i = istart; i >= istart - tail + 1; i--) {
		length = n - i - 1;
		if (length > k) length = k;

		if (length > 0) {
			AXPYU_K(length, 0, 0, B[i],
				a + 1, 1, B + i + 1, 1, NULL, 0);
		}
		
#ifndef UNIT
		B[i] *= a[0];
#endif
		a -= lda;
	}

	a -= lda * (blockLen - 1);

	for (i -= (blockLen - 1); i >= iend; i -= blockLen) {
		
		length = k;
		b_old = VLEV_FLOAT(B + i, blockLen);

#ifndef UNIT
		y = VLSEV_FLOAT(a, stride_a, blockLen);
		z = VFMULVV_FLOAT(b_old, y, blockLen);
		VSEV_FLOAT(B + i, z, blockLen);
#endif

		for (j = 1; j < k + 1; j++) {
			diag = VLSEV_FLOAT(a + j, stride_a, blockLen);
			z = VLEV_FLOAT(B + i + j, blockLen);
			z = VFMACCVV_FLOAT(z, diag, b_old, blockLen);
			VSEV_FLOAT(B + i + j, z, blockLen);
		}
		a -= lda * blockLen;
	}
	
	return 0;
}
