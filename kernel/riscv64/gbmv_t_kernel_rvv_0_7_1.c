#include "common.h"

#if defined(DOUBLE)

#define FLOAT_V_T    		vfloat64m4_t
#define VSETVL_MAX   		RISCV_RVV(vsetvlmax_e64m4)
#define VLEV_FLOAT 			RISCV_RVV(vle64_v_f64m4)
#define VFMULVF_FLOAT 		RISCV_RVV(vfmul_vf_f64m4)
#define VFMACCVV_FLOAT 		RISCV_RVV(vfmacc_vv_f64m4)
#define VLSEV_FLOAT 		RISCV_RVV(vlse64_v_f64m4)
#define VSEV_FLOAT 			RISCV_RVV(vse64_v_f64m4)

#else
	
#define FLOAT_V_T   		vfloat32m4_t
#define VSETVL_MAX          RISCV_RVV(vsetvlmax_e32m4)
#define VLEV_FLOAT 			RISCV_RVV(vle32_v_f32m4)
#define VFMULVF_FLOAT		RISCV_RVV(vfmul_vf_f32m4)
#define VFMACCVV_FLOAT 		RISCV_RVV(vfmacc_vv_f32m4)
#define VLSEV_FLOAT 		RISCV_RVV(vlse32_v_f32m4)
#define VSEV_FLOAT 			RISCV_RVV(vse32_v_f32m4)

#endif

int CNAME(BLASLONG i, BLASLONG tail, BLASLONG kl, BLASLONG ku, FLOAT* a, BLASLONG lda, FLOAT* X, FLOAT* Y, FLOAT alpha)
{
	BLASLONG j;
	BLASLONG VL = VSETVL_MAX();
	BLASLONG stride = lda * sizeof(FLOAT);
	FLOAT_V_T diag_i, x_copy, y_copy;

	for (; i + VL < tail; i += VL)
	{
		y_copy = VLEV_FLOAT(Y + i, VL);

		for (j = 0; j < kl + ku + 1; j++)
		{
			diag_i = VLSEV_FLOAT(a + j, stride, VL);
			x_copy = VLEV_FLOAT(X + i - ku + j, VL);

			x_copy = VFMULVF_FLOAT(x_copy, alpha, VL);

			y_copy = VFMACCVV_FLOAT(y_copy, diag_i, x_copy, VL);
		}

		VSEV_FLOAT(Y + i, y_copy, VL);
		a += lda * VL;
	}

	return 0;
}
	