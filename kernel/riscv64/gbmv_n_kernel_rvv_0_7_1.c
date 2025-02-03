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

int CNAME(BLASLONG j, BLASLONG tail, BLASLONG kl, BLASLONG ku, FLOAT* a, BLASLONG lda, FLOAT* X, FLOAT* Y, FLOAT alpha)
{
    BLASLONG i;
	BLASLONG VL = VSETVL_MAX();
    BLASLONG stride = lda * sizeof(FLOAT);
    FLOAT_V_T diag_i, x_copy, y_copy;

    for (; j + VL < tail; j += VL)
    {
        x_copy = VLEV_FLOAT(X + j, VL);
        x_copy = VFMULVF_FLOAT(x_copy, alpha, VL);

        for (i = 0; i < ku + kl + 1; i++)
        {
            y_copy = VLEV_FLOAT(Y + j - ku + i, VL);
            diag_i = VLSEV_FLOAT(a + i, stride, VL);

            y_copy = VFMACCVV_FLOAT(y_copy, diag_i, x_copy, VL);

            VSEV_FLOAT(Y + j - ku + i, y_copy, VL);
        }
        a += lda * VL;
    }

    return 0;
}