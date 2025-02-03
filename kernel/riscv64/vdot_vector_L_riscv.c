#include <stdio.h>
#include "common.h"

#if defined (RISCV64_ZVL256B)
#if !defined(DOUBLE)

#define VSETVL(n) __riscv_vsetvl_e32m2(n)
#define VSETVL_MAX __riscv_vsetvlmax_e32m1()
#define FLOAT_V_T vfloat32m2_t
#define FLOAT_V_T_M1 vfloat32m1_t
#define VLEV_FLOAT __riscv_vle32_v_f32m2
#define VLSEV_FLOAT __riscv_vlse32_v_f32m2
#define VFMACCVV_FLOAT __riscv_vfmacc_vv_f32m2
#define VFMVVF_FLOAT __riscv_vfmv_v_f_f32m2
#define VFMVVF_FLOAT_M1 __riscv_vfmv_v_f_f32m1
#define VFDOTVV_FLOAT __riscv_vfdot_vv_f32m2
#define VFMACCVF_FLOAT __riscv_vfmacc_vf_f32m2
#define VFMULVV_FLOAT __riscv_vfmul_vv_f32m2
#define VSEV_FLOAT __riscv_vse32_v_f32m2

#else

#define VSETVL(n) __riscv_vsetvl_e64m2(n)
#define VSETVL_MAX __riscv_vsetvlmax_e64m1()
#define FLOAT_V_T vfloat64m2_t
#define FLOAT_V_T_M1 vfloat64m1_t
#define VLEV_FLOAT __riscv_vle64_v_f64m2
#define VLSEV_FLOAT __riscv_vlse64_v_f64m2
#define VFMACCVV_FLOAT __riscv_vfmacc_vv_f64m2
#define VFMVVF_FLOAT __riscv_vfmv_v_f_f64m2
#define VFMVVF_FLOAT_M1 __riscv_vfmv_v_f_f64m1
#define VFDOTVV_FLOAT __riscv_vfdot_vv_f64m2
#define VFMACCVF_FLOAT __riscv_vfmacc_vf_f64m2
#define VFMULVV_FLOAT __riscv_vfmul_vv_f64m2
#define VSEV_FLOAT __riscv_vse64_v_f64m2

#endif
#else
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

#endif
#endif

int CNAME(BLASLONG n, BLASLONG k, FLOAT alpha, FLOAT *a, BLASLONG lda, FLOAT *x, FLOAT *y, BLASLONG length){
    BLASLONG i;
    BLASLONG stride_a = lda * sizeof(FLOAT);
    unsigned int gvl = 2;
    gvl = VSETVL(n);
    int n3 = (n / gvl) * gvl;
    for (i = 0; i < n3 ; i += gvl) {
        FLOAT_V_T vx, vy, va, mul;
        vy = VLEV_FLOAT(y + i, gvl);
        for (int ii = 0; ii < k ; ii++) {
            vx = VLEV_FLOAT(x + i + 1 + ii, gvl);
            va = VLSEV_FLOAT(a + 1 + ii, stride_a, gvl);
            mul = VFMULVV_FLOAT(vx, va, gvl);
            vy = VFMACCVF_FLOAT(vy, alpha, mul, gvl);
        }
        VSEV_FLOAT(y + i, vy, gvl);
        a += gvl * lda;
    }
    for (; i < n; i++) {
        y[i] += alpha * DOTU_K(k, a + 1, 1, x + i + 1, 1);
        a += lda;
    }
    return 0;
}
