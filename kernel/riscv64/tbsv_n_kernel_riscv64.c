#include "common.h"

#if defined(C910V)
#define GROUP_SIZE 1
#else
#define GROUP_SIZE 2
#endif

#define CONCAT(x,y) x##y
#define CONCAT2(x,y) CONCAT(x,y)

#if defined(DOUBLE)

#define FLOAT_V_T           CONCAT2(vfloat64m, CONCAT2(GROUP_SIZE,_t))
#define VSETVL              CONCAT2(RISCV_RVV(vsetvl_e64m), GROUP_SIZE)
#define VLEV_FLOAT          CONCAT2(RISCV_RVV(vle64_v_f64m), GROUP_SIZE)
#define VFMACCVF_FLOAT      CONCAT2(RISCV_RVV(vfmacc_vf_f64m), GROUP_SIZE)
#define VSEV_FLOAT          CONCAT2(RISCV_RVV(vse64_v_f64m), GROUP_SIZE)

#else

#define FLOAT_V_T           CONCAT2(vfloat32m, CONCAT2(GROUP_SIZE,_t))
#define VSETVL              CONCAT2(RISCV_RVV(vsetvl_e32m), GROUP_SIZE)
#define VLEV_FLOAT          CONCAT2(RISCV_RVV(vle32_v_f32m), GROUP_SIZE)
#define VFMACCVF_FLOAT      CONCAT2(RISCV_RVV(vfmacc_vf_f32m), GROUP_SIZE)
#define VSEV_FLOAT          CONCAT2(RISCV_RVV(vse32_v_f32m), GROUP_SIZE)

#endif

int CNAME(BLASLONG k, FLOAT* a, FLOAT* b, FLOAT alpha) {
  FLOAT_V_T va, vb, vtemp;

  for (BLASLONG j = k; j > 0; ) {
    BLASLONG vl = VSETVL(j);
    va = VLEV_FLOAT(a, vl);
    vb = VLEV_FLOAT(b, vl);
    vtemp = VFMACCVF_FLOAT(vb, alpha, va, vl);
    VSEV_FLOAT(b, vtemp, vl);
    j -= vl;
    a += vl;
    b += vl;
  }

  return 0;
}
