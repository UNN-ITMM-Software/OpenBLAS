#include "common.h"

#if defined(C910V)
#define GROUP_SIZE 1
#else
#define GROUP_SIZE 2
#endif

#define CONCAT(x,y) x##y
#define CONCAT2(x,y) CONCAT(x,y)

#if defined(DOUBLE)

#define FLOAT_V_SINGLE_T    vfloat64m1_t
#define FLOAT_V_T           CONCAT2(vfloat64m, CONCAT2(GROUP_SIZE,_t))
#define VSETVL_MAX_SINGLE   RISCV_RVV(vsetvlmax_e64m1)
#define VSETVL_MAX          CONCAT2(RISCV_RVV(vsetvlmax_e64m), GROUP_SIZE)
#define VSETVL(n)           CONCAT2(RISCV_RVV(vsetvl_e64m), GROUP_SIZE)(n)
#define VLEV_FLOAT          CONCAT2(RISCV_RVV(vle64_v_f64m), GROUP_SIZE)
#define VFMACCVV_FLOAT      CONCAT2(RISCV_RVV(vfmacc_vv_f64m), GROUP_SIZE)
#define VFMVVF_FLOAT        CONCAT2(RISCV_RVV(vfmv_v_f_f64m), GROUP_SIZE)
#define VFMVVF_FLOAT_SINGLE RISCV_RVV(vfmv_v_f_f64m1)
#define VFMVFS_FLOAT        RISCV_RVV(vfmv_f_s_f64m1_f64)
#define VSEV_FLOAT          CONCAT2(RISCV_RVV(vse64_v_f64m), GROUP_SIZE)
#if defined(C910V)
#define VFREDUSUMVS_FLOAT(vr, va, vb, vlen) \
                            CONCAT2(RISCV_RVV(vfredusum_vs_f64m),\
                                    CONCAT2(GROUP_SIZE, _f64m1))(vr, va, vb, vlen)
#else
#define VFREDUSUMVS_FLOAT(vr, va, vb, vlen) \
                            CONCAT2(RISCV_RVV(vfredusum_vs_f64m),\
                                    CONCAT2(GROUP_SIZE, _f64m1))(va, vb, vlen)
#endif

#else

#define FLOAT_V_SINGLE_T    vfloat32m1_t
#define FLOAT_V_T           CONCAT2(vfloat32m, CONCAT2(GROUP_SIZE,_t))
#define VSETVL_MAX_SINGLE   RISCV_RVV(vsetvlmax_e32m1)
#define VSETVL_MAX          CONCAT2(RISCV_RVV(vsetvlmax_e32m), GROUP_SIZE)
#define VSETVL(n)           CONCAT2(RISCV_RVV(vsetvl_e32m), GROUP_SIZE)(n)
#define VLEV_FLOAT          CONCAT2(RISCV_RVV(vle32_v_f32m), GROUP_SIZE)
#define VFMACCVV_FLOAT      CONCAT2(RISCV_RVV(vfmacc_vv_f32m), GROUP_SIZE)
#define VFMVVF_FLOAT        CONCAT2(RISCV_RVV(vfmv_v_f_f32m), GROUP_SIZE)
#define VFMVVF_FLOAT_SINGLE RISCV_RVV(vfmv_v_f_f32m1)
#define VFMVFS_FLOAT        RISCV_RVV(vfmv_f_s_f32m1_f32)
#define VSEV_FLOAT          CONCAT2(RISCV_RVV(vse32_v_f32m), GROUP_SIZE)
#if defined(C910V)
#define VFREDUSUMVS_FLOAT(vr, va, vb, vlen) \
                          CONCAT2(RISCV_RVV(vfredusum_vs_f32m), \
                                  CONCAT2(GROUP_SIZE, _f32m1))(vr, va, vb, vlen)
#else
#define VFREDUSUMVS_FLOAT(vr, va, vb, vlen) \
                          CONCAT2(RISCV_RVV(vfredusum_vs_f32m), \
                                  CONCAT2(GROUP_SIZE, _f32m1))(va, vb, vlen)
#endif

#endif

FLOAT CNAME(BLASLONG k, FLOAT* a, FLOAT* b) {
  FLOAT_V_T va, vb, vsum, vtemp;
  FLOAT_V_SINGLE_T vzero, vreduce;

  BLASLONG vl_group_max = VSETVL_MAX();
  BLASLONG vl_single_max = VSETVL_MAX_SINGLE();
  vsum = VFMVVF_FLOAT(ZERO, vl_group_max);
  vzero = VFMVVF_FLOAT_SINGLE(ZERO, vl_single_max);

  for (BLASLONG j = k; j > 0; ) {
    BLASLONG vl = VSETVL(j);
    va = VLEV_FLOAT(a, vl);
    vb = VLEV_FLOAT(b, vl);
#if defined(C910V)
    vsum = VFMACCVV_FLOAT(vsum, va, vb, vl_group_max);
#else
    vsum = VFMACCVV_FLOAT(vsum, va, vb, vl);
#endif

    j -= vl;
    a += vl;
    b += vl;
  }

  vreduce = VFREDUSUMVS_FLOAT(vreduce, vsum, vzero, vl_group_max);
  FLOAT dot = VFMVFS_FLOAT(vreduce);
  return dot;
}
