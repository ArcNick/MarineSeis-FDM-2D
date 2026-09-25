#include "tools_func.cuh"

/**
 * \brief ξ-direction flux (fluid)
 * \param Uf: fluid vector U
 * \param Jacb: Grid Jacobian parameters
 * \param idx: linear index of the grid point
 * \return float2 ξ-direction flux (fluid)
 */
__device__ float2 flux_xi_fluid(
    const U_fluid &Uf, const Grid &Jacb, int idx
) {
    return make_float2(
        -Jacb.J[idx] * Jacb.xi_x[idx] * Uf.p[idx],
        -Jacb.J[idx] * Jacb.xi_z[idx] * Uf.p[idx]
    );
}

__device__ float2 flux_xi_solid(
    const U_solid &Us, const Grid &Jacb, int idx
) {
    int idx_J = idx + (nz_f > 0 ? (nz_f - 1) : 0) * nx_all;
    return make_float2(
        Jacb.J[idx_J] * (Jacb.xi_x[idx_J] * Us.sx[idx] + Jacb.xi_z[idx_J] * Us.sxz[idx]),
        Jacb.J[idx_J] * (Jacb.xi_x[idx_J] * Us.sxz[idx] + Jacb.xi_z[idx_J] * Us.sz[idx])
    );
}

__device__
float2 flux_ze_fluid(
    const U_fluid &Uf, const Grid &Jacb, int ix, int off_z,
    int ifz, int dir, const float2 * __restrict__ Gf
) {
    if (off_z >= 0) {
        // 向介质内部第 off_z 行：行号 = ifz + dir*off_z
        int idx = (ifz + dir * off_z) * nx_all + ix;
        return make_float2(
            -Jacb.J[idx] * Jacb.ze_x[idx] * Uf.p[idx], 
            -Jacb.J[idx] * Jacb.ze_z[idx] * Uf.p[idx]
        );
    } else {
        return __ldg(Gf + (-off_z - 1) * nx_all + ix);
    }
}

__device__
float2 flux_ze_solid(
    const U_solid &Us, const Grid &Jacb, int ix, int off_z,
    int ifz, int dir, const float2 * __restrict__ Gs
) {
    if (off_z >= 0) {
        // 向介质内部第 off_z 行：行号 = ifz + dir*off_z
        int idx = (ifz + dir * off_z) * nx_all + ix;
        // 固体数组的行 -> 全局 Jacobian 行：流固耦合时固体整体下移 (nz_f-1) 行
        // 固体数组行 -> 全局 Jacobian 行：固体上方只有【流固耦合】时才有 nz_f-1 行；
    // 纯固体（SOLID，nz_f==0）时固体占满全域，偏移应为 0（否则会错一行/越界）。
        int idx_J = idx + (nz_f > 0 ? (nz_f - 1) : 0) * nx_all;
        return make_float2(
            Jacb.J[idx_J] * (Jacb.ze_x[idx_J] * Us.sx[idx] + Jacb.ze_z[idx_J] * Us.sxz[idx]),
            Jacb.J[idx_J] * (Jacb.ze_x[idx_J] * Us.sxz[idx] + Jacb.ze_z[idx_J] * Us.sz[idx])      
        );
    } else {
        return __ldg(Gs + (-off_z - 1) * nx_all + ix);
    }
}

