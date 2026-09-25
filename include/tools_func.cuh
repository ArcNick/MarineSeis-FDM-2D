#ifndef TOOLS_FUNC_CUH
#define TOOLS_FUNC_CUH

#include "elements.h"

constexpr float DRP_AM1 = -0.30874f;
constexpr float DRP_A0  = -0.63260f;
constexpr float DRP_A1  = +1.23300f;
constexpr float DRP_A2  = -0.33340f;
constexpr float DRP_A3  = +0.04168f;

constexpr float M42_A0 = -7.0f / 6.0f;
constexpr float M42_A1 = +8.0f / 6.0f;
constexpr float M42_A2 = -1.0f / 6.0f;

constexpr float M22_A0 = -1.0f;
constexpr float M22_A1 = +1.0f;

extern __constant__ float SRC_GAUSS_W[5][5];

__device__ __forceinline__
void add_gauss_source(
    float *f1, float *f2, int posx, int posz, int nz_rows, float source
) {
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            const int ix = posx + i, iz = posz + j;
            if (ix <= 2 || ix >= nx_all - 3 || iz <= 2 || iz >= nz_rows - 3) {
                continue;
            }
            const float w = source * SRC_GAUSS_W[j + 2][i + 2];
            const int p = iz * nx_all + ix;
            f1[p] += w;
            if (f2 != nullptr) f2[p] += w;
        }
    }
}

enum MacDir {
    F = 0,
    B = 1
};

enum RK4Dir {
    FBFB = 0,
    BFBF = 1
};

// DRP/opt MacCorMack
template<MacDir DIR>
__device__ __forceinline__
float DMac_xi(const float * __restrict__ f, int idx) {
    if constexpr (DIR == F) {
        return (
            + DRP_AM1 * f[idx - 1]
            + DRP_A0  * f[idx]
            + DRP_A1  * f[idx + 1]
            + DRP_A2  * f[idx + 2]
            + DRP_A3  * f[idx + 3]
        );
    } else {
        return (
            - DRP_AM1 * f[idx + 1]
            - DRP_A0  * f[idx]
            - DRP_A1  * f[idx - 1]
            - DRP_A2  * f[idx - 2]
            - DRP_A3  * f[idx - 3]
        );
    }
}

template<MacDir DIR>
__device__ __forceinline__
float DMac_ze(const float * __restrict__ f, int idx) {
    if constexpr (DIR == F) {
        return (
            + DRP_AM1 * f[idx - nx_all]
            + DRP_A0  * f[idx]
            + DRP_A1  * f[idx + nx_all]
            + DRP_A2  * f[idx + 2 * nx_all]
            + DRP_A3  * f[idx + 3 * nx_all]
        );
    } else {
        return (
            - DRP_AM1 * f[idx + nx_all]
            - DRP_A0  * f[idx]
            - DRP_A1  * f[idx - nx_all]
            - DRP_A2  * f[idx - 2 * nx_all]
            - DRP_A3  * f[idx - 3 * nx_all]
        );
    }
}

// 4-2 MacCormack
template<MacDir DIR>
__device__ __forceinline__
float M42_ze(const float * __restrict__ f, int idx) {
    if constexpr (DIR == F) {
        return (
            + M42_A0 * f[idx]
            + M42_A1 * f[idx + nx_all]
            + M42_A2 * f[idx + 2 * nx_all]
        );
    } else {
        return (
            - M42_A0 * f[idx]
            - M42_A1 * f[idx - nx_all]
            - M42_A2 * f[idx - 2 * nx_all]
        );
    }
}

// 2-2 MacCormack
template<MacDir DIR>
__device__ __forceinline__
float M22_ze(const float * __restrict__ f, int idx) {
    if constexpr (DIR == F) {
        return (
            + M22_A0 * f[idx]
            + M22_A1 * f[idx + nx_all]
        );
    } else {
        return (
            - M22_A0 * f[idx]
            - M22_A1 * f[idx - nx_all]
        );
    }
}

// correct func
template<MacDir DIR>
__device__ __forceinline__ 
float Mac_dv_dze(
    const float * __restrict__ f, int idx, int dist, bool solid_side
) {
    if (dist >= 3) return DMac_ze<DIR>(f, idx);
    if (dist == 2) return M42_ze<DIR>(f, idx);
    if (dist == 1) return M22_ze<DIR>(f, idx);
    return solid_side ? M22_ze<F>(f, idx) : M22_ze<B>(f, idx);
}

template<MacDir DIR, class F2>
__device__ __forceinline__
float2 d_xi_flux(F2 func, int idx) {
    if constexpr (DIR == F) {
        return make_float2(
            (DRP_AM1 * func(idx - 1).x + DRP_A0 * func(idx).x +
             DRP_A1 * func(idx + 1).x + DRP_A2 * func(idx + 2).x +
             DRP_A3 * func(idx + 3).x),
            (DRP_AM1 * func(idx - 1).y + DRP_A0 * func(idx).y +
             DRP_A1 * func(idx + 1).y + DRP_A2 * func(idx + 2).y +
             DRP_A3 * func(idx + 3).y)
        );
    } else {
        return make_float2(
            -(DRP_AM1 * func(idx + 1).x + DRP_A0 * func(idx).x +
              DRP_A1 * func(idx - 1).x + DRP_A2 * func(idx - 2).x +
              DRP_A3 * func(idx - 3).x),
            -(DRP_AM1 * func(idx + 1).y + DRP_A0 * func(idx).y +
              DRP_A1 * func(idx - 1).y + DRP_A2 * func(idx - 2).y +
              DRP_A3 * func(idx - 3).y)
        );
    }
}

template<MacDir DIR, class F2>
__device__ __forceinline__
float2 d_ze_flux(F2 func, int idx) {
    if constexpr (DIR == F) {
        return make_float2(
            (DRP_AM1 * func(idx - nx_all).x + DRP_A0 * func(idx).x +
             DRP_A1 * func(idx + nx_all).x + DRP_A2 * func(idx + 2 * nx_all).x +
             DRP_A3 * func(idx + 3 * nx_all).x),
            (DRP_AM1 * func(idx - nx_all).y + DRP_A0 * func(idx).y +
             DRP_A1 * func(idx + nx_all).y + DRP_A2 * func(idx + 2 * nx_all).y +
             DRP_A3 * func(idx + 3 * nx_all).y)
        );
    } else {
        return make_float2(
            -(DRP_AM1 * func(idx + nx_all).x + DRP_A0 * func(idx).x +
              DRP_A1 * func(idx - nx_all).x + DRP_A2 * func(idx - 2 * nx_all).x +
              DRP_A3 * func(idx - 3 * nx_all).x),
            -(DRP_AM1 * func(idx + nx_all).y + DRP_A0 * func(idx).y +
              DRP_A1 * func(idx - nx_all).y + DRP_A2 * func(idx - 2 * nx_all).y +
              DRP_A3 * func(idx - 3 * nx_all).y)
        );
    }
}

/**
 * \brief ξ-direction flux (fluid)
 * \param Uf: fluid vector U
 * \param Jacb: Grid Jacobian parameters
 * \param idx: linear index of the grid point
 * \return float2 ξ-direction flux (fluid)
 */
__device__ float2 flux_xi_fluid(
    const U_fluid &Uf, const Grid &Jacb, int idx
);

/**
 * \brief ξ-direction flux (solid)
 * \param Us: solid vector U
 * \param Jacb: Grid Jacobian parameters
 * \param idx: linear index of the grid point
 * \return float2 ξ-direction flux (solid)
 */
__device__ float2 flux_xi_solid(
    const U_solid &Us, const Grid &Jacb, int idx
);

/**
 * \brief ζ-direction flux (fluid)
 * \param Uf: fluid vector U
 * \param Jacb: Grid Jacobian parameters
 * \param ix: spatial index of the grid point
 * \param off_z: ζ 方向相对界面的行号：0 = 界面（或自由表面）行，
 *               正 = 向介质内部，负 = 镜像幽灵槽（与 dir 无关）
 * \param ifz: 界面/自由表面所在行（在流体数组内的行号）
 *             流固界面 -> nz_f-1；流体顶面自由表面 -> IFZ_FREE
 * \param dir: 介质在界面的哪一侧：+1 = 介质在界面之下（行号增大方向），
 *            -1 = 介质在界面之上。off_z>0 时实际行号 = ifz + dir*off_z
 * \param Gf: ghost flux of near interface (fluid)
 * \return float2 ζ-direction flux (fluid)
 */
__device__ float2 flux_ze_fluid(
    const U_fluid &Uf, const Grid &Jacb, int ix, int off_z, int ifz, int dir,
    const float2 * __restrict__ Gf
);

/**
 * \brief ζ-direction flux (solid)
 * \param Us: solid vector U
 * \param Jacb: Grid Jacobian parameters
 * \param ix: spatial index of the grid point
 * \param off_z: ζ 方向相对界面的行号（含义同上）
 * \param ifz: 界面/自由表面所在行（在固体数组内的行号）
 *             流固界面 -> 0；顶面自由表面 -> IFZ_FREE
 * \param dir: 介质在界面的哪一侧：+1 = 介质在界面之下，-1 = 介质在界面之上
 * \param Gs: ghost flux of near interface (solid)
 * \return float2 ζ-direction flux (solid)
 */
__device__ float2 flux_ze_solid(
    const U_solid &Us, const Grid &Jacb, int ix, int off_z, int ifz, int dir,
    const float2 * __restrict__ Gs
);

#endif
