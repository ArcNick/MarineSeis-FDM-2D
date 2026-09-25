#ifndef DIRICHLET_KERNELS_CUH
#define DIRICHLET_KERNELS_CUH

#include "elements.h"
#include "tools_func.cuh"

namespace Dirichlet {
// Dirichlet boundary is only offered for SOLID

template<MacDir DIR>
__global__ void update_stress(
    U_solid Us, Med_solid Ms, Grid Jacb, U_solid Us_k
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix <= 2 || iz <= 2 || ix >= nx_all - 3 || iz >= nz_all - 3) {
        return;
    }

    const float * __restrict__ vx = Us.vx;
    const float * __restrict__ vz = Us.vz;

    int idx = iz * nx_all + ix;

    float Dvx_dxi = DMac_xi<DIR>(vx, idx);
    float Dvx_dze = DMac_ze<DIR>(vx, idx);
    float Dvz_dxi = DMac_xi<DIR>(vz, idx);
    float Dvz_dze = DMac_ze<DIR>(vz, idx);

    Us_k.sx[idx] = (
        + Ms.C11[idx] * (Jacb.xi_x[idx] * Dvx_dxi + Jacb.ze_x[idx] * Dvx_dze)
        + Ms.C13[idx] * (Jacb.xi_z[idx] * Dvz_dxi + Jacb.ze_z[idx] * Dvz_dze)
    );

    Us_k.sz[idx] = (
        + Ms.C13[idx] * (Jacb.xi_x[idx] * Dvx_dxi + Jacb.ze_x[idx] * Dvx_dze)
        + Ms.C33[idx] * (Jacb.xi_z[idx] * Dvz_dxi + Jacb.ze_z[idx] * Dvz_dze)
    );

    Us_k.sxz[idx] = (
        + Ms.C55[idx] * (
            + Jacb.xi_z[idx] * Dvx_dxi
            + Jacb.ze_z[idx] * Dvx_dze
            + Jacb.xi_x[idx] * Dvz_dxi
            + Jacb.ze_x[idx] * Dvz_dze
        )
    );
}

template<MacDir DIR>
__global__ void update_velocity(
    U_solid Us, Med_solid Ms, Grid Jacb, U_solid Us_k
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix <= 2 || iz <= 2 || ix >= nx_all - 3 || iz >= nz_all - 3) {
        return;
    }

    const float * __restrict__ sx  = Us.sx;
    const float * __restrict__ sz  = Us.sz;
    const float * __restrict__ sxz = Us.sxz;

    int idx = iz * nx_all + ix;

    float Dsx_dxi  = DMac_xi<DIR>(sx, idx);
    float Dsx_dze  = DMac_ze<DIR>(sx, idx);
    float Dsz_dxi  = DMac_xi<DIR>(sz, idx);
    float Dsz_dze  = DMac_ze<DIR>(sz, idx);
    float Dsxz_dxi = DMac_xi<DIR>(sxz, idx);
    float Dsxz_dze = DMac_ze<DIR>(sxz, idx);

    Us_k.vx[idx] = (
        + Jacb.xi_x[idx] * Dsx_dxi
        + Jacb.ze_x[idx] * Dsx_dze
        + Jacb.xi_z[idx] * Dsxz_dxi
        + Jacb.ze_z[idx] * Dsxz_dze
    ) / Ms.rho[idx];

    Us_k.vz[idx] = (
        + Jacb.xi_x[idx] * Dsxz_dxi
        + Jacb.ze_x[idx] * Dsxz_dze
        + Jacb.xi_z[idx] * Dsz_dxi
        + Jacb.ze_z[idx] * Dsz_dze
    ) / Ms.rho[idx];
}

__global__ void add_source(
    U_solid Us, int posx, int posz, float source
);

void linear_operate(
    const MemManager &mem, MacDir mdir, int isrc, float time,
    cudaStream_t stream_vel, cudaStream_t stream_sig
);

void update_RK4(
    const MemManager &mem, RK4Dir rdir, int isrc, int it,
    cudaStream_t stream_vel, cudaStream_t stream_sig
);

// end namespace Dirichlet
}

#endif
