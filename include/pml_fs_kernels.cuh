#ifndef FULL_MARINE_KERNELS_CUH
#define FULL_MARINE_KERNELS_CUH

#include "elements.h"
#include "tools_func.cuh"

// The 4 sides are ADE CFS-PML
namespace PmlFs::SolidOnly {

template<MacDir DIR>
__global__ void update_stress(
    U_solid Us, Med_solid Ms, Grid Jacb, U_solid Us_k,
    ADE ADEs_xi, ADE ADEs_xi_k, ADE ADEs_ze, ADE ADEs_ze_k,
    PML_par pml, Fsurface fs
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix <= 2 || iz <= 2 || ix >= nx_all - 3 || iz >= nz_all - 3) {
        return;
    }
    int idx = iz * nx_all + ix;
    int dist = iz - 3;
    int distx = min(ix - 3, nx_all - 4 - ix);

    float dvx_dxi = DMac_xi<DIR>(Us.vx, idx);
    float dvz_dxi = DMac_xi<DIR>(Us.vz, idx);

    bool in_side = (unsigned)distx < (unsigned)pml_span_d;
    bool in_bottom = iz >= nz_all - 3 - pml_span_d;

    if (dist == 0) {
        Us_k.sx[idx] = (
            + (Ms.C11[idx] * fs.alm[ix].al1_xx + Ms.C13[idx] * fs.alm[ix].al1_zz) * dvx_dxi
            + (Ms.C11[idx] * fs.alm[ix].al2_xx + Ms.C13[idx] * fs.alm[ix].al2_zz) * dvz_dxi
        );

        Us_k.sz[idx] = (
            + (Ms.C13[idx] * fs.alm[ix].al1_xx + Ms.C33[idx] * fs.alm[ix].al1_zz) * dvx_dxi
            + (Ms.C13[idx] * fs.alm[ix].al2_xx + Ms.C33[idx] * fs.alm[ix].al2_zz) * dvz_dxi
        );

        Us_k.sxz[idx] = Ms.C55[idx] * (
            fs.alm[ix].al1_xz * dvx_dxi + fs.alm[ix].al2_xz * dvz_dxi
        );

        if (in_side) {
            float dp_be = __ldg(pml.dp_xi + ix) / __ldg(pml.be_xi + ix);  // d / beta
            float apdp_be = __ldg(pml.al_xi + ix) + dp_be;                // alpha + d / beta
            float invb = 1.0f / __ldg(pml.be_xi + ix);                    // 1 / beta

            ADEs_xi_k.Ex[idx] = -apdp_be * ADEs_xi.Ex[idx] + dp_be * dvx_dxi;
            ADEs_xi_k.Ez[idx] = -apdp_be * ADEs_xi.Ez[idx] + dp_be * dvz_dxi;

            const float cxx_x = Ms.C11[idx] * fs.alm[ix].al1_xx + Ms.C13[idx] * fs.alm[ix].al1_zz;
            const float cxx_z = Ms.C11[idx] * fs.alm[ix].al2_xx + Ms.C13[idx] * fs.alm[ix].al2_zz;
            Us_k.sx[idx] += (
                + (invb - 1.0f) * (cxx_x * dvx_dxi + cxx_z * dvz_dxi)
                - invb * (cxx_x * ADEs_xi.Ex[idx] + cxx_z * ADEs_xi.Ez[idx])
            );

            const float czz_x = Ms.C13[idx] * fs.alm[ix].al1_xx + Ms.C33[idx] * fs.alm[ix].al1_zz;
            const float czz_z = Ms.C13[idx] * fs.alm[ix].al2_xx + Ms.C33[idx] * fs.alm[ix].al2_zz;

            Us_k.sz[idx] += (
                + (invb - 1.0f) * (czz_x * dvx_dxi + czz_z * dvz_dxi)
                - invb * (czz_x * ADEs_xi.Ex[idx] + czz_z * ADEs_xi.Ez[idx])
            );

            Us_k.sxz[idx] += Ms.C55[idx] * (
                + (invb - 1.0f) * (fs.alm[ix].al1_xz * dvx_dxi + fs.alm[ix].al2_xz * dvz_dxi)
                - invb * (fs.alm[ix].al1_xz * ADEs_xi.Ex[idx]
                        + fs.alm[ix].al2_xz * ADEs_xi.Ez[idx])
            );
        }
        return;
    }

    float dvx_dze = Mac_dv_dze<DIR>(Us.vx, idx, dist, true);
    float dvz_dze = Mac_dv_dze<DIR>(Us.vz, idx, dist, true);

    Us_k.sx[idx] = (
        + Ms.C11[idx] * (Jacb.xi_x[idx] * dvx_dxi + Jacb.ze_x[idx] * dvx_dze)
        + Ms.C13[idx] * (Jacb.xi_z[idx] * dvz_dxi + Jacb.ze_z[idx] * dvz_dze)
    );

    Us_k.sz[idx] = (
        + Ms.C13[idx] * (Jacb.xi_x[idx] * dvx_dxi + Jacb.ze_x[idx] * dvx_dze)
        + Ms.C33[idx] * (Jacb.xi_z[idx] * dvz_dxi + Jacb.ze_z[idx] * dvz_dze)
    );

    Us_k.sxz[idx] = (
        + Ms.C55[idx] * (
            + Jacb.xi_z[idx] * dvx_dxi
            + Jacb.ze_z[idx] * dvx_dze
            + Jacb.xi_x[idx] * dvz_dxi
            + Jacb.ze_x[idx] * dvz_dze
        )
    );

    if (!in_side && !in_bottom) {
        return;
    }

    float gxixi = Jacb.xi_x[idx] * Jacb.xi_x[idx] + Jacb.xi_z[idx] * Jacb.xi_z[idx];
    float gxize = Jacb.xi_x[idx] * Jacb.ze_x[idx] + Jacb.xi_z[idx] * Jacb.ze_z[idx];
    float gzeze = Jacb.ze_x[idx] * Jacb.ze_x[idx] + Jacb.ze_z[idx] * Jacb.ze_z[idx];

    if (in_side) {
        float Dxi_vx = dvx_dxi + (gxize / gxixi) * dvx_dze;
        float Dxi_vz = dvz_dxi + (gxize / gxixi) * dvz_dze;

        float dp_be = __ldg(pml.dp_xi + ix) / __ldg(pml.be_xi + ix);  // d / beta
        float apdp_be = __ldg(pml.al_xi + ix) + dp_be;                // alpha + d / beta
        float invb = 1.0f / __ldg(pml.be_xi + ix);                    // 1 / beta

        ADEs_xi_k.Ex[idx] = -apdp_be * ADEs_xi.Ex[idx] + dp_be * Dxi_vx;
        ADEs_xi_k.Ez[idx] = -apdp_be * ADEs_xi.Ez[idx] + dp_be * Dxi_vz;

        Us_k.sx[idx] += (
            + Ms.C11[idx] * (invb - 1.0f) * Jacb.xi_x[idx] * Dxi_vx
            + Ms.C13[idx] * (invb - 1.0f) * Jacb.xi_z[idx] * Dxi_vz
            - invb * Ms.C11[idx] * Jacb.xi_x[idx] * ADEs_xi.Ex[idx]
            - invb * Ms.C13[idx] * Jacb.xi_z[idx] * ADEs_xi.Ez[idx]
        );

        Us_k.sz[idx] += (
            + Ms.C13[idx] * (invb - 1.0f) * Jacb.xi_x[idx] * Dxi_vx
            + Ms.C33[idx] * (invb - 1.0f) * Jacb.xi_z[idx] * Dxi_vz
            - invb * Ms.C13[idx] * Jacb.xi_x[idx] * ADEs_xi.Ex[idx]
            - invb * Ms.C33[idx] * Jacb.xi_z[idx] * ADEs_xi.Ez[idx]
        );

        Us_k.sxz[idx] += (
            + Ms.C55[idx] * (invb - 1.0f) * (
                Jacb.xi_z[idx] * Dxi_vx + Jacb.xi_x[idx] * Dxi_vz
            )
            - invb * Ms.C55[idx] * Jacb.xi_z[idx] * ADEs_xi.Ex[idx]
            - invb * Ms.C55[idx] * Jacb.xi_x[idx] * ADEs_xi.Ez[idx]
        );
    }

    if (in_bottom) {
        float Dze_vx = dvx_dze + (gxize / gzeze) * dvx_dxi;
        float Dze_vz = dvz_dze + (gxize / gzeze) * dvz_dxi;

        float dp_be = __ldg(pml.dp_ze_s + iz) / __ldg(pml.be_ze_s + iz);
        float apdp_be = __ldg(pml.al_ze_s + iz) + dp_be;
        float invb = 1.0f / __ldg(pml.be_ze_s + iz);

        ADEs_ze_k.Ex[idx] = -apdp_be * ADEs_ze.Ex[idx] + dp_be * Dze_vx;
        ADEs_ze_k.Ez[idx] = -apdp_be * ADEs_ze.Ez[idx] + dp_be * Dze_vz;

        Us_k.sx[idx] += (
            + Ms.C11[idx] * (invb - 1.0f) * Jacb.ze_x[idx] * Dze_vx
            + Ms.C13[idx] * (invb - 1.0f) * Jacb.ze_z[idx] * Dze_vz
            - invb * Ms.C11[idx] * Jacb.ze_x[idx] * ADEs_ze.Ex[idx]
            - invb * Ms.C13[idx] * Jacb.ze_z[idx] * ADEs_ze.Ez[idx]
        );

        Us_k.sz[idx] += (
            + Ms.C13[idx] * (invb - 1.0f) * Jacb.ze_x[idx] * Dze_vx
            + Ms.C33[idx] * (invb - 1.0f) * Jacb.ze_z[idx] * Dze_vz
            - invb * Ms.C13[idx] * Jacb.ze_x[idx] * ADEs_ze.Ex[idx]
            - invb * Ms.C33[idx] * Jacb.ze_z[idx] * ADEs_ze.Ez[idx]
        );

        Us_k.sxz[idx] += (
            + Ms.C55[idx] * (invb - 1.0f) * (
                Jacb.ze_z[idx] * Dze_vx + Jacb.ze_x[idx] * Dze_vz
            )
            - invb * Ms.C55[idx] * Jacb.ze_z[idx] * ADEs_ze.Ex[idx]
            - invb * Ms.C55[idx] * Jacb.ze_x[idx] * ADEs_ze.Ez[idx]
        );
    }
}

template<MacDir DIR>
__global__ void update_velocity(
    U_solid Us, Med_solid Ms, Grid Jacb, U_solid Us_k,
    ADE ADEs_xi, ADE ADEs_xi_k, ADE ADEs_ze, ADE ADEs_ze_k,
    PML_par pml, Fsurface fs
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix <= 2 || iz <= 2 || ix >= nx_all - 3 || iz >= nz_all - 3) {
        return;
    }
    int idx = iz * nx_all + ix;
    int dist = iz - 3;

    float rhodvx_dt, rhodvz_dt;
    float Div_xi_S_x, Div_xi_S_z;
    float Div_ze_S_x, Div_ze_S_z;

    float gxixi = Jacb.xi_x[idx] * Jacb.xi_x[idx] + Jacb.xi_z[idx] * Jacb.xi_z[idx];
    float gzeze = Jacb.ze_x[idx] * Jacb.ze_x[idx] + Jacb.ze_z[idx] * Jacb.ze_z[idx];

    if (dist < 3) {
        auto Fxi = [&](int k) {
            return flux_xi_solid(Us, Jacb, k);
        };
        auto Fze = [&](int k) {
            return flux_ze_solid(Us, Jacb, ix,
                                 (k - IFZ_FREE * nx_all - ix) / nx_all,
                                 IFZ_FREE, DIR_BELOW, fs.Gfs);
        };

        float2 dF_dxi = d_xi_flux<DIR>(Fxi, idx);
        float2 dF_dze = d_ze_flux<DIR>(Fze, idx);

        rhodvx_dt = (dF_dxi.x + dF_dze.x) / Jacb.J[idx];
        rhodvz_dt = (dF_dxi.y + dF_dze.y) / Jacb.J[idx];

        {
            const float gxize = Jacb.xi_x[idx] * Jacb.ze_x[idx]
                              + Jacb.xi_z[idx] * Jacb.ze_z[idx];
            const float dxsx  = DMac_xi<DIR>(Us.sx, idx);
            const float dxsz  = DMac_xi<DIR>(Us.sz, idx);
            const float dxsxz = DMac_xi<DIR>(Us.sxz, idx);
            const float dzsx  = Mac_dv_dze<DIR>(Us.sx,  idx, dist, true);
            const float dzsz  = Mac_dv_dze<DIR>(Us.sz,  idx, dist, true);
            const float dzsxz = Mac_dv_dze<DIR>(Us.sxz, idx, dist, true);

            Div_xi_S_x = (
                + Jacb.xi_x[idx] * (dxsx  + gxize / gxixi * dzsx)
                + Jacb.xi_z[idx] * (dxsxz + gxize / gxixi * dzsxz)
            );
            Div_xi_S_z = (
                + Jacb.xi_x[idx] * (dxsxz + gxize / gxixi * dzsxz)
                + Jacb.xi_z[idx] * (dxsz  + gxize / gxixi * dzsz)
            );
            Div_ze_S_x = 0.0f;
            Div_ze_S_z = 0.0f;
        }
    } else {
        float dsx_dxi  = DMac_xi<DIR>(Us.sx, idx);
        float dsx_dze  = DMac_ze<DIR>(Us.sx, idx);
        float dsz_dxi  = DMac_xi<DIR>(Us.sz, idx);
        float dsz_dze  = DMac_ze<DIR>(Us.sz, idx);
        float dsxz_dxi = DMac_xi<DIR>(Us.sxz, idx);
        float dsxz_dze = DMac_ze<DIR>(Us.sxz, idx);

        rhodvx_dt = (
            + Jacb.xi_x[idx] * dsx_dxi + Jacb.ze_x[idx] * dsx_dze
            + Jacb.xi_z[idx] * dsxz_dxi + Jacb.ze_z[idx] * dsxz_dze
        );
        rhodvz_dt = (
            + Jacb.xi_x[idx] * dsxz_dxi + Jacb.ze_x[idx] * dsxz_dze
            + Jacb.xi_z[idx] * dsz_dxi + Jacb.ze_z[idx] * dsz_dze            
        );

        float gxize = Jacb.xi_x[idx] * Jacb.ze_x[idx] + Jacb.xi_z[idx] * Jacb.ze_z[idx];
        Div_xi_S_x = (
            + Jacb.xi_x[idx] * (dsx_dxi + gxize / gxixi * dsx_dze)
            + Jacb.xi_z[idx] * (dsxz_dxi + gxize / gxixi * dsxz_dze)
        );
        Div_xi_S_z = (
            + Jacb.xi_x[idx] * (dsxz_dxi + gxize / gxixi * dsxz_dze)
            + Jacb.xi_z[idx] * (dsz_dxi + gxize / gxixi * dsz_dze)
        );
        Div_ze_S_x = (
            + Jacb.ze_x[idx] * (dsx_dze + gxize / gzeze * dsx_dxi)
            + Jacb.ze_z[idx] * (gxize / gzeze * dsxz_dxi + dsxz_dze)
        );
        Div_ze_S_z = (
            + Jacb.ze_x[idx] * (gxize / gzeze * dsxz_dxi + dsxz_dze)
            + Jacb.ze_z[idx] * (gxize / gzeze * dsz_dxi + dsz_dze)
        );
    }

    int distx = min(ix - 3, nx_all - 4 - ix);
    bool in_side   = (unsigned)distx < (unsigned)pml_span_d;
    bool in_bottom = iz >= nz_s - 3 - pml_span_d;

    Us_k.vx[idx] = rhodvx_dt / Ms.rho[idx];
    Us_k.vz[idx] = rhodvz_dt / Ms.rho[idx];

    if (!in_side && !in_bottom) {
        return;
    }

    if (in_side) {
        float dp_be = __ldg(pml.dp_xi + ix) / __ldg(pml.be_xi + ix);
        float apdp_be = __ldg(pml.al_xi + ix) + dp_be;
        float invb = 1.0f / __ldg(pml.be_xi + ix);

        ADEs_xi_k.Vx[idx] = -apdp_be * ADEs_xi.Vx[idx] + dp_be * Div_xi_S_x;
        ADEs_xi_k.Vz[idx] = -apdp_be * ADEs_xi.Vz[idx] + dp_be * Div_xi_S_z;
        
        Us_k.vx[idx] += ((invb - 1.0f) * Div_xi_S_x - invb * ADEs_xi.Vx[idx]) / Ms.rho[idx];
        Us_k.vz[idx] += ((invb - 1.0f) * Div_xi_S_z - invb * ADEs_xi.Vz[idx]) / Ms.rho[idx];
    }

    if (in_bottom) {
        float dp_be = __ldg(pml.dp_ze_s + iz) / __ldg(pml.be_ze_s + iz);
        float apdp_be = __ldg(pml.al_ze_s + iz) + dp_be;
        float invb = 1.0f / __ldg(pml.be_ze_s + iz);

        ADEs_ze_k.Vx[idx] = -apdp_be * ADEs_ze.Vx[idx] + dp_be * Div_ze_S_x;
        ADEs_ze_k.Vz[idx] = -apdp_be * ADEs_ze.Vz[idx] + dp_be * Div_ze_S_z;

        Us_k.vx[idx] += ((invb - 1.0f) * Div_ze_S_x - invb * ADEs_ze.Vx[idx]) / Ms.rho[idx];
        Us_k.vz[idx] += ((invb - 1.0f) * Div_ze_S_z - invb * ADEs_ze.Vz[idx]) / Ms.rho[idx];
    }
}

__global__ void build_ghost_flux(
    const U_solid Us, const Grid Jacb, float2 * __restrict__ Gfs
);

__global__ void add_source(
    U_solid Us, int posx, int posz, float source
);

void linear_operate(
    const MemManager &mem, MacDir mdir, int isrc, float time,
    cudaStream_t stream_vel, cudaStream_t stream_sig
);

void update_RK4(
    const MemManager &mem_manager, RK4Dir rdir, int isrc, int it,
    cudaStream_t stream_vel, cudaStream_t stream_sig
);

} // end of namespace PmlFs::SolidOnly

/******************************************************/

namespace PmlFs::FluidSolid {

template<MacDir DIR>
__global__ void update_velocity_fluid(
    U_fluid Uf, Med_fluid Mf, Grid Jacb, U_fluid Uf_k,
    ADE ADEf_xi, ADE ADEf_xi_k, ADE ADEf_ze, ADE ADEf_ze_k,
    PML_par pml, Flux flux, Fsurface fs
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix <= 2 || iz <= 2 || ix >= nx_all - 3 || iz >= nz_f) {
        return;
    }
    int idx = iz * nx_all + ix;
    int dist_flso = nz_f - 1 - iz;
    int dist_fs = iz - 3;

    float rhodvx_dt, rhodvz_dt;
    float Div_xi_P_x, Div_xi_P_z;
    float Div_ze_P_x, Div_ze_P_z;

    float gxixi = Jacb.xi_x[idx] * Jacb.xi_x[idx] + Jacb.xi_z[idx] * Jacb.xi_z[idx];
    float gzeze = Jacb.ze_x[idx] * Jacb.ze_x[idx] + Jacb.ze_z[idx] * Jacb.ze_z[idx];

    if (dist_flso < 3) {
        // conservation equation
        auto Fxi = [&](int k) {
            return flux_xi_fluid(Uf, Jacb, k);
        };
        auto Fze = [&](int k) {
            return flux_ze_fluid(
                Uf, Jacb, ix, (nz_f - 1) - k / nx_all, nz_f - 1, DIR_ABOVE, flux.Gf
            );
        };

        float2 dF_dxi = d_xi_flux<DIR>(Fxi, idx);
        float2 dF_dze = d_ze_flux<DIR>(Fze, idx);

        rhodvx_dt = (dF_dxi.x + dF_dze.x) / Jacb.J[idx];
        rhodvz_dt = (dF_dxi.y + dF_dze.y) / Jacb.J[idx];

        Div_xi_P_x = Jacb.xi_x[idx] * (
            Jacb.xi_x[idx] * (dF_dxi.x + dF_dze.x) + Jacb.xi_z[idx] * (dF_dxi.y + dF_dze.y)
        ) / (gxixi * Jacb.J[idx]);
        Div_xi_P_z = Jacb.xi_z[idx] * (
            Jacb.xi_x[idx] * (dF_dxi.x + dF_dze.x) + Jacb.xi_z[idx] * (dF_dxi.y + dF_dze.y)
        ) / (gxixi * Jacb.J[idx]);
        Div_ze_P_x = Jacb.ze_x[idx] * (
            Jacb.ze_x[idx] * (dF_dxi.x + dF_dze.x) + Jacb.ze_z[idx] * (dF_dxi.y + dF_dze.y)
        ) / (gzeze * Jacb.J[idx]);
        Div_ze_P_z = Jacb.ze_z[idx] * (
            Jacb.ze_x[idx] * (dF_dxi.x + dF_dze.x) + Jacb.ze_z[idx] * (dF_dxi.y + dF_dze.y)
        ) / (gzeze * Jacb.J[idx]);

    } else if (dist_fs < 3) {
        auto Fxi = [&](int k) {
            return flux_xi_fluid(Uf, Jacb, k);
        };
        auto Fze = [&](int k) {
            return flux_ze_fluid(
                Uf, Jacb, ix, (k - IFZ_FREE * nx_all - ix) / nx_all, 
                IFZ_FREE, DIR_BELOW, fs.Gfs
            );
        };

        float2 dF_dxi = d_xi_flux<DIR>(Fxi, idx);
        float2 dF_dze = d_ze_flux<DIR>(Fze, idx);

        rhodvx_dt = (dF_dxi.x + dF_dze.x) / Jacb.J[idx];
        rhodvz_dt = (dF_dxi.y + dF_dze.y) / Jacb.J[idx];

        Div_xi_P_x = Jacb.xi_x[idx] * (
            Jacb.xi_x[idx] * (dF_dxi.x + dF_dze.x) + Jacb.xi_z[idx] * (dF_dxi.y + dF_dze.y)
        ) / (gxixi * Jacb.J[idx]);
        Div_xi_P_z = Jacb.xi_z[idx] * (
            Jacb.xi_x[idx] * (dF_dxi.x + dF_dze.x) + Jacb.xi_z[idx] * (dF_dxi.y + dF_dze.y)
        ) / (gxixi * Jacb.J[idx]);
        Div_ze_P_x = Jacb.ze_x[idx] * (
            Jacb.ze_x[idx] * (dF_dxi.x + dF_dze.x) + Jacb.ze_z[idx] * (dF_dxi.y + dF_dze.y)
        ) / (gzeze * Jacb.J[idx]);
        Div_ze_P_z = Jacb.ze_z[idx] * (
            Jacb.ze_x[idx] * (dF_dxi.x + dF_dze.x) + Jacb.ze_z[idx] * (dF_dxi.y + dF_dze.y)
        ) / (gzeze * Jacb.J[idx]);

    } else {
        // non-conservation equation
        float dp_dxi = DMac_xi<DIR>(Uf.p, idx);
        float dp_dze = DMac_ze<DIR>(Uf.p, idx);

        rhodvx_dt = -(Jacb.xi_x[idx] * dp_dxi + Jacb.ze_x[idx] * dp_dze);
        rhodvz_dt = -(Jacb.xi_z[idx] * dp_dxi + Jacb.ze_z[idx] * dp_dze);

        float gxize = Jacb.xi_x[idx] * Jacb.ze_x[idx] + Jacb.xi_z[idx] * Jacb.ze_z[idx];

        Div_xi_P_x = -Jacb.xi_x[idx] * (gxixi * dp_dxi + gxize * dp_dze) / gxixi;
        Div_xi_P_z = -Jacb.xi_z[idx] * (gxixi * dp_dxi + gxize * dp_dze) / gxixi;
        Div_ze_P_x = -Jacb.ze_x[idx] * (gxize * dp_dxi + gzeze * dp_dze) / gzeze;
        Div_ze_P_z = -Jacb.ze_z[idx] * (gxize * dp_dxi + gzeze * dp_dze) / gzeze;
    }

    Uf_k.vx[idx] = rhodvx_dt / Mf.rho[idx];
    Uf_k.vz[idx] = rhodvz_dt / Mf.rho[idx];

    int distx = min(ix - 3, nx_all - 4 - ix);
    bool in_side = (unsigned)distx < (unsigned)pml_span_d;

    if (in_side) {
        float dp_be = __ldg(pml.dp_xi + ix) / __ldg(pml.be_xi + ix);
        float apdp_be = __ldg(pml.al_xi + ix) + dp_be;
        float invb = 1.0f / __ldg(pml.be_xi + ix);

        ADEf_xi_k.Vx[idx] = -apdp_be * ADEf_xi.Vx[idx] + dp_be * Div_xi_P_x;
        ADEf_xi_k.Vz[idx] = -apdp_be * ADEf_xi.Vz[idx] + dp_be * Div_xi_P_z;

        Uf_k.vx[idx] += ((invb - 1.0f) * Div_xi_P_x - invb * ADEf_xi.Vx[idx]) / Mf.rho[idx];
        Uf_k.vz[idx] += ((invb - 1.0f) * Div_xi_P_z - invb * ADEf_xi.Vz[idx]) / Mf.rho[idx];
    }

}

template<MacDir DIR>
__global__ void update_stress_fluid(
    U_fluid Uf, Med_fluid Mf, Grid Jacb, U_fluid Uf_k,
    ADE ADEf_xi, ADE ADEf_xi_k, ADE ADEf_ze, ADE ADEf_ze_k,
    PML_par pml, Fsurface fs
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix <= 2 || iz <= 2 || ix >= nx_all - 3 || iz >= nz_f) {
        return;
    }
    int idx = iz * nx_all + ix;
    int dist_fs = iz - 3;
    int distx = min(ix - 3, nx_all - 4 - ix);

    float dvx_dxi = DMac_xi<DIR>(Uf.vx, idx);
    float dvz_dxi = DMac_xi<DIR>(Uf.vz, idx);

    bool in_side = (unsigned)distx < (unsigned)pml_span_d;
    if (dist_fs == 0) {
        Uf_k.p[idx] = 0.0f;
        return;
    }
    int dz_near = min(iz - IFZ_FREE, nz_f - 1 - iz);
    float dvx_dze = Mac_dv_dze<DIR>(Uf.vx, idx, dz_near, false);
    float dvz_dze = Mac_dv_dze<DIR>(Uf.vz, idx, dz_near, false);

    // σ = -pI  =>  p,t = -K div v
    Uf_k.p[idx] = -Mf.K[idx] * (
        + Jacb.xi_x[idx] * dvx_dxi + Jacb.ze_x[idx] * dvx_dze
        + Jacb.xi_z[idx] * dvz_dxi + Jacb.ze_z[idx] * dvz_dze
    );

    if (in_side) {
        float gxixi = Jacb.xi_x[idx] * Jacb.xi_x[idx] + Jacb.xi_z[idx] * Jacb.xi_z[idx];
        float gxize = Jacb.xi_x[idx] * Jacb.ze_x[idx] + Jacb.xi_z[idx] * Jacb.ze_z[idx];

        float Dxi_vx = dvx_dxi + (gxize / gxixi) * dvx_dze;
        float Dxi_vz = dvz_dxi + (gxize / gxixi) * dvz_dze;

        float dp_be = __ldg(pml.dp_xi + ix) / __ldg(pml.be_xi + ix);        // d / beta
        float apdp_be = __ldg(pml.al_xi + ix) + dp_be;                      // alpha + d / beta
        float invb = 1.0f / __ldg(pml.be_xi + ix);                          // 1 / beta

        ADEf_xi_k.Ex[idx] = -apdp_be * ADEf_xi.Ex[idx] + dp_be * Dxi_vx;
        ADEf_xi_k.Ez[idx] = -apdp_be * ADEf_xi.Ez[idx] + dp_be * Dxi_vz;

        Uf_k.p[idx] += Mf.K[idx] * (
            - (invb - 1.0f) * Jacb.xi_x[idx] * Dxi_vx
            - (invb - 1.0f) * Jacb.xi_z[idx] * Dxi_vz
            + invb * Jacb.xi_x[idx] * ADEf_xi.Ex[idx]
            + invb * Jacb.xi_z[idx] * ADEf_xi.Ez[idx]
        );
    }
}

template<MacDir DIR>
__global__ void update_stress_solid(
    U_solid Us, Med_solid Ms, Grid Jacb, U_solid Us_k,
    ADE ADEs_xi, ADE ADEs_xi_k, ADE ADEs_ze, ADE ADEs_ze_k,
    PML_par pml
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix <= 2 || ix >= nx_all - 3 || iz >= nz_s - 3) {
        return;
    }

    int idx = iz * nx_all + ix;
    int idx_J = idx + (nz_f - 1) * nx_all;
    
    float dvx_dxi = DMac_xi<DIR>(Us.vx, idx);
    float dvx_dze = Mac_dv_dze<DIR>(Us.vx, idx, iz, true);
    float dvz_dxi = DMac_xi<DIR>(Us.vz, idx);
    float dvz_dze = Mac_dv_dze<DIR>(Us.vz, idx, iz, true);

    Us_k.sx[idx] = (
        + Ms.C11[idx] * (Jacb.xi_x[idx_J] * dvx_dxi + Jacb.ze_x[idx_J] * dvx_dze)
        + Ms.C13[idx] * (Jacb.xi_z[idx_J] * dvz_dxi + Jacb.ze_z[idx_J] * dvz_dze)
    );

    Us_k.sz[idx] = (
        + Ms.C13[idx] * (Jacb.xi_x[idx_J] * dvx_dxi + Jacb.ze_x[idx_J] * dvx_dze)
        + Ms.C33[idx] * (Jacb.xi_z[idx_J] * dvz_dxi + Jacb.ze_z[idx_J] * dvz_dze)
    );

    Us_k.sxz[idx] = (
        + Ms.C55[idx] * (
            + Jacb.xi_z[idx_J] * dvx_dxi
            + Jacb.ze_z[idx_J] * dvx_dze
            + Jacb.xi_x[idx_J] * dvz_dxi
            + Jacb.ze_x[idx_J] * dvz_dze
        )
    );

    int distx = min(ix - 3, nx_all - 4 - ix);
    bool in_side = (unsigned)distx < (unsigned)pml_span_d;
    bool in_bottom = iz >= nz_s - 3 - pml_span_d;

    if (!in_side && !in_bottom) {
        return;
    }

    float gxixi = Jacb.xi_x[idx_J] * Jacb.xi_x[idx_J] + Jacb.xi_z[idx_J] * Jacb.xi_z[idx_J];
    float gxize = Jacb.xi_x[idx_J] * Jacb.ze_x[idx_J] + Jacb.xi_z[idx_J] * Jacb.ze_z[idx_J];
    float gzeze = Jacb.ze_x[idx_J] * Jacb.ze_x[idx_J] + Jacb.ze_z[idx_J] * Jacb.ze_z[idx_J];

    if (in_side) {
        float Dxi_vx = dvx_dxi + (gxize / gxixi) * dvx_dze;
        float Dxi_vz = dvz_dxi + (gxize / gxixi) * dvz_dze;

        float dp_be = __ldg(pml.dp_xi + ix) / __ldg(pml.be_xi + ix);  // d / beta
        float apdp_be = __ldg(pml.al_xi + ix) + dp_be;                // alpha + d / beta
        float invb = 1.0f / __ldg(pml.be_xi + ix);                    // 1 / beta

        ADEs_xi_k.Ex[idx] = -apdp_be * ADEs_xi.Ex[idx] + dp_be * Dxi_vx;
        ADEs_xi_k.Ez[idx] = -apdp_be * ADEs_xi.Ez[idx] + dp_be * Dxi_vz;

        Us_k.sx[idx] += (
            + Ms.C11[idx] * (invb - 1.0f) * Jacb.xi_x[idx_J] * Dxi_vx
            + Ms.C13[idx] * (invb - 1.0f) * Jacb.xi_z[idx_J] * Dxi_vz
            - invb * Ms.C11[idx] * Jacb.xi_x[idx_J] * ADEs_xi.Ex[idx]
            - invb * Ms.C13[idx] * Jacb.xi_z[idx_J] * ADEs_xi.Ez[idx]
        );

        Us_k.sz[idx] += (
            + Ms.C13[idx] * (invb - 1.0f) * Jacb.xi_x[idx_J] * Dxi_vx
            + Ms.C33[idx] * (invb - 1.0f) * Jacb.xi_z[idx_J] * Dxi_vz
            - invb * Ms.C13[idx] * Jacb.xi_x[idx_J] * ADEs_xi.Ex[idx]
            - invb * Ms.C33[idx] * Jacb.xi_z[idx_J] * ADEs_xi.Ez[idx]
        );

        Us_k.sxz[idx] += (
            + Ms.C55[idx] * (invb - 1.0f) * (
                Jacb.xi_z[idx_J] * Dxi_vx + Jacb.xi_x[idx_J] * Dxi_vz
            )
            - invb * Ms.C55[idx] * Jacb.xi_z[idx_J] * ADEs_xi.Ex[idx]
            - invb * Ms.C55[idx] * Jacb.xi_x[idx_J] * ADEs_xi.Ez[idx]
        );
    }

    if (in_bottom) {
        float Dze_vx = dvx_dze + (gxize / gzeze) * dvx_dxi;
        float Dze_vz = dvz_dze + (gxize / gzeze) * dvz_dxi;

        float dp_be = __ldg(pml.dp_ze_s + iz) / __ldg(pml.be_ze_s + iz);
        float apdp_be = __ldg(pml.al_ze_s + iz) + dp_be;
        float invb = 1.0f / __ldg(pml.be_ze_s + iz);

        ADEs_ze_k.Ex[idx] = -apdp_be * ADEs_ze.Ex[idx] + dp_be * Dze_vx;
        ADEs_ze_k.Ez[idx] = -apdp_be * ADEs_ze.Ez[idx] + dp_be * Dze_vz;

        Us_k.sx[idx] += (
            + Ms.C11[idx] * (invb - 1.0f) * Jacb.ze_x[idx_J] * Dze_vx
            + Ms.C13[idx] * (invb - 1.0f) * Jacb.ze_z[idx_J] * Dze_vz
            - invb * Ms.C11[idx] * Jacb.ze_x[idx_J] * ADEs_ze.Ex[idx]
            - invb * Ms.C13[idx] * Jacb.ze_z[idx_J] * ADEs_ze.Ez[idx]
        );

        Us_k.sz[idx] += (
            + Ms.C13[idx] * (invb - 1.0f) * Jacb.ze_x[idx_J] * Dze_vx
            + Ms.C33[idx] * (invb - 1.0f) * Jacb.ze_z[idx_J] * Dze_vz
            - invb * Ms.C13[idx] * Jacb.ze_x[idx_J] * ADEs_ze.Ex[idx]
            - invb * Ms.C33[idx] * Jacb.ze_z[idx_J] * ADEs_ze.Ez[idx]
        );

        Us_k.sxz[idx] += (
            + Ms.C55[idx] * (invb - 1.0f) * (
                Jacb.ze_z[idx_J] * Dze_vx + Jacb.ze_x[idx_J] * Dze_vz
            )
            - invb * Ms.C55[idx] * Jacb.ze_z[idx_J] * ADEs_ze.Ex[idx]
            - invb * Ms.C55[idx] * Jacb.ze_x[idx_J] * ADEs_ze.Ez[idx]
        );
    }
}


template<MacDir DIR>
__global__ void update_velocity_solid(
    U_solid Us, Med_solid Ms, Grid Jacb, U_solid Us_k,
    ADE ADEs_xi, ADE ADEs_xi_k, ADE ADEs_ze, ADE ADEs_ze_k,
    PML_par pml, Flux flux
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix <= 2 || ix >= nx_all - 3 || iz >= nz_s - 3) {
        return;
    }
    int idx = iz * nx_all + ix;
    int idx_J = idx + (nz_f - 1) * nx_all;
    int dist = iz;

    float rhodvx_dt, rhodvz_dt;
    float Div_xi_S_x, Div_xi_S_z;
    float Div_ze_S_x, Div_ze_S_z;

    float gxixi = Jacb.xi_x[idx_J] * Jacb.xi_x[idx_J] + Jacb.xi_z[idx_J] * Jacb.xi_z[idx_J];
    float gzeze = Jacb.ze_x[idx_J] * Jacb.ze_x[idx_J] + Jacb.ze_z[idx_J] * Jacb.ze_z[idx_J];

    if (dist < 3) {
        // conservation equation
        auto Fxi = [&](int k) {
            return flux_xi_solid(Us, Jacb, k);
        };
        auto Fze = [&](int k) {
            return flux_ze_solid(Us, Jacb, ix, (k - ix) / nx_all, 0, DIR_BELOW, flux.Gs);
        };

        float2 dF_dxi = d_xi_flux<DIR>(Fxi, idx);
        float2 dF_dze = d_ze_flux<DIR>(Fze, idx);

        rhodvx_dt = (dF_dxi.x + dF_dze.x) / Jacb.J[idx_J];
        rhodvz_dt = (dF_dxi.y + dF_dze.y) / Jacb.J[idx_J];

        Div_xi_S_x = Jacb.xi_x[idx_J] * (
            Jacb.xi_x[idx_J] * (dF_dxi.x + dF_dze.x) + Jacb.xi_z[idx_J] * (dF_dxi.y + dF_dze.y)
        ) / (gxixi * Jacb.J[idx_J]);
        Div_xi_S_z = Jacb.xi_z[idx_J] * (
            Jacb.xi_x[idx_J] * (dF_dxi.x + dF_dze.x) + Jacb.xi_z[idx_J] * (dF_dxi.y + dF_dze.y)
        ) / (gxixi * Jacb.J[idx_J]);
        Div_ze_S_x = Jacb.ze_x[idx_J] * (
            Jacb.ze_x[idx_J] * (dF_dxi.x + dF_dze.x) + Jacb.ze_z[idx_J] * (dF_dxi.y + dF_dze.y)
        ) / (gzeze * Jacb.J[idx_J]);
        Div_ze_S_z = Jacb.ze_z[idx_J] * (
            Jacb.ze_x[idx_J] * (dF_dxi.x + dF_dze.x) + Jacb.ze_z[idx_J] * (dF_dxi.y + dF_dze.y)
        ) / (gzeze * Jacb.J[idx_J]);
    } else {
        // non-conservation equation
        float dsx_dxi = DMac_xi<DIR>(Us.sx, idx);
        float dsx_dze = DMac_ze<DIR>(Us.sx, idx);
        float dsz_dxi = DMac_xi<DIR>(Us.sz, idx);
        float dsz_dze = DMac_ze<DIR>(Us.sz, idx);
        float dsxz_dxi = DMac_xi<DIR>(Us.sxz, idx);
        float dsxz_dze = DMac_ze<DIR>(Us.sxz, idx);

        rhodvx_dt = (
            + Jacb.xi_x[idx_J] * dsx_dxi + Jacb.ze_x[idx_J] * dsx_dze
            + Jacb.xi_z[idx_J] * dsxz_dxi + Jacb.ze_z[idx_J] * dsxz_dze
        );
        rhodvz_dt = (
            + Jacb.xi_x[idx_J] * dsxz_dxi + Jacb.ze_x[idx_J] * dsxz_dze
            + Jacb.xi_z[idx_J] * dsz_dxi + Jacb.ze_z[idx_J] * dsz_dze            
        );

        float gxize = Jacb.xi_x[idx_J] * Jacb.ze_x[idx_J] + Jacb.xi_z[idx_J] * Jacb.ze_z[idx_J];
        Div_xi_S_x = (
            + Jacb.xi_x[idx_J] * (dsx_dxi + gxize / gxixi * dsx_dze)
            + Jacb.xi_z[idx_J] * (dsxz_dxi + gxize / gxixi * dsxz_dze)
        );
        Div_xi_S_z = (
            + Jacb.xi_x[idx_J] * (dsxz_dxi + gxize / gxixi * dsxz_dze)
            + Jacb.xi_z[idx_J] * (dsz_dxi + gxize / gxixi * dsz_dze)
        );
        Div_ze_S_x = (
            + Jacb.ze_x[idx_J] * (dsx_dze + gxize / gzeze * dsx_dxi)
            + Jacb.ze_z[idx_J] * (gxize / gzeze * dsxz_dxi + dsxz_dze)
        );
        Div_ze_S_z = (
            + Jacb.ze_x[idx_J] * (gxize / gzeze * dsxz_dxi + dsxz_dze)
            + Jacb.ze_z[idx_J] * (gxize / gzeze * dsz_dxi + dsz_dze)
        );
    }

    int distx = min(ix - 3, nx_all - 4 - ix);
    bool in_side   = (unsigned)distx < (unsigned)pml_span_d;
    bool in_bottom = iz >= nz_s - 3 - pml_span_d;

    Us_k.vx[idx] = rhodvx_dt / Ms.rho[idx];
    Us_k.vz[idx] = rhodvz_dt / Ms.rho[idx];

    if (!in_side && !in_bottom) {
        return;
    }

    if (in_side) {
        float dp_be = __ldg(pml.dp_xi + ix) / __ldg(pml.be_xi + ix);
        float apdp_be = __ldg(pml.al_xi + ix) + dp_be;
        float invb = 1.0f / __ldg(pml.be_xi + ix);

        ADEs_xi_k.Vx[idx] = -apdp_be * ADEs_xi.Vx[idx] + dp_be * Div_xi_S_x;
        ADEs_xi_k.Vz[idx] = -apdp_be * ADEs_xi.Vz[idx] + dp_be * Div_xi_S_z;
        
        Us_k.vx[idx] += ((invb - 1.0f) * Div_xi_S_x - invb * ADEs_xi.Vx[idx]) / Ms.rho[idx];
        Us_k.vz[idx] += ((invb - 1.0f) * Div_xi_S_z - invb * ADEs_xi.Vz[idx]) / Ms.rho[idx];
    }

    if (in_bottom) {
        float dp_be = __ldg(pml.dp_ze_s + iz) / __ldg(pml.be_ze_s + iz);
        float apdp_be = __ldg(pml.al_ze_s + iz) + dp_be;
        float invb = 1.0f / __ldg(pml.be_ze_s + iz);

        ADEs_ze_k.Vx[idx] = -apdp_be * ADEs_ze.Vx[idx] + dp_be * Div_ze_S_x;
        ADEs_ze_k.Vz[idx] = -apdp_be * ADEs_ze.Vz[idx] + dp_be * Div_ze_S_z;

        Us_k.vx[idx] += ((invb - 1.0f) * Div_ze_S_x - invb * ADEs_ze.Vx[idx]) / Ms.rho[idx];
        Us_k.vz[idx] += ((invb - 1.0f) * Div_ze_S_z - invb * ADEs_ze.Vz[idx]) / Ms.rho[idx];
    }
}

template<MacDir DIR>
__global__ void fl_so_boundary(
    U_fluid Uf_k, U_solid Us_k, Flux flux
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix <= 2 || ix >= nx_all - 3) {
        return;
    }

    int idx_f = (nz_f - 1) * nx_all + ix;
    int idx_s = ix;

    const float2 nif = __ldg(flux.nif_d + ix);

    float vx_f = Uf_k.vx[idx_f], vz_f = Uf_k.vz[idx_f];
    float p_f  = Uf_k.p[idx_f];

    float vx_s  = Us_k.vx[idx_s];
    float vz_s  = Us_k.vz[idx_s];
    float sxx_s = Us_k.sx[idx_s];
    float szz_s = Us_k.sz[idx_s];
    float sxz_s = Us_k.sxz[idx_s];

    float vn_f = nif.x * vx_f + nif.y * vz_f;
    float vt_f = nif.y * vx_f - nif.x * vz_f;
    float vn_s = nif.x * vx_s + nif.y * vz_s;
    float vt_s = nif.y * vx_s - nif.x * vz_s;

    float snn_s = (
        + nif.x * nif.x * sxx_s 
        + 2 * nif.x * nif.y * sxz_s 
        + nif.y * nif.y * szz_s
    );
    float snt_s = (
        + nif.x * nif.y * (sxx_s - szz_s) 
        + (nif.y * nif.y - nif.x * nif.x) * sxz_s
    );
    float stt_s = (
        + nif.y * nif.y * sxx_s 
        - 2 * nif.x * nif.y * sxz_s 
        + nif.x * nif.x * szz_s
    );

    if constexpr (DIR == F) {
        vn_f  = vn_s;
        p_f   = -snn_s;
        snt_s = 0;
    } else {
        vn_s  = vn_f;
        snn_s = -p_f;
        snt_s = 0;
    }

    Uf_k.vx[idx_f] = nif.x * vn_f + nif.y * vt_f;
    Uf_k.vz[idx_f] = nif.y * vn_f - nif.x * vt_f;
    Uf_k.p [idx_f] = p_f;

    Us_k.vx[idx_s]  = nif.x * vn_s + nif.y * vt_s;
    Us_k.vz[idx_s]  = nif.y * vn_s - nif.x * vt_s;
    Us_k.sx[idx_s]  = (
        + nif.x * nif.x * snn_s 
        + 2 * nif.x * nif.y * snt_s 
        + nif.y * nif.y * stt_s
    );
    Us_k.sz[idx_s]  = (
        + nif.y * nif.y * snn_s 
        - 2 * nif.x * nif.y * snt_s 
        + nif.x * nif.x * stt_s
    );
    Us_k.sxz[idx_s] = (
        -nif.x * nif.y * stt_s 
        + (nif.y * nif.y - nif.x * nif.x) * snt_s
    );
}

__global__ void build_ghost_flux(
    const U_fluid Uf, const U_solid Us, const Grid Jacb,
    float2 * __restrict__ Gf, float2 * __restrict__ Gs
);

__global__ void fl_so_filter(
    U_fluid Uf, U_solid Us, Flux flux
);

__global__ void add_source(
    U_fluid Uf, int posx, int posz, float source
);

void linear_operate(
    const MemManager &mem, MacDir mdir, int isrc, float time,
    cudaStream_t sm_vel_f, cudaStream_t sm_sig_f,
    cudaStream_t sm_vel_s, cudaStream_t sm_sig_s
);

void update_RK4(
    const MemManager &mem_manager, RK4Dir rdir, int isrc, int it,
    cudaStream_t sm_vel_f, cudaStream_t sm_sig_f, 
    cudaStream_t sm_vel_s, cudaStream_t sm_sig_s
);

void apply_cerjan(
    const MemManager &mem,
    cudaStream_t sm_vel_f, cudaStream_t sm_sig_f,
    cudaStream_t sm_vel_s, cudaStream_t sm_sig_s
);

} // end of namespace PmlFs::FluidSolid

#endif
