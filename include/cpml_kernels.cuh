#ifndef CPML_KERNELS_CUH
#define CPML_KERNELS_CUH

#include "elements.h"
#include "tools_func.cuh"

// The 4 sides are ADE CFS-PML
namespace Cpml::SolidOnly {

template<MacDir DIR>
__global__ void update_stress(
    U_solid Us, Med_solid Ms, Grid Jacb, U_solid Us_k,
    ADE ADEs_xi, ADE ADEs_xi_k, ADE ADEs_ze, ADE ADEs_ze_k,
    PML_par pml
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix <= 2 || iz <= 2 || ix >= nx_all - 3 || iz >= nz_all - 3) {
        return;
    }
    int idx = iz * nx_all + ix;

    float dvx_dxi = DMac_xi<DIR>(Us.vx, idx);
    float dvx_dze = DMac_ze<DIR>(Us.vx, idx);
    float dvz_dxi = DMac_xi<DIR>(Us.vz, idx);
    float dvz_dze = DMac_ze<DIR>(Us.vz, idx);

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

    int distx = min(ix - 3, nx_all - 4 - ix);
    const bool in_xi = (unsigned)distx < (unsigned)pml_span_d;
    const bool in_ze = (iz >= nz_all - 3 - pml_span_d) || (iz <= pml_span_d + 2);
    const PmlRole role = pml_role(
        in_xi, __ldg(pml.dp_xi + ix), in_ze, __ldg(pml.dp_ze_s + iz)
    );
    const bool in_side   = role.pri_xi;   // 主阻尼方向：ξ 族
    const bool in_bottom = role.pri_ze;   // 主阻尼方向：ζ 族
    const bool in_top    = false;

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

    if (in_top || in_bottom) {
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

    // ---- MPML 辅助（平行）阻尼：ξ 主阻尼层内沿 ζ 族方向 ----
    if (role.aux_ze) {
        const float g_zeze = Jacb.J[idx]*Jacb.J[idx] *
                          (Jacb.xi_x[idx]*Jacb.xi_x[idx] + Jacb.xi_z[idx]*Jacb.xi_z[idx]);
        const float x_ze = -Jacb.J[idx] * Jacb.xi_z[idx];
        const float z_ze =  Jacb.J[idx] * Jacb.xi_x[idx];
        const float d2 = __ldg(pml.dp_ze_par + ix);
        const float D2_vx = dvx_dze / g_zeze;
        const float D2_vz = dvz_dze / g_zeze;
        const float cx = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_xi + ix) : 0.0f, D2_vx, ADEs_ze.Ex[idx], ADEs_ze_k.Ex[idx]);
        const float cz = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_xi + ix) : 0.0f, D2_vz, ADEs_ze.Ez[idx], ADEs_ze_k.Ez[idx]);
        Us_k.sx[idx] += Ms.C11[idx] * x_ze * cx + Ms.C13[idx] * z_ze * cz;
        Us_k.sz[idx] += Ms.C13[idx] * x_ze * cx + Ms.C33[idx] * z_ze * cz;
        Us_k.sxz[idx] += Ms.C55[idx] * (z_ze * cx + x_ze * cz);
    }

    // ---- MPML 辅助（平行）阻尼：ζ 主阻尼层内沿 ξ 族方向 ----
    if (MPML_AUX_XI && role.aux_xi) {
        const float g_xixi = Jacb.J[idx]*Jacb.J[idx] *
                          (Jacb.ze_x[idx]*Jacb.ze_x[idx] + Jacb.ze_z[idx]*Jacb.ze_z[idx]);
        const float x_xi =  Jacb.J[idx] * Jacb.ze_z[idx];
        const float z_xi = -Jacb.J[idx] * Jacb.ze_x[idx];
        const float d2 = __ldg(pml.dp_xi_par_s + iz);
        const float D2_vx = dvx_dxi / g_xixi;
        const float D2_vz = dvz_dxi / g_xixi;
        const float cx = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_ze_s + iz) : 0.0f, D2_vx, ADEs_xi.Ex[idx], ADEs_xi_k.Ex[idx]);
        const float cz = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_ze_s + iz) : 0.0f, D2_vz, ADEs_xi.Ez[idx], ADEs_xi_k.Ez[idx]);
        Us_k.sx[idx] += Ms.C11[idx] * x_xi * cx + Ms.C13[idx] * z_xi * cz;
        Us_k.sz[idx] += Ms.C13[idx] * x_xi * cx + Ms.C33[idx] * z_xi * cz;
        Us_k.sxz[idx] += Ms.C55[idx] * (z_xi * cx + x_xi * cz);
    }
}

template<MacDir DIR>
__global__ void update_velocity(
    U_solid Us, Med_solid Ms, Grid Jacb, U_solid Us_k,
    ADE ADEs_xi, ADE ADEs_xi_k, ADE ADEs_ze, ADE ADEs_ze_k,
    PML_par pml
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix <= 2 || iz <= 2 || ix >= nx_all - 3 || iz >= nz_all - 3) {
        return;
    }
    int idx = iz * nx_all + ix;

    float dsx_dxi  = DMac_xi<DIR>(Us.sx, idx);
    float dsx_dze  = DMac_ze<DIR>(Us.sx, idx);
    float dsz_dxi  = DMac_xi<DIR>(Us.sz, idx);
    float dsz_dze  = DMac_ze<DIR>(Us.sz, idx);
    float dsxz_dxi = DMac_xi<DIR>(Us.sxz, idx);
    float dsxz_dze = DMac_ze<DIR>(Us.sxz, idx);

    Us_k.vx[idx] = (
        + Jacb.xi_x[idx] * dsx_dxi
        + Jacb.ze_x[idx] * dsx_dze
        + Jacb.xi_z[idx] * dsxz_dxi
        + Jacb.ze_z[idx] * dsxz_dze
    ) / Ms.rho[idx];

    Us_k.vz[idx] = (
        + Jacb.xi_x[idx] * dsxz_dxi
        + Jacb.ze_x[idx] * dsxz_dze
        + Jacb.xi_z[idx] * dsz_dxi
        + Jacb.ze_z[idx] * dsz_dze
    ) / Ms.rho[idx];

    int distx = min(ix - 3, nx_all - 4 - ix);
    const bool in_xi = (unsigned)distx < (unsigned)pml_span_d;
    const bool in_ze = (iz >= nz_all - 3 - pml_span_d) || (iz <= pml_span_d + 2);
    const PmlRole role = pml_role(
        in_xi, __ldg(pml.dp_xi + ix), in_ze, __ldg(pml.dp_ze_s + iz)
    );
    const bool in_side   = role.pri_xi;
    const bool in_bottom = role.pri_ze;
    const bool in_top    = false;

    if (!in_side && !in_bottom) {
        return;
    }

    float gxixi = Jacb.xi_x[idx] * Jacb.xi_x[idx] + Jacb.xi_z[idx] * Jacb.xi_z[idx];
    float gxize = Jacb.xi_x[idx] * Jacb.ze_x[idx] + Jacb.xi_z[idx] * Jacb.ze_z[idx];
    float gzeze = Jacb.ze_x[idx] * Jacb.ze_x[idx] + Jacb.ze_z[idx] * Jacb.ze_z[idx];

    if (in_side) {
        float Dxi_sx  = dsx_dxi  + (gxize / gxixi) * dsx_dze;
        float Dxi_sz  = dsz_dxi  + (gxize / gxixi) * dsz_dze;
        float Dxi_sxz = dsxz_dxi + (gxize / gxixi) * dsxz_dze;

        // [∇₁^ξ·σ]
        float DxiS_x = Jacb.xi_x[idx] * Dxi_sx  + Jacb.xi_z[idx] * Dxi_sxz;
        float DxiS_z = Jacb.xi_x[idx] * Dxi_sxz + Jacb.xi_z[idx] * Dxi_sz;

        float dp_be = __ldg(pml.dp_xi + ix) / __ldg(pml.be_xi + ix);
        float apdp_be = __ldg(pml.al_xi + ix) + dp_be;
        float invb = 1.0f / __ldg(pml.be_xi + ix);

        ADEs_xi_k.Vx[idx] = -apdp_be * ADEs_xi.Vx[idx] + dp_be * DxiS_x;
        ADEs_xi_k.Vz[idx] = -apdp_be * ADEs_xi.Vz[idx] + dp_be * DxiS_z;

        Us_k.vx[idx] += ((invb - 1.0f) * DxiS_x - invb * ADEs_xi.Vx[idx]) / Ms.rho[idx];
        Us_k.vz[idx] += ((invb - 1.0f) * DxiS_z - invb * ADEs_xi.Vz[idx]) / Ms.rho[idx];
    }

    if (in_top || in_bottom) {
        float Dze_sx  = dsx_dze  + (gxize / gzeze) * dsx_dxi;
        float Dze_sz  = dsz_dze  + (gxize / gzeze) * dsz_dxi;
        float Dze_sxz = dsxz_dze + (gxize / gzeze) * dsxz_dxi;

        // [∇₁^ζ·σ]
        float DzeS_x = Jacb.ze_x[idx] * Dze_sx  + Jacb.ze_z[idx] * Dze_sxz;
        float DzeS_z = Jacb.ze_x[idx] * Dze_sxz + Jacb.ze_z[idx] * Dze_sz;

        float dp_be = __ldg(pml.dp_ze_s + iz) / __ldg(pml.be_ze_s + iz);
        float apdp_be = __ldg(pml.al_ze_s + iz) + dp_be;
        float invb = 1.0f / __ldg(pml.be_ze_s + iz);

        ADEs_ze_k.Vx[idx] = -apdp_be * ADEs_ze.Vx[idx] + dp_be * DzeS_x;
        ADEs_ze_k.Vz[idx] = -apdp_be * ADEs_ze.Vz[idx] + dp_be * DzeS_z;

        Us_k.vx[idx] += ((invb - 1.0f) * DzeS_x - invb * ADEs_ze.Vx[idx]) / Ms.rho[idx];
        Us_k.vz[idx] += ((invb - 1.0f) * DzeS_z - invb * ADEs_ze.Vz[idx]) / Ms.rho[idx];
    }

    // ---- MPML 辅助（平行）阻尼：ξ 主阻尼层内沿 ζ 族方向 ----
    if (role.aux_ze) {
        const float g_zeze = Jacb.J[idx]*Jacb.J[idx] *
                          (Jacb.xi_x[idx]*Jacb.xi_x[idx] + Jacb.xi_z[idx]*Jacb.xi_z[idx]);
        const float x_ze = -Jacb.J[idx] * Jacb.xi_z[idx];
        const float z_ze =  Jacb.J[idx] * Jacb.xi_x[idx];
        const float d2 = __ldg(pml.dp_ze_par + ix);
        const float D2_sx  = dsx_dze  / g_zeze;
        const float D2_sz  = dsz_dze  / g_zeze;
        const float D2_sxz = dsxz_dze / g_zeze;
        const float D2S_x = x_ze * D2_sx + z_ze * D2_sxz;   // [∇₂^ξ·σ]_x
        const float D2S_z = x_ze * D2_sxz + z_ze * D2_sz;
        const float cvx = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_xi + ix) : 0.0f, D2S_x, ADEs_ze.Vx[idx], ADEs_ze_k.Vx[idx]);
        const float cvz = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_xi + ix) : 0.0f, D2S_z, ADEs_ze.Vz[idx], ADEs_ze_k.Vz[idx]);
        Us_k.vx[idx] += cvx / Ms.rho[idx];
        Us_k.vz[idx] += cvz / Ms.rho[idx];
    }

    // ---- MPML 辅助（平行）阻尼：ζ 主阻尼层内沿 ξ 族方向 ----
    if (MPML_AUX_XI && role.aux_xi) {
        const float g_xixi = Jacb.J[idx]*Jacb.J[idx] *
                          (Jacb.ze_x[idx]*Jacb.ze_x[idx] + Jacb.ze_z[idx]*Jacb.ze_z[idx]);
        const float x_xi =  Jacb.J[idx] * Jacb.ze_z[idx];
        const float z_xi = -Jacb.J[idx] * Jacb.ze_x[idx];
        const float d2 = __ldg(pml.dp_xi_par_s + iz);
        const float D2_sx  = dsx_dxi  / g_xixi;
        const float D2_sz  = dsz_dxi  / g_xixi;
        const float D2_sxz = dsxz_dxi / g_xixi;
        const float D2S_x = x_xi * D2_sx + z_xi * D2_sxz;   // [∇₂^ζ·σ]_x
        const float D2S_z = x_xi * D2_sxz + z_xi * D2_sz;
        const float cvx = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_ze_s + iz) : 0.0f, D2S_x, ADEs_xi.Vx[idx], ADEs_xi_k.Vx[idx]);
        const float cvz = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_ze_s + iz) : 0.0f, D2S_z, ADEs_xi.Vz[idx], ADEs_xi_k.Vz[idx]);
        Us_k.vx[idx] += cvx / Ms.rho[idx];
        Us_k.vz[idx] += cvz / Ms.rho[idx];
    }
}

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

} // end of namespace Cpml::SolidOnly


/******************************************************/
namespace Cpml::FluidSolid {

template<MacDir DIR>
__global__ void update_velocity_fluid(
    U_fluid Uf, Med_fluid Mf, Grid Jacb, U_fluid Uf_k,
    ADE ADEf_xi, ADE ADEf_xi_k, ADE ADEf_ze, ADE ADEf_ze_k,
    PML_par pml, Flux flux
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix <= 2 || iz <= 2 || ix >= nx_all - 3 || iz >= nz_f) {
        return;
    }
    int idx = iz * nx_all + ix;
    int dist = nz_f - 1 - iz;

    float rhodvx_dt, rhodvz_dt;
    float Div_xi_P_x, Div_xi_P_z;
    float Div_ze_P_x, Div_ze_P_z;

    float gxixi = Jacb.xi_x[idx] * Jacb.xi_x[idx] + Jacb.xi_z[idx] * Jacb.xi_z[idx];
    float gzeze = Jacb.ze_x[idx] * Jacb.ze_x[idx] + Jacb.ze_z[idx] * Jacb.ze_z[idx];

    if (dist < 3) {
        // conservation equation
        auto Fxi = [&](int k) {
            return flux_xi_fluid(Uf, Jacb, k);
        };
        auto Fze = [&](int k) {
            return flux_ze_fluid(
                Uf, Jacb, ix, (nz_f - 1) - k / nx_all, 
                nz_f - 1, DIR_ABOVE, flux.Gf
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
    const bool in_xi = (unsigned)distx < (unsigned)pml_span_d;
    const bool in_ze = iz <= pml_span_d + 2;
    const PmlRole role = pml_role(
        in_xi, __ldg(pml.dp_xi + ix), in_ze, __ldg(pml.dp_ze_f + iz)
    );
    const bool in_side = role.pri_xi;   // 主阻尼方向：ξ 族
    const bool in_top  = role.pri_ze;   // 主阻尼方向：ζ 族

    if (!in_side && !in_top) {
        return;
    }

    if (in_side) {
        float dp_be = __ldg(pml.dp_xi + ix) / __ldg(pml.be_xi + ix);
        float apdp_be = __ldg(pml.al_xi + ix) + dp_be;
        float invb = 1.0f / __ldg(pml.be_xi + ix);

        ADEf_xi_k.Vx[idx] = -apdp_be * ADEf_xi.Vx[idx] + dp_be * Div_xi_P_x;
        ADEf_xi_k.Vz[idx] = -apdp_be * ADEf_xi.Vz[idx] + dp_be * Div_xi_P_z;

        Uf_k.vx[idx] += ((invb - 1.0f) * Div_xi_P_x - invb * ADEf_xi.Vx[idx]) / Mf.rho[idx];
        Uf_k.vz[idx] += ((invb - 1.0f) * Div_xi_P_z - invb * ADEf_xi.Vz[idx]) / Mf.rho[idx];
    }

    if (in_top) {
        float dp_be = __ldg(pml.dp_ze_f + iz) / __ldg(pml.be_ze_f + iz);
        float apdp_be = __ldg(pml.al_ze_f + iz) + dp_be;
        float invb = 1.0f / __ldg(pml.be_ze_f + iz);

        ADEf_ze_k.Vx[idx] = -apdp_be * ADEf_ze.Vx[idx] + dp_be * Div_ze_P_x;
        ADEf_ze_k.Vz[idx] = -apdp_be * ADEf_ze.Vz[idx] + dp_be * Div_ze_P_z;

        Uf_k.vx[idx] += ((invb - 1.0f) * Div_ze_P_x - invb * ADEf_ze.Vx[idx]) / Mf.rho[idx];
        Uf_k.vz[idx] += ((invb - 1.0f) * Div_ze_P_z - invb * ADEf_ze.Vz[idx]) / Mf.rho[idx];
    }

    // ---- MPML 辅助（平行）阻尼：ξ 主阻尼层内沿 ζ 族方向 ----
    if (MPML_AUX_ZE && role.aux_ze && nz_f - 1 - iz >= MPML_AUX_GUARD) {
        const float g_zeze = Jacb.J[idx]*Jacb.J[idx] *
                          (Jacb.xi_x[idx]*Jacb.xi_x[idx] + Jacb.xi_z[idx]*Jacb.xi_z[idx]);
        const float x_ze = -Jacb.J[idx] * Jacb.xi_z[idx];
        const float z_ze =  Jacb.J[idx] * Jacb.xi_x[idx];
        const float d2 = __ldg(pml.dp_ze_par + ix);
        const float dp_dze = Mac_dv_dze<DIR>(Uf.p, idx, nz_f - 1 - iz, false);
        const float D2S_x = -x_ze * (dp_dze / g_zeze);   // [∇₂^ξ p]_x
        const float D2S_z = -z_ze * (dp_dze / g_zeze);
        const float cvx = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_xi + ix) : 0.0f, D2S_x, ADEf_ze.Vx[idx], ADEf_ze_k.Vx[idx]);
        const float cvz = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_xi + ix) : 0.0f, D2S_z, ADEf_ze.Vz[idx], ADEf_ze_k.Vz[idx]);
        Uf_k.vx[idx] += cvx / Mf.rho[idx];
        Uf_k.vz[idx] += cvz / Mf.rho[idx];
    }

    // ---- MPML 辅助（平行）阻尼：ζ 主阻尼层内沿 ξ 族方向 ----
    if (MPML_AUX_XI && role.aux_xi) {
        const float g_xixi = Jacb.J[idx]*Jacb.J[idx] *
                          (Jacb.ze_x[idx]*Jacb.ze_x[idx] + Jacb.ze_z[idx]*Jacb.ze_z[idx]);
        const float x_xi =  Jacb.J[idx] * Jacb.ze_z[idx];
        const float z_xi = -Jacb.J[idx] * Jacb.ze_x[idx];
        const float d2 = __ldg(pml.dp_xi_par_f + iz);
        const float dp_dxi = DMac_xi<DIR>(Uf.p, idx);
        const float D2S_x = -x_xi * (dp_dxi / g_xixi);   // [∇₂^ζ p]_x
        const float D2S_z = -z_xi * (dp_dxi / g_xixi);
        const float cvx = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_ze_f + iz) : 0.0f, D2S_x, ADEf_xi.Vx[idx], ADEf_xi_k.Vx[idx]);
        const float cvz = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_ze_f + iz) : 0.0f, D2S_z, ADEf_xi.Vz[idx], ADEf_xi_k.Vz[idx]);
        Uf_k.vx[idx] += cvx / Mf.rho[idx];
        Uf_k.vz[idx] += cvz / Mf.rho[idx];
    }
}

template<MacDir DIR>
__global__ void update_stress_fluid(
    U_fluid Uf, Med_fluid Mf, Grid Jacb, U_fluid Uf_k,
    ADE ADEf_xi, ADE ADEf_xi_k, ADE ADEf_ze, ADE ADEf_ze_k,
    PML_par pml
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix <= 2 || iz <= 2 || ix >= nx_all - 3 || iz >= nz_f) {
        return;
    }
    int idx = iz * nx_all + ix;

    float dvx_dxi = DMac_xi<DIR>(Uf.vx, idx);
    float dvx_dze = Mac_dv_dze<DIR>(Uf.vx, idx, nz_f - 1 - iz, false);
    float dvz_dxi = DMac_xi<DIR>(Uf.vz, idx);
    float dvz_dze = Mac_dv_dze<DIR>(Uf.vz, idx, nz_f - 1 - iz, false);

    Uf_k.p[idx] = -Mf.K[idx] * (
        + Jacb.xi_x[idx] * dvx_dxi + Jacb.ze_x[idx] * dvx_dze
        + Jacb.xi_z[idx] * dvz_dxi + Jacb.ze_z[idx] * dvz_dze
    );

    int distx = min(ix - 3, nx_all - 4 - ix);
    const bool in_xi = (unsigned)distx < (unsigned)pml_span_d;
    const bool in_ze = iz <= pml_span_d + 2;
    const PmlRole role = pml_role(
        in_xi, __ldg(pml.dp_xi + ix), in_ze, __ldg(pml.dp_ze_f + iz)
    );
    const bool in_side = role.pri_xi;
    const bool in_top  = role.pri_ze;

    if (!in_side && !in_top) {
        return;
    }

    float gxixi = Jacb.xi_x[idx] * Jacb.xi_x[idx] + Jacb.xi_z[idx] * Jacb.xi_z[idx];
    float gxize = Jacb.xi_x[idx] * Jacb.ze_x[idx] + Jacb.xi_z[idx] * Jacb.ze_z[idx];
    float gzeze = Jacb.ze_x[idx] * Jacb.ze_x[idx] + Jacb.ze_z[idx] * Jacb.ze_z[idx];

    if (in_side) {
        float Dxi_vx = dvx_dxi + (gxize / gxixi) * dvx_dze;
        float Dxi_vz = dvz_dxi + (gxize / gxixi) * dvz_dze;

        float dp_be = __ldg(pml.dp_xi + ix) / __ldg(pml.be_xi + ix);        // d / beta
        float apdp_be = __ldg(pml.al_xi + ix) + dp_be;                      // alpha + d / beta
        float invb = 1.0f / __ldg(pml.be_xi + ix);                          // 1 / beta

        ADEf_xi_k.Ex[idx] = -apdp_be * ADEf_xi.Ex[idx] + dp_be * Dxi_vx;
        ADEf_xi_k.Ez[idx] = -apdp_be * ADEf_xi.Ez[idx] + dp_be * Dxi_vz;

        Uf_k.p[idx] += (
            - Mf.K[idx] * (invb - 1.0f) * Jacb.xi_x[idx] * Dxi_vx
            - Mf.K[idx] * (invb - 1.0f) * Jacb.xi_z[idx] * Dxi_vz
            + invb * Mf.K[idx] * Jacb.xi_x[idx] * ADEf_xi.Ex[idx]
            + invb * Mf.K[idx] * Jacb.xi_z[idx] * ADEf_xi.Ez[idx]
        );
    }

    if (in_top) {
        float Dze_vx = dvx_dze + (gxize / gzeze) * dvx_dxi;
        float Dze_vz = dvz_dze + (gxize / gzeze) * dvz_dxi;

        float dp_be = __ldg(pml.dp_ze_f + iz) / __ldg(pml.be_ze_f + iz);    // d / beta
        float apdp_be = __ldg(pml.al_ze_f + iz) + dp_be;                    // alpha + d / beta
        float invb = 1.0f / __ldg(pml.be_ze_f + iz);                        // 1 / beta

        ADEf_ze_k.Ex[idx] = -apdp_be * ADEf_ze.Ex[idx] + dp_be * Dze_vx;
        ADEf_ze_k.Ez[idx] = -apdp_be * ADEf_ze.Ez[idx] + dp_be * Dze_vz;

        Uf_k.p[idx] += Mf.K[idx] * (
            -(invb - 1.0f) * (Jacb.ze_x[idx] * Dze_vx + Jacb.ze_z[idx] * Dze_vz)
            + invb * (Jacb.ze_x[idx] * ADEf_ze.Ex[idx] + Jacb.ze_z[idx] * ADEf_ze.Ez[idx])
        );
    }

    // ---- MPML 辅助（平行）阻尼：ξ 主阻尼层内沿 ζ 族方向 ----
    if (MPML_AUX_ZE && role.aux_ze && nz_f - 1 - iz >= MPML_AUX_GUARD) {
        const float g_zeze = Jacb.J[idx]*Jacb.J[idx] *
                          (Jacb.xi_x[idx]*Jacb.xi_x[idx] + Jacb.xi_z[idx]*Jacb.xi_z[idx]);
        const float x_ze = -Jacb.J[idx] * Jacb.xi_z[idx];
        const float z_ze =  Jacb.J[idx] * Jacb.xi_x[idx];
        const float d2 = __ldg(pml.dp_ze_par + ix);
        const float D2_vx = dvx_dze / g_zeze;    // D₂^ξ v_x = ∂ζ v_x / g_ζζ
        const float D2_vz = dvz_dze / g_zeze;
        const float cx = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_xi + ix) : 0.0f, D2_vx, ADEf_ze.Ex[idx], ADEf_ze_k.Ex[idx]);
        const float cz = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_xi + ix) : 0.0f, D2_vz, ADEf_ze.Ez[idx], ADEf_ze_k.Ez[idx]);
            // 压力方程符号：p,t = -K*div(v)，散度修正整体取负
        Uf_k.p[idx] -= Mf.K[idx] * (x_ze * cx + z_ze * cz);
    }

    // ---- MPML 辅助（平行）阻尼：ζ 主阻尼层内沿 ξ 族方向 ----
    if (MPML_AUX_XI && role.aux_xi) {
        const float g_xixi = Jacb.J[idx]*Jacb.J[idx] *
                          (Jacb.ze_x[idx]*Jacb.ze_x[idx] + Jacb.ze_z[idx]*Jacb.ze_z[idx]);
        const float x_xi =  Jacb.J[idx] * Jacb.ze_z[idx];
        const float z_xi = -Jacb.J[idx] * Jacb.ze_x[idx];
        const float d2 = __ldg(pml.dp_xi_par_f + iz);
        const float D2_vx = dvx_dxi / g_xixi;    // D₂^ζ v_x = ∂ξ v_x / g_ξξ
        const float D2_vz = dvz_dxi / g_xixi;
        const float cx = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_ze_f + iz) : 0.0f, D2_vx, ADEf_xi.Ex[idx], ADEf_xi_k.Ex[idx]);
        const float cz = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_ze_f + iz) : 0.0f, D2_vz, ADEf_xi.Ez[idx], ADEf_xi_k.Ez[idx]);
            // 压力方程符号：p,t = -K*div(v)，散度修正整体取负
        Uf_k.p[idx] -= Mf.K[idx] * (x_xi * cx + z_xi * cz);
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
            + Jacb.xi_z[idx] * dvx_dxi
            + Jacb.ze_z[idx] * dvx_dze
            + Jacb.xi_x[idx] * dvz_dxi
            + Jacb.ze_x[idx] * dvz_dze
        )
    );

    int distx = min(ix - 3, nx_all - 4 - ix);
    const bool in_xi = (unsigned)distx < (unsigned)pml_span_d;
    const bool in_ze = iz >= nz_s - 3 - pml_span_d;
    const PmlRole role = pml_role(
        in_xi, __ldg(pml.dp_xi + ix), in_ze, __ldg(pml.dp_ze_s + iz)
    );
    const bool in_side   = role.pri_xi;
    const bool in_bottom = role.pri_ze;

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

    // ---- MPML 辅助（平行）阻尼：ξ 主阻尼层内沿 ζ 族方向 ----
    if (MPML_AUX_ZE && role.aux_ze && iz >= MPML_AUX_GUARD) {
        const float g_zeze = Jacb.J[idx_J]*Jacb.J[idx_J] *
                          (Jacb.xi_x[idx_J]*Jacb.xi_x[idx_J] + Jacb.xi_z[idx_J]*Jacb.xi_z[idx_J]);
        const float x_ze = -Jacb.J[idx_J] * Jacb.xi_z[idx_J];
        const float z_ze =  Jacb.J[idx_J] * Jacb.xi_x[idx_J];
        const float d2 = __ldg(pml.dp_ze_par + ix);
        const float D2_vx = dvx_dze / g_zeze;
        const float D2_vz = dvz_dze / g_zeze;
        const float cx = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_xi + ix) : 0.0f, D2_vx, ADEs_ze.Ex[idx], ADEs_ze_k.Ex[idx]);
        const float cz = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_xi + ix) : 0.0f, D2_vz, ADEs_ze.Ez[idx], ADEs_ze_k.Ez[idx]);
        Us_k.sx[idx] += Ms.C11[idx] * x_ze * cx + Ms.C13[idx] * z_ze * cz;
        Us_k.sz[idx] += Ms.C13[idx] * x_ze * cx + Ms.C33[idx] * z_ze * cz;
        Us_k.sxz[idx] += Ms.C55[idx] * (z_ze * cx + x_ze * cz);
    }

    // ---- MPML 辅助（平行）阻尼：ζ 主阻尼层内沿 ξ 族方向 ----
    if (MPML_AUX_XI && role.aux_xi) {
        const float g_xixi = Jacb.J[idx_J]*Jacb.J[idx_J] *
                          (Jacb.ze_x[idx_J]*Jacb.ze_x[idx_J] + Jacb.ze_z[idx_J]*Jacb.ze_z[idx_J]);
        const float x_xi =  Jacb.J[idx_J] * Jacb.ze_z[idx_J];
        const float z_xi = -Jacb.J[idx_J] * Jacb.ze_x[idx_J];
        const float d2 = __ldg(pml.dp_xi_par_s + iz);
        const float D2_vx = dvx_dxi / g_xixi;
        const float D2_vz = dvz_dxi / g_xixi;
        const float cx = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_ze_s + iz) : 0.0f, D2_vx, ADEs_xi.Ex[idx], ADEs_xi_k.Ex[idx]);
        const float cz = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_ze_s + iz) : 0.0f, D2_vz, ADEs_xi.Ez[idx], ADEs_xi_k.Ez[idx]);
        Us_k.sx[idx] += Ms.C11[idx] * x_xi * cx + Ms.C13[idx] * z_xi * cz;
        Us_k.sz[idx] += Ms.C13[idx] * x_xi * cx + Ms.C33[idx] * z_xi * cz;
        Us_k.sxz[idx] += Ms.C55[idx] * (z_xi * cx + x_xi * cz);
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
            return flux_ze_solid(
                Us, Jacb, ix, (k - ix) / nx_all, 0, DIR_BELOW, flux.Gs
            );
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
    const bool in_xi = (unsigned)distx < (unsigned)pml_span_d;
    const bool in_ze = iz >= nz_s - 3 - pml_span_d;
    const PmlRole role = pml_role(
        in_xi, __ldg(pml.dp_xi + ix), in_ze, __ldg(pml.dp_ze_s + iz)
    );
    const bool in_side   = role.pri_xi;
    const bool in_bottom = role.pri_ze;

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

    // ---- MPML 辅助（平行）阻尼：ξ 主阻尼层内沿 ζ 族方向 ----
    if (MPML_AUX_ZE && role.aux_ze && iz >= MPML_AUX_GUARD) {
        const float g_zeze = Jacb.J[idx_J]*Jacb.J[idx_J] *
                          (Jacb.xi_x[idx_J]*Jacb.xi_x[idx_J] + Jacb.xi_z[idx_J]*Jacb.xi_z[idx_J]);
        const float x_ze = -Jacb.J[idx_J] * Jacb.xi_z[idx_J];
        const float z_ze =  Jacb.J[idx_J] * Jacb.xi_x[idx_J];
        const float d2 = __ldg(pml.dp_ze_par + ix);
        const float D2_sx  = Mac_dv_dze<DIR>(Us.sx,  idx, dist, true) / g_zeze;
        const float D2_sz  = Mac_dv_dze<DIR>(Us.sz,  idx, dist, true) / g_zeze;
        const float D2_sxz = Mac_dv_dze<DIR>(Us.sxz, idx, dist, true) / g_zeze;
        const float D2S_x = x_ze * D2_sx + z_ze * D2_sxz;
        const float D2S_z = x_ze * D2_sxz + z_ze * D2_sz;
        const float cvx = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_xi + ix) : 0.0f, D2S_x, ADEs_ze.Vx[idx], ADEs_ze_k.Vx[idx]);
        const float cvz = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_xi + ix) : 0.0f, D2S_z, ADEs_ze.Vz[idx], ADEs_ze_k.Vz[idx]);
        Us_k.vx[idx] += cvx / Ms.rho[idx];
        Us_k.vz[idx] += cvz / Ms.rho[idx];
    }

    // ---- MPML 辅助（平行）阻尼：ζ 主阻尼层内沿 ξ 族方向 ----
    if (MPML_AUX_XI && role.aux_xi) {
        const float g_xixi = Jacb.J[idx_J]*Jacb.J[idx_J] *
                          (Jacb.ze_x[idx_J]*Jacb.ze_x[idx_J] + Jacb.ze_z[idx_J]*Jacb.ze_z[idx_J]);
        const float x_xi =  Jacb.J[idx_J] * Jacb.ze_z[idx_J];
        const float z_xi = -Jacb.J[idx_J] * Jacb.ze_x[idx_J];
        const float d2 = __ldg(pml.dp_xi_par_s + iz);
        const float D2_sx  = DMac_xi<DIR>(Us.sx,  idx) / g_xixi;
        const float D2_sz  = DMac_xi<DIR>(Us.sz,  idx) / g_xixi;
        const float D2_sxz = DMac_xi<DIR>(Us.sxz, idx) / g_xixi;
        const float D2S_x = x_xi * D2_sx + z_xi * D2_sxz;
        const float D2S_z = x_xi * D2_sxz + z_xi * D2_sz;
        const float cvx = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_ze_s + iz) : 0.0f, D2S_x, ADEs_xi.Vx[idx], ADEs_xi_k.Vx[idx]);
        const float cvz = ade_aux_step(d2, MPML_A2_CFS ? __ldg(pml.al_ze_s + iz) : 0.0f, D2S_z, ADEs_xi.Vz[idx], ADEs_xi_k.Vz[idx]);
        Us_k.vx[idx] += cvx / Ms.rho[idx];
        Us_k.vz[idx] += cvz / Ms.rho[idx];
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

} // end of namespace Cpml::FluidSolid

#endif
