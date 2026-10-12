#include "cpml_kernels.cuh"
#include "common_kernels.cuh"
#include "cuda_check.h"

namespace Cpml::SolidOnly {

__global__ void add_source(
    U_solid Us, int posx, int posz, float source
) {
    add_gauss_source(Us.sx, Us.sz, posx, posz, nz_s, source);
}

void linear_operate(
    const MemManager &mem, MacDir mdir, int isrc, float time,
    cudaStream_t stream_vel, cudaStream_t stream_sig
) {
    static dim3 gridSize((mem.nx_solid + 15) / 16, (mem.nz_solid + 15) / 16);
    static dim3 blockSize(16, 16);

    static cudaEvent_t ev_in = nullptr, ev_in_sig = nullptr;
    if (ev_in == nullptr) {
        CHECK(cudaEventCreateWithFlags(&ev_in, cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&ev_in_sig, cudaEventDisableTiming));
    }
    CHECK(cudaEventRecord(ev_in, stream_vel));
    CHECK(cudaEventRecord(ev_in_sig, stream_sig));
    CHECK(cudaStreamWaitEvent(stream_vel, ev_in_sig, 0));
    CHECK(cudaStreamWaitEvent(stream_sig, ev_in, 0));

    static cudaEvent_t ev_vel = nullptr, ev_sig = nullptr;
    if (ev_vel == nullptr) {
        CHECK(cudaEventCreateWithFlags(&ev_vel, cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&ev_sig, cudaEventDisableTiming));
    }
    switch (mdir) {
    case F:
        update_velocity<F><<<gridSize, blockSize, 0, stream_vel>>>(
            mem.Us_tmp, mem.Ms_d, mem.Jacb_d, mem.Us_k, 
            mem.ADEs_xi_tmp, mem.ADEs_xi_k,
            mem.ADEs_ze_tmp, mem.ADEs_ze_k,
            mem.pml_par_d
        );
        update_stress<F><<<gridSize, blockSize, 0, stream_sig>>>(
            mem.Us_tmp, mem.Ms_d, mem.Jacb_d, mem.Us_k,
            mem.ADEs_xi_tmp, mem.ADEs_xi_k,
            mem.ADEs_ze_tmp, mem.ADEs_ze_k,
            mem.pml_par_d
        );
        break;
    case B:
        update_velocity<B><<<gridSize, blockSize, 0, stream_vel>>>(
            mem.Us_tmp, mem.Ms_d, mem.Jacb_d, mem.Us_k, 
            mem.ADEs_xi_tmp, mem.ADEs_xi_k,
            mem.ADEs_ze_tmp, mem.ADEs_ze_k,
            mem.pml_par_d
        );
        update_stress<B><<<gridSize, blockSize, 0, stream_sig>>>(
            mem.Us_tmp, mem.Ms_d, mem.Jacb_d, mem.Us_k,
            mem.ADEs_xi_tmp, mem.ADEs_xi_k,
            mem.ADEs_ze_tmp, mem.ADEs_ze_k,
            mem.pml_par_d
        );
        break;
    }
    CHECK(cudaEventRecord(ev_vel, stream_vel));
    CHECK(cudaEventRecord(ev_sig, stream_sig));
    CHECK(cudaStreamWaitEvent(stream_vel, ev_sig, 0));
    CHECK(cudaStreamWaitEvent(stream_sig, ev_vel, 0));

    add_source<<<1, 1, 0, stream_sig>>>(
        mem.Us_k, mem.src_pos[isrc].x, 
        mem.src_pos[isrc].y, mem.ricker(time)
    );
}

void update_RK4(
    const MemManager &mem, RK4Dir rdir, int isrc, int it,
    cudaStream_t stream_vel, cudaStream_t stream_sig
) {
    float time = it * mem.dt;
    float hf_dt = mem.dt * 0.5;
    switch (rdir) {
    case FBFB:
        // =========== step 1 ===========
        linear_operate(mem, F, isrc, time, stream_vel, stream_sig);

        // vector U
        MemManager::vector_add_Us(
            mem.Us_d, mem.Us_k, mem.Us_tmp, 
            1.0, mem.dt / 2, mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_Us(
            mem.Us_b, mem.Us_k, mem.dt / 6, 
            mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );

        // vector ADEs xi-direction & ze-direction
        MemManager::vector_add_ADE(
            mem.ADEs_xi, mem.ADEs_xi_k, mem.ADEs_xi_tmp,
            1.0, mem.dt / 2, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_ADE(
            mem.ADEs_xi_b, mem.ADEs_xi_k, mem.dt / 6,
            mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );

        MemManager::vector_add_ADE(
            mem.ADEs_ze, mem.ADEs_ze_k, mem.ADEs_ze_tmp,
            1.0, mem.dt / 2, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_ADE(
            mem.ADEs_ze_b, mem.ADEs_ze_k, mem.dt / 6,
            mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );

        // =========== step 2 ===========
        linear_operate(mem, B, isrc, time + hf_dt, stream_vel, stream_sig);

        // vector U
        MemManager::vector_add_Us(
            mem.Us_d, mem.Us_k, mem.Us_tmp, 
            1.0, mem.dt / 2, mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_Us(
            mem.Us_b, mem.Us_k, mem.dt / 3, 
            mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );

        // vector ADEs xi-direction & ze-direction
        MemManager::vector_add_ADE(
            mem.ADEs_xi, mem.ADEs_xi_k, mem.ADEs_xi_tmp,
            1.0, mem.dt / 2, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_ADE(
            mem.ADEs_xi_b, mem.ADEs_xi_k, mem.dt / 3,
            mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );

        MemManager::vector_add_ADE(
            mem.ADEs_ze, mem.ADEs_ze_k, mem.ADEs_ze_tmp,
            1.0, mem.dt / 2, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_ADE(
            mem.ADEs_ze_b, mem.ADEs_ze_k, mem.dt / 3,
            mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );

        // =========== step 3 ===========
        linear_operate(mem, F, isrc, time + hf_dt, stream_vel, stream_sig);

        // vector U
        MemManager::vector_add_Us(
            mem.Us_d, mem.Us_k, mem.Us_tmp, 
            1.0, mem.dt, mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_Us(
            mem.Us_b, mem.Us_k, mem.dt / 3, 
            mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );

        // vector ADEs xi-direction & ze-direction
        MemManager::vector_add_ADE(
            mem.ADEs_xi, mem.ADEs_xi_k, mem.ADEs_xi_tmp,
            1.0, mem.dt, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_ADE(
            mem.ADEs_xi_b, mem.ADEs_xi_k, mem.dt / 3,
            mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );

        MemManager::vector_add_ADE(
            mem.ADEs_ze, mem.ADEs_ze_k, mem.ADEs_ze_tmp,
            1.0, mem.dt, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_ADE(
            mem.ADEs_ze_b, mem.ADEs_ze_k, mem.dt / 3,
            mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        
        // =========== step 4 ===========
        linear_operate(mem, B, isrc, time + mem.dt, stream_vel, stream_sig);
        
        // vector U
        MemManager::vector_add_Us(
            mem.Us_b, mem.Us_k, mem.Us_d, 
            1.0, mem.dt / 6, mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );

        // vector ADEs xi-direction & ze-direction
        MemManager::vector_add_ADE(
            mem.ADEs_xi_b, mem.ADEs_xi_k, mem.ADEs_xi,
            1.0, mem.dt / 6, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_add_ADE(
            mem.ADEs_ze_b, mem.ADEs_ze_k, mem.ADEs_ze,
            1.0, mem.dt / 6, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        break;

    case BFBF:
        // =========== step 1 ===========
        linear_operate(mem, B, isrc, time, stream_vel, stream_sig);

        // vector U
        MemManager::vector_add_Us(
            mem.Us_d, mem.Us_k, mem.Us_tmp, 
            1.0, mem.dt / 2, mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_Us(
            mem.Us_b, mem.Us_k, mem.dt / 6, 
            mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );

        // vector ADEs xi-direction & ze-direction
        MemManager::vector_add_ADE(
            mem.ADEs_xi, mem.ADEs_xi_k, mem.ADEs_xi_tmp,
            1.0, mem.dt / 2, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_ADE(
            mem.ADEs_xi_b, mem.ADEs_xi_k, mem.dt / 6,
            mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );

        MemManager::vector_add_ADE(
            mem.ADEs_ze, mem.ADEs_ze_k, mem.ADEs_ze_tmp,
            1.0, mem.dt / 2, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_ADE(
            mem.ADEs_ze_b, mem.ADEs_ze_k, mem.dt / 6,
            mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );

        // =========== step 2 ===========
        linear_operate(mem, F, isrc, time + hf_dt, stream_vel, stream_sig);

        // vector U
        MemManager::vector_add_Us(
            mem.Us_d, mem.Us_k, mem.Us_tmp, 
            1.0, mem.dt / 2, mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_Us(
            mem.Us_b, mem.Us_k, mem.dt / 3, 
            mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );

        // vector ADEs xi-direction & ze-direction
        MemManager::vector_add_ADE(
            mem.ADEs_xi, mem.ADEs_xi_k, mem.ADEs_xi_tmp,
            1.0, mem.dt / 2, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_ADE(
            mem.ADEs_xi_b, mem.ADEs_xi_k, mem.dt / 3,
            mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );

        MemManager::vector_add_ADE(
            mem.ADEs_ze, mem.ADEs_ze_k, mem.ADEs_ze_tmp,
            1.0, mem.dt / 2, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_ADE(
            mem.ADEs_ze_b, mem.ADEs_ze_k, mem.dt / 3,
            mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );

        // =========== step 3 ===========
        linear_operate(mem, B, isrc, time + hf_dt, stream_vel, stream_sig);

        // vector U
        MemManager::vector_add_Us(
            mem.Us_d, mem.Us_k, mem.Us_tmp, 
            1.0, mem.dt, mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_Us(
            mem.Us_b, mem.Us_k, mem.dt / 3, 
            mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );

        // vector ADEs xi-direction & ze-direction
        MemManager::vector_add_ADE(
            mem.ADEs_xi, mem.ADEs_xi_k, mem.ADEs_xi_tmp,
            1.0, mem.dt, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_ADE(
            mem.ADEs_xi_b, mem.ADEs_xi_k, mem.dt / 3,
            mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );

        MemManager::vector_add_ADE(
            mem.ADEs_ze, mem.ADEs_ze_k, mem.ADEs_ze_tmp,
            1.0, mem.dt, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_self_add_ADE(
            mem.ADEs_ze_b, mem.ADEs_ze_k, mem.dt / 3,
            mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        
        // =========== step 4 ===========
        linear_operate(mem, F, isrc, time + mem.dt, stream_vel, stream_sig);
        
        // vector U
        MemManager::vector_add_Us(
            mem.Us_b, mem.Us_k, mem.Us_d, 
            1.0, mem.dt / 6, mem.nz_solid, mem.nx_solid, 
            stream_vel, stream_sig
        );

        // vector ADEs xi-direction & ze-direction
        MemManager::vector_add_ADE(
            mem.ADEs_xi_b, mem.ADEs_xi_k, mem.ADEs_xi,
            1.0, mem.dt / 6, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        MemManager::vector_add_ADE(
            mem.ADEs_ze_b, mem.ADEs_ze_k, mem.ADEs_ze,
            1.0, mem.dt / 6, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        break;
    }
}
    
} // end of namespace Cpml::SolidOnly

namespace Cpml::FluidSolid {

__global__ void build_ghost_flux(
    const U_fluid Uf, const U_solid Us, const Grid Jacb,
    float2 * __restrict__ Gf, float2 * __restrict__ Gs
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix <= 2 || ix >= nx_all - 3) return;

    float2 Fz_f = flux_ze_fluid(Uf, Jacb, ix, 0, nz_f - 1, DIR_ABOVE, Gf);
    float2 Fz_s = flux_ze_solid(Us, Jacb, ix, 0, 0, DIR_BELOW, Gs);

    // fluid-solid interface: fluid zone
    do {
        for (int n = 1; n <= 3; n++) {
            // DIR_ABOVE: fluid flux is above the fluid-solid interface
            float2 Fn = flux_ze_fluid(Uf, Jacb, ix, n, nz_f - 1, DIR_ABOVE, Gf);
            Gf[(n - 1) * nx_all + ix] = make_float2(
                2 * Fz_f.x - Fn.x, 2 * Fz_f.y - Fn.y
            );
        }
    } while(0);

    // fluid-solid interface: solid zone
    do {
        for (int n = 1; n <= 3; n++) {
            // DIR_BELOW: solid flux is below the fluid-solid interface
            float2 Fn = flux_ze_solid(Us, Jacb, ix, n, 0, DIR_BELOW, Gs);
            Gs[(n - 1) * nx_all + ix] = make_float2(
                2 * Fz_s.x - Fn.x, 2 * Fz_s.y - Fn.y
            );
        }
    } while(0);
}

__global__ void fl_so_filter(
    U_fluid Uf, U_solid Us, Flux flux
) {
    constexpr int B = 128;
    constexpr int HALO = 3;
    constexpr int NF = 8;

    __shared__ float tile[NF][B + 2 * HALO];

    const int row_f = (nz_f - 1) * nx_all;
    const int row_s = 0;

    const float* src[NF] = {
        Uf.vx + row_f, Uf.vz + row_f, Uf.p + row_f,
        Us.vx + row_s, Us.vz + row_s,
        Us.sx + row_s, Us.sz + row_s, Us.sxz + row_s
    };

    const int le = 3;
    const int ri = nx_all - 4;
    const int pad = pml_span_d + 3;          // 外侧 3 冻结 + npd PML
    const int base = blockIdx.x * B;

    for (int f = 0; f < NF; ++f) {
        for (int t = threadIdx.x; t < B + 2 * HALO; t += blockDim.x) {
            int ix = base + t - HALO;
            ix = max(le, min(ri, ix));
            tile[f][t] = src[f][ix];
        }
    }

    __syncthreads();

    const int ix = base + threadIdx.x;
    if (ix < le || ix > ri) {
        return;
    }

    const int c = threadIdx.x + HALO;

    const int k_lo = max(-HALO, le - ix);
    const int k_hi = min(HALO, ri - ix);

    // 整条流固界面行统一用 7 点高斯滤波（σ=0.55）
    const float *W = GAUSS_W7_PML;

    float weight_sum = 0.0f;
    for (int k = k_lo; k <= k_hi; ++k) {
        weight_sum += W[k + HALO];
    }

    float val[NF];

    for (int f = 0; f < NF; ++f) {
        float sum = 0.0f;

        for (int k = k_lo; k <= k_hi; ++k) {
            sum += W[k + HALO] * tile[f][c + k];
        }

        val[f] = sum / weight_sum;
    }

    // | val[0] | val[1] | val[2] | val[3] | val[4] | val[5] | val[6] | val[7] |
    // |  vx_f  |  vz_f  |  p_f   |  vx_s  |  vz_s  | sxx_s  | szz_s  | sxz_s  |
    const float2 nif = __ldg(flux.nif_d + ix);

    float vn_f = nif.x * val[0] + nif.y * val[1];
    float vt_f = nif.y * val[0] - nif.x * val[1];

    float vn_s = nif.x * val[3] + nif.y * val[4];
    float vt_s = nif.y * val[3] - nif.x * val[4];

    float snn_s = (
        + nif.x * nif.x * val[5]
        + 2.0f * nif.x * nif.y * val[7]
        + nif.y * nif.y * val[6]
    );
    float stt_s = (
        + nif.y * nif.y * val[5] 
        - 2.0f * nif.x * nif.y * val[7] 
        + nif.x * nif.x * val[6]
    );

    float vn_avg = 0.5f * (vn_f + vn_s);
    float snn_avg = 0.5f * (snn_s - val[2]);

    Uf.vx[row_f + ix] = nif.x * vn_avg + nif.y * vt_f;
    Uf.vz[row_f + ix] = nif.y * vn_avg - nif.x * vt_f;
    Uf.p[row_f + ix] = -snn_avg;

    Us.vx[ix] = nif.x * vn_avg + nif.y * vt_s;
    Us.vz[ix] = nif.y * vn_avg - nif.x * vt_s;

    Us.sx[ix] = nif.x * nif.x * snn_avg + nif.y * nif.y * stt_s;
    Us.sz[ix] = nif.y * nif.y * snn_avg + nif.x * nif.x * stt_s;
    Us.sxz[ix] = nif.x * nif.y * (snn_avg - stt_s);
}

__global__ void add_source(
    U_fluid Uf, int posx, int posz, float source
) {
    add_gauss_source(Uf.p, nullptr, posx, posz, nz_f, source);
}

void linear_operate(
    const MemManager &mem, MacDir mdir, int isrc, float time,
    cudaStream_t sm_vel_f, cudaStream_t sm_sig_f,
    cudaStream_t sm_vel_s, cudaStream_t sm_sig_s
) {
    static dim3 gridSize_s((mem.nx_solid + 15) / 16, (mem.nz_solid + 15) / 16);
    static dim3 gridSize_f((mem.nx_fluid + 15) / 16, (mem.nz_fluid + 15) / 16);
    static dim3 blockSize(16, 16);
    static dim3 gridGhost((mem.nx + 255) / 256);

    static cudaEvent_t ev_in[4] = {nullptr, nullptr, nullptr, nullptr};
    if (ev_in[0] == nullptr) {
        for (int i = 0; i < 4; ++i) {
            CHECK(cudaEventCreateWithFlags(&ev_in[i], cudaEventDisableTiming));
        }
    }
    CHECK(cudaEventRecord(ev_in[0], sm_vel_f));
    CHECK(cudaEventRecord(ev_in[1], sm_sig_f));
    CHECK(cudaEventRecord(ev_in[2], sm_vel_s));
    CHECK(cudaEventRecord(ev_in[3], sm_sig_s));
    CHECK(cudaStreamWaitEvent(sm_vel_f, ev_in[3], 0));
    CHECK(cudaStreamWaitEvent(sm_sig_f, ev_in[2], 0));
    CHECK(cudaStreamWaitEvent(sm_vel_s, ev_in[1], 0));
    CHECK(cudaStreamWaitEvent(sm_sig_s, ev_in[0], 0));

    static cudaEvent_t ev_ghost = nullptr;
    if (ev_ghost == nullptr) {
        CHECK(cudaEventCreateWithFlags(&ev_ghost, cudaEventDisableTiming));
    }
    build_ghost_flux<<<gridGhost, 256, 0, sm_vel_f>>>(
        mem.Uf_tmp, mem.Us_tmp, mem.Jacb_d, mem.flux.Gf, mem.flux.Gs
    );
    CHECK(cudaEventRecord(ev_ghost, sm_vel_f));
    CHECK(cudaStreamWaitEvent(sm_vel_s, ev_ghost, 0));

    switch (mdir) {
    case F:
        update_velocity_fluid<F><<<gridSize_f, blockSize, 0, sm_vel_f>>>(
            mem.Uf_tmp, mem.Mf_d, mem.Jacb_d, mem.Uf_k,
            mem.ADEf_xi_tmp, mem.ADEf_xi_k, mem.ADEf_ze_tmp, mem.ADEf_ze_k,
            mem.pml_par_d, mem.flux
        );
        update_stress_fluid<F><<<gridSize_f, blockSize, 0, sm_sig_f>>>(
            mem.Uf_tmp, mem.Mf_d, mem.Jacb_d, mem.Uf_k,
            mem.ADEf_xi_tmp, mem.ADEf_xi_k, mem.ADEf_ze_tmp, mem.ADEf_ze_k,
            mem.pml_par_d
        );
        update_velocity_solid<F><<<gridSize_s, blockSize, 0, sm_vel_s>>>(
            mem.Us_tmp, mem.Ms_d, mem.Jacb_d, mem.Us_k, 
            mem.ADEs_xi_tmp, mem.ADEs_xi_k,
            mem.ADEs_ze_tmp, mem.ADEs_ze_k,
            mem.pml_par_d, mem.flux
        );
        update_stress_solid<F><<<gridSize_s, blockSize, 0, sm_sig_s>>>(
            mem.Us_tmp, mem.Ms_d, mem.Jacb_d, mem.Us_k,
            mem.ADEs_xi_tmp, mem.ADEs_xi_k,
            mem.ADEs_ze_tmp, mem.ADEs_ze_k,
            mem.pml_par_d
        );
        break;
    case B:
        update_velocity_fluid<B><<<gridSize_f, blockSize, 0, sm_vel_f>>>(
            mem.Uf_tmp, mem.Mf_d, mem.Jacb_d, mem.Uf_k,
            mem.ADEf_xi_tmp, mem.ADEf_xi_k, mem.ADEf_ze_tmp, mem.ADEf_ze_k,
            mem.pml_par_d, mem.flux
        );
        update_stress_fluid<B><<<gridSize_f, blockSize, 0, sm_sig_f>>>(
            mem.Uf_tmp, mem.Mf_d, mem.Jacb_d, mem.Uf_k,
            mem.ADEf_xi_tmp, mem.ADEf_xi_k, mem.ADEf_ze_tmp, mem.ADEf_ze_k,
            mem.pml_par_d
        );
        update_velocity_solid<B><<<gridSize_s, blockSize, 0, sm_vel_s>>>(
            mem.Us_tmp, mem.Ms_d, mem.Jacb_d, mem.Us_k, 
            mem.ADEs_xi_tmp, mem.ADEs_xi_k,
            mem.ADEs_ze_tmp, mem.ADEs_ze_k,
            mem.pml_par_d, mem.flux
        );
        update_stress_solid<B><<<gridSize_s, blockSize, 0, sm_sig_s>>>(
            mem.Us_tmp, mem.Ms_d, mem.Jacb_d, mem.Us_k,
            mem.ADEs_xi_tmp, mem.ADEs_xi_k,
            mem.ADEs_ze_tmp, mem.ADEs_ze_k,
            mem.pml_par_d
        );
        break;
    }
    add_source<<<1, 1, 0, sm_sig_f>>>(
        mem.Uf_k, mem.src_pos[isrc].x, 
        mem.src_pos[isrc].y, mem.ricker(time)
    );
}

static void fl_so_apply(
    const MemManager &mem, MacDir mdir,
    cudaStream_t sm_vel_f, cudaStream_t sm_sig_f,
    cudaStream_t sm_vel_s, cudaStream_t sm_sig_s
) {
    static cudaEvent_t ev_k[4] = {nullptr, nullptr, nullptr, nullptr};
    static cudaEvent_t ev_done = nullptr;
    if (ev_k[0] == nullptr) {
        for (int i = 0; i < 4; ++i) {
            CHECK(cudaEventCreateWithFlags(&ev_k[i], cudaEventDisableTiming));
        }
        CHECK(cudaEventCreateWithFlags(&ev_done, cudaEventDisableTiming));
    }
    CHECK(cudaEventRecord(ev_k[0], sm_vel_f));
    CHECK(cudaEventRecord(ev_k[1], sm_sig_f));
    CHECK(cudaEventRecord(ev_k[2], sm_vel_s));
    CHECK(cudaEventRecord(ev_k[3], sm_sig_s));
    CHECK(cudaStreamWaitEvent(sm_sig_f, ev_k[0], 0));
    CHECK(cudaStreamWaitEvent(sm_sig_f, ev_k[2], 0));
    CHECK(cudaStreamWaitEvent(sm_sig_f, ev_k[3], 0));

    const dim3 grid((mem.nx + 127) / 128);
    const dim3 block(128);
    if (mdir == F) {
        fl_so_boundary<F><<<grid, block, 0, sm_sig_f>>>(
            mem.Uf_k, mem.Us_k, mem.flux
        );
    } else {
        fl_so_boundary<B><<<grid, block, 0, sm_sig_f>>>(
            mem.Uf_k, mem.Us_k, mem.flux
        );
    }
    CHECK(cudaEventRecord(ev_done, sm_sig_f));
    CHECK(cudaStreamWaitEvent(sm_vel_f, ev_done, 0));
    CHECK(cudaStreamWaitEvent(sm_vel_s, ev_done, 0));
    CHECK(cudaStreamWaitEvent(sm_sig_s, ev_done, 0));
}

// tmp = U^n + c_tmp * k，b += c_b * k (fluid, solid and ADEs)
static void rk4_accumulate(
    const MemManager &mem, float c_tmp, float c_b,
    cudaStream_t sm_vel_f, cudaStream_t sm_sig_f,
    cudaStream_t sm_vel_s, cudaStream_t sm_sig_s
) {
    const int nxf = mem.nx_fluid, nzf = mem.nz_fluid;
    const int nxs = mem.nx_solid, nzs = mem.nz_solid;

    // U_fluid
    MemManager::vector_add_Uf(
        mem.Uf_d, mem.Uf_k, mem.Uf_tmp, 1.0, c_tmp, nzf, nxf,
        sm_vel_f, sm_sig_f
    );
    MemManager::vector_self_add_Uf(
        mem.Uf_b, mem.Uf_k, c_b, nzf, nxf,
        sm_vel_f, sm_sig_f
    );

    // U_solid
    MemManager::vector_add_Us(
        mem.Us_d, mem.Us_k, mem.Us_tmp, 1.0, c_tmp, nzs, nxs,
        sm_vel_s, sm_sig_s
    );
    MemManager::vector_self_add_Us(
        mem.Us_b, mem.Us_k, c_b, nzs, nxs,
        sm_vel_s, sm_sig_s
    );

    // ADE_fluid
    MemManager::vector_add_ADE(
        mem.ADEf_xi, mem.ADEf_xi_k, mem.ADEf_xi_tmp, 1.0, c_tmp, nzf, nxf,
        sm_vel_f, sm_sig_f
    );
    MemManager::vector_self_add_ADE(
        mem.ADEf_xi_b, mem.ADEf_xi_k, c_b, nzf, nxf,
        sm_vel_f, sm_sig_f
    );
    MemManager::vector_add_ADE(
        mem.ADEf_ze, mem.ADEf_ze_k, mem.ADEf_ze_tmp, 1.0, c_tmp, nzf, nxf,
        sm_vel_f, sm_sig_f
    );
    MemManager::vector_self_add_ADE(
        mem.ADEf_ze_b, mem.ADEf_ze_k, c_b, nzf, nxf,
        sm_vel_f, sm_sig_f
    );

    // ADE_solid
    MemManager::vector_add_ADE(
        mem.ADEs_xi, mem.ADEs_xi_k, mem.ADEs_xi_tmp, 1.0, c_tmp, nzs, nxs,
        sm_vel_s, sm_sig_s
    );
    MemManager::vector_self_add_ADE(
        mem.ADEs_xi_b, mem.ADEs_xi_k, c_b, nzs, nxs,
        sm_vel_s, sm_sig_s
    );
    MemManager::vector_add_ADE(
        mem.ADEs_ze, mem.ADEs_ze_k, mem.ADEs_ze_tmp, 1.0, c_tmp, nzs, nxs,
        sm_vel_s, sm_sig_s
    );
    MemManager::vector_self_add_ADE(
        mem.ADEs_ze_b, mem.ADEs_ze_k, c_b, nzs, nxs,
        sm_vel_s, sm_sig_s
    );
}

// U^{n+1} = U_b + c * k 
// step 4 in a RK4 round
static void rk4_finalize(
    const MemManager &mem,
    cudaStream_t sm_vel_f, cudaStream_t sm_sig_f,
    cudaStream_t sm_vel_s, cudaStream_t sm_sig_s
) {
    const float c  = mem.dt / 6.0f;
    const int nxf = mem.nx_fluid, nzf = mem.nz_fluid;
    const int nxs = mem.nx_solid, nzs = mem.nz_solid;

    MemManager::vector_add_Uf(
        mem.Uf_b, mem.Uf_k, mem.Uf_d, 1.0, c, nzf, nxf,
        sm_vel_f, sm_sig_f
    );
    MemManager::vector_add_Us(
        mem.Us_b, mem.Us_k, mem.Us_d, 1.0, c, nzs, nxs,
        sm_vel_s, sm_sig_s
    );
    MemManager::vector_add_ADE(
        mem.ADEf_xi_b, mem.ADEf_xi_k, mem.ADEf_xi, 1.0, c, nzf, nxf,
        sm_vel_f, sm_sig_f
    );
    MemManager::vector_add_ADE(
        mem.ADEf_ze_b, mem.ADEf_ze_k, mem.ADEf_ze, 1.0, c, nzf, nxf,
        sm_vel_f, sm_sig_f
    );
    MemManager::vector_add_ADE(
        mem.ADEs_xi_b, mem.ADEs_xi_k, mem.ADEs_xi, 1.0, c, nzs, nxs,
        sm_vel_s, sm_sig_s
    );
    MemManager::vector_add_ADE(
        mem.ADEs_ze_b, mem.ADEs_ze_k, mem.ADEs_ze, 1.0, c, nzs, nxs,
        sm_vel_s, sm_sig_s
    );
}

void update_RK4(
    const MemManager &mem, RK4Dir rdir, int isrc, int it,
    cudaStream_t sm_vel_f, cudaStream_t sm_sig_f,
    cudaStream_t sm_vel_s, cudaStream_t sm_sig_s
) {
    const float time  = it * mem.dt;
    const float hf_dt = mem.dt * 0.5;
    const float dt    = mem.dt;

    switch (rdir) {
    case FBFB:
        // =========== step 1 ===========
        linear_operate(
            mem, F, isrc, time, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s
        );
        fl_so_apply(mem, F, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s);
        rk4_accumulate(
            mem, dt / 2, dt / 6, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s
        );

        // =========== step 2 ===========
        linear_operate(
            mem, B, isrc, time + hf_dt, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s
        );
        fl_so_apply(mem, B, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s);
        rk4_accumulate(
            mem, dt / 2, dt / 3, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s
        );

        // =========== step 3 ===========
        linear_operate(
            mem, F, isrc, time + hf_dt, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s
        );
        fl_so_apply(mem, F, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s);
        rk4_accumulate(mem, dt, dt / 3, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s);

        // =========== step 4 ===========
        linear_operate(
            mem, B, isrc, time + dt, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s
        );
        fl_so_apply(mem, B, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s);
        rk4_finalize(mem, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s);
        break;

    case BFBF:
        // =========== step 1 ===========
        linear_operate(
            mem, B, isrc, time, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s
        );
        fl_so_apply(mem, B, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s);
        rk4_accumulate(
            mem, dt / 2, dt / 6, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s
        );

        // =========== step 2 ===========
        linear_operate(
            mem, F, isrc, time + hf_dt, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s
        );
        fl_so_apply(mem, F, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s);
        rk4_accumulate(
            mem, dt / 2, dt / 3, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s
        );

        // =========== step 3 ===========
        linear_operate(
            mem, B, isrc, time + hf_dt, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s
        );
        fl_so_apply(mem, B, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s);
        rk4_accumulate(
            mem, dt, dt / 3, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s
        );

        // =========== step 4 ===========
        linear_operate(
            mem, F, isrc, time + dt, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s
        );
        fl_so_apply(mem, F, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s);
        rk4_finalize(mem, sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s);
        break;
    }

    do {
        const dim3 grid((mem.nx + 127) / 128);
        static cudaEvent_t ev_fin[4] = {nullptr, nullptr, nullptr, nullptr};
        static cudaEvent_t ev_filt = nullptr;
        if (ev_fin[0] == nullptr) {
            for (int i = 0; i < 4; ++i) {
                CHECK(cudaEventCreateWithFlags(&ev_fin[i], cudaEventDisableTiming));
            }
            CHECK(cudaEventCreateWithFlags(&ev_filt, cudaEventDisableTiming));
        }
        CHECK(cudaEventRecord(ev_fin[0], sm_vel_f));
        CHECK(cudaEventRecord(ev_fin[2], sm_vel_s));
        CHECK(cudaEventRecord(ev_fin[3], sm_sig_s));
        CHECK(cudaStreamWaitEvent(sm_sig_f, ev_fin[0], 0));
        CHECK(cudaStreamWaitEvent(sm_sig_f, ev_fin[2], 0));
        CHECK(cudaStreamWaitEvent(sm_sig_f, ev_fin[3], 0));
        fl_so_filter<<<grid, 128, 0, sm_sig_f>>>(mem.Uf_d, mem.Us_d, mem.flux);
        CHECK(cudaEventRecord(ev_filt, sm_sig_f));
        CHECK(cudaStreamWaitEvent(sm_vel_f, ev_filt, 0));
        CHECK(cudaStreamWaitEvent(sm_vel_s, ev_filt, 0));
        CHECK(cudaStreamWaitEvent(sm_sig_s, ev_filt, 0));
    } while(0); 
}

__global__ void sponge_interface_rows(
    U_fluid Uf, U_solid Us, const float * __restrict__ gx
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= nx_all) return;

    float g = __ldg(gx + ix);
    if (g >= 1.0f) return;

    const int fi = (nz_f - 1) * nx_all + ix;
    Uf.vx[fi] *= g;
    Uf.vz[fi] *= g;
    Uf.p [fi] *= g;

    Us.vx [ix] *= g;
    Us.vz [ix] *= g;
    Us.sx [ix] *= g;
    Us.sz [ix] *= g;
    Us.sxz[ix] *= g;
}

void apply_cerjan(
    const MemManager &mem,
    cudaStream_t sm_vel_f, cudaStream_t sm_sig_f,
    cudaStream_t sm_vel_s, cudaStream_t sm_sig_s
) {
    if (mem.sponge_x_d == nullptr) return;

    static cudaEvent_t ev[4] = {nullptr, nullptr, nullptr, nullptr};
    static cudaEvent_t ev_done = nullptr;
    if (ev[0] == nullptr) {
        for (int i = 0; i < 4; ++i) {
            CHECK(cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming));
        }
        CHECK(cudaEventCreateWithFlags(&ev_done, cudaEventDisableTiming));
    }
    cudaStream_t all[4] = {sm_vel_f, sm_sig_f, sm_vel_s, sm_sig_s};
    CHECK(cudaEventRecord(ev[1], sm_sig_f));
    CHECK(cudaEventRecord(ev[2], sm_vel_s));
    CHECK(cudaEventRecord(ev[3], sm_sig_s));
    for (int i = 1; i < 4; ++i) CHECK(cudaStreamWaitEvent(sm_vel_f, ev[i], 0));

    const int block = 256;
    const int grid = (mem.nx + block - 1) / block;
    // sponge_interface_rows<<<grid, block, 0, sm_vel_f>>>(
    //     mem.Uf_d, mem.Us_d, mem.sponge_x_d
    // );

    CHECK(cudaEventRecord(ev_done, sm_vel_f));
    for (int i = 1; i < 4; ++i) CHECK(cudaStreamWaitEvent(all[i], ev_done, 0));
}

} // end of namespace Cpml::FluidSolid
