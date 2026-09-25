#include "dirichlet_kernels.cuh"
#include "common_kernels.cuh"
#include "cuda_check.h"

namespace Dirichlet {

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

    static cudaEvent_t ev_vel = nullptr, ev_sig = nullptr;
    if (ev_vel == nullptr) {
        CHECK(cudaEventCreateWithFlags(&ev_vel, cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&ev_sig, cudaEventDisableTiming));
    }
    CHECK(cudaEventRecord(ev_vel, stream_vel));
    CHECK(cudaEventRecord(ev_sig, stream_sig));
    CHECK(cudaStreamWaitEvent(stream_vel, ev_sig, 0));
    CHECK(cudaStreamWaitEvent(stream_sig, ev_vel, 0));

    switch (mdir) {
    case F:
        update_velocity<F><<<gridSize, blockSize, 0, stream_vel>>>(
            mem.Us_tmp, mem.Ms_d, mem.Jacb_d, mem.Us_k
        );
        update_stress<F><<<gridSize, blockSize, 0, stream_sig>>>(
            mem.Us_tmp, mem.Ms_d, mem.Jacb_d, mem.Us_k
        );
        break;
    case B:
        update_velocity<B><<<gridSize, blockSize, 0, stream_vel>>>(
            mem.Us_tmp, mem.Ms_d, mem.Jacb_d, mem.Us_k
        );
        update_stress<B><<<gridSize, blockSize, 0, stream_sig>>>(
            mem.Us_tmp, mem.Ms_d, mem.Jacb_d, mem.Us_k
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
        // step 1
        linear_operate(mem, F, isrc, time, stream_vel, stream_sig);
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

        // step 2
        linear_operate(mem, B, isrc, time + hf_dt, stream_vel, stream_sig);
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

        // step 3
        linear_operate(mem, F, isrc, time + hf_dt, stream_vel, stream_sig);
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

        // step 4
        linear_operate(mem, B, isrc, time + mem.dt, stream_vel, stream_sig);
        MemManager::vector_add_Us(
            mem.Us_b, mem.Us_k, mem.Us_d,
            1.0, mem.dt / 6, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        break;

    case BFBF:
        // step 1
        linear_operate(mem, B, isrc, time, stream_vel, stream_sig);
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

        // step 2
        linear_operate(mem, F, isrc, time + hf_dt, stream_vel, stream_sig);
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

        // step 3
        linear_operate(mem, B, isrc, time + hf_dt, stream_vel, stream_sig);
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
        
        // step 4
        linear_operate(mem, F, isrc, time + mem.dt, stream_vel, stream_sig);
        MemManager::vector_add_Us(
            mem.Us_b, mem.Us_k, mem.Us_d,
            1.0, mem.dt / 6, mem.nz_solid, mem.nx_solid,
            stream_vel, stream_sig
        );
        break;
    }
}

}