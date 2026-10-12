#include "cuda_check.h"
#include "json_config.hpp"
#include "elements.h"
#include "common_kernels.cuh"

#include <algorithm>
#include <cstdio>
#include <cmath>

__constant__ int nx_all;
__constant__ int nz_all;
__constant__ int nx_f;
__constant__ int nz_f;
__constant__ int nx_s;
__constant__ int nz_s;
__constant__ int pml_span_d;

__constant__ float GAUSS_W[5] = {
    2.1311488e-08f, 0.011866694f, 0.97626656f, 0.011866694f, 2.1311488e-08f
};
// 7 点版本：物理域用的就是孙耀充论文那套 5 点（外两点为 0）。
__constant__ float GAUSS_W7[7] = {
    0.0f, 2.1311488e-08f, 0.011866694f, 0.97626656f,
    0.011866694f, 2.1311488e-08f, 0.0f
};
// 流固界面行统一使用的滤波：7 点高斯 σ=0.55（中心 0.9316，Nyquist 响应 0.863，
// 比论文那套 0.952 强约 3 倍）。整条界面（含 PML 段）都用它，不用 Cerjan。
// σ 越小越接近论文滤波、虚假反射越小，但 σ=0.4（中心 0.996）会失稳，
// 0.5（0.965）是实测能稳住的最弱一档。
__constant__ float GAUSS_W7_PML[7] = {
    1.1e-13f, 1.678e-06f, 0.03419f, 0.93163f,
    0.03419f, 1.678e-06f, 1.1e-13f
};
__constant__ float SRC_GAUSS_W[5][5] = {
    {0.0030f, 0.0133f, 0.0219f, 0.0133f, 0.0030f},
    {0.0133f, 0.0596f, 0.0983f, 0.0596f, 0.0133f},
    {0.0219f, 0.0983f, 0.1621f, 0.0983f, 0.0219f},
    {0.0133f, 0.0596f, 0.0983f, 0.0596f, 0.0133f},
    {0.0030f, 0.0133f, 0.0219f, 0.0133f, 0.0030f}
};

// helper functions
static void alloc_ade_field(float *&p, int n) {
    CHECK(cudaMalloc((void**)&p, size_t(n) * sizeof(float)));
    CHECK(cudaMemset(p, 0, size_t(n) * sizeof(float)));
}

static void alloc_ade(ADE &a, int n) {
    alloc_ade_field(a.Vx, n);
    alloc_ade_field(a.Vz, n);
    alloc_ade_field(a.Ex, n);
    alloc_ade_field(a.Ez, n);
}

static void copy_ade(const ADE &dst, const ADE &src, int n) {
    const size_t bytes = size_t(n) * sizeof(float);
    CHECK(cudaMemcpy(dst.Vx, src.Vx, bytes, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(dst.Vz, src.Vz, bytes, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(dst.Ex, src.Ex, bytes, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(dst.Ez, src.Ez, bytes, cudaMemcpyDeviceToDevice));
}

// constructor
MemManager::MemManager(const char *filename) {
    const char *default_filename = "./models/params.json";
    if (filename == nullptr) {
        filename = default_filename;
    }
    JsonConfig Json(filename);

    // ========== basic parameters ==========
    fpeak = Json.get<float>("base.fpeak");
    dt = Json.get<float>("base.dt");
    nt = Json.get<int>("base.nt");
    snapshot = Json.get<int>("base.snapshot");

    // ========== grid parameters ===========
    calc_type = static_cast<Calc_type>(Json.get<int>("grid.calc_type"));
    bdy_type = static_cast<Boundary_type>(Json.get<int>("grid.boundary_type"));

    const int nsrc = static_cast<int>(Json.size("grid.src_pos.posx"));
    for (int i = 0; i < nsrc; i++) {
        src_pos.push_back((int2){
            Json.get<int>("grid.src_pos.posx", i),
            Json.get<int>("grid.src_pos.posz", i)
        });
    }

    nx_solid = Json.get<int>("grid.solid.nx");
    nz_solid = Json.get<int>("grid.solid.nz");
    nx = nx_solid;
    
    if (calc_type == FANDS) {
        nx_fluid = Json.get<int>("grid.fluid.nx");
        nz_fluid = Json.get<int>("grid.fluid.nz");
        if (nx_fluid != nx_solid) {
            printf("nx_fluid doesn't match nx_solid!\n");
            exit(1);
        }
        nz = nz_fluid + nz_solid - 1;
    } else {
        nz = nz_solid;
    }

    // ========== PML parameters ============
    npd       = Json.get<int>("PML.npd");
    thickness = Json.get<float>("PML.thickness");
    NPOW      = Json.get<int>("PML.NPOW");
    Vp_max    = Json.get<float>("PML.Vp_max");
    Rc        = Json.get<float>("PML.Rc");
    beta0     = Json.get<float>("PML.beta0");

    if (bdy_type == DIRI || calc_type != FANDS || npd < 2) {
        cerjan_width = 0;
    } else {
        cerjan_width = std::max(3, (int)(npd * 3.0 / 10));
    }
    cerjan_a     = 0.025f;
    pml_span     = npd;

    pml_cfs.beta0   = beta0;
    pml_cfs.p_beta  = 2.0f;
    pml_cfs.p_alpha = 1.0f;
    pml_cfs.alpha0  = 2 * float(M_PI) * fpeak;

    // ========== allocate memory ===========
    mem_allocate_U();
    mem_allocate_Med();
    mem_allocate_grid();
    mem_allocate_PML();
    mem_allocate_sponge();
    mem_allocate_flux();
    mem_allocate_free_surface();

    // ========= read in array data =========
    struct Elements {
        const char *JSONkey;
        float *ptr;
        size_t count;
        int flag;
    } elements[] = {
        {"grid.solid.C11", Ms_h.C11, size_t(nx_solid * nz_solid), SOLID},
        {"grid.solid.C13", Ms_h.C13, size_t(nx_solid * nz_solid), SOLID},
        {"grid.solid.C33", Ms_h.C33, size_t(nx_solid * nz_solid), SOLID},
        {"grid.solid.C55", Ms_h.C55, size_t(nx_solid * nz_solid), SOLID},
        {"grid.solid.rho", Ms_h.rho, size_t(nx_solid * nz_solid), SOLID},
        {"grid.fluid.K",   Mf_h.K,   size_t(nx_fluid * nz_fluid), FANDS},
        {"grid.fluid.rho", Mf_h.rho, size_t(nx_fluid * nz_fluid), FANDS},
        {"grid.jacob.J",    Jacb_h.J,    size_t(nx * nz), SOLID},
        {"grid.jacob.xi_x", Jacb_h.xi_x, size_t(nx * nz), SOLID},
        {"grid.jacob.xi_z", Jacb_h.xi_z, size_t(nx * nz), SOLID},
        {"grid.jacob.ze_x", Jacb_h.ze_x, size_t(nx * nz), SOLID},
        {"grid.jacob.ze_z", Jacb_h.ze_z, size_t(nx * nz), SOLID}
    };

    FILE *fp = nullptr;
    for (const auto &elem : elements) {
        // Skip fluid elements if calc_type is SOLID
        if (elem.flag == FANDS && calc_type == SOLID) continue;

        fp = fopen(Json.get<std::string>(elem.JSONkey).c_str(), "rb");
        if (!fp) {
            throw std::runtime_error(
                "Failed to open file: " + Json.get<std::string>(elem.JSONkey) 
            );
        }

        size_t read_cnt = fread(elem.ptr, sizeof(float), elem.count, fp);
        if (read_cnt != elem.count) {
            fclose(fp);
            throw std::runtime_error(
                "Failed to read data from file: " + Json.get<std::string>(elem.JSONkey)
            );
        }
        fclose(fp);
    }

    calc_pml_par();
    build_sponge();
    build_if_normal();
    build_free_surface();
    build_constants();
}

// destructor
MemManager::~MemManager() {
    auto free_host = [](auto *&ptr) {
        delete[] ptr;
        ptr = nullptr;
    };
    auto free_device = [](auto *&ptr) {
        if (ptr) {
            CHECK(cudaFree((void*)ptr));
            ptr = nullptr;
        }
    };
    auto free_u_fluid = [&](U_fluid &u, bool device) {
        if (device) {
            free_device(u.vx);
            free_device(u.vz);
            free_device(u.p);
        } else {
            free_host(u.vx);
            free_host(u.vz);
            free_host(u.p);
        }
    };
    auto free_u_solid = [&](U_solid &u, bool device) {
        if (device) {
            free_device(u.vx);
            free_device(u.vz);
            free_device(u.sx);
            free_device(u.sz);
            free_device(u.sxz);
        } else {
            free_host(u.vx);
            free_host(u.vz);
            free_host(u.sx);
            free_host(u.sz);
            free_host(u.sxz);
        }
    };
    auto free_ade = [&](ADE &ade) {
        free_device(ade.Vx);
        free_device(ade.Vz);
        free_device(ade.Ex);
        free_device(ade.Ez);
    };

    free_u_fluid(Uf_h, false);
    free_u_solid(Us_h, false);
    free_u_fluid(Uf_d, true);
    free_u_solid(Us_d, true);
    free_u_fluid(Uf_b, true);
    free_u_fluid(Uf_tmp, true);
    free_u_fluid(Uf_k, true);
    free_u_solid(Us_b, true);
    free_u_solid(Us_tmp, true);
    free_u_solid(Us_k, true);

    free_host(Mf_h.rho);
    free_host(Mf_h.K);
    free_host(Ms_h.rho);
    free_host(Ms_h.C11);
    free_host(Ms_h.C13);
    free_host(Ms_h.C33);
    free_host(Ms_h.C55);
    free_device(Mf_d.rho);
    free_device(Mf_d.K);
    free_device(Ms_d.rho);
    free_device(Ms_d.C11);
    free_device(Ms_d.C13);
    free_device(Ms_d.C33);
    free_device(Ms_d.C55);

    free_host(Jacb_h.J);
    free_host(Jacb_h.xi_x);
    free_host(Jacb_h.xi_z);
    free_host(Jacb_h.ze_x);
    free_host(Jacb_h.ze_z);
    free_device(Jacb_d.J);
    free_device(Jacb_d.xi_x);
    free_device(Jacb_d.xi_z);
    free_device(Jacb_d.ze_x);
    free_device(Jacb_d.ze_z);

    free_host(pml_par_h.al_xi);
    free_host(pml_par_h.be_xi);
    free_host(pml_par_h.dp_xi);
    free_host(pml_par_h.al_ze_f);
    free_host(pml_par_h.be_ze_f);
    free_host(pml_par_h.dp_ze_f);
    free_host(pml_par_h.al_ze_s);
    free_host(pml_par_h.be_ze_s);
    free_host(pml_par_h.dp_ze_s);

    free_device(pml_par_d.al_xi);
    free_device(pml_par_d.be_xi);
    free_device(pml_par_d.dp_xi);
    free_device(pml_par_d.al_ze_f);
    free_device(pml_par_d.be_ze_f);
    free_device(pml_par_d.dp_ze_f);
    free_device(pml_par_d.al_ze_s);
    free_device(pml_par_d.be_ze_s);
    free_device(pml_par_d.dp_ze_s);
    free_device(sponge_x_d);

    free_ade(ADEs_xi);    free_ade(ADEs_ze);
    free_ade(ADEs_xi_b);  free_ade(ADEs_xi_tmp);  free_ade(ADEs_xi_k);
    free_ade(ADEs_ze_b);  free_ade(ADEs_ze_tmp);  free_ade(ADEs_ze_k);
    free_ade(ADEf_xi);    free_ade(ADEf_ze);
    free_ade(ADEf_xi_b);  free_ade(ADEf_xi_tmp);  free_ade(ADEf_xi_k);
    free_ade(ADEf_ze_b);  free_ade(ADEf_ze_tmp);  free_ade(ADEf_ze_k);

    free_device(flux.Gf);
    free_device(flux.Gs);
    free_device(flux.nif_d);
    free_host(flux.nif_h);

    free_host(fs_h.Gfs);
    free_host(fs_h.alm);
    free_device(fs_d.Gfs);
    free_device(fs_d.alm);
}

// =========== tool functions ===========
void MemManager::grid_cp_h2d() {
    if (calc_type == FANDS) {
        CHECK(cudaMemcpy(Mf_d.rho, Mf_h.rho, nx_fluid * nz_fluid * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(Mf_d.K,   Mf_h.K,   nx_fluid * nz_fluid * sizeof(float), cudaMemcpyHostToDevice));

        if (Mf_h.rho) { delete[] Mf_h.rho; Mf_h.rho = nullptr; }
        if (Mf_h.K)   { delete[] Mf_h.K;   Mf_h.K   = nullptr; }
    }

    CHECK(cudaMemcpy(Ms_d.rho, Ms_h.rho, nx_solid * nz_solid * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Ms_d.C11, Ms_h.C11, nx_solid * nz_solid * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Ms_d.C13, Ms_h.C13, nx_solid * nz_solid * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Ms_d.C33, Ms_h.C33, nx_solid * nz_solid * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Ms_d.C55, Ms_h.C55, nx_solid * nz_solid * sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(Jacb_d.J,      Jacb_h.J,      nx * nz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Jacb_d.xi_x,   Jacb_h.xi_x,   nx * nz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Jacb_d.xi_z,   Jacb_h.xi_z,   nx * nz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Jacb_d.ze_x, Jacb_h.ze_x, nx * nz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Jacb_d.ze_z, Jacb_h.ze_z, nx * nz * sizeof(float), cudaMemcpyHostToDevice));

    if (Ms_h.rho) { delete[] Ms_h.rho; Ms_h.rho = nullptr; }
    if (Ms_h.C11) { delete[] Ms_h.C11; Ms_h.C11 = nullptr; }
    if (Ms_h.C13) { delete[] Ms_h.C13; Ms_h.C13 = nullptr; }
    if (Ms_h.C33) { delete[] Ms_h.C33; Ms_h.C33 = nullptr; }
    if (Ms_h.C55) { delete[] Ms_h.C55; Ms_h.C55 = nullptr; }
}

void MemManager::U_cp_d2h() const {
    CHECK(cudaMemcpy(Us_h.vx, Us_d.vx, nx_solid * nz_solid * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(Us_h.vz, Us_d.vz, nx_solid * nz_solid * sizeof(float), cudaMemcpyDeviceToHost));
    if (calc_type == FANDS) {
        CHECK(cudaMemcpy(Uf_h.vx, Uf_d.vx, nx_fluid * nz_fluid * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(Uf_h.vz, Uf_d.vz, nx_fluid * nz_fluid * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(Uf_h.p,  Uf_d.p,  nx_fluid * nz_fluid * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

void MemManager::RK4_reset() {
    // reset vector U
    if (calc_type == FANDS) {
        CHECK(cudaMemcpy(Uf_b.vx, Uf_d.vx, nx_fluid * nz_fluid * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(Uf_b.vz, Uf_d.vz, nx_fluid * nz_fluid * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(Uf_b.p,  Uf_d.p,  nx_fluid * nz_fluid * sizeof(float), cudaMemcpyDeviceToDevice));

        CHECK(cudaMemcpy(Uf_tmp.vx, Uf_d.vx, nx_fluid * nz_fluid * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(Uf_tmp.vz, Uf_d.vz, nx_fluid * nz_fluid * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(Uf_tmp.p,  Uf_d.p,  nx_fluid * nz_fluid * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    CHECK(cudaMemcpy(Us_b.vx,  Us_d.vx,  nx_solid * nz_solid * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(Us_b.vz,  Us_d.vz,  nx_solid * nz_solid * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(Us_b.sx,  Us_d.sx,  nx_solid * nz_solid * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(Us_b.sz,  Us_d.sz,  nx_solid * nz_solid * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(Us_b.sxz, Us_d.sxz, nx_solid * nz_solid * sizeof(float), cudaMemcpyDeviceToDevice));

    CHECK(cudaMemcpy(Us_tmp.vx,  Us_d.vx,  nx_solid * nz_solid * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(Us_tmp.vz,  Us_d.vz,  nx_solid * nz_solid * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(Us_tmp.sx,  Us_d.sx,  nx_solid * nz_solid * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(Us_tmp.sz,  Us_d.sz,  nx_solid * nz_solid * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(Us_tmp.sxz, Us_d.sxz, nx_solid * nz_solid * sizeof(float), cudaMemcpyDeviceToDevice));

    if (bdy_type == DIRI) return;

    const int ns = nx_solid * nz_solid;
    copy_ade(ADEs_xi_b,   ADEs_xi, ns);
    copy_ade(ADEs_xi_tmp, ADEs_xi, ns);
    copy_ade(ADEs_ze_b,   ADEs_ze, ns);
    copy_ade(ADEs_ze_tmp, ADEs_ze, ns);
    if (calc_type == FANDS) {
        const int nf = nx_fluid * nz_fluid;
        copy_ade(ADEf_xi_b,   ADEf_xi, nf);
        copy_ade(ADEf_xi_tmp, ADEf_xi, nf);
        copy_ade(ADEf_ze_b,   ADEf_ze, nf);
        copy_ade(ADEf_ze_tmp, ADEf_ze, nf);
    }
}

float MemManager::ricker(float time) const {
    static float T = 1.2 / fpeak;
    float t = time - T;
    if (t > T) return 0.0f;
    float temp = M_PI * fpeak * t;
    temp *= temp;
    return (1 - 2 * temp) * exp(-temp);
}

// =========== vector operations ===========
void MemManager::vector_add_Us(
    const U_solid &Us1, 
    const U_solid &Us2, 
    const U_solid &Us3,
    float c1, float c2,
    int row, int col,
    cudaStream_t stream_vel,
    cudaStream_t stream_sig
) {
    dim3 gridSize((col + 15) / 16, (row + 15) / 16);
    dim3 blockSize(16, 16);
    matrix_add<<<gridSize, blockSize, 0, stream_sig>>>(
        Us1.sx, Us2.sx, Us3.sx, 
        c1, c2, row, col
    );
    matrix_add<<<gridSize, blockSize, 0, stream_vel>>>(
        Us1.vx, Us2.vx, Us3.vx, 
        c1, c2, row, col
    );
    matrix_add<<<gridSize, blockSize, 0, stream_sig>>>(
        Us1.sz, Us2.sz, Us3.sz, 
        c1, c2, row, col
    );
    matrix_add<<<gridSize, blockSize, 0, stream_vel>>>(
        Us1.vz, Us2.vz, Us3.vz, 
        c1, c2, row, col
    );
    matrix_add<<<gridSize, blockSize, 0, stream_sig>>>(
        Us1.sxz, Us2.sxz, Us3.sxz, 
        c1, c2, row, col
    );
}

void MemManager::vector_add_Uf(
    const U_fluid &Uf1, 
    const U_fluid &Uf2, 
    const U_fluid &Uf3,
    float c1, float c2,
    int row, int col,
    cudaStream_t stream_vel,
    cudaStream_t stream_sig
) {
    dim3 gridSize((col + 15) / 16, (row + 15) / 16);
    dim3 blockSize(16, 16);
    matrix_add<<<gridSize, blockSize, 0, stream_vel>>>(
        Uf1.vx, Uf2.vx, Uf3.vx, 
        c1, c2, row, col
    );
    matrix_add<<<gridSize, blockSize, 0, stream_sig>>>(
        Uf1.p, Uf2.p, Uf3.p, 
        c1, c2, row, col
    );
    matrix_add<<<gridSize, blockSize, 0, stream_vel>>>(
        Uf1.vz, Uf2.vz, Uf3.vz, 
        c1, c2, row, col
    );
}

void MemManager::vector_add_ADE(
    const ADE &ADE1,
    const ADE &ADE2,
    const ADE &ADE3,
    float c1, float c2,
    int row, int col,
    cudaStream_t stream_V,
    cudaStream_t stream_E
) {
    dim3 gridSize((col + 15) / 16, (row + 15) / 16);
    dim3 blockSize(16, 16);
    matrix_add<<<gridSize, blockSize, 0, stream_V>>>(
        ADE1.Vx, ADE2.Vx, ADE3.Vx,
        c1, c2, row, col
    );
    matrix_add<<<gridSize, blockSize, 0, stream_E>>>(
        ADE1.Ex, ADE2.Ex, ADE3.Ex,
        c1, c2, row, col
    );
    matrix_add<<<gridSize, blockSize, 0, stream_V>>>(
        ADE1.Vz, ADE2.Vz, ADE3.Vz,
        c1, c2, row, col
    );
    matrix_add<<<gridSize, blockSize, 0, stream_E>>>(
        ADE1.Ez, ADE2.Ez, ADE3.Ez,
        c1, c2, row, col
    );
}

void MemManager::vector_self_add_Us(
    const U_solid &Us1,
    const U_solid &Us2,
    float cB,
    int row, int col,
    cudaStream_t stream_vel,
    cudaStream_t stream_sig
) {
    dim3 gridSize((col + 15) / 16, (row + 15) / 16);
    dim3 blockSize(16, 16);
    matrix_self_add<<<gridSize, blockSize, 0, stream_sig>>>(
        Us1.sx, Us2.sx, cB, row, col
    );
    matrix_self_add<<<gridSize, blockSize, 0, stream_vel>>>(
        Us1.vx, Us2.vx, cB, row, col
    );
    matrix_self_add<<<gridSize, blockSize, 0, stream_sig>>>(
        Us1.sz, Us2.sz, cB, row, col
    );
    matrix_self_add<<<gridSize, blockSize, 0, stream_vel>>>(
        Us1.vz, Us2.vz, cB, row, col
    );
    matrix_self_add<<<gridSize, blockSize, 0, stream_sig>>>(
        Us1.sxz, Us2.sxz, cB, row, col
    );
}

void MemManager::vector_self_add_Uf(
    const U_fluid &Uf1,
    const U_fluid &Uf2,
    float cB,
    int row, int col,
    cudaStream_t stream_vel,
    cudaStream_t stream_sig
) {
    dim3 gridSize((col + 15) / 16, (row + 15) / 16);
    dim3 blockSize(16, 16);

    matrix_self_add<<<gridSize, blockSize, 0, stream_vel>>>(
        Uf1.vx, Uf2.vx, cB, row, col
    );
    matrix_self_add<<<gridSize, blockSize, 0, stream_sig>>>(
        Uf1.p, Uf2.p, cB, row, col
    );
    matrix_self_add<<<gridSize, blockSize, 0, stream_vel>>>(
        Uf1.vz, Uf2.vz, cB, row, col
    );
}

void MemManager::vector_self_add_ADE(
    const ADE &ADE1,
    const ADE &ADE2,
    float cB,
    int row, int col,
    cudaStream_t stream_V,
    cudaStream_t stream_E
) {
    dim3 gridSize((col + 15) / 16, (row + 15) / 16);
    dim3 blockSize(16, 16);

    matrix_self_add<<<gridSize, blockSize, 0, stream_V>>>(
        ADE1.Vx, ADE2.Vx, cB, row, col
    );
    matrix_self_add<<<gridSize, blockSize, 0, stream_E>>>(
        ADE1.Ex, ADE2.Ex, cB, row, col
    );
    matrix_self_add<<<gridSize, blockSize, 0, stream_V>>>(
        ADE1.Vz, ADE2.Vz, cB, row, col
    );
    matrix_self_add<<<gridSize, blockSize, 0, stream_E>>>(
        ADE1.Ez, ADE2.Ez, cB, row, col
    );
}

// =========== allocate memory ===========
void MemManager::mem_allocate_U() {
    // field vectors (host)
    if (calc_type == FANDS) {
        Uf_h.vx  = new float[nx_fluid * nz_fluid];
        Uf_h.vz  = new float[nx_fluid * nz_fluid];
        Uf_h.p   = new float[nx_fluid * nz_fluid];
    }

    Us_h.vx  = new float[nx_solid * nz_solid];
    Us_h.vz  = new float[nx_solid * nz_solid];
    Us_h.sx  = new float[nx_solid * nz_solid];
    Us_h.sz  = new float[nx_solid * nz_solid];
    Us_h.sxz = new float[nx_solid * nz_solid];

    // field vectors (device)
    if (calc_type == FANDS) {
        // fluid
        CHECK(cudaMalloc((void**)&Uf_d.vx, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMalloc((void**)&Uf_d.vz, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMalloc((void**)&Uf_d.p,  nx_fluid * nz_fluid * sizeof(float)));

        CHECK(cudaMalloc((void**)&Uf_b.vx, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMalloc((void**)&Uf_b.vz, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMalloc((void**)&Uf_b.p,  nx_fluid * nz_fluid * sizeof(float)));

        CHECK(cudaMalloc((void**)&Uf_tmp.vx, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMalloc((void**)&Uf_tmp.vz, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMalloc((void**)&Uf_tmp.p,  nx_fluid * nz_fluid * sizeof(float)));

        CHECK(cudaMalloc((void**)&Uf_k.vx, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMalloc((void**)&Uf_k.vz, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMalloc((void**)&Uf_k.p,  nx_fluid * nz_fluid * sizeof(float)));

        CHECK(cudaMemset(Uf_d.vx, 0, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMemset(Uf_d.vz, 0, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMemset(Uf_d.p,  0, nx_fluid * nz_fluid * sizeof(float)));

        CHECK(cudaMemset(Uf_b.vx, 0, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMemset(Uf_b.vz, 0, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMemset(Uf_b.p,  0, nx_fluid * nz_fluid * sizeof(float)));

        CHECK(cudaMemset(Uf_tmp.vx, 0, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMemset(Uf_tmp.vz, 0, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMemset(Uf_tmp.p,  0, nx_fluid * nz_fluid * sizeof(float)));

        CHECK(cudaMemset(Uf_k.vx, 0, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMemset(Uf_k.vz, 0, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMemset(Uf_k.p,  0, nx_fluid * nz_fluid * sizeof(float)));
    }

    CHECK(cudaMalloc((void**)&Us_d.vx,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_d.vz,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_d.sx,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_d.sz,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_d.sxz, nx_solid * nz_solid * sizeof(float)));

    CHECK(cudaMalloc((void**)&Us_b.vx,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_b.vz,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_b.sx,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_b.sz,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_b.sxz, nx_solid * nz_solid * sizeof(float)));

    CHECK(cudaMalloc((void**)&Us_tmp.vx,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_tmp.vz,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_tmp.sx,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_tmp.sz,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_tmp.sxz, nx_solid * nz_solid * sizeof(float)));

    CHECK(cudaMalloc((void**)&Us_k.vx,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_k.vz,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_k.sx,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_k.sz,  nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Us_k.sxz, nx_solid * nz_solid * sizeof(float)));

    CHECK(cudaMemset(Us_d.vx,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_d.vz,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_d.sx,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_d.sz,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_d.sxz, 0, nx_solid * nz_solid * sizeof(float)));

    CHECK(cudaMemset(Us_b.vx,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_b.vz,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_b.sx,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_b.sz,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_b.sxz, 0, nx_solid * nz_solid * sizeof(float)));

    CHECK(cudaMemset(Us_tmp.vx,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_tmp.vz,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_tmp.sx,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_tmp.sz,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_tmp.sxz, 0, nx_solid * nz_solid * sizeof(float)));

    CHECK(cudaMemset(Us_k.vx,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_k.vz,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_k.sx,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_k.sz,  0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Us_k.sxz, 0, nx_solid * nz_solid * sizeof(float)));

    printf("Vector U memory allocation finished.\n");
}

void MemManager::mem_allocate_Med() {
    if (calc_type == FANDS) {
        Mf_h.rho = new float[nx_fluid * nz_fluid];
        Mf_h.K   = new float[nx_fluid * nz_fluid];

        CHECK(cudaMalloc((void**)&Mf_d.rho, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMalloc((void**)&Mf_d.K,   nx_fluid * nz_fluid * sizeof(float)));

        CHECK(cudaMemset(Mf_d.rho, 0, nx_fluid * nz_fluid * sizeof(float)));
        CHECK(cudaMemset(Mf_d.K,   0, nx_fluid * nz_fluid * sizeof(float)));
    }

    Ms_h.rho = new float[nx_solid * nz_solid];
    Ms_h.C11 = new float[nx_solid * nz_solid];
    Ms_h.C13 = new float[nx_solid * nz_solid];
    Ms_h.C33 = new float[nx_solid * nz_solid];
    Ms_h.C55 = new float[nx_solid * nz_solid];

    CHECK(cudaMalloc((void**)&Ms_d.rho, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Ms_d.C11, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Ms_d.C13, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Ms_d.C33, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&Ms_d.C55, nx_solid * nz_solid * sizeof(float)));

    CHECK(cudaMemset(Ms_d.rho, 0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Ms_d.C11, 0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Ms_d.C13, 0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Ms_d.C33, 0, nx_solid * nz_solid * sizeof(float)));
    CHECK(cudaMemset(Ms_d.C55, 0, nx_solid * nz_solid * sizeof(float)));

    printf("Media memory allocation finished.\n");
}

void MemManager::mem_allocate_grid() {
    Jacb_h.J    = new float[nx * nz];
    Jacb_h.xi_x = new float[nx * nz];
    Jacb_h.xi_z = new float[nx * nz];
    Jacb_h.ze_x = new float[nx * nz];
    Jacb_h.ze_z = new float[nx * nz];

    CHECK(cudaMalloc((void**)&Jacb_d.J,    nx * nz * sizeof(float)));
    CHECK(cudaMalloc((void**)&Jacb_d.xi_x, nx * nz * sizeof(float)));
    CHECK(cudaMalloc((void**)&Jacb_d.xi_z, nx * nz * sizeof(float)));
    CHECK(cudaMalloc((void**)&Jacb_d.ze_x, nx * nz * sizeof(float)));
    CHECK(cudaMalloc((void**)&Jacb_d.ze_z, nx * nz * sizeof(float)));

    CHECK(cudaMemset(Jacb_d.J,    0, nx * nz * sizeof(float)));
    CHECK(cudaMemset(Jacb_d.xi_x, 0, nx * nz * sizeof(float)));
    CHECK(cudaMemset(Jacb_d.xi_z, 0, nx * nz * sizeof(float)));
    CHECK(cudaMemset(Jacb_d.ze_x, 0, nx * nz * sizeof(float)));
    CHECK(cudaMemset(Jacb_d.ze_z, 0, nx * nz * sizeof(float)));

    printf("Jacobian parameters memory allocation finished.\n");
}

void MemManager::mem_allocate_PML() {
    if (bdy_type == DIRI) return;

    pml_par_h.al_xi   = new float[nx];
    pml_par_h.be_xi   = new float[nx];
    pml_par_h.dp_xi   = new float[nx];
    pml_par_h.al_ze_s = new float[nz_solid];
    pml_par_h.be_ze_s = new float[nz_solid];
    pml_par_h.dp_ze_s = new float[nz_solid];
    CHECK(cudaMalloc((void**)&pml_par_d.al_xi, nx * sizeof(float)));
    CHECK(cudaMalloc((void**)&pml_par_d.be_xi, nx * sizeof(float)));
    CHECK(cudaMalloc((void**)&pml_par_d.dp_xi, nx * sizeof(float)));
    CHECK(cudaMalloc((void**)&pml_par_d.al_ze_s, nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&pml_par_d.be_ze_s, nz_solid * sizeof(float)));
    CHECK(cudaMalloc((void**)&pml_par_d.dp_ze_s, nz_solid * sizeof(float)));
    if (calc_type == FANDS) {
        pml_par_h.al_ze_f = new float[nz_fluid];
        pml_par_h.be_ze_f = new float[nz_fluid];
        pml_par_h.dp_ze_f = new float[nz_fluid];
        CHECK(cudaMalloc((void**)&pml_par_d.al_ze_f, nz_fluid * sizeof(float)));
        CHECK(cudaMalloc((void**)&pml_par_d.be_ze_f, nz_fluid * sizeof(float)));
        CHECK(cudaMalloc((void**)&pml_par_d.dp_ze_f, nz_fluid * sizeof(float)));
    }

    // ADE memory: one set per computational direction, full-size arrays (2D is acceptable)
    const int ns = nx_solid * nz_solid;
    alloc_ade(ADEs_xi,     ns);
    alloc_ade(ADEs_xi_b,   ns);
    alloc_ade(ADEs_xi_tmp, ns);
    alloc_ade(ADEs_xi_k,   ns);
    alloc_ade(ADEs_ze,     ns);
    alloc_ade(ADEs_ze_b,   ns);
    alloc_ade(ADEs_ze_tmp, ns);
    alloc_ade(ADEs_ze_k,   ns);

    if (calc_type == FANDS) {
        const int nf = nx_fluid * nz_fluid;
        alloc_ade(ADEf_xi,     nf);
        alloc_ade(ADEf_xi_b,   nf);
        alloc_ade(ADEf_xi_tmp, nf);
        alloc_ade(ADEf_xi_k,   nf);
        alloc_ade(ADEf_ze,     nf);
        alloc_ade(ADEf_ze_b,   nf);
        alloc_ade(ADEf_ze_tmp, nf);
        alloc_ade(ADEf_ze_k,   nf);
    }

    printf("PML memory allocation finished.\n");
}

void MemManager::mem_allocate_sponge() {
    if (bdy_type == DIRI || cerjan_width <= 0) return;
    CHECK(cudaMalloc((void**)&sponge_x_d, nx * sizeof(float)));
}

void MemManager::build_sponge() {
    if (sponge_x_d == nullptr) return;

    float *gx = new float[nx];
    for (int i = 0; i < nx; ++i) gx[i] = 1.0f;

    for (int j = 0; j < cerjan_width; ++j) {
        const float d = float(cerjan_width - j);
        const int il = 3 + j;
        const int ir = nx - 4 - j;
        if (il >= 0 && il < nx) gx[il] = expf(-cerjan_a * d * d);
        if (ir >= 0 && ir < nx) gx[ir] = expf(-cerjan_a * d * d);
    }

    CHECK(cudaMemcpy(sponge_x_d, gx, nx * sizeof(float), cudaMemcpyHostToDevice));
    printf("[Cerjan/FSI] interface rows + %d Cerjan cells (= 3/10 npd), a = %.3f, "
           "edge factor = %.6e\n",
           cerjan_width, cerjan_a,
           expf(-cerjan_a * float(cerjan_width * cerjan_width)));
    delete[] gx;
}

void MemManager::mem_allocate_flux() {
    if (calc_type != FANDS) return;

    struct FluxElem {
        const char *name;
        float2 **ptr;
        size_t count;
    };
    FluxElem table[] = {
        {"flux.Gf",    &flux.Gf,    size_t(3 * nx)},
        {"flux.Gs",    &flux.Gs,    size_t(3 * nx)},
        {"flux.nif_d", &flux.nif_d, size_t(nx)},
    };

    for (const auto &e : table) {
        CHECK(cudaMalloc((void**)e.ptr, e.count * sizeof(float2)));
        CHECK(cudaMemset(*e.ptr, 0, e.count * sizeof(float2)));
    }
    flux.nif_h = new float2[nx];

    printf("Fluid-solid flux memory allocation finished.\n");
}

void MemManager::mem_allocate_free_surface() {
    if (bdy_type != PANDF) return;
    fs_h.Gfs = new float2[nx * 3];
    fs_h.alm = new ALM[nx];
    CHECK(cudaMalloc((void**)&fs_d.Gfs, nx * 3 * sizeof(float2)));
    CHECK(cudaMalloc((void**)&fs_d.alm, nx * sizeof(ALM)));
    CHECK(cudaMemset(fs_d.Gfs, 0, nx * 3 * sizeof(float2)));
    CHECK(cudaMemset(fs_d.alm, 0, nx * sizeof(ALM)));
    printf("Free surface memory allocation finished.\n");
}

void MemManager::calc_pml_par() {
    if (bdy_type == DIRI) {
        return;
    }
    // 余弦剖面 d(r)=d0/2*[1-cos(pi*r)] 的积分为 1/2，
    // 为保持与幂律剖面相同的 Rc，d0 = -Vp*ln(Rc)/L。
    // const float damp0 = -Vp_max * std::log(Rc) / thickness;
    // 原幂律剖面 d(r)=d0*r^NPOW 的标定：
    const float damp0 = -(NPOW + 1) * Vp_max * std::log(Rc) / (2.0f * thickness);
    printf("[PML] beta0 = %.3f, p_beta = %.2f, p_alpha = %.2f,"
           " alpha0 = %.1f 1/s, d0 = %.1f 1/s\n",
           pml_cfs.beta0, pml_cfs.p_beta, pml_cfs.p_alpha,
           pml_cfs.alpha0, damp0);

    auto profile = [&](float *al, float *be, float *dp, int n, bool lo_pml, bool hi_pml) {
        for (int i = 0; i < n; ++i) {
            int k = -1;
            if (lo_pml && i >= 3 && i <= 2 + pml_span)         k = 3 + pml_span - i;
            if (hi_pml && i >= n - 3 - pml_span && i <= n - 4) k = i - (n - 4 - pml_span);
            if (k <= 0) {
                dp[i] = 0.0f;
                be[i] = 1.0f;
                al[i] = pml_cfs.alpha0;
            } else {
                float r  = float(k) / float(npd);
                // 原幂律剖面：
                dp[i] = damp0 * std::pow(r, float(NPOW));
                // dp[i] = 0.5f * damp0 * (1.0f - std::cos(float(M_PI) * r));
                be[i] = 1.0f + (pml_cfs.beta0 - 1.0f) * std::pow(r, pml_cfs.p_beta);
                al[i] = pml_cfs.alpha0 * (1.0f - std::pow(r, pml_cfs.p_alpha));
            }
        }
    };

    // xi axis: left and right are PML for both CPML and PANDF
    profile(pml_par_h.al_xi, pml_par_h.be_xi, pml_par_h.dp_xi, nx, true, true);

    // zeta axis
    if (calc_type == SOLID) {
        // top: PML only for CPML (PANDF top is a free surface); bottom: PML
        profile(pml_par_h.al_ze_s, pml_par_h.be_ze_s, pml_par_h.dp_ze_s,
                nz_solid, bdy_type == CPML, true);
    } else { // FANDS
        // fluid (upper zone): top PML only for CPML, bottom is the interface
        profile(pml_par_h.al_ze_f, pml_par_h.be_ze_f, pml_par_h.dp_ze_f,
                nz_fluid, bdy_type == CPML, false);
        // solid (lower zone): top is the interface, bottom is PML
        profile(pml_par_h.al_ze_s, pml_par_h.be_ze_s, pml_par_h.dp_ze_s,
                nz_solid, false, true);
    }

    // copy PML parameters from host to device
    auto upload = [](float *dst, const float *src, int n) {
        CHECK(cudaMemcpy(dst, src, size_t(n) * sizeof(float), cudaMemcpyHostToDevice));
    };
    upload(pml_par_d.al_xi, pml_par_h.al_xi, nx);
    upload(pml_par_d.be_xi, pml_par_h.be_xi, nx);
    upload(pml_par_d.dp_xi, pml_par_h.dp_xi, nx);
    upload(pml_par_d.al_ze_s, pml_par_h.al_ze_s, nz_solid);
    upload(pml_par_d.be_ze_s, pml_par_h.be_ze_s, nz_solid);
    upload(pml_par_d.dp_ze_s, pml_par_h.dp_ze_s, nz_solid);
    if (calc_type == FANDS) {
        upload(pml_par_d.al_ze_f, pml_par_h.al_ze_f, nz_fluid);
        upload(pml_par_d.be_ze_f, pml_par_h.be_ze_f, nz_fluid);
        upload(pml_par_d.dp_ze_f, pml_par_h.dp_ze_f, nz_fluid);
    }
}

void MemManager::build_if_normal() {
    if (calc_type != FANDS) return;

    int row = nz_fluid - 1;
    for (int ix = 0; ix < nx; ++ix) {
        int idx = row * nx + ix;
        float zx = Jacb_h.ze_x[idx];
        float zz = Jacb_h.ze_z[idx];
        float s  = std::sqrt(zx * zx + zz * zz);
        flux.nif_h[ix] = make_float2(zx / s, zz / s);
    }
    CHECK(cudaMemcpy(flux.nif_d, flux.nif_h, nx * sizeof(float2), cudaMemcpyHostToDevice));
    printf("Fluid-solid interface normal built.\n");
}

void MemManager::build_free_surface() {
    if (bdy_type != PANDF) return;

    for (int ix = 0; ix < nx; ix++) {
        int idx = ix + 3 * nx;

        double xi_x = Jacb_h.xi_x[idx];
        double xi_z = Jacb_h.xi_z[idx];
        double ze_x = Jacb_h.ze_x[idx];
        double ze_z = Jacb_h.ze_z[idx];

        double m11 = 0.0, m12 = 0.0, m21 = 0.0, m22 = 0.0;

        if (calc_type == SOLID) {
            double Dz11 = Ms_h.C11[idx] * ze_x * ze_x + Ms_h.C55[idx] * ze_z * ze_z;
            double Dz12 = (Ms_h.C13[idx] + Ms_h.C55[idx]) * ze_x * ze_z;
            double Dz22 = Ms_h.C55[idx] * ze_x * ze_x + Ms_h.C33[idx] * ze_z * ze_z;

            double Dx11 = Ms_h.C11[idx] * ze_x * xi_x + Ms_h.C55[idx] * ze_z * xi_z;
            double Dx12 = Ms_h.C13[idx] * ze_x * xi_z + Ms_h.C55[idx] * ze_z * xi_x;
            double Dx21 = Ms_h.C13[idx] * ze_z * xi_x + Ms_h.C55[idx] * ze_x * xi_z;
            double Dx22 = Ms_h.C55[idx] * ze_x * xi_x + Ms_h.C33[idx] * ze_z * xi_z;

            double det = Dz11 * Dz22 - Dz12 * Dz12;
            if (std::abs(det) < 1e-14) {
                m11 = m12 = m21 = m22 = 0.0;
            } else {
                double inv11 =  Dz22 / det;
                double inv12 = -Dz12 / det;
                double inv22 =  Dz11 / det;

                m11 = inv11 * Dx11 + inv12 * Dx21;
                m12 = inv11 * Dx12 + inv12 * Dx22;
                m21 = inv12 * Dx11 + inv22 * Dx21;
                m22 = inv12 * Dx12 + inv22 * Dx22;
            }
        } else {
            double norm2 = ze_x * ze_x + ze_z * ze_z;
            if (norm2 < 1e-14) {
                m11 = m12 = m21 = m22 = 0.0;
            } else {
                double dot = ze_x * xi_x + ze_z * xi_z;
                double J   = ze_x * xi_z - ze_z * xi_x;

                m11 =  dot / norm2;
                m12 =  J   / norm2;
                m21 = -J   / norm2;
                m22 =  dot / norm2;
            }
        }

        double al1_xx = xi_x - ze_x * m11;
        double al1_zz = -ze_z * m21;
        double al1_xz = xi_z - ze_z * m11 - ze_x * m21;
        double al2_xx = -ze_x * m12;
        double al2_zz = xi_z - ze_z * m22;
        double al2_xz = xi_x - ze_z * m12 - ze_x * m22;

        fs_h.alm[ix] = (ALM){
            static_cast<float>(al1_xx),
            static_cast<float>(al1_zz),
            static_cast<float>(al1_xz),
            static_cast<float>(al2_xx),
            static_cast<float>(al2_zz),
            static_cast<float>(al2_xz)
        };
        fs_h.Gfs[ix] = make_float2(0.0f, 0.0f);
    }

    CHECK(cudaMemcpy(fs_d.alm, fs_h.alm, nx * sizeof(ALM), cudaMemcpyHostToDevice));
}

void MemManager::build_constants() {
    CHECK(cudaMemcpyToSymbol(pml_span_d, &pml_span, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(nx_all, &nx, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(nz_all, &nz, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(nx_f, &nx_fluid, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(nz_f, &nz_fluid, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(nx_s, &nx_solid, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(nz_s, &nz_solid, sizeof(int)));
}
