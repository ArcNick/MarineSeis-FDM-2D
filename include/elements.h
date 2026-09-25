#ifndef ELEMENTS_H
#define ELEMENTS_H

#include <vector>

struct U_fluid {
    float *vx = nullptr;        // vx
    float *vz = nullptr;        // vz
    float *p  = nullptr;        // p
};

struct U_solid {
    float *vx  = nullptr;       // vx
    float *vz  = nullptr;       // vz
    float *sx  = nullptr;       // sigmax
    float *sz  = nullptr;       // sigmaz
    float *sxz = nullptr;       // sigmaxz
};

struct ADE {
    float *Vx = nullptr;
    float *Vz = nullptr;
    float *Ex = nullptr;
    float *Ez = nullptr;
};

struct Med_fluid {
    float *rho = nullptr;        // density
    float *K   = nullptr;        // bulk module
};

struct Med_solid {
    float *rho = nullptr;        // density
    float *C11 = nullptr;        // stiffness for VTI/HTI
    float *C13 = nullptr;        // stiffness for VTI/HTI        
    float *C33 = nullptr;        // stiffness for VTI/HTI
    float *C55 = nullptr;        // stiffness for VTI/HTI
};

struct Grid {
    float *J    = nullptr;     // Jacobian
    float *xi_x = nullptr;     // (partial xi) / (partial x)
    float *xi_z = nullptr;     // (partial xi) / (partial z)
    float *ze_x = nullptr;     // (partial zeta) / (partial x)
    float *ze_z = nullptr;     // (partial zeta) / (partial z)
};

struct PML_par {
    float *al_xi   = nullptr;     // alpha
    float *al_ze_f = nullptr;     // alpha
    float *al_ze_s = nullptr;     // alpha
    float *be_xi   = nullptr;     // beta
    float *be_ze_f = nullptr;     // beta
    float *be_ze_s = nullptr;     // beta
    float *dp_xi   = nullptr;     // damp
    float *dp_ze_f = nullptr;     // damp
    float *dp_ze_s = nullptr;     // damp
};

struct ALM {
    float al1_xx;
    float al1_zz;
    float al1_xz;
    float al2_xx;
    float al2_zz;
    float al2_xz;
};

struct PML_cfs {
    float beta0;
    float p_beta;
    float p_alpha;
    float alpha0;
    int   p_d;
};

// ---------------------------------------------------------------------------
// ζ 方向"界面 / 自由表面"定位（flux_ze_fluid / flux_ze_solid 用）
//   ifz : 界面（或自由表面）所在行号，在【本介质数组内】
//   dir : 介质相对界面在哪一侧
//           +1 = 介质在界面之下（行号增大方向）：固体（界面在 0 行 / 自由表面在 3 行）
//           -1 = 介质在界面之上（行号减小方向）：流固耦合里的流体（界面在 nz_f-1 行）
//   off_z > 0 时实际行号 = ifz + dir*off_z；off_z < 0 时读镜像幽灵槽
// 自由表面为什么是 3：最上面 0/1/2 行是恒 0 行（Dirichlet），自由表面只能落在第 3 行。
// ---------------------------------------------------------------------------
constexpr int DIR_BELOW = +1;      // Medium are below the interface
constexpr int DIR_ABOVE = -1;      // Medium are above the interface
constexpr int IFZ_FREE  = 3;       // Free surface row number

struct Fsurface {
    float2 *Gfs = nullptr;
    ALM    *alm = nullptr;
};

struct Flux {
    float2 *Gf    = nullptr;
    float2 *Gs    = nullptr;
    float2 *nif_h = nullptr;    // interface normal vector (host)
    float2 *nif_d = nullptr;    // interface normal vector (device)
};

enum Calc_type {
    SOLID = 0,                  // the whole zone is solid
    FANDS = 1,                  // the upper is fluid and the lower is solid
};

enum Boundary_type {
    DIRI  = 0,                  // Dirichlet boundary condition
    CPML  = 1,                  // CFS-PML for 4 sides
    PANDF = 2,                  // CFS-PML for left, right and bottom; free surface for top
};

class MemManager {
public:
    // basic parameters
    float fpeak;
    float dt;
    float nt;
    int snapshot;
    
    // grid parameters
    int nx, nz;
    int nx_fluid = 0;               // length x for fluid zone
    int nx_solid;                   // length x for solid zone
    int nz_fluid = 0;               // length z for fluid zone
    int nz_solid;                   // length z for solid zone

    std::vector<int2> src_pos;      // source positions (host)

    U_fluid Uf_h;                   // vector U for fluid zone (host)
    U_solid Us_h;                   // vector U for solid zone (host)
    U_fluid Uf_d;                   // vector U for fluid zone (device)
    U_solid Us_d;                   // vector U for solid zone (device)

    U_fluid Uf_b, Uf_tmp, Uf_k;     // RK4 vector for fluid (device)
    U_solid Us_b, Us_tmp, Us_k;     // RK4 vector for solid (device)

    Med_fluid Mf_h;                 // rho & stiffness for fluid zone (host)
    Med_solid Ms_h;                 // rho & stiffness for solid zone (host)
    Med_fluid Mf_d;                 // rho & stiffness for fluid zone (device)
    Med_solid Ms_d;                 // rho & stiffness for solid zone (device)

    Grid Jacb_h;                    // grid parameters (host)
    Grid Jacb_d;                    // grid parameters (device)
    Calc_type calc_type;            // calculation types
    Boundary_type bdy_type;         // boundary types

    Flux flux;                      // interface fluxes / ghosts / normal (FANDS)
    Fsurface fs_h;                  // free surface parameters (host)
    Fsurface fs_d;                  // free surface parameters (device)

    // PML parameters
    int npd;                        // PML thickness (grid points)
    float thickness;                // PML thickness (m)
    int NPOW;                       // PML damping order
    float Vp_max;                   // maximum P-wave velocity
    float Rc;                       // reflection coefficient
    float beta0;                    // PML coefficient beta
    int   cerjan_width;             // 流固界面行在 PML 外侧叠加 Cerjan 的层数
    int   pml_span;                 // 设备判定 PML 的跨度（= npd）
    float cerjan_a;                 // Cerjan damping coefficient
    float *sponge_x_d = nullptr;    // Cerjan factor gx(ix)
    
    PML_par pml_par_h;              // PML parameters array (host)
    PML_par pml_par_d;              // PML parameters array (device)
    PML_cfs pml_cfs;                // CFS-PML 形状参数（β0、剖面指数、α0）

    ADE ADEs_xi, ADEs_ze;                   // solid: xi / zeta memory
    ADE ADEs_xi_b, ADEs_xi_tmp, ADEs_xi_k;
    ADE ADEs_ze_b, ADEs_ze_tmp, ADEs_ze_k;

    ADE ADEf_xi, ADEf_ze;                   // fluid: xi / zeta memory
    ADE ADEf_xi_b, ADEf_xi_tmp, ADEf_xi_k;
    ADE ADEf_ze_b, ADEf_ze_tmp, ADEf_ze_k;

    MemManager(const char *filename = nullptr);
    ~MemManager();

    // tool functions
    void grid_cp_h2d();
    void U_cp_d2h() const;
    void RK4_reset();
    float ricker(float time) const;

    /** vector operations
     * \brief linear combination of solid vectors
     * \brief Us3 = c1 * Us1 + c2 * Us2
     * \param Us: solid vectors
     * \param c: coefficients of Us
     * \param row row index
     * \param col column index
     * \param stream_vel: streams for velocity
     * \param stream_sig: streams for stress
     */
    static void vector_add_Us(
        const U_solid &Us1, 
        const U_solid &Us2, 
        const U_solid &Us3,
        float c1, float c2,
        int row, int col,
        cudaStream_t stream_vel = nullptr,
        cudaStream_t stream_sig = nullptr
    );
    
    /** vector operations
     * \brief linear combination of fluid vectors
     * \brief Uf3 = c1 * Uf1 + c2 * Uf2
     * \param Uf: fluid vectors
     * \param c: coefficients of Uf
     * \param row row index
     * \param col column index
     * \param stream_vel: streams for velocity
     * \param stream_sig: streams for stress
     */
    static void vector_add_Uf(
        const U_fluid &Uf1, 
        const U_fluid &Uf2, 
        const U_fluid &Uf3,
        float c1, float c2,
        int row, int col,
        cudaStream_t stream_vel = nullptr,
        cudaStream_t stream_sig = nullptr
    );

    /** vector operations
     * \brief linear combination of ADE vectors
     * \brief ADE3 = c1 * ADE1 + c2 * ADE2
     * \param ADE: ADE vectors
     * \param c: coefficients of ADE
     * \param row row index
     * \param col column index
     * \param stream_V: streams for variable memory
     * \param stream_E: streams for variable memory
     */
    static void vector_add_ADE(
        const ADE &ADE1,
        const ADE &ADE2,
        const ADE &ADE3,
        float c1, float c2,
        int row, int col,
        cudaStream_t stream_V = nullptr,
        cudaStream_t stream_E = nullptr
    );
    
    /** vector operations
     * \brief self-add of solid vectors
     * \brief Us1 += cB * Us2
     * \param cB: coefficients of Us
     * \param row row index
     * \param col column index
     * \param stream_vel: streams for velocity
     * \param stream_sig: streams for stress
     */
    static void vector_self_add_Us(
        const U_solid &Us1,
        const U_solid &Us2,
        float cB,
        int row, int col,
        cudaStream_t stream_vel = nullptr,
        cudaStream_t stream_sig = nullptr
    );

    /** vector operations
     * \brief self-add of fluid vectors
     * \brief Uf1 += cB * Uf2
     * \param cB: coefficients of Uf
     * \param row row index
     * \param col column index
     * \param stream_vel: streams for velocity
     * \param stream_sig: streams for stress
     */
    static void vector_self_add_Uf(
        const U_fluid &Uf1,
        const U_fluid &Uf2,
        float cB,
        int row, int col,
        cudaStream_t stream_vel = nullptr,
        cudaStream_t stream_sig = nullptr
    );
    
    /** vector operations
     * \brief self-add of ADE vectors
     * \brief ADE1 += cB * ADE2
     * \param cB: coefficients of ADE
     * \param row row index
     * \param col column index
     * \param stream_V: streams for variable memory
     * \param stream_E: streams for variable memory
     */
    static void vector_self_add_ADE(
        const ADE &ADE1,
        const ADE &ADE2,
        float cB,
        int row, int col,
        cudaStream_t stream_V = nullptr,
        cudaStream_t stream_E = nullptr
    );

private:
    // allocate memory
    void mem_allocate_U();
    void mem_allocate_Med();
    void mem_allocate_grid();
    void mem_allocate_PML();
    void mem_allocate_sponge();
    void mem_allocate_flux();
    void mem_allocate_free_surface();

    void calc_pml_par();
    void build_sponge();
    void build_if_normal();
    void build_free_surface();
    void build_constants();
};
extern __constant__ int pml_span_d;
extern __constant__ int nx_all, nz_all;
extern __constant__ int nx_f, nz_f;
extern __constant__ int nx_s, nz_s;
extern __constant__ float GAUSS_W[5];       // k=-2..2，已归一化

#endif
