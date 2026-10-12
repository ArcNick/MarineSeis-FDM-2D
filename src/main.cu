#include "cuda_check.h"
#include "json_config.hpp"
#include "elements.h"
#include "stream_manager.hpp"
#include "dirichlet_kernels.cuh"
#include "cpml_kernels.cuh"
#include "pml_fs_kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace Diri = Dirichlet;
namespace fs = std::filesystem;

/** 
 * output structure (argv[2] = output root directory, e.g. output/iso_SOLID_PANDF_ortho):
 * <root>/<shot_id>/snapshot/solid/vz/vz_%06d.bin
 * <root>/<shot_id>/snapshot/solid/vx/vx_%06d.bin
 * <root>/<shot_id>/snapshot/fluid/vz/vz_%06d.bin   (FANDS)
 * <root>/<shot_id>/snapshot/fluid/vx/vx_%06d.bin   (FANDS)
 * <root>/<shot_id>/record/fluid_p.bin              (FANDS)
 * <root>/<shot_id>/record/solid_vx.bin             (FANDS)
 * <root>/<shot_id>/record/solid_vz.bin             (FANDS)
 * <root>/<shot_id>/record/solid_vz.bin             (SOLID)
 * <root> will be cleared before running.
 */

static float buffer1[(1 << 12)];
static float buffer2[(1 << 12)];
static float buffer3[(1 << 12)];

static void require_create_dirs(const fs::path &path);
static void write_field(
    const fs::path &dir, 
    const char *field, 
    int it, 
    const float *data, 
    size_t n
);
static void output_record(
    const MemManager &mem, 
    std::ofstream *solid_vz_record, 
    std::ofstream *fluid_vz_record, 
    std::ofstream *fluid_p_record
);
static void output_snapshot(
    const MemManager &mem, 
    const fs::path &shot_dir, 
    int it
);

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(
            stderr,
            "usage: %s <model.json> <output_dir>\n"
            "example: %s models/iso_SOLID_PANDF_ortho/"
            "iso_SOLID_PANDF_ortho.json "
            "output/iso_SOLID_PANDF_ortho\n",
            argv[0], argv[0]
        );
        return 1;
    }
    const fs::path out_root = argv[2];
    const std::string out_root_s = out_root.string();
    if (out_root_s.empty() || out_root_s == "/"
            || out_root_s == "." || out_root_s == "..") {
        fprintf(stderr, "Invalid output dir: %s\n", out_root_s.c_str());
        return 1;
    }

    MemManager mem(argv[1]);
    StreamManager sm;

    mem.grid_cp_h2d();
    std::error_code ec;
    fs::remove_all(out_root, ec);
    if (ec) {
        std::cerr << "cannot clear " << out_root << ": " << ec.message() << "\n";
        return 1;
    }
    require_create_dirs(out_root);
    printf("[output] cleared %s\n", out_root_s.c_str());

    const int nsrc = static_cast<int>(mem.src_pos.size());
    const int gap = static_cast<int>(1e-3 / mem.dt + 0.5);
    // const int gap = 1;
    for (int isrc = 0; isrc < nsrc; isrc++) {
        const fs::path shot_dir = out_root / std::to_string(isrc);
        require_create_dirs(shot_dir / "snapshot");
        require_create_dirs(shot_dir / "record");

        std::ofstream solid_vx_record;
        std::ofstream solid_vz_record;
        std::ofstream fluid_p_record;

        const fs::path record_dir = shot_dir / "record";
        if (mem.calc_type == FANDS) {
            solid_vx_record.open(
                record_dir / "solid_vx.bin",
                std::ios::binary | std::ios::trunc
            );
            solid_vz_record.open(
                record_dir / "solid_vz.bin",
                std::ios::binary | std::ios::trunc
            );
            fluid_p_record.open(
                record_dir / "fluid_p.bin",
                std::ios::binary | std::ios::trunc
            );
            if (!fluid_p_record) {
                std::cerr << "cannot open fluid record files\n";
                return 1;
            }
            if (!solid_vx_record || !solid_vz_record) {
                std::cerr << "cannot open solid record files\n";
                return 1;
            }
        } else {
            solid_vz_record.open(
                record_dir / "solid_vz.bin",
                std::ios::binary | std::ios::trunc
            );
            if (!solid_vz_record) {
                std::cerr << "cannot open solid record files\n";
                return 1;
            }
        }

        for (int it = 0; it < mem.nt; it++) {
            mem.RK4_reset();
            RK4Dir rk4dire = static_cast<RK4Dir>(it & 1);

            if (mem.bdy_type == DIRI) {
                Diri::update_RK4(
                    mem, rk4dire, isrc, it,
                    sm.vel, sm.sig
                );
            }
            else if (mem.bdy_type == CPML) {
                switch (mem.calc_type) {
                case SOLID:
                    Cpml::SolidOnly::update_RK4(
                        mem, rk4dire, isrc, it,
                        sm.vel, sm.sig
                    );
                    break;
                case FANDS:
                    Cpml::FluidSolid::update_RK4(
                        mem, rk4dire, isrc, it,
                        sm.vel_fluid, sm.sig_fluid,
                        sm.vel_solid, sm.sig_solid
                    );
                    break;
                }
            }
            else if (mem.bdy_type == PANDF) {
                switch (mem.calc_type) {
                case SOLID:
                    PmlFs::SolidOnly::update_RK4(
                        mem, rk4dire, isrc, it,
                        sm.vel, sm.sig
                    );
                    break;
                case FANDS:
                    PmlFs::FluidSolid::update_RK4(
                        mem, rk4dire, isrc, it,
                        sm.vel_fluid, sm.sig_fluid,
                        sm.vel_solid, sm.sig_solid
                    );
                    break;
                }
            }

            // if (mem.calc_type == FANDS) {
            //     if (mem.bdy_type == CPML) {
            //         Cpml::FluidSolid::apply_cerjan(
            //             mem, sm.vel_fluid, sm.sig_fluid,
            //             sm.vel_solid, sm.sig_solid
            //         );
            //     } else if (mem.bdy_type == PANDF) {
            //         PmlFs::FluidSolid::apply_cerjan(
            //             mem, sm.vel_fluid, sm.sig_fluid,
            //             sm.vel_solid, sm.sig_solid
            //         );
            //     }
            // }
            const bool take_snapshot = it % mem.snapshot == 0;
            const bool take_record = it % gap == 0;
            if (take_snapshot || take_record) {
                CHECK(cudaDeviceSynchronize());
                if (take_snapshot) { 
                    printf("finished %0.2f%%\r", 100.0 * it / mem.nt);
                    fflush(stdout);
                    output_snapshot(mem, shot_dir, it);
                }
                if (take_record && mem.bdy_type != DIRI) {
                    if (mem.calc_type == FANDS) {
                        output_record(
                            mem, &solid_vx_record, &solid_vz_record, &fluid_p_record
                        );
                    } else {
                        output_record(mem, nullptr, &solid_vz_record, nullptr);
                    }
                }
            }
        }
    }
    return 0;
}

static void require_create_dirs(const fs::path &path) {
    std::error_code ec;
    fs::create_directories(path, ec);
    if (ec) {
        std::cerr << "cannot create " << path << ": " << ec.message() << "\n";
        exit(1);
    }
}

static void write_field(
    const fs::path &dir, const char *field, int it, const float *data, size_t n
) {
    require_create_dirs(dir);
    std::ostringstream name;
    name << field << "_" << std::setw(6) << std::setfill('0') << it << ".bin";
    const fs::path file = dir / name.str();
    std::ofstream fp(file, std::ios::binary);
    if (!fp) {
        std::cerr << "cannot write " << file << "\n";
        exit(1);
    }
    fp.write(
        reinterpret_cast<const char *>(data), 
        static_cast<std::streamsize>(n * sizeof(float))
    );
}

static void output_record(
    const MemManager &mem, 
    std::ofstream *solid_vx_record, 
    std::ofstream *solid_vz_record, 
    std::ofstream *fluid_p_record
) {
    // fluid-solid case
    if (mem.calc_type == FANDS) {
        if (fluid_p_record == nullptr) {
            std::cerr << "fluid record files are not open\n";
            exit(EXIT_FAILURE);
        }
        if (solid_vx_record == nullptr || solid_vz_record == nullptr) {
            std::cerr << "solid record file is not open\n";
            exit(EXIT_FAILURE);
        }

        const int fluid_row = mem.nz_fluid - 1;
        const int solid_row = 0;

        const size_t offset_fluid = fluid_row * mem.nx;
        const size_t offset_solid = solid_row * mem.nx;
        const size_t bytes = mem.nx * sizeof(float);

        CHECK(cudaMemcpy(buffer1, mem.Us_d.vx + offset_solid, bytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(buffer2, mem.Us_d.vz + offset_solid, bytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(buffer3, mem.Uf_d.p  + offset_fluid, bytes, cudaMemcpyDeviceToHost));

        solid_vx_record->write(
            reinterpret_cast<const char *>(buffer1), 
            static_cast<std::streamsize>(bytes)
        );
        solid_vz_record->write(
            reinterpret_cast<const char *>(buffer2), 
            static_cast<std::streamsize>(bytes)
        );
        fluid_p_record->write(
            reinterpret_cast<const char *>(buffer3), 
            static_cast<std::streamsize>(bytes)
        );

        if (!(*fluid_p_record)) {
            std::cerr << "failed to write fluid record data\n";
            exit(EXIT_FAILURE);
        }
        if (!(*solid_vx_record) || !(*solid_vz_record)) {
            std::cerr << "failed to write solid record data\n";
            exit(EXIT_FAILURE);
        }
        return;
    }

    // solid-only case
    if (solid_vz_record == nullptr) {
        std::cerr << "solid record file is not open\n";
        exit(EXIT_FAILURE);
    }

    int solid_row;
    switch (mem.bdy_type) {
    case CPML:
        solid_row = 30;
        break;
    case PANDF:
        solid_row = 3;
        break;
    }

    const size_t offset = solid_row * mem.nx;
    const size_t bytes = mem.nx * sizeof(float);
    CHECK(cudaMemcpy(buffer1, mem.Us_d.vz + offset, bytes, cudaMemcpyDeviceToHost));

    solid_vz_record->write(
        reinterpret_cast<const char *>(buffer1), 
        static_cast<std::streamsize>(bytes)
    );
    if (!(*solid_vz_record)) {
        std::cerr << "failed to write solid record data\n";
        exit(EXIT_FAILURE);
    }
}

static void output_snapshot(const MemManager &mem, const fs::path &shot_dir, int it) {
    mem.U_cp_d2h();
    const fs::path snap = shot_dir / "snapshot";
    const size_t ns = size_t(mem.nz_solid) * mem.nx_solid;
    write_field(snap / "solid" / "vz", "vz", it, mem.Us_h.vz, ns);
    write_field(snap / "solid" / "vx", "vx", it, mem.Us_h.vx, ns);

    if (mem.calc_type == FANDS) {
        const size_t nf = size_t(mem.nz_fluid) * mem.nx_fluid;
        write_field(snap / "fluid" / "vz", "vz", it, mem.Uf_h.vz, nf);
        write_field(snap / "fluid" / "vx", "vx", it, mem.Uf_h.vx, nf);
    }
}
