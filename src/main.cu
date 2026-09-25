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
 * <root>/<shot_id>/snapshot/fluid/vz/vz_%06d.bin   (FANDS)
 * <root>/<shot_id>/snapshot/fluid/p/p_%06d.bin     (FANDS, for seismic record)
 * <root>/<shot_id>/record/                         (seismic record)
 * <root> will be cleared before running.
 */

static void require_create_dirs(const fs::path &path);
static void write_field(const fs::path &dir, const char *field, int it, const float *data, size_t n);
static void output_snapshot(const MemManager &mem, const fs::path &shot_dir, int it);

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(
            stderr,
            "usage: %s <model.json> <output_dir>\n"
            "  例如: %s models/iso_SOLID_PANDF_ortho/"
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
        fprintf(stderr, "output dir must be valid: %s\n", out_root_s.c_str());
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
    for (int isrc = 0; isrc < nsrc; isrc++) {
        const fs::path shot_dir = out_root / std::to_string(isrc);
        require_create_dirs(shot_dir / "snapshot");
        require_create_dirs(shot_dir / "record");
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

            if (mem.calc_type == FANDS) {
                if (mem.bdy_type == CPML) {
                    Cpml::FluidSolid::apply_cerjan(
                        mem, sm.vel_fluid, sm.sig_fluid,
                        sm.vel_solid, sm.sig_solid
                    );
                } else if (mem.bdy_type == PANDF) {
                    PmlFs::FluidSolid::apply_cerjan(
                        mem, sm.vel_fluid, sm.sig_fluid,
                        sm.vel_solid, sm.sig_solid
                    );
                }
            }

            if (it % mem.snapshot == 0) {
                printf("finished %0.2f%%\r", 100.0 * it / mem.nt);
                fflush(stdout);
                output_snapshot(mem, shot_dir, it);
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
    fp.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(n * sizeof(float)));
}

static void output_snapshot(const MemManager &mem, const fs::path &shot_dir, int it) {
    mem.U_cp_d2h();
    const fs::path snap = shot_dir / "snapshot";
    write_field(snap / "solid" / "vz", "vz", it, mem.Us_h.vz, size_t(mem.nz_solid) * mem.nx_solid);

    if (mem.calc_type == FANDS) {
        write_field(snap / "fluid" / "vz", "vz", it, mem.Uf_h.vz, size_t(mem.nz_fluid) * mem.nx_fluid);
        write_field(snap / "fluid" / "p", "p", it, mem.Uf_h.p, size_t(mem.nz_fluid) * mem.nx_fluid);
    }
}
