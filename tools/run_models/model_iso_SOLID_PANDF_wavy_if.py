"""两层各向同性固体 + 自由表面 + 起伏固固界面（矩形域、贴体网格）。

几何与 model_iso_FANDS_PANDF_wavy.py 完全相同：
    x(ξ,ζ) = h ξ
    z(ξ,ζ) = h ζ + A_i sin(2πx/λ_i) w_i(ζ) + A_f sin(2πx/λ_f) w_f(ζ)
自由表面在 ζ=3；固固界面在 ζ=nz_upper。上下两层用同一套 C11/C13/C33/C55/rho
场表示，因此界面就是正常 welded solid-solid 边界。

在项目根目录直接运行：
    python3 tools/run_models/model_iso_SOLID_PANDF_wavy_if.py
"""
import json
import os
import subprocess

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ============================ 算例参数 ============================
case = "iso_SOLID_PANDF_wavy_if"

DIRI, CPML, PANDF = 0, 1, 2
SOLID, FANDS = 0, 1

calc_type = SOLID
boundary_type = PANDF

fpeak = 30.0
dt = 1e-4
nt = 30001
snapshot = 200

nx = 501
nz_upper = 200
nz_lower = 381
nz = nz_upper + nz_lower
h = 2.5
src_pos_x = [250]
src_pos_z = [60]                    # 震源在上层固体

# 与流固 wavy 模型保持同一套几何起伏
amp_interface = 50.0
wavelength_interface = 500.0
amp_surface = 12.5
wavelength_surface = 1000.0

# 上层软固体：Vp 由合理泊松比给出，rho 取常见浅层沉积物值
vp_upper = 2500.0
vs_upper = 1500.0
rho_upper = 1800.0

# 下层硬固体：用户给定
vp_lower = 3200.0
vs_lower = 1732.0
rho_lower = 2000.0

npd = 20
NPOW = 2
vp_max = 3200.0
Rc = 1e-3
ppw0 = 8

show_model = True
cuda_bin = "./bin/main_debug"
# ==================================================================

# 供 plot_wavefield.py import：界面名 -> 全局 ζ 行号
interface = {
    "free_surface": 3,
    "solid_solid": [nz_upper],
}

if __name__ == "__main__":
    if not (os.path.isdir("src") and os.path.isfile(cuda_bin)):
        raise SystemExit("请在项目根目录下运行，并先 `make` 生成 " + cuda_bin)
    if amp_surface and boundary_type != PANDF:
        raise SystemExit("起伏自由表面只在 PANDF 下有意义")

    zeta0 = float(nz_upper)                  # 固固界面网格行
    b_pml = float(nz - 4 - npd)              # 起伏包络在 PML 内归零
    zeta = np.arange(nz, dtype=np.float64)
    xive = np.arange(nx, dtype=np.float64)

    X = np.broadcast_to((h * xive)[None, :], (nz, nx)).copy()
    xi = h * xive / wavelength_interface
    xf = h * xive / wavelength_surface

    # 界面起伏包络：自由表面行 3 以下逐渐打开，底部 PML 内归零
    w_i = np.zeros(nz)
    dw_i = np.zeros(nz)
    if amp_interface:
        seg = (zeta >= 3.0) & (zeta <= b_pml)
        den = np.sin(np.pi * (zeta0 - 3.0) / (b_pml - 3.0)) ** 2
        t = (zeta[seg] - 3.0) / (b_pml - 3.0)
        w_i[seg] = np.sin(np.pi * t) ** 2 / den
        dw_i[seg] = (2 * np.pi / (b_pml - 3.0)
                     * np.sin(np.pi * t) * np.cos(np.pi * t) / den)

    # 自由表面起伏：在 ζ=3 处最大，向下到 PML 归零
    w_f = np.ones(nz)
    dw_f = np.zeros(nz)
    if amp_surface:
        seg = (zeta >= 3.0) & (zeta <= b_pml)
        t = (zeta[seg] - 3.0) / (b_pml - 3.0)
        w_f[seg] = np.cos(0.5 * np.pi * t) ** 2
        dw_f[seg] = -0.5 * np.pi / (b_pml - 3.0) * np.sin(np.pi * t)
        w_f[zeta > b_pml] = 0.0

    Si = np.sin(2 * np.pi * xi)
    Ci = np.cos(2 * np.pi * xi)
    Sf = np.sin(2 * np.pi * xf)
    Cf = np.cos(2 * np.pi * xf)

    Z = (h * zeta[:, None]
         + amp_interface * Si[None, :] * w_i[:, None]
         + amp_surface * Sf[None, :] * w_f[:, None])
    dz_dxi = (
        amp_interface * 2 * np.pi * h / wavelength_interface
        * Ci[None, :] * w_i[:, None]
        + amp_surface * 2 * np.pi * h / wavelength_surface
        * Cf[None, :] * w_f[:, None]
    )
    dz_dze = (
        h
        + amp_interface * Si[None, :] * dw_i[:, None]
        + amp_surface * Sf[None, :] * dw_f[:, None]
    )

    J = h * dz_dze
    xi_x = np.full((nz, nx), 1.0 / h)
    xi_z = np.zeros((nz, nx))
    ze_x = -dz_dxi / J
    ze_z = h / J

    # 两层各向同性弹性参数
    C11_u = rho_upper * vp_upper ** 2
    C55_u = rho_upper * vs_upper ** 2
    C13_u = C11_u - 2.0 * C55_u
    C11_l = rho_lower * vp_lower ** 2
    C55_l = rho_lower * vs_lower ** 2
    C13_l = C11_l - 2.0 * C55_l

    upper = np.broadcast_to(zeta[:, None] < zeta0, (nz, nx))
    C11 = np.where(upper, C11_u, C11_l).astype("<f4")
    C13 = np.where(upper, C13_u, C13_l).astype("<f4")
    C33 = C11.copy()
    C55 = np.where(upper, C55_u, C55_l).astype("<f4")
    rho = np.where(upper, rho_upper, rho_lower).astype("<f4")

    model_dir = os.path.join("models", case)
    solid_dir = os.path.join(model_dir, "solid")
    jacob_dir = os.path.join(model_dir, "jacob")
    os.makedirs(solid_dir, exist_ok=True)
    os.makedirs(jacob_dir, exist_ok=True)

    for name, arr in (("C11", C11), ("C13", C13), ("C33", C33),
                      ("C55", C55), ("rho", rho)):
        arr.astype("<f4").tofile(os.path.join(solid_dir, name + ".bin"))
    for name, arr in (("J", J), ("xi_x", xi_x), ("xi_z", xi_z),
                      ("ze_x", ze_x), ("ze_z", ze_z), ("x", X), ("z", Z)):
        arr.astype("<f4").tofile(os.path.join(jacob_dir, name + ".bin"))

    # 式(41)：纯固体取两层里较小的 Vs
    beta0 = min(vs_upper, vs_lower) / (0.5 * ppw0 * h * fpeak)
    thickness = (npd - 1) * h

    json_path = os.path.join(model_dir, case + ".json")
    with open(json_path, "w") as f:
        json.dump({
            "base": {"fpeak": fpeak, "dt": dt, "nt": nt,
                     "snapshot": snapshot},
            "grid": {
                "calc_type": calc_type,
                "boundary_type": boundary_type,
                "src_pos": {"posx": src_pos_x, "posz": src_pos_z},
                "solid": {
                    "nx": nx, "nz": nz,
                    "C11": os.path.join(solid_dir, "C11.bin"),
                    "C13": os.path.join(solid_dir, "C13.bin"),
                    "C33": os.path.join(solid_dir, "C33.bin"),
                    "C55": os.path.join(solid_dir, "C55.bin"),
                    "rho": os.path.join(solid_dir, "rho.bin"),
                },
                "jacob": {
                    "J": os.path.join(jacob_dir, "J.bin"),
                    "xi_x": os.path.join(jacob_dir, "xi_x.bin"),
                    "xi_z": os.path.join(jacob_dir, "xi_z.bin"),
                    "ze_x": os.path.join(jacob_dir, "ze_x.bin"),
                    "ze_z": os.path.join(jacob_dir, "ze_z.bin"),
                },
            },
            "PML": {"npd": npd, "thickness": thickness, "NPOW": NPOW,
                    "Vp_max": vp_max, "Rc": Rc, "beta0": round(beta0, 4)},
        }, f, indent=2)
        f.write("\n")
    print(f"[model] {nx} x {nz} (upper {nz_upper} + lower {nz_lower}), "
          f"h = {h} m")
    print(f"[model] upper: Vp={vp_upper}, Vs={vs_upper}, rho={rho_upper}")
    print(f"[model] lower: Vp={vp_lower}, Vs={vs_lower}, rho={rho_lower}")
    print(f"[model] solid-solid interface at zeta={int(zeta0)}, "
          f"free surface at zeta=3")
    print(f"[model] beta0 = {beta0:.4f} (eq.41, C = min Vs), npd = {npd}")
    print(f"[model] json   : {json_path}")

    if show_model and plt is not None:
        vp_map = np.where(upper, vp_upper, vp_lower)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.pcolormesh(X, Z, vp_map, cmap="rainbow",
                           vmin=vp_upper * 0.9, vmax=vp_lower * 1.1,
                           shading="auto", rasterized=True)
        fig.colorbar(im, ax=ax, label="Vp [m/s]")
        step = max(1, nx // 50)
        ax.plot(X[::step, ::step].T, Z[::step, ::step].T,
                "-", color="0.85", lw=0.3)
        ax.plot(X[::step, ::step], Z[::step, ::step],
                "-", color="0.85", lw=0.3)
        ax.plot(X[3], Z[3], "b-", lw=1.2, label="free surface")
        ax.plot(X[int(zeta0)], Z[int(zeta0)], "r-", lw=1.2,
                label="solid-solid interface")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(case)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(model_dir, case + "_model.png"), dpi=110)
        plt.show(block=False)
        plt.pause(0.1)
    elif show_model:
        print("[model] matplotlib unavailable; skipped model preview")

    out_root = os.path.join("output", case)
    subprocess.run([cuda_bin, json_path, out_root], check=True)
    print(f"[model] snapshots: {out_root}/<shot>/snapshot/solid/vz/")
