"""流固耦合 + 自由表面 + 纯正交网格（Scholte 波）：生成模型并跑正演。

在**项目根目录**下直接运行（不需要命令行参数）：
    python3 tools/run_models/model_ortho_freesurface_scholte.py

网格是完全正交、均匀的： x = h·(ξ − pad_x)，z = h·ζ。度规因此是常数
    J = h²,  ξ_x = 1/h,  ζ_z = 1/h,  ξ_z = ζ_x = 0
布局（nx、nz_fluid、nz_solid 都给**物理域**尺寸，外侧 PML 由脚本自己加）：
    左右各 pad_x = npd+3 列（外侧 3 列冻结 + npd 列 CFS-PML）；
    顶部 3 行冻结，第 3 行 = 自由表面（PANDF）；
    流固界面在流体最后一行 ζ = nz_fluid + 2；
    底部 npd+3 行（冰冻 3 + CFS-PML npd）。
"""
import json
import os
import subprocess

import numpy as np

try:                                # 只为可选的模型预览；没装 matplotlib 也能跑
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 13,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
})

#==============================
# 算例参数
#==============================
case = "ortho_freesurface_scholte"

DIRI, CPML, PANDF = 0, 1, 2         # boundary_type
SOLID, FANDS = 0, 1                 # calc_type

calc_type = FANDS
boundary_type = PANDF

fpeak = 30.0                        # 震源主频 [Hz]
dt = 2e-4                           # 时间步长 [s]
nt = 16401                          # 总步数（4.04 s）
snapshot = 400                      # 每多少步存一帧快照

nx = 601                           # 物理域列数（不含左右 PML）
nz_fluid = 51                      # 物理域流体行数（不含顶部 3 行冻结）
nz_solid = 81                   # 物理域固体行数（不含底部 npd+3）
h = 2                             # 网格间距 [m]
src_pos_x = [10]                   # 物理域内索引（左端第 0 列）
src_pos_z = [30]                   # 物理域内索引（自由表面算第 0 行）

rho_e = 2000.0                      # 固体
vp = 3200.0
vs = 1800.0
rho_fluid = 1000.0                  # 流体
vp_fluid = 1500.0

npd = 30                            # PML 格数（含外侧 3 行/列冻结带）
NPOW = 2                            # d(x) 剖面指数（= 论文 p_d）
vp_max = 3200                       # None = 自动取 max(固体 Vp, 流体 Vp)
Rc = 1e-3                           # 目标反射系数
ppw0 = 12                           # 论文式(41) 的最少每波长点数

show_model = True                   # 生成后画一张几何+物性预览
cuda_bin = "./bin/main_debug"       # 必须在项目根目录下运行

#==============================
# 绘图接口（供 plot_wavefield.py 用）
#==============================
# 供 plot_wavefield.py import：界面名 -> 全局 ζ 行号
interface = {
    "free_surface": 3,
    "fluid_solid": nz_fluid + 2,
}

if __name__ == "__main__":
    if not (os.path.isdir("src") and os.path.isfile(cuda_bin)):
        raise SystemExit("请在项目根目录下运行，并先 `make` 生成 " + cuda_bin)
    if boundary_type != PANDF:
        raise SystemExit("这个模型是 FANDS + PANDF（顶部自由表面），别改 boundary_type")

#==============================
# 网格派生量：物理域 / 求解域 / pad
#==============================
    top_frozen = 3
    bottom_span = npd + 3
    pad_x = npd + 3                      # 左右：3 列冻结 + npd 列 PML
    nz_fluid_full = nz_fluid + top_frozen
    nz_solid_full = nz_solid + bottom_span
    nz = nz_fluid_full + nz_solid_full - 1
    zeta0 = nz_fluid_full - 1            # 流固界面所在全局行
    nx_full = nx + 2 * pad_x             # 求解器列数
    xive = np.arange(nx_full) - pad_x    # 物理域左端 ξ = pad_x ↔ x = 0

    # ---------- 纯正交网格 + 常数度规 ----------
    X, Z = np.meshgrid(xive * h, np.arange(nz) * h)
    J = np.full((nz, nx_full), h * h, dtype="<f4")
    xi_x = np.full((nz, nx_full), 1.0 / h, dtype="<f4")
    ze_z = np.full((nz, nx_full), 1.0 / h, dtype="<f4")
    xi_z = np.zeros((nz, nx_full), dtype="<f4")
    ze_x = np.zeros((nz, nx_full), dtype="<f4")

    # ---------- 介质参数（各向同性） ----------
#==============================
# 介质参数
#==============================
    mu = rho_e * vs ** 2
    lam = rho_e * (vp ** 2 - 2.0 * vs ** 2)
    K_f = rho_fluid * vp_fluid ** 2

#==============================
# 写模型二进制（solid / fluid / jacob）
#==============================
    model_dir = os.path.join("models", case)
    for sub in ("solid", "fluid", "jacob"):
        os.makedirs(os.path.join(model_dir, sub), exist_ok=True)
    for name, value, n in (("C11", lam + 2 * mu, nz_solid_full),
                           ("C13", lam, nz_solid_full),
                           ("C33", lam + 2 * mu, nz_solid_full),
                           ("C55", mu, nz_solid_full)):
        np.full((n, nx_full), value, dtype="<f4").tofile(
            os.path.join(model_dir, "solid", name + ".bin"))
    np.full((nz_solid_full, nx_full), rho_e, dtype="<f4").tofile(
        os.path.join(model_dir, "solid", "rho.bin"))
    np.full((nz_fluid_full, nx_full), K_f, dtype="<f4").tofile(
        os.path.join(model_dir, "fluid", "K.bin"))
    np.full((nz_fluid_full, nx_full), rho_fluid, dtype="<f4").tofile(
        os.path.join(model_dir, "fluid", "rho.bin"))
    for name, arr in (("J", J), ("xi_x", xi_x), ("xi_z", xi_z),
                      ("ze_x", ze_x), ("ze_z", ze_z),
                      ("x", X.astype("<f4")), ("z", Z.astype("<f4"))):
        arr.astype("<f4").tofile(os.path.join(model_dir, "jacob", name + ".bin"))

    # ---------- PML 参数 ----------
    c_min = min(vs, vp_fluid)
    beta0 = c_min / (0.5 * ppw0 * h * fpeak)
    thickness = (npd - 1) * h
    vp_layer_max = max(vp, vp_fluid)
    if vp_max is None:
        vp_max = vp_layer_max
        print(f"[model] vp_max 自动取 {vp_max:.0f} m/s（固体与流体最大 Vp）")
    elif vp_max < max(vp, vp_fluid):
        print(f"[model] 提醒: vp_max = {vp_max:.0f} m/s 比实际最大 Vp "
              f"{vp_layer_max:.0f} m/s 小，PML 阻尼建议按最快介质取")
    elif vp_max > vp_layer_max * 1.05:
        print(f"[model] 提醒: vp_max = {vp_max:.0f} m/s 比实际最大 Vp "
              f"{vp_layer_max:.0f} m/s 大，PML 会过阻尼（反射/晚时伪场会变大）")

    cfg = {
        "base": {"fpeak": fpeak, "dt": dt, "nt": nt, "snapshot": snapshot},
        "grid": {
            "calc_type": calc_type,
            "boundary_type": boundary_type,
            "src_pos": {"posx": [x + pad_x for x in src_pos_x],
                        "posz": [z + top_frozen for z in src_pos_z]},
            "solid": {
                "nx": nx_full, "nz": nz_solid_full,
                "C11": os.path.join(model_dir, "solid", "C11.bin"),
                "C13": os.path.join(model_dir, "solid", "C13.bin"),
                "C33": os.path.join(model_dir, "solid", "C33.bin"),
                "C55": os.path.join(model_dir, "solid", "C55.bin"),
                "rho": os.path.join(model_dir, "solid", "rho.bin"),
            },
            "fluid": {
                "nx": nx_full, "nz": nz_fluid_full,
                "K": os.path.join(model_dir, "fluid", "K.bin"),
                "rho": os.path.join(model_dir, "fluid", "rho.bin"),
            },
            "jacob": {
                k: os.path.join(model_dir, "jacob", k + ".bin")
                for k in ("J", "xi_x", "xi_z", "ze_x", "ze_z")
            },
        },
        "PML": {"npd": npd, "thickness": thickness, "NPOW": NPOW,
                "Vp_max": vp_max, "Rc": Rc, "beta0": round(beta0, 4)},
    }
#==============================
# 写 JSON 与 pad.txt
#==============================
    json_path = os.path.join(model_dir, case + ".json")
    with open(json_path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    # 网格分区记录（和其它模型脚本同格式：物理域列范围、左右 pad、冻结/PML 行数）
    with open(os.path.join(model_dir, "pad.txt"), "w") as f:
        f.write(f"nx_orig {nx}\n")
        f.write(f"shift {pad_x}\n")
        f.write(f"pad_flat {pad_x}\n")
        f.write(f"outer {pad_x}\n")
        f.write(f"npd {npd}\n")
        f.write(f"nz_fluid_phys {nz_fluid}\n")
        f.write(f"nz_solid_phys {nz_solid}\n")
        f.write(f"nz_fluid_full {nz_fluid_full}\n")
        f.write(f"nz_solid_full {nz_solid_full}\n")
        f.write(f"src_z_offset {top_frozen}\n")
        f.write(f"phys0 {pad_x}\n")
        f.write(f"phys1 {nx_full - pad_x - 1}\n")

#==============================
# 自检输出
#==============================
    print(f"[model] 物理域 {nx} x {(nz_fluid + nz_solid - 1)} "
          f"(fluid {nz_fluid} + solid {nz_solid}), h = {h} m, 纯正交")
    print(f"[model] 求解网格 {nx_full} x {nz} "
          f"(左右各 {pad_x} 列 pad，fluid {nz_fluid_full} + solid {nz_solid_full})")
    print(f"[model] 物理域列 {pad_x}..{nx_full - pad_x - 1}"
          f"（x = 0..{(nx - 1) * h:.1f} m）；左右 pad = {pad_x} 列"
          f"（外侧 3 列冻结 + {npd} 列 CFS-PML），外侧 x = "
          f"{-pad_x * h:.1f}..{(nx_full - 1 - pad_x) * h:.1f} m")
    print(f"[model] 自由表面 ζ=3，流固界面 ζ={zeta0}，"
          f"PML {npd} 格（厚 {(npd - 1) * h:.1f} m，含外侧 3 冻结）")
    print(f"[model] 源 (ξ={src_pos_x[0] + pad_x}, ζ={src_pos_z[0] + top_frozen})，"
          f"即 z = {(src_pos_z[0] + top_frozen) * h:.1f} m")
    print(f"[model] beta0 = {beta0:.4f} (式(41), C = {c_min:.0f} m/s), "
          f"NPOW = {NPOW}, Rc = {Rc:g}")
    print(f"[model] json   : {json_path}")

#==============================
# 模型预览（可选）
#==============================
    if show_model and plt is not None:
        vp_map = np.full((nz, nx_full), float(vp))
        vp_map[:nz_fluid_full] = vp_fluid
        vmin = min(vp, vp_fluid) * 0.9
        vmax = max(vp, vp_fluid) * 1.1
        fig, ax = plt.subplots(figsize=(18, 6))
        im = ax.pcolormesh(X, Z, vp_map, cmap="rainbow", vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        fig.colorbar(im, ax=ax, label="Vp (m/s)")
        step = max(1, nx_full // 50)
        ax.plot(X[::step, ::step].T, Z[::step, ::step].T, "-", color="0.85", lw=0.3)
        ax.plot(X[::step, ::step], Z[::step, ::step], "-", color="0.85", lw=0.3)
        ax.plot(X[3], Z[3], "b-", lw=1.3, label="free surface (ζ=3)")
        ax.plot(X[3], Z[3], "b-", lw=1.3)
        # ax.plot(X[zeta0], Z[zeta0], "r-", lw=1.3,
        #         label=f"fluid-solid interface (ζ={zeta0})")
        ax.plot(X[zeta0], Z[zeta0], "r-", lw=1.3)
        # ax.plot(X[:, npd + 2], Z[:, npd + 2], "k--", lw=0.8)
        # ax.plot(X[:, nx_full - npd - 3], Z[:, nx_full - npd - 3], "k--", lw=0.8)
        # ax.plot(X[nz - npd - 3], Z[nz - npd - 3], "k--", lw=0.8)
        ax.invert_yaxis()               # z 向下为正
        ax.set_aspect("equal")
        ax.set_title(f"{case}\n{nx_full} x {nz} cells, h = {h} m, "
                     f"Vp = {vp:.0f}/{vp_fluid:.0f}, Vs = {vs:.0f} m/s")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        # ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(model_dir, case + "_model.png"), dpi=110)
        plt.show(block=True)
        plt.pause(0.1)
    elif show_model:
        print("[model] 没装 matplotlib，跳过模型预览")

#==============================
# 运行求解器
#==============================
    out_root = os.path.join("output", case)
    subprocess.run([cuda_bin, json_path, out_root], check=True)
    print(f"[model] snapshots: {out_root}/<炮号>/snapshot/  记录: .../record/")
