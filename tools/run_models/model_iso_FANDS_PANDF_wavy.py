"""各向同性介质 + 矩形域 + 贴体起伏网格（起伏流固界面 / 起伏自由表面）。

在**项目根目录**下直接运行（不需要命令行参数）:
    python3 model_iso_FANDS_PANDF_wavy.py

几何（矩形物理域，只有某条 ζ=const 网格线在物理空间里起伏）：
    x(ξ,ζ) = h ξ
    z(ξ,ζ) = h ζ + A_i sin(2πx/λ_i) w_i(ζ) + A_f sin(2πx/λ_f) w_f(ζ)
两个包络在**底部 PML 带内（ζ≥b=nz-4-npd）恒为 0、导数也为 0**，所以带内行距
严格等于 h ⇒ 底部 PML 厚度逐列相同；左右带是竖直列（x=hξ）⇒ 逐行相同。
w_f 在 ζ≤3 恒为 1（自由表面在第 3 行，上面 3 行是恒零缓冲）。
脚本末尾会把"逐列/逐行 PML 厚度"打印出来自检。
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
    "font.serif": ["Arial"],    # 衬线字体：Times New Roman
    "font.size": 10,                      # 期刊常用字号10pt，可改为11/12
    "axes.unicode_minus": False,          # 解决负号显示方块
    "mathtext.fontset": "stix",           # 数学公式STIX字体，和Times风格匹配
    "axes.linewidth": 0.8,                # 坐标轴线条粗细（论文常用）
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,              # 图例无边框（很多SCI期刊偏好）
})

# ============================ 算例参数 ============================
case = "iso_FANDS_PANDF_wavy_if"    # models/<case>/、output/<case>/ 都用它

DIRI, CPML, PANDF = 0, 1, 2         # boundary_type
SOLID, FANDS = 0, 1                 # calc_type

calc_type = FANDS
boundary_type = PANDF

fpeak = 30.0                        # 震源主频 [Hz]
dt = 1e-4                        # 时间步长 [s]
nt = 30001                          # 总步数（10 s）
snapshot = 200                     # 每多少步存一帧快照

nx = 501
nz_fluid = 201                      # 含界面行
nz_solid = 381                      # 含界面行
h = 2.5                             # 网格间距 [m]
src_pos_x = [250]
src_pos_z = [60]                    # 在流体里

amp_interface = 50.0                # 流固界面起伏幅值 [m]（0 = 平界面）
wavelength_interface = 500.0
amp_surface = 12.5                   # 自由表面起伏幅值 [m]（0 = 平表面）
wavelength_surface = 1000.0

rho_e = 2000.0                      # 固体
vp = 3200.0
vs = 1732.0
rho_fluid = 1000.0                  # 流体
vp_fluid = 1500.0

npd = 20                            # PML 格数（含外侧 3 行/列冻结带）
NPOW = 2                            # 论文：p_d = 2（师兄 src2d 同）
vp_max = 3200.0
Rc = 1e-3
ppw0 = 8                           # 论文式(41) 的最少每波长点数

show_model = True                   # 是否画模型预览（批处理时设 False）
cuda_bin = "./bin/main_debug"       # 必须在项目根目录下运行
# ==================================================================

# 供 plot_wavefield.py import：界面名 -> 全局 ζ 行号
interface = {
    "free_surface": 3,
    "fluid_solid": nz_fluid - 1,
}

if __name__ == "__main__":
    if not (os.path.isdir("src") and os.path.isfile(cuda_bin)):
        raise SystemExit("请在项目根目录下运行，并先 `make` 生成 " + cuda_bin)
    if amp_surface and boundary_type == CPML:
        raise SystemExit("起伏自由表面只在 PANDF 下有意义")

    nz = nz_fluid + nz_solid - 1 if calc_type == FANDS else nz_solid
    zeta0 = (nz_fluid - 1) if calc_type == FANDS else nz // 2
    b_pml = float(nz - 4 - npd)      # 包络归零处（比首阻尼行再往里 1 行）
    a_i = 3.0 if boundary_type != CPML else float(npd + 3)   # CPML 时顶部 PML 也要等厚

    zeta = np.arange(nz, dtype=np.float64)
    xive = np.arange(nx, dtype=np.float64)
    X = np.broadcast_to((h * xive)[None, :], (nz, nx)).copy()
    xi = h * xive / wavelength_interface
    xf = h * xive / wavelength_surface

    w_i = np.zeros(nz); dw_i = np.zeros(nz)
    if amp_interface:
        seg = (zeta >= a_i) & (zeta <= b_pml)
        den = np.sin(np.pi * (zeta0 - a_i) / (b_pml - a_i)) ** 2
        t = (zeta[seg] - a_i) / (b_pml - a_i)
        w_i[seg] = np.sin(np.pi * t) ** 2 / den
        dw_i[seg] = 2 * np.pi / (b_pml - a_i) * np.sin(np.pi * t) * np.cos(np.pi * t) / den
    w_f = np.ones(nz); dw_f = np.zeros(nz)
    if amp_surface:
        seg = (zeta >= 3.0) & (zeta <= b_pml)
        t = (zeta[seg] - 3.0) / (b_pml - 3.0)
        w_f[seg] = np.cos(0.5 * np.pi * t) ** 2
        dw_f[seg] = -0.5 * np.pi / (b_pml - 3.0) * np.sin(np.pi * t)
        w_f[zeta > b_pml] = 0.0

    Si = np.sin(2 * np.pi * xi); Ci = np.cos(2 * np.pi * xi)
    Sf = np.sin(2 * np.pi * xf); Cf = np.cos(2 * np.pi * xf)
    Z = (h * zeta[:, None] + amp_interface * Si[None, :] * w_i[:, None]
         + amp_surface * Sf[None, :] * w_f[:, None])
    dz_dxi = (amp_interface * 2 * np.pi * h / wavelength_interface * Ci[None, :] * w_i[:, None]
              + amp_surface * 2 * np.pi * h / wavelength_surface * Cf[None, :] * w_f[:, None])
    dz_dze = (h + amp_interface * Si[None, :] * dw_i[:, None]
              + amp_surface * Sf[None, :] * dw_f[:, None])

    # 逆度规：x = hξ ⇒ ξ,z = 0；ζ,x = -z,ξ/J ≠ 0（别写反）
    J = h * dz_dze
    xi_x = np.full((nz, nx), 1.0 / h)
    xi_z = np.zeros((nz, nx))
    ze_x = -dz_dxi / J
    ze_z = h / J

    mu = rho_e * vs ** 2
    lam = rho_e * (vp ** 2 - 2 * vs ** 2)
    K_f = rho_fluid * vp_fluid ** 2

    model_dir = os.path.join("models", case)
    for sub in (["solid", "fluid", "jacob"] if calc_type == FANDS else ["solid", "jacob"]):
        os.makedirs(os.path.join(model_dir, sub), exist_ok=True)
    for name, value, n in [("C11", lam + 2 * mu, nz_solid), ("C13", lam, nz_solid),
                           ("C33", lam + 2 * mu, nz_solid), ("C55", mu, nz_solid)]:
        np.full((n, nx), value, dtype="<f4").tofile(
            os.path.join(model_dir, "solid", name + ".bin"))
    np.full((nz_solid, nx), rho_e, dtype="<f4").tofile(
        os.path.join(model_dir, "solid", "rho.bin"))
    if calc_type == FANDS:
        np.full((nz_fluid, nx), K_f, dtype="<f4").tofile(
            os.path.join(model_dir, "fluid", "K.bin"))
        np.full((nz_fluid, nx), rho_fluid, dtype="<f4").tofile(
            os.path.join(model_dir, "fluid", "rho.bin"))
    for name, arr in [("J", J), ("xi_x", xi_x), ("xi_z", xi_z),
                      ("ze_x", ze_x), ("ze_z", ze_z), ("x", X), ("z", Z)]:
        arr.astype("<f4").tofile(os.path.join(model_dir, "jacob", name + ".bin"))

    # 式(41)：C 取最短波长对应的波速（流体/固体取较小者）
    c_min = min(vs, vp_fluid) if calc_type == FANDS else vs
    beta0 = c_min / (0.5 * ppw0 * h * fpeak)
    thickness = (npd - 1) * h

    grid = {
        "calc_type": calc_type, "boundary_type": boundary_type,
        "src_pos": {"posx": src_pos_x, "posz": src_pos_z},
        "solid": {"nx": nx, "nz": nz_solid,
                  "C11": os.path.join(model_dir, "solid", "C11.bin"),
                  "C13": os.path.join(model_dir, "solid", "C13.bin"),
                  "C33": os.path.join(model_dir, "solid", "C33.bin"),
                  "C55": os.path.join(model_dir, "solid", "C55.bin"),
                  "rho": os.path.join(model_dir, "solid", "rho.bin")},
                "jacob": {"J": os.path.join(model_dir, "jacob", "J.bin"),
                  "xi_x": os.path.join(model_dir, "jacob", "xi_x.bin"),
                  "xi_z": os.path.join(model_dir, "jacob", "xi_z.bin"),
                  "ze_x": os.path.join(model_dir, "jacob", "ze_x.bin"),
                  "ze_z": os.path.join(model_dir, "jacob", "ze_z.bin")},
    }
    if calc_type == FANDS:
        grid["fluid"] = {"nx": nx, "nz": nz_fluid,
                         "K": os.path.join(model_dir, "fluid", "K.bin"),
                         "rho": os.path.join(model_dir, "fluid", "rho.bin")}

    json_path = os.path.join(model_dir, case + ".json")
    with open(json_path, "w") as f:
        json.dump({"base": {"fpeak": fpeak, "dt": dt, "nt": nt, "snapshot": snapshot},
                   "grid": grid,
                   "PML": {"npd": npd, "thickness": thickness, "NPOW": NPOW,
                           "Vp_max": vp_max, "Rc": Rc, "beta0": round(beta0, 4)}},
                  f, indent=2)
        f.write("\n")
    print(f"[model] {nx} x {nz} (fluid {nz_fluid} + solid {nz_solid}), h = {h} m")
    print(f"[model] 起伏: 界面 A={amp_interface} m λ={wavelength_interface} m, "
          f"自由表面 A={amp_surface} m λ={wavelength_surface} m")
    print(f"[model] beta0 = {beta0:.4f} (式(41), C = {c_min} m/s), npd = {npd}")
    # PML 带宽统一口径：从"PML 带内边界"到最外层的墙，包含 冻结3 + 阻尼(npd-3) + 过渡3 格
    bot = np.abs(Z[nz - 3 - npd] - Z[nz - 1])[2:-2]      # 底：逐列
    side = np.abs(X[:, npd + 2] - X[:, 0])               # 侧：逐行
    print(f"[model] PML 带宽（冻结3+阻尼{npd - 3}+过渡3 = {npd + 3} 格）: "
          f"底部 {bot.min():.4f}~{bot.max():.4f} m, 侧向 {side.min():.4f}~{side.max():.4f} m"
          f"（理想 {(npd + 2) * h:.1f} m；纯阻尼段 {(npd - 3) * h:.1f} m）")
    print(f"[model] json   : {json_path}")

    if show_model and plt is not None:
        # 背景：Vp 分布（流体行用流体 Vp，其余用固体 Vp），叠加几何
        vp_map = np.full((nz, nx), float(vp))
        if calc_type == FANDS:
            vp_map[:nz_fluid] = vp_fluid
        vmin = min(vp, vp_fluid if calc_type == FANDS else vp) * 0.9
        vmax = max(vp, vp_fluid if calc_type == FANDS else vp) * 1.1
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.pcolormesh(X, Z, vp_map, cmap="rainbow", vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        fig.colorbar(im, ax=ax, label="Vp [m/s]")
        step = max(1, nx // 50)             # 网格线太密就抽稀
        ax.plot(X[::step, ::step].T, Z[::step, ::step].T, "-", color="0.85", lw=0.3)
        ax.plot(X[::step, ::step], Z[::step, ::step], "-", color="0.85", lw=0.3)
        # ax.plot(X[3], Z[3], "b-", lw=1.3, label="free surface (ζ=3)")
        if calc_type == FANDS:
            ax.plot(X[zeta0], Z[zeta0], "r-", lw=1.3,
                    label=f"fluid-solid interface (ζ={zeta0})")
        # ax.plot(X[nz - 3 - npd], Z[nz - 3 - npd], "g--", lw=1.1,
        #         label=f"bottom PML inner edge (ζ={nz - 3 - npd} → {(npd + 2) * h:.0f} m)")
        # ax.plot(X[:, npd + 2], Z[:, npd + 2], "m--", lw=1.1,
        #         label=f"side PML inner edge (ξ={npd + 2} → {npd + 2}·h = {(npd + 2) * h:.0f} m)")
        ax.invert_yaxis()                   # z 向下为正
        ax.set_aspect("equal")
        ax.set_title(f"{case}\n{nx} x {nz} cells, h = {h} m, "
                     f"interface A = {amp_interface} m, free surface A = {amp_surface} m")
        ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
        # ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(model_dir, case + "_model.png"), dpi=110)
        plt.show(block=True)               # 看一眼就继续，不等你关窗口
        plt.pause(0.1)
    elif show_model:
        print("[model] 没装 matplotlib，跳过模型预览")

    # sys.exit(0)
    out_root = os.path.join("output", case)
    subprocess.run([cuda_bin, json_path, out_root], check=True)
    print(f"[model] snapshots: {out_root}/<炮号>/snapshot/  记录: .../record/")
