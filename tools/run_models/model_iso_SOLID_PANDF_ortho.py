"""
各向同性均匀固体 + 自由表面(PANDF) + 正交网格：生成模型并跑正演。

在项目根目录下直接运行:
python model_iso_SOLID_PANDF_ortho.py

产出:
    models/<case>/{solid,jacob}/*.bin, <case>.json  模型与度规、求解器输入
    output/<case>/<炮号>/snapshot/{solid,fluid}/...  波场快照二进制
    output/<case>/<炮号>/record/                     预留：地震记录
    figures/<case>/                                 出图脚本写这里
"""
import json
import os
import subprocess

import numpy as np

try:                                # 只为可选的模型预览；没装 matplotlib 也能跑
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ============================ 算例参数 ============================
case = "iso_SOLID_PANDF_ortho"      # models/<case>/ 与 output/<case>/ 都用它

DIRI, CPML, PANDF = 0, 1, 2         # boundary_type
SOLID, FANDS = 0, 1                 # calc_type

calc_type = SOLID
boundary_type = CPML

fpeak = 30.0                        # 震源主频 [Hz]
dt = 1e-4                           # 时间步长 [s]
nt = 25000                          # 总步数（1.0 s）
snapshot = 200                      # 每多少步存一帧快照

nx_solid = 501
nz_solid = 501
h = 2.5                             # 网格间距 [m]
src_pos_x = [250]
src_pos_z = [250]

rho_e = 2000.0                      # 各向同性介质参数
vp = 3200.0
vs = 1732.0
epsilon = 0.0                       # VTI 参数（各向同性时都为 0）
delta = 0.0

npd = 15                           # PML 格数（含外侧 3 行/列冻结带）
NPOW = 2                            # d(x) 剖面指数（= 论文 p_d）
vp_max = 3200.0                     # d0 用的最大速度
Rc = 0.001                           # 目标反射系数
ppw0 = 8                            # 论文式(41) 要求的最少每波长点数

show_model = True                   # 是否弹窗看模型（批处理时设 False）
cuda_bin = "./bin/main_debug"       # 必须在项目根目录下运行
# ==================================================================

# 供 plot_wavefield.py import：当前模型没有内部界面/自由表面线
interface = {}

if __name__ == "__main__":
    if not (os.path.isdir("src") and os.path.isfile(cuda_bin)):
        raise SystemExit("请在项目根目录下运行，并先 `make` 生成 " + cuda_bin)

    C33_e = rho_e * vp ** 2
    C55_e = rho_e * vs ** 2
    C11_e = C33_e * (1 + 2 * epsilon)
    C13_e = ((C33_e - C55_e) * (2 * C33_e * delta + (C33_e - C55_e))) ** 0.5 - C55_e

    thickness = (npd - 1) * h
    beta0 = vs / (0.5 * ppw0 * h * fpeak)
    # beta0 = 4
    model_dir = os.path.join("models", case)
    solid_dir = os.path.join(model_dir, "solid")
    jacob_dir = os.path.join(model_dir, "jacob")
    os.makedirs(solid_dir, exist_ok=True)
    os.makedirs(jacob_dir, exist_ok=True)

    for name, value in [("rho", rho_e), ("C11", C11_e), ("C13", C13_e),
                        ("C33", C33_e), ("C55", C55_e)]:
        np.full((nz_solid, nx_solid), value, dtype="<f4").tofile(
            os.path.join(solid_dir, name + ".bin"))
    for name, value in [("J", h * h), ("xi_x", 1.0 / h), ("xi_z", 0.0),
                        ("ze_x", 0.0), ("ze_z", 1.0 / h)]:
        np.full((nz_solid, nx_solid), value, dtype="<f4").tofile(
            os.path.join(jacob_dir, name + ".bin"))
    X, Z = np.meshgrid(np.arange(nx_solid) * h, np.arange(nz_solid) * h)
    X.astype("<f4").tofile(os.path.join(jacob_dir, "x.bin"))    # 后处理用
    Z.astype("<f4").tofile(os.path.join(jacob_dir, "z.bin"))

    json_path = os.path.join(model_dir, case + ".json")
    with open(json_path, "w") as f:
        json.dump({
            "base": {"fpeak": fpeak, "dt": dt, "nt": nt, "snapshot": snapshot},
            "grid": {
                "calc_type": calc_type,
                "boundary_type": boundary_type,
                "src_pos": {"posx": src_pos_x, "posz": src_pos_z},
                "solid": {
                    "nx": nx_solid, "nz": nz_solid,
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
    print(f"[model] {nx_solid} x {nz_solid}, h = {h} m, dt = {dt} s, nt = {nt} "
          f"({nt * dt:.3f} s), CFL = {vp * dt / h:.3f}")
    print(f"[model] beta0 = {beta0:.4f} (式(41), C = Vs = {vs}), npd = {npd}, "
          f"thickness = {thickness} m")
    print(f"[model] json   : {json_path}")

    if show_model and plt is not None:
        # 背景：Vp 分布（均匀模型就是常数；将来分层/起伏模型直接换成真实 Vp 场）
        Xm, Zm = np.meshgrid(np.arange(nx_solid) * h, np.arange(nz_solid) * h)
        vp_map = np.full((nz_solid, nx_solid), float(vp))
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.pcolormesh(Xm, Zm, vp_map, cmap="viridis",
                           vmin=vp * 0.85, vmax=vp * 1.15, shading="auto")
        fig.colorbar(im, ax=ax, label="Vp [m/s]")
        # 叠加几何：网格线 + 自由表面 + PML 内边界
        step = max(1, nx_solid // 50)
        ax.plot(Xm[::step, ::step].T, Zm[::step, ::step].T, "-", color="0.85", lw=0.3)
        ax.plot(Xm[::step, ::step], Zm[::step, ::step], "-", color="0.85", lw=0.3)
        ax.plot(Xm[3], Zm[3], "b-", lw=1.3, label="free surface (ζ=3)")
        ax.plot(Xm[nz_solid - 3 - npd], Zm[nz_solid - 3 - npd], "g--", lw=1.1,
                label=f"bottom PML inner edge (ζ={nz_solid - 3 - npd} → {(npd + 2) * h:.0f} m)")
        ax.plot(Xm[:, npd + 2], Zm[:, npd + 2], "m--", lw=1.1,
                label=f"side PML inner edge (ξ={npd + 2} → {(npd + 2) * h:.0f} m)")
        ax.invert_yaxis()               # z 向下为正
        ax.set_aspect("equal")
        ax.set_title(f"{case}\n{nx_solid} x {nz_solid} cells, h = {h} m, "
                     f"Vp = {vp:.0f} m/s, Vs = {vs:.0f} m/s, ρ = {rho_e:.0f}")
        ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(model_dir, case + "_model.png"), dpi=110)
        plt.show(block=False)       # 看一眼就继续，不等你关窗口
        plt.pause(0.1)
    elif show_model:
        print("[model] 没装 matplotlib，跳过模型预览")
    
    out_root = os.path.join("output", case)
    subprocess.run([cuda_bin, json_path, out_root], check=True)
    print(f"[model] snapshots: {out_root}/<炮号>/snapshot/  记录: .../record/")
