"""流固起伏界面：原 501 列整体作为物理域，冻结点+PML 放到外侧 pad 区。

在**项目根目录**下直接运行（不需要命令行参数）:
    python3 tools/run_models/model_FANDS_PANDF_wavy_pad.py

几何（矩形物理域，只有某条 ζ=const 网格线在物理空间里起伏）：
    x(ξ,ζ) = h ξ
    z(ξ,ζ) = h ζ + A_i cos(2πx/λ_i) w_i(ζ) + A_f cos(2πx/λ_f) w_f(ζ)
物理域内部用窗函数把两端值和各阶斜率压到 0；左右两侧各加
`npd+3` 列平整冻结/PML，因此物理域和 pad 光滑连接。
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

#==============================
# 算例参数
#==============================
case = "FANDS_PANDF_wavy_padmove"  # models/<case>/、output/<case>/ 都用它

DIRI, CPML, PANDF = 0, 1, 2         # boundary_type
SOLID, FANDS = 0, 1                 # calc_type

calc_type = FANDS
boundary_type = CPML

fpeak = 25.0                        # 震源主频 [Hz]
dt = 1e-4                        # 时间步长 [s]
nt = 126401                          # 总步数（10 s）
snapshot = 4000                      # 每多少步存一帧快照

nx = 1501
nz_fluid = 601                      # 物理域行数（不含顶部 3 行冻结/缓冲）
nz_solid = 601                      # 物理域行数（不含底部 npd+3）
h = 1.5                             # 网格间距 [m]
src_pos_x = [250]
src_pos_z = [570]                    # 物理域内索引（自由表面算第 0 行）

amp_interface = 0.0                # 流固界面起伏幅值 [m]（0 = 平界面）
wavelength_interface = 625.0
# wavelength_interface = 417
amp_surface = 0.0                    # 自由表面起伏幅值 [m]（0 = 平表面）
wavelength_surface = 250.0

rho_e = 1800.0                      # 固体
vp = 2100.0
vs = 1000.0
rho_fluid = 1000.0                  # 流体
vp_fluid = 1500.0

# 固体分层：除自由表面和流固界面外，其余固体界面都是**水平**的（物理 z = 常数）。
# solid_interface_z 给层间水平界面的深度 [m]（物理 z，最上面一行 z = 0，向下为正，
# 自由表面大约在 z = 6 m）；solid_layers 从上到下给每层的 (vp, vs, rho)，
# 个数必须是 len(solid_interface_z) + 1。留空 [] 就是单层
# （只有流固界面 + 自由表面，没有水平固体界面）。
solid_interface_z = []
solid_layers = [(vp, vs, rho_e)]
npd = 30                            # PML 格数（含外侧 3 行/列冻结带）
n_taper = 0                        # 物理域两端用于把斜率压到 0 的窗函数宽度
NPOW = 2                            # 论文：p_d = 2（师兄 src2d 同）
vp_max = 2800                       # None = 自动取各层与流体的最大 Vp；填数则按你给的值
Rc = 1e-3
ppw0 = 10                           # 论文式(41) 的最少每波长点数

show_model = True                  # 批处理测试
cuda_bin = "./bin/main_debug"       # 必须在项目根目录下运行

#==============================
# 绘图接口（供 plot_wavefield.py 用）
#==============================
# 供 plot_wavefield.py import：界面名 -> 全局 ζ 行号
interface = {
    "free_surface": 3,
    "fluid_solid": nz_fluid + 2,
    "solid_layers_z": list(solid_interface_z),   # 水平固体界面深度 [m]（给 plot 画横线）
}

if __name__ == "__main__":
    if not (os.path.isdir("src") and os.path.isfile(cuda_bin)):
        raise SystemExit("请在项目根目录下运行，并先 `make` 生成 " + cuda_bin)
    if amp_surface and boundary_type == CPML:
        raise SystemExit("起伏自由表面只在 PANDF 下有意义")

#==============================
# 网格派生量：物理域 / 求解域 / pad
#==============================
    top_frozen = 3
    bottom_span = npd + 3
    nz_fluid_full = nz_fluid + top_frozen
    nz_solid_full = nz_solid + bottom_span
    nz = nz_fluid_full + nz_solid_full - 1
    zeta0 = nz_fluid_full - 1
    b_pml = float(nz - 4 - npd)      # 包络归零处（比首阻尼行再往里 1 行）
    a_i = 3.0 if boundary_type != CPML else float(npd + 3)   # CPML 时顶部 PML 也要等厚

#==============================
# 网格与坐标
#==============================
    zeta = np.arange(nz, dtype=np.float64)
    pad_flat = npd + 3               # 外侧平整 pad = 3 冻结点 + npd PML
    outer = pad_flat
    shift = outer                    # 原 501 列物理域整体右移
    nx_full = nx
    nx_phys = nx_full
    nx = nx_phys + 2 * outer
    xive = np.arange(nx, dtype=np.float64)
    # 原物理域 x 坐标不变；物理域外只在连接层过渡，外侧 pad 严格水平。
    xphys = h * (xive - shift)
    X = np.broadcast_to(xphys[None, :], (nz, nx)).copy()
    xL = 0.0
    xR = h * (nx_phys - 1)

    # ζ 方向：在自由表面曲线和界面曲线之间做混合（不能用"各自加窗再相加"，
    # 那样自由表面的窗会压到界面行、界面的窗会压到自由表面行，两个形状互相扭曲）。
    #   u: 自由表面行 0 → 界面行 1；v: 界面行 1 → b_pml 处 0（只在界面以下起作用）
#==============================
# 贴体网格：ζ 方向混合
#==============================
    u = np.clip((zeta - a_i) / (zeta0 - a_i), 0.0, 1.0)
    u = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
    v = np.ones(nz)
    if b_pml > zeta0:
        seg = zeta > zeta0
        t = np.clip((zeta[seg] - zeta0) / (b_pml - zeta0), 0.0, 1.0)
        v[seg] = np.cos(0.5 * np.pi * t) ** 2
        v[zeta > b_pml] = 0.0

    def smootherstep(t):
        t = np.clip(t, 0.0, 1.0)
        return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3

    def base_z(xq):
        xq = np.asarray(xq, dtype=np.float64)
        d_itf = amp_interface * np.cos(2 * np.pi * xq / wavelength_interface)
        d_surf = amp_surface * np.cos(2 * np.pi * xq / wavelength_surface)
        return ((h * zeta)[:, None]
                + d_surf[None, :] * (1.0 - u)[:, None]
                + d_itf[None, :] * (u * v)[:, None])

    Z = np.broadcast_to((h * zeta)[:, None], (nz, nx)).copy()
    in_phys = (xphys >= xL) & (xphys <= xR)
    xp = xphys[in_phys]
    zp = base_z(xp)
    zL = base_z(np.array([xL]))[:, 0]
    zR = base_z(np.array([xR]))[:, 0]

    Wl = np.ones_like(xp)
    left = xp < n_taper * h
    Wl[left] = smootherstep(xp[left] / (n_taper * h))
    Wr = np.ones_like(xp)
    right = xp > xR - n_taper * h
    Wr[right] = smootherstep((xR - xp[right]) / (n_taper * h))

    Zphys = zp.copy()
    Zphys[:, left] = zL[:, None] + (zp[:, left] - zL[:, None]) * Wl[left][None, :]
    Zphys[:, right] = zR[:, None] + (zp[:, right] - zR[:, None]) * Wr[right][None, :]
    Z[:, in_phys] = Zphys
    Z[:, xphys < xL] = zL[:, None]
    Z[:, xphys > xR] = zR[:, None]

    # ---------- 固体分层：按物理 z 判定归属，所以层界面严格水平 ----------
    if len(solid_layers) != len(solid_interface_z) + 1:
        raise SystemExit("solid_layers 的个数必须 = len(solid_interface_z) + 1")
    if calc_type != FANDS and len(solid_layers) > 1:
        raise SystemExit("固体分层目前只支持 FANDS（流固耦合）模型")
#==============================
# 介质参数：固体分层
#==============================
    z_if = np.sort(np.asarray(solid_interface_z, dtype=np.float64))
    z_solid = Z[zeta0:, :]                       # 固体每个单元的物理 z
    z_top_min = float(Z[zeta0, :].min())         # 流固界面最浅/最深（决定会不会尖灭）
    z_top_max = float(Z[zeta0, :].max())
    if z_if.size:
        if np.any(np.diff(z_if) <= 0.0):
            raise SystemExit("solid_interface_z 必须严格递增")
        if z_if[0] <= z_top_min:
            raise SystemExit(
                f"第一个固体界面 {z_if[0]:.1f} m 比流固界面最浅处 {z_top_min:.1f} m 还浅，"
                "第一层会没有单元")
        if z_if[-1] >= float(Z[nz - 1].max()):
            raise SystemExit(f"最后一个固体界面 {z_if[-1]:.1f} m 已到模型底部以下")
        pinch = z_if[(z_if > z_top_min) & (z_if < z_top_max)]
        if pinch.size:
            print(f"[model] 注意: 界面 {np.round(pinch, 1).tolist()} m 比流固界面最深处 "
                  f"{z_top_max:.1f} m 浅，对应层在深的一侧会尖灭（被起伏界面截掉）")
    layer_of = np.searchsorted(z_if, z_solid, side="right")     # 0 = 最上面一层
    vp_l = np.asarray([p[0] for p in solid_layers], dtype=np.float64)[layer_of]
    vs_l = np.asarray([p[1] for p in solid_layers], dtype=np.float64)[layer_of]
    rho_l = np.asarray([p[2] for p in solid_layers], dtype=np.float64)[layer_of]
    mu_s = rho_l * vs_l ** 2
    lam_s = rho_l * (vp_l ** 2 - 2.0 * vs_l ** 2)

    dz_dxi = np.gradient(Z, axis=1, edge_order=2)
    dz_dze = np.gradient(Z, axis=0, edge_order=2)

    # 逆度规：x = hξ ⇒ ξ,z = 0；ζ,x = -z,ξ/J ≠ 0（别写反）
    J = h * dz_dze
    xi_x = np.full((nz, nx), 1.0 / h)
    xi_z = np.zeros((nz, nx))
    ze_x = -dz_dxi / J
    ze_z = h / J

    K_f = rho_fluid * vp_fluid ** 2

#==============================
# 写模型二进制（solid / fluid / jacob）
#==============================
    model_dir = os.path.join("models", case)
    for sub in (["solid", "fluid", "jacob"] if calc_type == FANDS else ["solid", "jacob"]):
        os.makedirs(os.path.join(model_dir, sub), exist_ok=True)
    for name, arr in [("C11", lam_s + 2 * mu_s), ("C13", lam_s),
                      ("C33", lam_s + 2 * mu_s), ("C55", mu_s),
                      ("rho", rho_l)]:
        arr.astype("<f4").tofile(
            os.path.join(model_dir, "solid", name + ".bin"))
    if calc_type == FANDS:
        np.full((nz_fluid_full, nx), K_f, dtype="<f4").tofile(
            os.path.join(model_dir, "fluid", "K.bin"))
        np.full((nz_fluid_full, nx), rho_fluid, dtype="<f4").tofile(
            os.path.join(model_dir, "fluid", "rho.bin"))
    for name, arr in [("J", J), ("xi_x", xi_x), ("xi_z", xi_z),
                      ("ze_x", ze_x), ("ze_z", ze_z), ("x", X), ("z", Z)]:
        arr.astype("<f4").tofile(os.path.join(model_dir, "jacob", name + ".bin"))

    # 式(41)：C 取最短波长对应的波速（流体/固体取较小者）
    vs_min = min(p[1] for p in solid_layers)
    c_min = min(vs_min, vp_fluid) if calc_type == FANDS else vs_min
    beta0 = c_min / (0.5 * ppw0 * h * fpeak)
    thickness = (npd - 1) * h
    # PML 阻尼 d0 按 Vp_max 标定：这里如果给大了，PML 会过阻尼，
    # 前几秒的反射和 4 s 以后的晚时伪场都会变大（实测 3800 → 905.2 1/s 比真值大 1.8 倍）。
    vp_layer_max = max(p[0] for p in solid_layers)
    if vp_max is None:
        vp_max = max([vp_layer_max] + ([vp_fluid] if calc_type == FANDS else []))
        print(f"[model] vp_max 自动取 {vp_max:.0f} m/s（各层与流体最大 Vp）")
    elif vp_max < vp_layer_max:
        print(f"[model] 提醒: vp_max = {vp_max:.0f} m/s 比最深层 Vp "
              f"{vp_layer_max:.0f} m/s 小，PML 阻尼建议按最快层取")
    elif vp_max > vp_layer_max * 1.05:
        print(f"[model] 提醒: vp_max = {vp_max:.0f} m/s 比实际最大 Vp "
              f"{vp_layer_max:.0f} m/s 大，PML 会过阻尼（反射/晚时伪场会变大）")

    grid = {
        "calc_type": calc_type, "boundary_type": boundary_type,
        "src_pos": {"posx": [x + shift for x in src_pos_x],
                    "posz": [z + top_frozen for z in src_pos_z]},
        "solid": {"nx": nx, "nz": nz_solid_full,
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
        grid["fluid"] = {"nx": nx, "nz": nz_fluid_full,
                         "K": os.path.join(model_dir, "fluid", "K.bin"),
                         "rho": os.path.join(model_dir, "fluid", "rho.bin")}

#==============================
# 写 JSON 与 pad.txt
#==============================
    json_path = os.path.join(model_dir, case + ".json")
    with open(json_path, "w") as f:
        json.dump({"base": {"fpeak": fpeak, "dt": dt, "nt": nt, "snapshot": snapshot},
                   "grid": grid,
                   "PML": {"npd": npd, "thickness": thickness, "NPOW": NPOW,
                           "Vp_max": vp_max, "Rc": Rc, "beta0": round(beta0, 4)}},
                  f, indent=2)
        f.write("\n")
    with open(os.path.join(model_dir, "pad.txt"), "w") as f:
        f.write(f"nx_orig {nx_phys}\n")
        f.write(f"shift {shift}\n")
        f.write(f"n_taper {n_taper}\n")
        f.write(f"pad_flat {pad_flat}\n")
        f.write(f"outer {outer}\n")
        f.write(f"npd {npd}\n")
        f.write(f"nz_fluid_phys {nz_fluid}\n")
        f.write(f"nz_solid_phys {nz_solid}\n")
        f.write(f"nz_fluid_full {nz_fluid_full}\n")
        f.write(f"nz_solid_full {nz_solid_full}\n")
        f.write(f"src_z_offset {top_frozen}\n")
        f.write(f"phys0 {pad_flat}\n")
        f.write(f"phys1 {nx - pad_flat - 1}\n")
#==============================
# 自检输出
#==============================
    print(f"[model] physical: {nx_phys} x {(nz_fluid + nz_solid - 1)} "
          f"(fluid {nz_fluid} + solid {nz_solid}), h = {h} m")
    print(f"[model] solver grid: {nx} x {nz} "
          f"(fluid {nz_fluid_full} + solid {nz_solid_full}), h = {h} m")
    print(f"[model] physical columns {pad_flat}..{nx - pad_flat - 1}; "
          f"edge taper = {n_taper} cells; flat pad = {pad_flat} each side")
    print(f"[model] 起伏: 界面 A={amp_interface} m λ={wavelength_interface} m, "
          f"自由表面 A={amp_surface} m λ={wavelength_surface} m")
    print(f"[model] 解耦检查: u(自由表面行 {int(a_i)}) = {u[int(a_i)]:.3f}, "
          f"u(界面行 {zeta0}) = {u[zeta0]:.3f}, v(界面行) = {v[zeta0]:.3f}"
          "  (前两项应为 0/1，第三项应为 1)")
    print(f"[model] beta0 = {beta0:.4f} (式(41), C = {c_min} m/s), npd = {npd}")
    print(f"[model] 固体分层: {len(solid_layers)} 层, 水平界面 z = "
          f"{np.round(z_if, 1).tolist()} m (流固界面 {z_top_min:.1f}~{z_top_max:.1f} m)")
    for k, (vpk, vsk, rhok) in enumerate(solid_layers):
        cnt = (layer_of == k).sum(axis=0)
        print(f"         第{k + 1}层 vp={vpk:.0f} vs={vsk:.0f} rho={rhok:.0f}"
              f"  每列 {cnt.min()}~{cnt.max()} 单元 ({cnt.min() * h:.0f}~{cnt.max() * h:.0f} m)")
    # PML 带宽统一口径：从"PML 带内边界"到最外层的墙，包含 冻结3 + 阻尼(npd-3) + 过渡3 格
    bot = np.abs(Z[nz - 3 - npd] - Z[nz - 1])[2:-2]      # 底：逐列
    side = np.abs(X[:, npd + 2] - X[:, 0])               # 侧：逐行
    print(f"[model] PML 带宽（冻结3+阻尼{npd - 3}+过渡3 = {npd + 3} 格）: "
          f"底部 {bot.min():.4f}~{bot.max():.4f} m, 侧向 {side.min():.4f}~{side.max():.4f} m"
          f"（理想 {(npd + 2) * h:.1f} m；纯阻尼段 {(npd - 3) * h:.1f} m）")
    print(f"[model] json   : {json_path}")

#==============================
# 模型预览（可选）
#==============================
    if show_model and plt is not None:
        # 背景：Vp 分布（流体行用流体 Vp，固体按分层），叠加几何
        vp_all = np.asarray([p[0] for p in solid_layers], dtype=np.float64)
        vp_map = np.empty((nz, nx))
        if calc_type == FANDS:
            vp_map[:nz_fluid_full] = vp_fluid
            vp_map[nz_fluid_full - 1:] = vp_all[layer_of]
        else:
            vp_map[:] = vp_all[0]
        all_vp = np.concatenate([vp_all, [vp_fluid] if calc_type == FANDS else []])
        vmin = all_vp.min() * 0.9
        vmax = all_vp.max() * 1.1
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.pcolormesh(X, Z, vp_map, cmap="rainbow", vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        fig.colorbar(im, ax=ax, label="Vp [m/s]")
        step = max(1, nx // 50)             # 网格线太密就抽稀
        ax.plot(X[::step, ::step].T, Z[::step, ::step].T, "-", color="0.85", lw=0.3)
        ax.plot(X[::step, ::step], Z[::step, ::step], "-", color="0.85", lw=0.3)
        # ax.plot(X[3], Z[3], "b-", lw=1.3, label="free surface (ζ=3)")
        for k, zk in enumerate(z_if):
            ax.axhline(zk, color="black", lw=1.1, ls="--",
                       label="solid layer interface" if k == 0 else None)
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
#==============================
# 运行求解器
#==============================
    out_root = os.path.join("output", case)
    subprocess.run([cuda_bin, json_path, out_root], check=True)
    print(f"[model] snapshots: {out_root}/<炮号>/snapshot/  记录: .../record/")
