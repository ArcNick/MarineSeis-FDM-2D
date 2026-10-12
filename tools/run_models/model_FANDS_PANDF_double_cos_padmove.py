"""流固界面 = 一正一负两个余弦包（关于中心对称），自由表面 = 可控振幅/周期的余弦。

在**项目根目录**下直接运行（不需要命令行参数）:
    python3 tools/run_models/model_FANDS_PANDF_double_cos_padmove.py

几何（矩形物理域，只有 ζ=const 的网格线在物理空间里起伏）：
    x(ξ,ζ) = h ξ
    z(ξ,ζ) = h ζ + A_i g(x) w_i(ζ) + A_f cos(2πx/λ_f) w_f(ζ)
    g(x)   = −B(x−x_c+s/2) + B(x−x_c−s/2)      左包向上（水浅）、右包向下（水深）
    B(t)   = ½[1+cos(2πt/λ_i)]，|t|≤λ_i/2，其余 0（升余弦包：两端值和斜率都是 0）
两个包关于物理域中心 x_c 奇对称，包外水平，所以物理域两端斜率天然为 0。

自由表面的 λ_f 会自动取整成"物理域宽度 = 半波长的整数倍"（脚本打印实际用的值），
两端因此是余弦的极值点（斜率 0），能和外侧平整 pad（npd+3 列冻结+PML）光滑相接。

ζ 方向不用"各自加窗再相加"，而是在两条边界曲线之间做混合：权 u 从自由表面行的 0
走到界面行的 1，界面以下再乘一个衰减 v（界面行 1 → b_pml 处 0）。
因此 ζ=自由表面行 上就是纯自由表面余弦、ζ=界面行 上就是纯双余弦包，两个界面互不干扰；
两者之间的水层是两条曲线的平滑过渡，固体内部只有界面形状并向外衰减。
脚本末尾会把"逐列/逐行 PML 厚度"打印出来自检。
"""
import json
import os
import subprocess
import sys
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
case = "FANDS_PANDF_double_cos_padmove"

DIRI, CPML, PANDF = 0, 1, 2         # boundary_type
SOLID, FANDS = 0, 1                 # calc_type

calc_type = FANDS
boundary_type = PANDF

fpeak = 30.0                        # 震源主频 [Hz]
dt = 1e-4                           # 时间步长 [s]
nt = 20000                          # 总步数（1 s）
snapshot = 200                      # 每多少步存一帧快照

nx = 751                            # 物理域列数
nz_fluid = 251                      # 物理域行数（不含顶部 3 行冻结/缓冲）
nz_solid = 301                      # 物理域行数（不含底部 npd+3）
h = 2.0                             # 网格间距 [m]
src_pos_x = [nx // 2]               # 放在两个包之间的水平段（正中心）
src_pos_z = [5]                    # 物理域内索引（自由表面算第 0 行）

amp_interface = 100.0                # 界面起伏幅值 [m]（左包向上、右包向下；取负号交换）
wavelength_interface = 400.0        # 单个余弦包的宽度 [m]
sep_interface = 600.0               # 两个包中心的间距 [m]（中间水平段 = sep − wavelength）

amp_surface = 4.0                  # 自由表面起伏幅值 [m]
wavelength_surface = 200.0          # 自由表面波长 [m]（自动取整到整除值，见打印）

rho_e = 2000.0                      # 固体
vp = 3200.0
vs = 1800.0
rho_fluid = 1000.0                  # 流体
vp_fluid = 1500.0

npd = 30                            # PML 格数（含外侧 3 行/列冻结带）
NPOW = 2                            # 论文：p_d = 2（师兄 src2d 同）
vp_max = 3200.0
Rc = 1e-3
ppw0 = 8                            # 论文式(41) 的最少每波长点数

show_model = True                   # 生成模型后画一张几何+物性预览
cuda_bin = "./bin/main_debug"       # 必须在项目根目录下运行

#==============================
# 绘图接口（供 plot_wavefield.py 用）
#==============================
# 供 plot_wavefield.py import：界面名 -> 全局 ζ 行号
interface = {
    "free_surface": 3,
    "fluid_solid": nz_fluid + 2,
}

#==============================
# 工具函数
#==============================
def bump(x, center, width):
    """升余弦包：中心处 +1，|x−center| ≥ width/2 处为 0，两端值和斜率都为 0。"""
    u = (np.asarray(x, dtype=np.float64) - center) / width
    return np.where(np.abs(u) <= 0.5, 0.5 * (1.0 + np.cos(2.0 * np.pi * u)), 0.0)

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
# 物理坐标（水平方向 + 外侧 pad）
#==============================
    # ---------- 水平方向：物理域 + 外侧平整 pad ----------
    pad_flat = npd + 3               # 外侧平整 pad = 3 冻结点 + npd PML
    outer = pad_flat
    shift = outer
    nx_phys = nx
    nx = nx_phys + 2 * outer
    xive = np.arange(nx, dtype=np.float64)
    xphys = h * (xive - shift)       # 物理域 x = 0 … h(nx_phys−1)，两侧 pad 为负/超出
    X = np.broadcast_to(xphys[None, :], (nz, nx)).copy()
    xL, xR = 0.0, h * (nx_phys - 1)
    x_c = 0.5 * (xL + xR)

#==============================
# 界面形状
#==============================
    # ---------- 界面形状：一正一负两个余弦包 ----------
    half_span = 0.5 * sep_interface + 0.5 * wavelength_interface
    if half_span > x_c:
        raise SystemExit(
            f"两个余弦包放不下：需要 |x−x_c| ≤ {x_c:.1f} m，"
            f"当前 sep/2+λ/2 = {half_span:.1f} m"
        )
    if amp_interface:
        def iface_shape(x):
            return (-bump(x, x_c - 0.5 * sep_interface, wavelength_interface)
                    + bump(x, x_c + 0.5 * sep_interface, wavelength_interface))
    else:
        def iface_shape(x):
            return np.zeros_like(np.asarray(x, dtype=np.float64))

#==============================
# 自由表面形状
#==============================
    # ---------- 自由表面：基本余弦，波长取整到"宽度 = 半波长整数倍" ----------
    n_half = max(1, int(round(2.0 * (xR - xL) / wavelength_surface)))
    lam_surf = 2.0 * (xR - xL) / n_half
    if abs(lam_surf - wavelength_surface) > 1e-9:
        print(f"[model] 自由表面波长取整: {wavelength_surface} → {lam_surf:.3f} m "
              f"(={n_half} 个半波长, 保证两端斜率为 0)")

    def surface_shape(x):
        return np.cos(2.0 * np.pi * np.asarray(x, dtype=np.float64) / lam_surf)

#==============================
# 贴体网格：ζ 方向混合
#==============================
    # ---------- ζ 方向：两条边界曲线之间做混合，而不是各自加窗后相加 ----------
    # u: 0（自由表面行）→ 1（界面行），smootherstep，两端斜率为 0
    # v: 1（界面行）→ 0（b_pml），让界面形状往固体里平滑衰减
    # 这样 ζ=自由表面行 上就是纯自由表面形状，ζ=界面行 上就是纯界面形状，互不干扰。
    zeta = np.arange(nz, dtype=np.float64)
    u = np.clip((zeta - a_i) / (zeta0 - a_i), 0.0, 1.0)
    u = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
    v = np.ones(nz)
    if b_pml > zeta0:
        seg = zeta > zeta0
        t = np.clip((zeta[seg] - zeta0) / (b_pml - zeta0), 0.0, 1.0)
        v[seg] = np.cos(0.5 * np.pi * t) ** 2
        v[zeta > b_pml] = 0.0

    def base_z(xq):
        xq = np.asarray(xq, dtype=np.float64)
        z_surf = amp_surface * surface_shape(xq)
        z_itf = amp_interface * iface_shape(xq)
        return ((h * zeta)[:, None]
                + z_surf[None, :] * (1.0 - u)[:, None]
                + z_itf[None, :] * (u * v)[:, None])

    in_phys = (xphys >= xL) & (xphys <= xR)
    xp = xphys[in_phys]
    zp = base_z(xp)
    zL = base_z(np.array([xL]))[:, 0]
    zR = base_z(np.array([xR]))[:, 0]

    Z = np.broadcast_to((h * zeta)[:, None], (nz, nx)).copy()
    Z[:, in_phys] = zp
    Z[:, xphys < xL] = zL[:, None]      # 外侧严格水平
    Z[:, xphys > xR] = zR[:, None]

    dz_dxi = np.gradient(Z, axis=1, edge_order=2)
    dz_dze = np.gradient(Z, axis=0, edge_order=2)

    # 逆度规：x = hξ ⇒ ξ,z = 0；ζ,x = −z,ξ/J ≠ 0（别写反）
    J = h * dz_dze
    xi_x = np.full((nz, nx), 1.0 / h)
    xi_z = np.zeros((nz, nx))
    ze_x = -dz_dxi / J
    ze_z = h / J

#==============================
# 介质参数
#==============================
    mu = rho_e * vs ** 2
    lam = rho_e * (vp ** 2 - 2 * vs ** 2)
    K_f = rho_fluid * vp_fluid ** 2

#==============================
# 写模型二进制（solid / fluid / jacob）
#==============================
    model_dir = os.path.join("models", case)
    for sub in (["solid", "fluid", "jacob"] if calc_type == FANDS else ["solid", "jacob"]):
        os.makedirs(os.path.join(model_dir, sub), exist_ok=True)
    for name, value, n in [("C11", lam + 2 * mu, nz_solid_full),
                           ("C13", lam, nz_solid_full),
                           ("C33", lam + 2 * mu, nz_solid_full),
                           ("C55", mu, nz_solid_full)]:
        np.full((n, nx), value, dtype="<f4").tofile(
            os.path.join(model_dir, "solid", name + ".bin")
        )

    np.full((nz_solid_full, nx), rho_e, dtype="<f4").tofile(
        os.path.join(model_dir, "solid", "rho.bin")
    )
    if calc_type == FANDS:
        np.full((nz_fluid_full, nx), K_f, dtype="<f4").tofile(
            os.path.join(model_dir, "fluid", "K.bin")
        )
        np.full((nz_fluid_full, nx), rho_fluid, dtype="<f4").tofile(
            os.path.join(model_dir, "fluid", "rho.bin")
        )

    for name, arr in [("J", J), ("xi_x", xi_x), ("xi_z", xi_z),
                      ("ze_x", ze_x), ("ze_z", ze_z), ("x", X), ("z", Z)]:
        arr.astype("<f4").tofile(os.path.join(model_dir, "jacob", name + ".bin"))

    # 式(41)：C 取最短波长对应的波速（流体/固体取较小者）
    c_min = min(vs, vp_fluid) if calc_type == FANDS else vs
    beta0 = c_min / (0.5 * ppw0 * h * fpeak)
    thickness = (npd - 1) * h

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
        grid["fluid"] = {
            "nx": nx, "nz": nz_fluid_full,
            "K": os.path.join(model_dir, "fluid", "K.bin"),
            "rho": os.path.join(model_dir, "fluid", "rho.bin")
        }

#==============================
# 写 JSON 与 pad.txt
#==============================
    json_path = os.path.join(model_dir, case + ".json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "base": { "fpeak": fpeak, "dt": dt, "nt": nt, "snapshot": snapshot },
                "grid": grid,
                "PML": {
                    "npd": npd, 
                    "thickness": thickness, 
                    "NPOW": NPOW,
                    "Vp_max": vp_max, 
                    "Rc": Rc, 
                    "beta0": round(beta0, 4) 
                }
            },
            f, indent=2
        )
        f.write("\n")
    with open(os.path.join(model_dir, "pad.txt"), "w") as f:
        f.write(f"nx_phys {nx_phys}\n")
        f.write(f"shift {shift}\n")
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

    # ============ 界面点坐标输出（给 SPECFEM2D 建模用） ============
    # z 的零点统一取在"自由表面所在行（ζ=3）"上，所以：
    #     z = Z(row, col) − 3h          （上面 3 行冻结带不计入，自由表面 z=0+起伏量）
    # 物理坐标：x = 0 在物理域左端，只写物理域内的列（pad 不写）。
    # 真实坐标：x = 0 在最左求解列（含 3 冻结列 + npd 列 PML），写整个求解域。
    # 固固界面：本算例只有一层固体，所以只有自由表面 + 流固界面；
    #           以后要加多层固体，在这里往 interfaces 里按行号追加即可
    #           （注意：差分是同位网格，两层介质点之间的真正界面在"半格点"上，
    #             所以固固界面的行号要么取上层最后一行、要么取下层第一行，
    #             两者相差半格 h/2；SEM 那边一个界面点就是材料分界，见下方说明）。
#==============================
# 界面点坐标输出（给 SPECFEM2D 用）
#==============================
    z_free = h * 3.0
    interfaces = [("free surface", 3), ("fluid-solid", zeta0)]

    def iface_blocks(cols):
        return [(label, X[row, cols], Z[row, cols] - z_free)
                for label, row in interfaces]

    def write_iface_dat(path, header, blks):
        with open(path, "w") as f:
            f.write(header + "\n\n")
            for k, (label, xk, zk) in enumerate(blks, 1):
                f.write(f"interface{k}: {label}\n{len(xk)}\n")
                for xi, zi in zip(xk, zk):
                    f.write(f"{xi:.4f}    {zi:.4f}\n")
                f.write("\n")
        return path

    cols_phys = np.arange(shift, shift + nx_phys)
    cols_all = np.arange(nx)
    iface_phys = write_iface_dat(
        os.path.join(model_dir, "interface_physical.dat"),
        "physical coordinates", iface_blocks(cols_phys))
    iface_real = write_iface_dat(
        os.path.join(model_dir, "interface_real.dat"),
        "real coordinates", iface_blocks(cols_all))

    # SPECFEM2D 格式：z 从模型底往上为正（自上而下递减），界面自下而上排列。
    # 这里已经按 SEM 的习惯把最底界（z=0 的平线）也写进去；分层单元数由 SEM 网格
    # 分辨率决定，需要你自己接在文件末尾。
    z_bottom_real = float(np.max(Z[nz - 1, :] - z_free))     # 真实坐标下最底一行的 z
    iface_sem = os.path.join(model_dir, "interface_real_sem.dat")
    with open(iface_sem, "w") as f:
        blks_bu = list(reversed(iface_blocks(cols_all)))     # 自下而上
        f.write("# number of interfaces\n")
        f.write(f" {len(blks_bu) + 1}\n")
        f.write("# interface number 1 (bottom of the mesh)\n")
        f.write(f" 2\n 0.0 0.0\n {h * (nx - 1):.4f} 0.0\n")
        for k, (label, xk, zk) in enumerate(blks_bu, 2):
            f.write(f"# interface number {k} ({label})\n")
            f.write(f" {len(xk)}\n")
            for xi, zi in zip(xk, zk):
                f.write(f" {xi:.4f} {z_bottom_real - zi:.4f}\n")
        f.write("# 下面是每层垂向谱元数，按 SEM 分辨率自己填（自下而上）\n")
        f.write(f"# 真实域 z 范围 0~{z_bottom_real:.1f} m（含底部 PML）；"
                f"差分格: 固相 {nz_solid_full - npd - 3} 格 + "
                f"流体 {nz_fluid_full - 3} 格, h = {h} m\n")
        f.write(f"# 物理域 z 范围 0~{h * (nz_solid + nz_fluid - 2):.1f} m"
                f"（不含底部 PML，也不含顶部 3 行冻结带）\n")
        
#==============================
# 自检输出
#==============================
    print(f"[model] physical: {nx_phys} x {(nz_fluid + nz_solid - 1)} "
          f"(fluid {nz_fluid} + solid {nz_solid}), h = {h} m")
    print(f"[model] solver grid: {nx} x {nz} "
          f"(fluid {nz_fluid_full} + solid {nz_solid_full}), h = {h} m")
    print(f"[model] 物理域 {xL:.0f}~{xR:.0f} m，中心 {x_c:.0f} m；"
          f"左包 {x_c - 0.5 * sep_interface:.0f} m 向上、"
          f"右包 {x_c + 0.5 * sep_interface:.0f} m 向下，幅值 {abs(amp_interface)} m，"
          f"包宽 {wavelength_interface} m")
    print(f"[model] 包外水平段: 左侧 {x_c - half_span:.0f} m, "
          f"中间 {sep_interface - wavelength_interface:.0f} m, "
          f"右侧 {xL + xR - x_c - half_span:.0f} m")
    print(f"[model] 自由表面: A = {amp_surface} m, λ = {lam_surf:.3f} m")
    print(f"[model] 解耦检查: u(自由表面行 {int(a_i)}) = {u[int(a_i)]:.3f}, "
          f"u(界面行 {zeta0}) = {u[zeta0]:.3f}, v(界面行) = {v[zeta0]:.3f}"
          "  (前两项应为 0/1，第三项应为 1)")
    # 自检：两条边界曲线各自的实际起伏量（理想 = ±A，互不干扰）
    itf = Z[zeta0][in_phys] - h * zeta0
    fsurf = Z[3][in_phys] - h * 3.0
    print(f"[自检] 界面行 ζ={zeta0}: {itf.min():+.2f} ~ {itf.max():+.2f} m "
          f"(应为 ∓{abs(amp_interface):.2f} m); "
          f"自由表面行 ζ=3: {fsurf.min():+.2f} ~ {fsurf.max():+.2f} m "
          f"(应为 ∓{abs(amp_surface):.2f} m)")
    print(f"[model] beta0 = {beta0:.4f} (式(41), C = {c_min} m/s), npd = {npd}")
    # PML 带宽统一口径：从"PML 带内边界"到最外层的墙，包含 冻结3 + 阻尼(npd-3) + 过渡3 格
    bot = np.abs(Z[nz - 3 - npd] - Z[nz - 1])[2:-2]      # 底：逐列
    side = np.abs(X[:, npd + 2] - X[:, 0])               # 侧：逐行
    print(f"[model] PML 带宽（冻结3+阻尼{npd - 3}+过渡3 = {npd + 3} 格）: "
          f"底部 {bot.min():.4f}~{bot.max():.4f} m, 侧向 {side.min():.4f}~{side.max():.4f} m"
          f"（理想 {(npd + 2) * h:.1f} m；纯阻尼段 {(npd - 3) * h:.1f} m）")
    print(f"[model] json   : {json_path}")
    print(f"[model] 界面坐标: {iface_phys}")
    print(f"[model]           {iface_real}")
    print(f"[model]           {iface_sem}  (SPECFEM2D 方向：z 自下往上)")

#==============================
# 模型预览（可选）
#==============================
    if show_model and plt is not None:
        # 背景：Vp 分布（流体行用流体 Vp，其余用固体 Vp），叠加几何
        vp_map = np.full((nz, nx), float(vp))
        if calc_type == FANDS:
            vp_map[:nz_fluid_full] = vp_fluid
        vmin = min(vp, vp_fluid if calc_type == FANDS else vp) * 0.9
        vmax = max(vp, vp_fluid if calc_type == FANDS else vp) * 1.1
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(X, Z, vp_map, cmap="rainbow", vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        fig.colorbar(im, ax=ax, label="Vp [m/s]")
        step = max(1, nx // 50)             # 网格线太密就抽稀
        ax.plot(X[::step, ::step].T, Z[::step, ::step].T, "-", color="0.85", lw=0.3)
        ax.plot(X[::step, ::step], Z[::step, ::step], "-", color="0.85", lw=0.3)
        ax.plot(X[3], Z[3], "-", color="blue", lw=1.3,
                label=f"free surface (ζ=3, A={amp_surface} m)")
        if calc_type == FANDS:
            ax.plot(X[zeta0], Z[zeta0], "-", color="red", lw=1.3,
                    label=f"fluid-solid interface (ζ={zeta0}, A={amp_interface} m)")
        ax.invert_yaxis()                   # z 向下为正
        ax.set_aspect("equal")
        ax.set_title(f"{case}\n{nx} x {nz} cells, h = {h} m")
        ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
        # ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(model_dir, case + "_model.png"), dpi=110)
        plt.show(block=True)
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
