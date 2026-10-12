"""波场快照和地震记录出图。

在**项目根目录**下直接运行（不需要任何命令行参数）:
    python3 plot_wavefield.py

读:  models/<case>/<case>.json、models/<case>/jacob/x.bin, z.bin
写:  figures/<case>/shot<炮号>_<分量>_<步号>.png        （快照：vz、vx 各一套）
     figures/<case>/shot<炮号>_record_<分量>.png
     （figures 每次运行先清空；多线程出图，线程数 = CPU 总线程数 - 2，线程池复用）
"""
import glob
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

try:
    from run_models import INTERFACES
except ImportError:
    from tools.run_models import INTERFACES

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],    # 衬线字体：Times New Roman
    "font.size": 13,                      # 期刊常用字号10pt，可改为11/12
    "axes.unicode_minus": False,          # 解决负号显示方块
    "mathtext.fontset": "stix",           # 数学公式STIX字体，和Times风格匹配
    "axes.linewidth": 0.8,                # 坐标轴线条粗细（论文常用）
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,              # 图例无边框（很多SCI期刊偏好）
})
#==============================
# 出图参数
#==============================
case = "FANDS_PANDF_wavy_if"
case = "SOLID_PANDF_wavy_if"

case = "FANDS_PANDF_wavy_padmove"
case = "ortho_classic_scholte"
case = "ortho_freesurface_scholte"
# case = "Zhao_scholte"
# case = "FANDS_PANDF_wavy_padmove_center"
case = "FANDS_PANDF_double_cos_padmove"
shot = 0                            # 第几炮（snapshot 下的数字文件夹）
crop_pml = True                       # True: 只画物理内区（裁掉 PML 与外侧冻结带）
clip_percentile = 98              # 色标分位裁剪（整个序列定一次，所有帧共用）
# 色标分两组：vx / vz 共用一套（同一 cmap + 同一 ±vlim，因为量级相同），
#            p 单独一套（压力比速度大好几个数量级）。
cmap = "seismic"                    # vx / vz 用
vlim_fixed = 2e-12                  # vx / vz 共用的固定色标 ±；None = 按分位自动（两者仍共用同一个值）
cmap_p = "seismic"                  # p 用
vlim_fixed_p = None                 # p 的固定色标 ±；None = 按 p 自己的分位自动定
interface_color = "black"
record_clip_percentile = 98
record_vlim_fixed = 2e-12           # 记录里 vx / vz 共用
record_vlim_fixed_p = None          # 记录里 p 单独；None = 自动
dpi = 100

#==============================
# 主程序
#==============================
if __name__ == "__main__":
#==============================
# 读配置与几何
#==============================
    model_dir = os.path.join("models", case)
    cfg_path = os.path.join(model_dir, case + ".json")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(model_dir, "params.json")   # 兼容旧模型
    with open(cfg_path) as f:
        cfg = json.load(f)

    fands = cfg["grid"]["calc_type"] == 1
    nx = cfg["grid"]["solid"]["nx"]
    nzs = cfg["grid"]["solid"]["nz"]
    nzf = cfg["grid"]["fluid"]["nz"] if fands else 0
    nz = nzf + nzs - 1 if fands else nzs
    npd = cfg["PML"]["npd"]
    dt = cfg["base"]["dt"]
    boundary_type = cfg["grid"]["boundary_type"]

#==============================
# 界面行号（来自模型脚本的 interface 字典）
#==============================
    # 界面几何直接来自模型脚本里的 interface 字典，不写进 solver JSON。
    interface_rows, layer_z = [], []
    for key, value in INTERFACES.get(case, {}).items():
        if key == "solid_layers_z":         # 水平固体界面：存物理深度，画横线
            layer_z = [float(v) for v in np.ravel(value) if v]
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            interface_rows.extend(int(v) for v in np.ravel(value))
        else:
            interface_rows.append(int(value))
    if not interface_rows:
        # 兼容还没有登记 interface 字典的旧模型
        if fands:
            interface_rows.append(nzf - 1)
        if boundary_type == 2:
            interface_rows.append(3)
    interface_rows = sorted(set(interface_rows))

#==============================
# 物理坐标与裁剪范围
#==============================
    # 物理坐标：优先读模型里的 x.bin/z.bin（曲面网格也能画对），
    # 没有就按正交均匀网格用 h = sqrt(J) 现算。
    xp = os.path.join(model_dir, "jacob", "x.bin")
    if os.path.isfile(xp):
        X = np.fromfile(xp, dtype="<f4").reshape(nz, nx)
        Z = np.fromfile(os.path.join(model_dir, "jacob", "z.bin"),
                        dtype="<f4").reshape(nz, nx)
    else:
        h = float(np.sqrt(np.fromfile(os.path.join(model_dir, "jacob", "J.bin"),
                                      dtype="<f4", count=1)[0]))
        X, Z = np.meshgrid(np.arange(nx) * h, np.arange(nz) * h)

    # 裁剪范围：外侧 3 行/列是冻结带，往里 npd 行/列是阻尼带
    if crop_pml:
        i0 = 3 if boundary_type == 2 else npd + 3   # PANDF 顶面是自由表面
        i1 = nz - 3 - npd
        j0, j1 = npd + 3, nx - npd - 3
    else:
        i0, i1, j0, j1 = 0, nz, 0, nx

    bin_dir = os.path.join("output", case, str(shot), "snapshot")
    fig_dir = os.path.join("figures", case)
    if os.path.isdir(fig_dir):
        shutil.rmtree(fig_dir)              # 目标文件夹有东西就先清空
    os.makedirs(fig_dir)
    record_dir = os.path.join("output", case, str(shot), "record")
    frames = sorted(glob.glob(os.path.join(bin_dir, "solid", "*", "*.bin")))  # 二进制
    record_files = sorted(glob.glob(os.path.join(record_dir, "*.bin")))
    if not frames and not record_files:
        raise SystemExit(f"没有找到快照或记录: {bin_dir}, {record_dir}")

#==============================
# 读快照（vz / vx / p）与分组色标
#==============================
    # 每个分量各拼一个场：vx/vz = 流体全部 + 固体（界面行不重复）；p 只有流体那几行。
    # 色标按组定：vx / vz 共用一套，p 单独一套（量级差好几个数量级）。
    raw = {}
    for comp in ("vz", "vx", "p"):
        if comp == "p":
            if not fands:
                continue
            paths = sorted(glob.glob(
                os.path.join(bin_dir, "fluid", "p", "p_*.bin")))
            if not paths:
                continue
            stack, steps = [], []
            for path in paths:
                it = int(os.path.basename(path)[len(comp) + 1:-4])
                stack.append(np.fromfile(path, dtype="<f4").reshape(nzf, nx))
                steps.append(it)
            raw[comp] = (stack, steps, 0, nzf)      # 只覆盖流体那几行
            continue
        paths = sorted(glob.glob(
            os.path.join(bin_dir, "solid", comp, f"{comp}_*.bin")))
        if not paths:
            continue
        stack, steps = [], []
        for path in paths:
            it = int(os.path.basename(path)[len(comp) + 1:-4])
            vs = np.fromfile(path, dtype="<f4").reshape(nzs, nx)
            if fands:
                vf = np.fromfile(
                    os.path.join(bin_dir, "fluid", comp, f"{comp}_{it:06d}.bin"),
                    dtype="<f4").reshape(nzf, nx)
                stack.append(np.vstack([vf, vs[1:]]))     # 界面行不重复
            else:
                stack.append(vs)
            steps.append(it)
        raw[comp] = (stack, steps, 0, nz)

    snaps = {}
    for group, members, cm, fix in (
        ("vel", ["vz", "vx"], cmap, vlim_fixed),
        ("p", ["p"], cmap_p, vlim_fixed_p),
    ):
        members = [c for c in members if c in raw]
        if not members:
            continue
        if fix is not None:
            vlim = float(fix)
            how = "固定"
        else:
            vlim = max(
                float(np.percentile(np.abs(v), clip_percentile))
                for c in members for v in raw[c][0]
            ) or 1.0
            how = f"自动（{clip_percentile} 分位）"
        print(f"[plot] {group} ({'/'.join(members)}): {how}色标 ±{vlim:.3g}, cmap={cm}")
        for c in members:
            stack, steps, r0, r1 = raw[c]
            snaps[c] = (stack, steps, vlim, cm, r0, r1)

#==============================
# 地震记录：色标分组
#==============================
    # 记录采样间隔与主程序一致：dt 上约 1 ms 一个样点
    record_gap = max(1, int(1e-3 / dt + 0.5))
    record_dt_ms = record_gap * dt * 1000.0
    record_x = X[0, j0:j1]

#==============================
# 出图函数：快照
#==============================
    def plot_one(comp, k):
        stack, steps, vlim, cm, r0, r1 = snaps[comp]
        a0, a1 = max(i0, r0), min(i1, r1)      # p 只有流体那几行
        v = stack[k][a0 - r0:a1 - r0, j0:j1]
        fig = Figure(figsize=(6.5, 6))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        # 正交网格用 imshow+bilinear：比 pcolormesh(gouraud) 快 ~5 倍且放大看是光滑的
        # （nearest 会显出格状纹理）；曲面网格只能用 pcolormesh(gouraud) 才对。
        xr, zr = X[a0:a1, j0:j1], Z[a0:a1, j0:j1]
        cartesian = (np.allclose(xr, xr[0][None, :]) and
                     np.allclose(zr, zr[:, 0][:, None]))
        if cartesian:
            im = ax.imshow(v, cmap=cm, vmin=-vlim, vmax=vlim,
                           extent=[xr[0, 0], xr[0, -1], zr[-1, 0], zr[0, 0]],
                           interpolation="bilinear", aspect="equal",
                           rasterized=True)
        else:
            im = ax.pcolormesh(xr, zr, v, cmap=cm,
                               vmin=-vlim, vmax=vlim,
                               shading="gouraud", rasterized=True)
        for iz in interface_rows:
            if a0 <= iz < a1:
                ax.plot(X[iz, j0:j1], Z[iz, j0:j1],
                        color=interface_color, lw=1.2)
        for zk in layer_z:              # 水平固体界面（物理 z = 常数）
            if zr.min() < zk < zr.max():
                ax.axhline(zk, color=interface_color, lw=1.0, ls="--")
        ax.set_aspect("equal")          # 坐标轴框按真实长宽比
        if not cartesian:
            ax.invert_yaxis()           # imshow 的 extent 已经给了 z 向下
        ax.set_xlabel("x (m)", fontsize=16)
        ax.set_ylabel("z (m)", fontsize=16)
        # ax.set_title(f"shot {shot}, t = {steps[k] * dt * 1000:.1f} ms, "
        #              f"max|{comp}| = {np.abs(v).max():.2e}", fontsize=16)
        # ax.set_title(f"t = {steps[k] * dt * 1000:.1f} ms", fontsize=16)
        fig.colorbar(im, ax=ax, shrink=0.8, label=None)
        fig.tight_layout()
        out = os.path.join(fig_dir, f"shot{shot}_{comp}_{steps[k]:06d}.png")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        return out

#==============================
# 记录色标分组
#==============================
    # 记录也分两组定色标：vx / vz 共用一套，p 单独一套
    def rec_group(path):
        label = os.path.splitext(os.path.basename(path))[0]
        return label, ("p" if label.endswith("_p") else "vel")

    rec_cfg = {}
    for group, cm, fix in (("vel", cmap, record_vlim_fixed),
                           ("p", cmap_p, record_vlim_fixed_p)):
        files = [p for p in record_files if rec_group(p)[1] == group]
        if not files:
            continue
        if fix is not None:
            vlim, how = float(fix), "固定"
        else:
            vlim = max(
                float(np.percentile(
                    np.abs(np.fromfile(p, dtype="<f4")), record_clip_percentile))
                for p in files) or 1.0
            how = f"自动（{record_clip_percentile} 分位）"
        rec_cfg[group] = (vlim, cm)
        print(f"[plot] 记录 {group}（{len(files)} 个）: {how}色标 ±{vlim:.3g}, cmap={cm}")

#==============================
# 出图函数：记录
#==============================
    def plot_record(path):
        label, group = rec_group(path)
        vlim, cm = rec_cfg[group]
        data = np.fromfile(path, dtype="<f4")
        if data.size % nx != 0:
            raise ValueError(f"{path}: 数据长度不是 nx 的整数倍")
        data = data.reshape(-1, nx)[:, j0:j1]
        t = np.arange(data.shape[0]) * record_dt_ms

        fig = Figure(figsize=(8, 8))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(
            record_x, t, data,
            cmap=cm, vmin=-vlim, vmax=vlim,
            shading="auto", rasterized=True
        )
        ax.set_xlabel("x (m)", fontsize=13)
        ax.set_ylabel("t (ms)", fontsize=13)
        ax.set_ylim(t[-1] if len(t) > 1 else record_dt_ms, 0)
        ax.set_title(
            f"shot {shot}, record {label}, dt = {record_dt_ms:.3f} ms, "
            f"max = {np.abs(data).max():.2e}",
            fontsize=13
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
        fig.tight_layout()
        out = os.path.join(fig_dir, f"shot{shot}_record_{label}.png")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        return out

#==============================
# 多线程出图（线程数 = CPU-2，线程池复用）
#==============================
    nthread = max(1, (os.cpu_count() or 4) - 2)     # 总线程数 - 2
    with ThreadPoolExecutor(max_workers=nthread) as pool:   # 线程池建一次，复用
        if record_files:
            for out in pool.map(plot_record, record_files):
                print(f"[plot] {out}")
        for comp, (stack, steps, *_) in snaps.items():
            for out in pool.map(lambda k, c=comp: plot_one(c, k), range(len(stack))):
                print(f"[plot] {out}")
    nsnap = sum(len(s[0]) for s in snaps.values())
    print(f"[plot] {nsnap} snapshots ({'+'.join(snaps) or '无'}) + "
          f"{len(record_files)} records -> {fig_dir}  ({nthread} threads)")
