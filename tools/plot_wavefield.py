"""波场快照出图：把 fluid/solid 的 vz 拼起来，每个时间点出一张图。

在**项目根目录**下直接运行（不需要任何命令行参数）:
    python3 plot_wavefield.py

读:  models/<case>/<case>.json、models/<case>/jacob/x.bin, z.bin
写:  figures/<case>/shot<炮号>_vz_<步号>.png
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

# ============================ 出图参数 ============================
case = "iso_FANDS_PANDF_wavy_if"
# case = "iso_SOLID_PANDF_wavy_if"
shot = 0                            # 第几炮（snapshot 下的数字文件夹）
crop_pml = True                      # True: 只画物理内区（裁掉 PML 与外侧冻结带）
clip_percentile = 98.5              # 色标分位裁剪（整个序列定一次，所有帧共用）
vlim_fixed = None                  # 固定色标范围 ±vlim_fixed；None = 自动（用上面的分位值）
cmap = "seismic"
interface_color = "black"
dpi = 90
# =================================================================

if __name__ == "__main__":
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

    # 界面几何直接来自模型脚本里的 interface 字典，不写进 solver JSON。
    interface_rows = []
    for value in INTERFACES.get(case, {}).values():
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
    frames = sorted(glob.glob(os.path.join(bin_dir, "solid", "vz", "vz_*.bin")))  # 二进制
    if not frames:
        raise SystemExit(f'没有找到快照: {bin_dir}/solid/vz （先跑模型脚本）')

    stack, steps = [], []
    for path in frames:
        it = int(os.path.basename(path)[3:-4])
        vs = np.fromfile(path, dtype="<f4").reshape(nzs, nx)
        if fands:
            vf = np.fromfile(os.path.join(bin_dir, "fluid", "vz", f"vz_{it:06d}.bin"),
                             dtype="<f4").reshape(nzf, nx)
            stack.append(np.vstack([vf, vs[1:]]))     # 界面行不重复
        else:
            stack.append(vs)
        steps.append(it)

    # 颜色条固定：每帧都用同一个 ±vlim
    #   vlim_fixed 给了值  -> 所有图都固定在这个值（超出 ±vlim 的部分按色标饱和显示）
    #   vlim_fixed = None -> 用整个序列的最大分位值定一次
    if vlim_fixed is not None:
        vlim = float(vlim_fixed)
        print(f"[plot] 固定色标 ±{vlim:.3g}")
    else:
        vlim = max(float(np.percentile(np.abs(v), clip_percentile)) for v in stack) or 1.0
        print(f"[plot] 自动色标 ±{vlim:.3g}（{clip_percentile} 分位）")

    def plot_one(k):
        v = stack[k][i0:i1, j0:j1]
        fig = Figure(figsize=(6.5, 6.5))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        # 正交网格用 imshow+bilinear：比 pcolormesh(gouraud) 快 ~5 倍且放大看是光滑的
        # （nearest 会显出格状纹理）；曲面网格只能用 pcolormesh(gouraud) 才对。
        xr, zr = X[i0:i1, j0:j1], Z[i0:i1, j0:j1]
        cartesian = (np.allclose(xr, xr[0][None, :]) and
                     np.allclose(zr, zr[:, 0][:, None]))
        if cartesian:
            im = ax.imshow(v, cmap=cmap, vmin=-vlim, vmax=vlim,
                           extent=[xr[0, 0], xr[0, -1], zr[-1, 0], zr[0, 0]],
                           interpolation="bilinear", aspect="equal",
                           rasterized=True)
        else:
            im = ax.pcolormesh(xr, zr, v, cmap=cmap, vmin=-vlim, vmax=vlim,
                               shading="gouraud", rasterized=True)
        for iz in interface_rows:
            if i0 <= iz < i1:
                ax.plot(X[iz, j0:j1], Z[iz, j0:j1],
                        color=interface_color, lw=1.2)
        ax.set_aspect("equal")          # 坐标轴框按真实长宽比
        if not cartesian:
            ax.invert_yaxis()           # imshow 的 extent 已经给了 z 向下
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(f"shot {shot}, t = {steps[k] * dt * 1000:.1f} ms, "
                     f"max|vz| = {np.abs(v).max():.2e}", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8, label="vz")
        fig.tight_layout()
        out = os.path.join(fig_dir, f"shot{shot}_vz_{steps[k]:06d}.png")
        fig.savefig(out, dpi=dpi)
        return out

    nthread = max(1, (os.cpu_count() or 4) - 2)     # 总线程数 - 2
    with ThreadPoolExecutor(max_workers=nthread) as pool:   # 线程池建一次，复用
        for out in pool.map(plot_one, range(len(stack))):
            print(f"[plot] {out}")
    print(f"[plot] {len(stack)} figures -> {fig_dir}  ({nthread} threads)")
