"""FDM(FANDS_PANDF_double_cos_padmove) 与 SPECFEM2D(double_cos) 逐道对比。

直接 `python tools/compare_double_cos.py`（在项目根目录下执行）。

对比约定
--------
* 坐标系：FDM 用 z 向下为正（深度），SPECFEM 用 z 向上为正。
* 源极性：本算例 SEM 的 SOURCE 用 factor = -1e10，是整体符号，属于自由约定。
  两者合起来正好使 vz 不用翻号、vx 要翻号：
      vz_FDM = +vz_SEM ，  vx_FDM = -vx_SEM
  （已在 29 道整体 NCC 上验证：翻错号时平均 NCC 约 -0.95，翻对后约 +0.95。）
* 时间轴：两边都是从 t=0 起、1 ms 采样，逐采样点对应；SEM 文件里打印的
  t 只是绝对值减掉了 1.2/f0 = 40 ms，不影响对齐。
* 接收器：两边都用同一条流固界面上的 29 个点（x = 50~1450 m，间隔 50 m）。
  FDM 记录的是界面固体侧的量；SEM 的接收器要放在界面下方 0.1 m（进固体单元），
  否则正好落在流固共用的界面节点上时会被归到流体单元，切向（vx）无意义。
  实测（同网格只改接收器 z）：界面正上 vz/vx 平均 +0.957/+0.553，
  界面上方 2.5 m（流体侧）+0.688/+0.259，界面下方 0.1 m（固体侧）+0.988/+0.988。
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.family": "serif",
                            "font.serif": ["DejaVu Serif"],
                            "font.size": 12, "axes.unicode_minus": False})

#==============================
# 参数
#==============================
CASE = "FANDS_PANDF_double_cos_padmove"
SEM_DIR = "/home/arcy0/software/specfem_runs/double_cos/OUTPUT_FILES"
H = 2.0                                       # FDM 网格间距 [m]
REC_X = np.arange(50.0, 1451.0, 50.0)         # 29 道，和 SEM 的 STATIONS 一一对应
PLOT_X = [50.0, 300.0, 550.0, 800.0, 1050.0, 1300.0]
T_END = 1.2                                   # 画图时间窗 [s]
T_SKIP = 0.04                                 # 跳过源时移段，再算 NCC
FIG = "figures/double_cos_traces.png"
#==============================


#==============================
# 读数据
#==============================
def load():
    cfg = json.load(open(f"models/{CASE}/{CASE}.json"))
    nx = cfg["grid"]["solid"]["nx"]
    pad = int(open(f"models/{CASE}/pad.txt").read().split("shift")[1].split()[0])
    cols = [int(round(x / H)) + pad for x in REC_X]

    fd = {}
    for comp in ("vz", "vx"):
        a = np.fromfile(f"output/{CASE}/0/record/solid_{comp}.bin", dtype="<f4")
        fd[comp] = a.reshape(-1, nx)

    sem = {}
    for comp, ext in (("vz", "FXZ"), ("vx", "FXX")):
        sem[comp] = np.array([np.loadtxt(f"{SEM_DIR}/AA.S{k + 1:04d}.{ext}.semv")[:, 1]
                              for k in range(len(REC_X))])
    return fd, sem, cols


#==============================
# 对比 + 出图
#==============================
def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def main():
    fd, sem, cols = load()
    sem["vz"] = sem["vz"]          # z 轴翻转与源极性相消 -> 不翻
    sem["vx"] = -sem["vx"]         # x 分量只受源极性影响 -> 翻

    n0 = int(round(T_SKIP * 1e3))
    t = np.arange(fd["vz"].shape[0] - n0) * 1e-3

    # 逐道 NCC，顺便算一个全局幅值比例（最小二乘）让两条曲线同量纲
    comps = ("vz", "vx")
    F = {}
    S = {}
    val = {}
    for comp in comps:
        F[comp] = np.array([fd[comp][n0:, cols[k]] for k in range(len(REC_X))])
        S[comp] = sem[comp][:, n0:]
        k = float(np.dot(S[comp].ravel(), F[comp].ravel())
                  / np.dot(S[comp].ravel(), S[comp].ravel()))
        S[comp] = S[comp] * k
        val[comp] = [ncc(F[comp][i], S[comp][i]) for i in range(len(REC_X))]

        print(f"--- {comp}   (SEM 全局幅值比 {k:.3e}) ---")
        print(f"{'x[m]':>6} {'NCC':>8}")
        for x, v in zip(REC_X, val[comp]):
            print(f"{x:6.0f} {v:+8.4f}")
        print(f"{'mean':>6} {np.mean(val[comp]):+8.4f}   "
              f"min {min(val[comp]):+8.4f}")

    # # 曲线图：行=检波器，列=分量；先画实线(FDM)，再画虚线(SEM)不被盖住
    # fig, axes = plt.subplots(len(PLOT_X), 2, figsize=(11, 14), sharex=True)
    # for r, x in enumerate(PLOT_X):
    #     i = int(np.where(REC_X == x)[0][0])
    #     for c, comp in enumerate(comps):
    #         ax = axes[r, c]
    #         ref = np.abs(F[comp][i]).max()
    #         ax.plot(t, F[comp][i] / ref, "-", color="tab:red", lw=1.3, zorder=2,
    #                 label="FDM")
    #         ax.plot(t, S[comp][i] / ref, "--", color="k", lw=1.0, zorder=3,
    #                 label="SPECFEM2D")
    #         ax.text(0.97, 0.86, f"NCC {val[comp][i]:+.3f}", ha="right",
    #                 transform=ax.transAxes, fontsize=9)
    #         ax.grid(alpha=0.3)
    #     axes[r, 0].set_ylabel(f"x = {x:.0f} m\n{comps[0]}", fontsize=10)
    #     axes[r, 1].set_ylabel(comps[1], fontsize=10)
    # for ax in axes[-1]:
    #     ax.set_xlabel("t (s)")
    # axes[0, 0].set_title(comps[0], fontsize=12)
    # axes[0, 1].set_title(comps[1], fontsize=12)
    # axes[0, 0].legend(fontsize=9)
    # fig.suptitle("double_cos interface:  solid line : FDM, "
    #              "dashed line : SPECFEM2D", fontsize=12)
    # fig.tight_layout(rect=(0, 0, 1, 0.975))
    # fig.savefig(FIG, dpi=150)
    # print("saved", FIG)

    # ===== 新版布局：3 行 × 4 列 =====
    # 列 0-1: vx 块(6 个检波器, 行优先)
    # 列 2-3: vz 块(6 个检波器, 行优先)
    fig, axes = plt.subplots(3, 4, figsize=(22, 8), sharex=True)

    # (分量, 属于该块的列号)
    block_specs = [("vx", [0, 1]), ("vz", [2, 3])]

    for comp, col_pair in block_specs:
        for idx, x in enumerate(PLOT_X):          # idx: 0..5, 行优先
            r = idx // 2                          # 行索引 0,1,2
            c = col_pair[idx % 2]                 # 块内列索引 0,1
            ax = axes[r, c]
            i = int(np.where(REC_X == x)[0][0])

            ref = np.abs(F[comp][i]).max()
            ax.plot(t, F[comp][i] / ref, "-", color="tab:red", lw=1.2,
                    zorder=2, label="FDM")
            ax.plot(t, S[comp][i] / ref, "--", color="k", lw=0.9,
                    zorder=3, label="SPECFEM2D")
            ax.text(0.97, 0.86, f"NCC {val[comp][i]:+.3f}",
                    ha="right", transform=ax.transAxes, fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylabel(f"x = {x:.0f} m", fontsize=9)

    # 列标题
    for c, lab in enumerate(["vx", "vx", "vz", "vz"]):
        axes[0, c].set_title(lab, fontsize=12)
    axes[0, 0].legend(fontsize=8, loc="upper left")

    # x 轴标签只在最底行
    for ax in axes[-1]:
        ax.set_xlabel("t (s)")

    fig.suptitle("double_cos interface:  solid line : FDM, "
                 "dashed line : SPECFEM2D", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG, dpi=150)
    print("saved", FIG)

if __name__ == "__main__":
    main()
