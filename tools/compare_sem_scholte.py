"""SDM/SPECFEM 对比：ortho_classic_scholte(FDM) ↔ flat_freesurface_scholte(SEM)。

两个算例的绝对时间都是 0 起、1 ms 采样，所以**逐样点对齐（shift=0）**。
直接 `python tools/compare_sem_scholte.py`（在项目根目录下）即可。
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.family": "serif",
                            "font.serif": ["Times New Roman"],
                            "font.size": 12, "axes.unicode_minus": False})

#==============================
# 参数
#==============================
CASE = "ortho_classic_scholte"
SHOT = 0
SEM_DIR = "/home/arcy0/software/specfem_runs/flat_freesurface_scholte/OUTPUT_FILES"
REC_X = [30.0, 180.0, 330.0, 480.0, 630.0, 780.0]     # FDM 坐标；SEM = +75 m
WINDOWS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
FIG_DIR = "figures"
# ---------------------------------------------------------------------


#==============================
# 读数据（FDM 记录 / SEM 地震道）
#==============================
def load():
    cfg_path = os.path.join("models", CASE, CASE + ".json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    nx = cfg["grid"]["solid"]["nx"]
    h = float(np.sqrt(np.fromfile(os.path.join("models", CASE, "jacob", "J.bin"),
                                  dtype="<f4", count=1)[0]))
    rec_dir = os.path.join("output", CASE, str(SHOT), "record")
    vz = np.fromfile(os.path.join(rec_dir, "solid_vz.bin"), dtype="<f4")
    n = vz.size // nx
    vz = vz.reshape(n, nx)
    sem = {}
    for i in range(1, 7):
        a = np.loadtxt(os.path.join(SEM_DIR, "AA.S%04d.FXZ.semv" % i))
        sem[i] = a[:, 1][:n]
    return cfg, h, vz, sem, n


#==============================
# 对比与出图
#==============================
def main():
    cfg, h, vz, sem, n = load()
    dt = float(cfg["base"]["dt"])
    gap = max(1, int(round(1e-3 / dt)))
    t = np.arange(n) * gap * dt
    # FDM 物理域 x = (col - pad) * h；pad 从 pad.txt 读
    pad = 33
    with open(os.path.join("models", CASE, "pad.txt")) as f:
        for line in f:
            if line.startswith("pad_flat"):
                pad = int(line.split()[1])
    cols = [int(round(x / h)) + pad for x in REC_X]

    def ncc(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))

    print("逐样点对齐（绝对时间都从 0 起）。窗口 NCC / 幅比 / 残差：")
    rows = []
    for k, x in enumerate(REC_X):
        f = np.array(vz[:, cols[k]], dtype=np.float64)
        s = sem[k + 1]
        g = float(np.dot(f, s) / np.dot(s, s))
        r = float(np.linalg.norm(f - g * s) / np.linalg.norm(f))
        wl = []
        for w0, w1 in WINDOWS:
            m = (t >= w0) & (t < w1)
            wl.append(ncc(f[m], s[m]))
        rows.append((x, f, s, g, r, wl))
        print("x=%4.0f  全段 NCC %.4f  幅比 %.3e  残差 %.4f | " % (x, ncc(f, s), g, r)
              + "  ".join("%.1f-%.1fs %.3f" % (w0, w1, c)
                          for (w0, w1), c in zip(WINDOWS, wl)))

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10.5), sharex=True)
    for ax, (x, f, s, g, r, wl) in zip(axes.ravel(), rows):
        ax.plot(t, f / np.abs(f).max(), "r--", lw=1.0, label="FDM sigma=0.55")
        ax.plot(t, s / np.abs(s).max(), "k-", lw=1.2, label="SEM")
        ax.set_title("x = %.0f m   NCC %.4f   residual %.1f%%" % (x, ncc(f, s), r * 100),
                     fontsize=11)
        ax.set_ylabel("norm. vz")
        ax.grid(alpha=0.3)
        if ax is axes.ravel()[0]:
            ax.legend(fontsize=9)
    for ax in axes.ravel():
        ax.set_xlabel("t (s)")
    fig.suptitle("2 s comparison, 1 ms sampling, aligned at absolute time 0",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG_DIR, CASE, "sem_vs_fdm_2s.png"), dpi=150)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10.5))
    for ax, (x, f, s, g, r, wl) in zip(axes.ravel(), rows):
        # 每个接收点各取自己首个波包前后 ±50/+150 ms（都换算到 FDM 的量纲）
        ipk = int(np.abs(f[:600]).argmax())
        m = (t >= t[ipk] - 0.05) & (t <= t[ipk] + 0.15)
        ax.plot(t[m], f[m], "r--", lw=1.2, label="FDM sigma=0.55")
        ax.plot(t[m], g * s[m], "k-", lw=1.5, label="SEM (scaled)")
        ax.set_title("x = %.0f m   (t = %.2f s)" % (x, t[ipk]), fontsize=11)
        ax.set_ylabel("vz")
        ax.grid(alpha=0.3)
        if ax is axes.ravel()[0]:
            ax.legend(fontsize=9)
    for ax in axes.ravel():
        ax.set_xlabel("t (s)")
    fig.suptitle("zoom: first arrival wavelet at each receiver (SEM scaled by the "
                 "least-squares gain)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG_DIR, CASE, "sem_vs_fdm_zoom.png"), dpi=150)
    print("saved", os.path.join(FIG_DIR, CASE))


if __name__ == "__main__":
    main()
