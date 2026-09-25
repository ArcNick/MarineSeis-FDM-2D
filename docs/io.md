# 输入输出约定（2026-09-19）

## 1. 目录

模型（生成脚本写）:

```
models/<case>/<case>.json          求解器输入
models/<case>/solid/*.bin          C11 C13 C33 C55 rho
models/<case>/fluid/*.bin          K rho            （仅 FANDS）
models/<case>/jacob/*.bin          J xi_x xi_z ze_x ze_z (+ x z，后处理用）
```

输出（求解器写，命令行第二个参数 = `output/<case>`）:
```
output/<case>/<炮号>/snapshot/solid/vz/vz_<步号>.bin
output/<case>/<炮号>/snapshot/fluid/vz/vz_<步号>.bin   （仅 FANDS）
output/<case>/<炮号>/snapshot/fluid/p/p_<步号>.bin     （仅 FANDS，供地震记录）
output/<case>/<炮号>/record/                           （预留：地震记录）
```

图不放在 output 里，直接放仓库根的 `figures/<case>/`：
```
figures/<case>/shot<炮号>_vz_<步号>.png
```

每次输出二进制前先清空 `output/<case>/`（整棵），每次出图前先清空 `figures/<case>/`。

* 文件名里的 `<步号>` 是时间步 `it`（6 位补零），不是毫秒——`dt*snapshot < 1 ms`
  时毫秒会撞名覆盖。
* 一次运行可含多炮（JSON 里 `grid.src_pos` 的数组长度），炮号就是数组下标。

## 2. 求解器命令行（只有这两个外部参数）

```
./bin/main_debug <model.json> <out_root>          # 必须在项目根目录下运行
# 例: ./bin/main_debug models/iso_SOLID_PANDF_ortho/iso_SOLID_PANDF_ortho.json \
                      output/iso_SOLID_PANDF_ortho
```

不再使用 `MFDM_SNAP_DIR` 环境变量。

## 3. 脚本写法（Python）

* 顶部就是算例参数（一眼能看出这个脚本干什么），下面 `if __name__ == "__main__":` 干活；
  不写命令行参数解析，直接 `python3 xxx.py` 就能跑；工具函数不要堆在文件开头。
* 路径用相对路径，运行位置固定在项目根目录（脚本会检查 `src/` 与 `bin/main_debug`）。
* 已按此标准的脚本：
  * `model_iso_SOLID_PANDF_ortho.py`（纯固体 + 自由表面 + 正交网格）
  * `model_iso_FANDS_PANDF_wavy.py`（流固 + 起伏界面/起伏自由表面）
  * `plot_wavefield.py`（拼接 fluid/solid 的 vz，每个时间步一张图；色标整段固定，
    多线程出图，线程数 = CPU 总线程数 - 2，线程池复用）

## 4. 几个容易踩的约定

* **PANDF 的自由表面在第 3 行**：0/1/2 行是恒零缓冲，不是物理域。
* **npd 含外侧 3 行/列冻结带**：真正加阻尼的宽度是 `npd-3` 格；
  `thickness=(npd-1)h` 只是 `d0` 公式里的标称厚度。
* **beta0 由生成脚本按式(41) 算**：`beta0 = C/(0.5*PPW0*h*fpeak)`，C 取最短波长
  对应的波速（固体取 Vs，同时有流体时取 min(Vs, Vp_fluid)）。npd 里的 3 行冻结带
  没有积分，实际用到的 beta 最大值比 beta0 小（npd=10 时约 6 成），需要更强阻尼
  就直接把 beta0 改大。
* **J 存正雅可比** `J=det(∂(x,z)/∂(ξ,ζ))`；正交网格 `J=h²`、`ξ_x=ζ_z=1/h`、
  `ξ_z=ζ_x=0`（竖直列网格上 `ξ_z` 必须为 0，别和 `ζ_x` 写反）。
* 曲线网格（起伏地形）**必须保证 PML 物理厚度沿边界处处相同**：起伏包络要在
  PML 带内归零，`model_iso_FANDS_PANDF_wavy.py` 生成时会打印逐列/逐行厚度自检。
