# 曲线网格下 ADE CFS-MPML 与自由表面的分量表达式

本文件给出代码中实际实现的分量表达式（`include/tools_func.cuh`、`include/elements.h`、
`include/cpml_kernels.cuh`、`include/pml_fs_kernels.cuh`、`src/elements.cu`），可直接放进毕设。

## 1 记号与度量

$\xi,\zeta$ 为计算坐标（模型里 $x=h\xi$，故对 $\xi,\zeta$ 的导数即网格差分），$x,z$ 为物理坐标；场量一律取物理直角坐标分量。

$$
\xi_x=\frac{\partial\xi}{\partial x},\quad \xi_z=\frac{\partial\xi}{\partial z},\quad \zeta_x=\frac{\partial\zeta}{\partial x},\quad \zeta_z=\frac{\partial\zeta}{\partial z},\quad J=x_\xi z_\zeta-x_\zeta z_\xi .
\tag{M-1}
$$

协变基矢量 $\boldsymbol e_\xi=(x_\xi,z_\xi)$、$\boldsymbol e_\zeta=(x_\zeta,z_\zeta)$，逆变基矢量 $\boldsymbol e^\xi=\nabla\xi$、$\boldsymbol e^\zeta=\nabla\zeta$。逆雅可比关系

$$
x_\xi=J\zeta_z,\qquad x_\zeta=-J\xi_z,\qquad z_\xi=-J\zeta_x,\qquad z_\zeta=J\xi_x .
\tag{M-2}
$$

代码里存的是逆变度量 $g^{\xi\xi}=\xi_x^2+\xi_z^2$、$g^{\zeta\zeta}=\zeta_x^2+\zeta_z^2$、$g^{\xi\zeta}=\xi_x\zeta_x+\xi_z\zeta_z$；协变度量 $g_{\xi\xi}=x_\xi^2+z_\xi^2$、$g_{\zeta\zeta}=x_\zeta^2+z_\zeta^2$ 与它们的关系是

$$
g_{\zeta\zeta}=x_\zeta^2+z_\zeta^2=J^2g^{\xi\xi},\qquad g_{\xi\xi}=x_\xi^2+z_\xi^2=J^2g^{\zeta\zeta}.
\tag{M-3}
$$

**这两个因子不能混用**：辅助方向算子 $D_2$ 的分母是协变度量 $g_{\zeta\zeta},g_{\xi\xi}$，而主方向算子 $D_1$ 中出现的是逆变度量 $g^{\xi\zeta},g^{\xi\xi}$。

方向导数算子：

$$
D_1^\xi=\partial_\xi+\frac{g^{\xi\zeta}}{g^{\xi\xi}}\partial_\zeta,\qquad D_2^\xi=\frac{1}{g_{\zeta\zeta}}\partial_\zeta=\frac{1}{J^2g^{\xi\xi}}\partial_\zeta,
\tag{M-4}
$$

$$
D_1^\zeta=\partial_\zeta+\frac{g^{\xi\zeta}}{g^{\zeta\zeta}}\partial_\xi,\qquad D_2^\zeta=\frac{1}{g_{\xi\xi}}\partial_\xi=\frac{1}{J^2g^{\zeta\zeta}}\partial_\xi .
\tag{M-5}
$$

$\nabla$ 的正交分解 $\nabla=\nabla_1+\nabla_2$，其中 $\nabla_1=\boldsymbol e^\xi D_1^\xi$（主阻尼，PML 面法向）、$\nabla_2=\boldsymbol e_\zeta D_2^\xi$（辅助阻尼，平行于 PML 面）：

$$
[\nabla_1^\xi\cdot\boldsymbol\sigma]_x=\xi_xD_1^\xi\sigma_{xx}+\xi_zD_1^\xi\sigma_{xz},\qquad [\nabla_1^\xi\cdot\boldsymbol\sigma]_z=\xi_xD_1^\xi\sigma_{xz}+\xi_zD_1^\xi\sigma_{zz},
\tag{M-6}
$$

$$
[\nabla_2^\xi\cdot\boldsymbol\sigma]_x=x_\zeta D_2^\xi\sigma_{xx}+z_\zeta D_2^\xi\sigma_{xz},\qquad [\nabla_2^\xi\cdot\boldsymbol\sigma]_z=x_\zeta D_2^\xi\sigma_{xz}+z_\zeta D_2^\xi\sigma_{zz}.
\tag{M-7}
$$

$\xi\leftrightarrow\zeta$ 互换即得主阻尼轴为 $\zeta$ 时的相应表达式。

## 2 自由表面：局部应变率法（只需要一个标量记忆）

自由表面是 $\zeta=0$ 等值线，其上的局部正交基为

$$
\boldsymbol\tau=\frac{\boldsymbol e_\xi}{|\boldsymbol e_\xi|},\qquad \boldsymbol n=\frac{\boldsymbol e^\zeta}{|\boldsymbol e^\zeta|},\qquad \boldsymbol\tau\cdot\boldsymbol n=0 .
\tag{M-8}
$$

由 $\boldsymbol e_\xi\cdot\nabla=\partial_\xi$ 与 $\boldsymbol e_\xi\cdot\boldsymbol e^\zeta=0$ 得

$$
\frac{\partial}{\partial\tau}=\boldsymbol\tau\cdot\nabla=\frac{1}{|\boldsymbol e_\xi|}\partial_\xi=\frac{1}{J|\boldsymbol e^\zeta|}\partial_\xi ,
\tag{M-9}
$$

即**沿自由表面的切向导数是纯 $\xi$ 导数**，与 $\zeta$ 导数无关。因此自由表面上只有 $\varepsilon_{\tau\tau}$ 需要写 PML 方程：

$$
\frac{\partial\varepsilon_{\tau\tau}}{\partial t}=\frac{1}{|\boldsymbol e_\xi|}\left(\tau_x\frac{\partial v_x}{\partial\xi}+\tau_z\frac{\partial v_z}{\partial\xi}\right)=\frac{1}{|\boldsymbol e_\xi|}\left(\boldsymbol\tau\cdot\frac{\partial\boldsymbol v}{\partial\xi}\right),
\tag{M-10}
$$

$$
\frac{\partial\varepsilon_{\tau\tau}}{\partial t}=\frac{1}{\beta_\xi^1}\left[\frac{1}{|\boldsymbol e_\xi|}\left(\boldsymbol\tau\cdot\frac{\partial\boldsymbol v}{\partial\xi}\right)-V\right],\qquad \frac{\partial V}{\partial t}+\left(\alpha_\xi^1+\frac{d_\xi^1}{\beta_\xi^1}\right)V=\frac{d_\xi^1}{\beta_\xi^1}\cdot\frac{1}{|\boldsymbol e_\xi|}\left(\boldsymbol\tau\cdot\frac{\partial\boldsymbol v}{\partial\xi}\right).
\tag{M-11}
$$

其余两个局部应变率由自由表面条件 $\sigma_{nn}=0$、$\sigma_{n\tau}=0$ 给出（各向同性介质）：

$$
\frac{\partial\varepsilon_{nn}}{\partial t}=-\frac{C_{13}}{C_{33}}\frac{\partial\varepsilon_{\tau\tau}}{\partial t},\qquad \frac{\partial\varepsilon_{n\tau}}{\partial t}=0 .
\tag{M-12}
$$

旋回 $x$–$z$ 直角坐标（$\varepsilon_{n\tau}=0$）：

$$
\frac{\partial\varepsilon_{xx}}{\partial t}=\tau_x^2\frac{\partial\varepsilon_{\tau\tau}}{\partial t}+n_x^2\frac{\partial\varepsilon_{nn}}{\partial t},\quad \frac{\partial\varepsilon_{zz}}{\partial t}=\tau_z^2\frac{\partial\varepsilon_{\tau\tau}}{\partial t}+n_z^2\frac{\partial\varepsilon_{nn}}{\partial t},\quad \frac{\partial\varepsilon_{xz}}{\partial t}=\tau_x\tau_z\frac{\partial\varepsilon_{\tau\tau}}{\partial t}+n_xn_z\frac{\partial\varepsilon_{nn}}{\partial t},
\tag{M-13}
$$

$$
\frac{\partial\sigma_{xx}}{\partial t}=C_{11}\frac{\partial\varepsilon_{xx}}{\partial t}+C_{13}\frac{\partial\varepsilon_{zz}}{\partial t},\quad \frac{\partial\sigma_{zz}}{\partial t}=C_{13}\frac{\partial\varepsilon_{xx}}{\partial t}+C_{33}\frac{\partial\varepsilon_{zz}}{\partial t},\quad \frac{\partial\sigma_{xz}}{\partial t}=2C_{55}\frac{\partial\varepsilon_{xz}}{\partial t}.
\tag{M-14}
$$

(M-13)(M-14) 与"面内平面应力"形式等价：

$$
\frac{\partial\sigma_{ij}}{\partial t}=\frac{C^{\mathrm{eff}}}{\beta_\xi^1}\tau_i\tau_j\left[\frac{1}{|\boldsymbol e_\xi|}\left(\boldsymbol\tau\cdot\frac{\partial\boldsymbol v}{\partial\xi}\right)-V\right],\qquad C^{\mathrm{eff}}=C_{11}-\frac{C_{13}^2}{C_{33}}\ \Big(=\frac{4\mu(\lambda+\mu)}{\lambda+2\mu}\ \text{各向同性}\Big).
\tag{M-15}
$$

要点：

1. **只有一个标量记忆 $V$**：把消去关系中的未知量重标定为 $w=(s_1/s_2)\partial_\zeta\boldsymbol v$ 后，每个应变分量都成为 $1/s_1$ 乘实系数组合，辅助方向的 $s_2$ 被完全吸收，自由表面行只剩主阻尼轴的 $s_1$，其复数部分即 (M-11) 的 $V$。该行不施加辅助（平行）阻尼，也不更新 $\zeta$ 族记忆。
2. 逐列只预存 $\tau_x,\tau_z,1/|\boldsymbol e_\xi|$ 三个量（代码 `struct FS_par` 另存了 $n_x,n_z$，由 $\boldsymbol\tau$ 转 90° 得到）。
3. 平直表面（$\boldsymbol\tau=(1,0)$）时 $C^{\mathrm{eff}}/|\boldsymbol e_\xi|$ 退化为 $(C_{11}-C_{13}^2/C_{33})/h$，且 $\partial\sigma_{zz}/\partial t=\partial\sigma_{xz}/\partial t=0$。
4. 动量方程一侧不变：仍用守恒形式 + 面力通量镜像（Zhang et al. 2014 §4.1），主阻尼方向的 PML 修正照旧。
5. 流体自由表面退化为 $p\equiv0$，不涉及记忆变量。

## 3 内部点：主阻尼方向（ADE CFS-PML）

复拉伸函数与剖面：

$$
s_\xi=\beta_\xi+\frac{d_\xi}{\alpha_\xi+i\omega},\qquad \frac{1}{s_\xi}=\frac{1}{\beta_\xi}-\frac{1}{\beta_\xi}\cdot\frac{d_\xi/\beta_\xi}{\alpha_\xi+d_\xi/\beta_\xi+i\omega},
\tag{M-16}
$$

于是 $(1/s_\xi-1)X=(\beta_\xi^{-1}-1)X-\beta_\xi^{-1}E$，记忆变量 $E$ 满足

$$
\frac{\partial E}{\partial t}+\left(\alpha_\xi+\frac{d_\xi}{\beta_\xi}\right)E=\frac{d_\xi}{\beta_\xi}X .
\tag{M-17}
$$

### 3.1 动量方程

$$
\rho\frac{\partial v_x}{\partial t}=[\nabla\cdot\boldsymbol\sigma]_x+\left(\frac{1}{\beta_\xi^1}-1\right)[\nabla_1^\xi\cdot\boldsymbol\sigma]_x-\frac{1}{\beta_\xi^1}V_x^1,\qquad \frac{\partial V_x^1}{\partial t}+\left(\alpha_\xi^1+\frac{d_\xi^1}{\beta_\xi^1}\right)V_x^1=\frac{d_\xi^1}{\beta_\xi^1}[\nabla_1^\xi\cdot\boldsymbol\sigma]_x ,
\tag{M-18}
$$

$v_z$ 分量只需把 $[\nabla_1^\xi\cdot\boldsymbol\sigma]_x$ 换成 (M-6) 的 $[\nabla_1^\xi\cdot\boldsymbol\sigma]_z$。记忆变量共 4 个分量（2 个速度分量 $\times$ 2 个阻尼方向）。

### 3.2 本构方程

$$
\frac{\partial\sigma_{xx}}{\partial t}=C_{11}v_{x,x}+C_{13}v_{z,z}+\left(\frac{1}{\beta_\xi^1}-1\right)\left(C_{11}\xi_xD_1^\xi v_x+C_{13}\xi_zD_1^\xi v_z\right)-\frac{1}{\beta_\xi^1}\left(C_{11}\xi_xE_x^1+C_{13}\xi_zE_z^1\right),
\tag{M-19}
$$

$$
\frac{\partial E_x^1}{\partial t}+\left(\alpha_\xi^1+\frac{d_\xi^1}{\beta_\xi^1}\right)E_x^1=\frac{d_\xi^1}{\beta_\xi^1}D_1^\xi v_x,\qquad \frac{\partial E_z^1}{\partial t}+\left(\alpha_\xi^1+\frac{d_\xi^1}{\beta_\xi^1}\right)E_z^1=\frac{d_\xi^1}{\beta_\xi^1}D_1^\xi v_z .
\tag{M-20}
$$

$\sigma_{zz}$ 把系数换成 $(C_{13},C_{33})$，$\sigma_{xz}$ 为

$$
\frac{\partial\sigma_{xz}}{\partial t}=C_{55}(v_{x,z}+v_{z,x})+C_{55}\left(\frac{1}{\beta_\xi^1}-1\right)\left(\xi_zD_1^\xi v_x+\xi_xD_1^\xi v_z\right)-\frac{C_{55}}{\beta_\xi^1}\left(\xi_zE_x^1+\xi_xE_z^1\right).
\tag{M-21}
$$

主阻尼轴取 $\zeta$ 时，把 (M-18)–(M-21) 中 $\xi_x\to\zeta_x$、$\xi_z\to\zeta_z$、$D_1^\xi\to D_1^\zeta$ 替换。

### 3.3 声学（流体）侧

$$
\rho\frac{\partial v_x}{\partial t}=-\partial_xp+\left(\frac{1}{\beta_\xi^1}-1\right)\left(-\xi_xD_1^\xi p\right)-\frac{1}{\beta_\xi^1}V_x^1,\qquad \frac{\partial V_x^1}{\partial t}+\left(\alpha_\xi^1+\frac{d_\xi^1}{\beta_\xi^1}\right)V_x^1=\frac{d_\xi^1}{\beta_\xi^1}\left(-\xi_xD_1^\xi p\right),
\tag{M-22}
$$

$$
\frac{\partial p}{\partial t}=-K\,\nabla\cdot\boldsymbol v+K\left[-\left(\frac{1}{\beta_\xi^1}-1\right)\left(\xi_xD_1^\xi v_x+\xi_zD_1^\xi v_z\right)+\frac{1}{\beta_\xi^1}\left(\xi_xE_x^1+\xi_zE_z^1\right)\right],
\tag{M-23}
$$

$$
\frac{\partial E_x^1}{\partial t}+\left(\alpha_\xi^1+\frac{d_\xi^1}{\beta_\xi^1}\right)E_x^1=\frac{d_\xi^1}{\beta_\xi^1}D_1^\xi v_x,\qquad \frac{\partial E_z^1}{\partial t}+\left(\alpha_\xi^1+\frac{d_\xi^1}{\beta_\xi^1}\right)E_z^1=\frac{d_\xi^1}{\beta_\xi^1}D_1^\xi v_z .
\tag{M-24}
$$

## 4 内部点：辅助（平行）方向

辅助方向取 $\beta_2=1$（论文式 (32)），拉伸函数与频移为

$$
s_2=1+\frac{d_2}{\alpha_2+i\omega},\qquad
\frac{1}{s_2}=1-\frac{d_2}{\alpha_2+d_2+i\omega},\qquad
d_2=p_{12}\,d_1,\qquad p_{12}=\left(\frac{x}{L}\right)^2p_{\max},\quad p_{\max}=0.3 ,
\tag{M-25}
$$

其中 $x/L$ 为该点到主阻尼方向 PML 内边界的归一化距离。记忆变量满足

$$
\frac{\partial E^2}{\partial t}+(\alpha_2+d_2)E^2=d_2X
\quad\Longleftrightarrow\quad
\frac{\partial E^2}{\partial t}=d_2\left(X-E^2\right)-\alpha_2E^2 ,
\tag{M-25b}
$$

而该方向"拉伸后算子与原始算子之差"为 $\beta_2^{-1}=1$、$\beta_2^{-1}-1=0$，故辅助方向只贡献记忆项 $-E^2$。

**辅助方向的频移 $\alpha_2$**：代码取 $\alpha_2$ 等于对应主阻尼层同一位置的 $\alpha$
（$\xi$ 主阻尼层的平行方向取 $\alpha_\xi(ix)$，$\zeta$ 主阻尼层的平行方向取 $\alpha_\zeta(iz)$），
即辅助方向也用 CFS 拉伸。论文式 (32) 取 $\alpha_2=0$（辅助方向是"普通 PML"），
这一取法对只有传播波的均匀介质是合适的；但流固界面上的界面波在 $\zeta$ 方向是衰减型
（evanescent）尾部，普通 PML 对该类波不适宜，实测会在界面 $\times$ 左右 PML 区域出现
缓慢残余增长，且增长速率随辅助阻尼强度增大（$p_{\max}$ 取 0.5、取常数 $p_{12}=0.3$ 时更快，
关掉辅助阻尼则不出现）。改用 CFS 的 $\alpha_2$ 后残余与关掉辅助阻尼时一致。

### 4.1 本构方程（固体）

$$
\frac{\partial E_x^2}{\partial t}=d_2\left(D_2^\xi v_x-E_x^2\right)-\alpha_2E_x^2,\qquad \frac{\partial E_z^2}{\partial t}=d_2\left(D_2^\xi v_z-E_z^2\right)-\alpha_2E_z^2,
\tag{M-26}
$$

$$
\frac{\partial\sigma_{xx}}{\partial t}\mathrel{+}=-\left(C_{11}x_\zeta E_x^2+C_{13}z_\zeta E_z^2\right),\quad \frac{\partial\sigma_{zz}}{\partial t}\mathrel{+}=-\left(C_{13}x_\zeta E_x^2+C_{33}z_\zeta E_z^2\right),\quad \frac{\partial\sigma_{xz}}{\partial t}\mathrel{+}=-C_{55}\left(z_\zeta E_x^2+x_\zeta E_z^2\right).
\tag{M-27}
$$

主阻尼轴取 $\zeta$ 时把 $\boldsymbol e_\zeta$ 换成 $\boldsymbol e_\xi$（$x_\zeta\to x_\xi=J\zeta_z$、$z_\zeta\to z_\xi=-J\zeta_x$）、$D_2^\xi\to D_2^\zeta$，$d_2$ 改由 $\zeta$ 方向剖面导出。

### 4.2 动量方程（固体）

$$
\frac{\partial V_x^2}{\partial t}=d_2\left([\nabla_2^\xi\cdot\boldsymbol\sigma]_x-V_x^2\right)-\alpha_2V_x^2,\qquad \rho\frac{\partial v_x}{\partial t}\mathrel{+}=-V_x^2 ,
\tag{M-28}
$$

$z$ 分量用 (M-7) 的 $[\nabla_2^\xi\cdot\boldsymbol\sigma]_z$；几何因子 $x_\zeta,z_\zeta$ 由 (M-2) 给出。

### 4.3 声学（流体）侧

$$
\frac{\partial V_x^2}{\partial t}=d_2\left(-\frac{x_\zeta}{g_{\zeta\zeta}}\partial_\zeta p-V_x^2\right)-\alpha_2V_x^2,\qquad \rho\frac{\partial v_x}{\partial t}\mathrel{+}=-V_x^2 ,
\tag{M-29}
$$

$$
\frac{\partial E_x^2}{\partial t}=d_2\left(\frac{1}{g_{\zeta\zeta}}\partial_\zeta v_x-E_x^2\right)-\alpha_2E_x^2,\qquad
\frac{\partial p}{\partial t}\mathrel{-}=K\left(x_\zeta E_x^2+z_\zeta E_z^2\right).
\tag{M-30}
$$

注意压力方程的符号：$p_{,t}=-K\nabla\cdot\boldsymbol v$，所以辅助方向对散度的修正整体取负
（而动量方程与本构方程里是 $+\,(1/\beta_2-1)$ 的形式）。

## 5 主/辅助方向的选择与边界处理（实现约定）

1. **主方向的选择**：逐点比较两侧的归一化阻尼 $d_\xi(ix)$ 与 $d_\zeta(iz)$，取较大者为主阻尼方向，另一个轴作辅助（平行）方向；角区（两侧 PML 重叠）也按此规则，这样在 PML 内边界附近不会出现两侧阻尼同时趋零的无阻尼角点。
2. **两个记忆数组**：`ADEs_xi`（或 `ADEf_xi`）存 $\xi$ 族算子的记忆——主方向为 $\xi$ 时是 $D_1^\xi$、作辅助时是 $D_2^\zeta$；`ADEs_ze`（或 `ADEf_ze`）存 $\zeta$ 族算子的记忆——主方向为 $\zeta$ 时是 $D_1^\zeta$、作辅助时是 $D_2^\xi$。每个数组含 $V_x,V_z$（动量路）与 $E_x,E_z$（本构路）四个分量，与单轴情形下的数组完全复用。
3. **辅助阻尼的适用范围**：辅助（平行）阻尼只施加在与介质物理边界（自由表面、流固界面）相隔至少 3 行的点上，即 `role.aux_ze && dist >= 3` 之类。这与程序原有做法一致（近自由表面的 $\zeta$ 方向 PML 修正本来也置 0）；否则平行阻尼会与界面处的单侧差分和显式赋值冲突而失稳。
4. **自由表面行**：只用 (M-9)–(M-15)，标量记忆 $V$ 存入 `ADEs_xi.Ex`，不施加辅助阻尼，也不更新 $\zeta$ 族记忆。
5. **近界面处的 $\zeta$ 方向导数**：速度用前后交替的 2-2 / 4-2 MacCormack 单侧算子，应力用 DRP/opt MacCormack 单侧算子，与 (M-19) 中的 $v_{x,x}=\xi_xv_{x,\xi}+\zeta_xv_{x,\zeta}$、$v_{z,z}=\xi_zv_{z,\xi}+\zeta_zv_{z,\zeta}$ 一致。

## 6 参数取值

$$
\beta_\xi=1+(\beta_0-1)\left(\frac{x}{L}\right)^{p_\beta},\qquad d_\xi=d_0\left(\frac{x}{L}\right)^{p_d},\qquad \alpha_\xi=\alpha_0\left[1-\left(\frac{x}{L}\right)^{p_\alpha}\right],
\tag{M-31}
$$

$$
d_0=-\frac{(p_d+1)c_P}{2L}\ln R,\qquad \alpha_0=2\pi f_c,\qquad p_d=2,\quad p_\beta=2,\quad p_\alpha=1,
\tag{M-32}
$$

$$
\beta_0=\frac{C}{0.5\,\mathrm{PPW}_0\,\Delta h\,f_c},\qquad C=\min\left(c_S^{\text{solid}},c_P^{\text{fluid}}\right),\qquad \mathrm{PPW}_0=8 .
\tag{M-33}
$$

辅助方向：$d_2=p_{12}d_1$、$p_{12}=(x/L)^2\cdot0.3$、$\beta_2=1$、$\alpha_2=0$，见 (M-25)。

## 7 数值验证（副本 `M-FDM-2D-mpml-impl`，10 s / 按各模型 dt）

| 算例 | 类型 | 网格 | 稳定性（10 s） |
|---|---|---|---|
| `ortho_classic_scholte` | 流固 + 自由表面 + PANDF | 正交 | 残余衰减 2.4e-16 $\to$ 8.7e-18 |
| `FANDS_PANDF_double_cos_padmove` | 流固 + 起伏自由表面/界面 + PANDF | 曲线 | 残余衰减 1.3e-13 $\to$ 1.2e-14 |
| `SOLID_PANDF_ortho` | 纯固体 + 四面 CPML | 正交 | 残余 2.6e-16 $\to$ 1.5e-19 |
| 起伏自由表面纯固体（临时算例） | 纯固体 + PANDF | 曲线 | 残余 5.1e-15 $\to$ 5.4e-20 |

波场阶段与不带 MPML 的原程序对比（`FANDS_PANDF_double_cos_padmove`，0.6 s 内三个时刻）：
流体/固体 $v_x,v_z$ 的最大相对误差 0.02 %–0.04 %，即辅助阻尼只改变 PML 内的吸收，
物理域波场与原程序一致。自由表面的局部应变率法与原来的 $c_1\sim c_6$ 消去法在平直
自由表面上逐位一致。

## 8 适用范围、失稳机制与参数选择

### 8.1 两组不同的不稳定，需要不同的措施

**(A) 曲线网格 PML 本身的失稳**（Zhang et al. 2014 讨论的那一种）——由辅助阻尼解决。
60° 平行四边形网格、四面 PML、纯固体、$\Delta t=4\times10^{-4}$（约 40 % CFL）、10 s：

| 配置 | 残余 |
|---|---|
| 单轴 ADE CFS-PML（`MPML_PMAX=0`，与原程序逐位相同） | 4.7e-19 → 3.3e-16（1 万步涨 700 倍，持续增长） |
| MPML（本文配置） | 2.3e-19 → 2.3e-20，完全平稳 |

**(B) 流固界面行 × 左右 PML 的棋盘（Nyquist）模式**——只能由界面行的空间滤波压制，
辅助阻尼无效。正交流固 `ortho_classic_scholte`、1 s、只改界面滤波：

| 界面滤波 | 结果 |
|---|---|
| $\sigma=0.55$ | 干净 |
| 孙耀充原始 5 点（等价 $\sigma\approx0.476$） | 0.8 s 起界面行×PML 出现棋盘，1 s 时达 $2.6\times10^{-8}$（波场 $10^{-13}$ 量级） |

把震源抬高使肖尔特波减弱（界面以上 20/83/133 行）：棋盘振幅按约 $d^{-0.9}$ 减小
（$4.5\times10^{-8}\to1.4\times10^{-8}\to5.6\times10^{-9}$），但**增长率不变**
（每千步 16.3 倍），出现位置固定在界面行与 PML 列的交界。故肖尔特波只是激励源，
界面行上的离散棋盘模式本身是放大的。

### 8.2 辅助方向为什么必须带 CFS 频移

论文式 (32) 取 $\alpha_2=0$（辅助方向是"普通 PML"）。由 $1/s_2=1-d_2/(i\omega+d_2)$ 可见
普通 PML 对 $\omega\to0$ 与衰减型（evanescent）波不适宜；流固界面上的界面波在 $\zeta$
方向的尾部正属于这一类。实测（正交流固 + $\sigma=0.55$ 滤波）：
$\alpha_2=0$ 时残余在第 7.3 s 涨到 100 倍（失稳）；把 $\alpha_2$ 取成对应主阻尼层同一
剖面的 $\alpha$（$\beta_2$ 仍为 1、$p_{12}$ 仍为 $(x/L)^2\cdot0.3$）后，10 s 残余与完全
关掉辅助阻尼时一致。故本实现取 $\alpha_2=\alpha_\xi(ix)$（$\xi$ 主阻尼层的平行方向）
或 $\alpha_2=\alpha_\zeta(iz)$（$\zeta$ 主阻尼层的平行方向）。

### 8.3 界面滤波 $\sigma$ 的极限（主仓库代码，无 MPML）

滤波系数 $w_k\propto\exp(-k^2/\sigma^2)$（$k=-3..3$，归一化；$\sigma=0.55$ 即原系数）。
判定 10 s、残余相对平台增长 100 倍为不稳：

| 模型 | $\Delta t$ | $\Delta t/(h/v_P)$ | $\sigma$ 极限 |
|---|---|---|---|
| 小平直流固（h=2 m，PML 60 m） | 2e-4 | ≈20 % | ≈0.556（0.55 不稳、0.5625 稳） |
| 同上 | 1e-4 | ≈10 % | < 0.25（几乎不需要滤波） |
| `FANDS_PANDF_double_cos_padmove` | 2e-4 | ≈32 % | ≈0.522（0.51875 不稳、0.525 稳） |
| `FANDS_PANDF_double_cos_padmove` | 1e-4 | ≈16 % | 孙原始滤波（≈0.476）仍不稳，3.47 s 失稳；0.53 稳 |

阈值附近失稳出现得越来越晚（$\sigma=0.50$：1.96 s；0.5125：3.3 s；0.51875：5.3 s；
0.525 撑过 10 s），因此实用上应留裕量。`double_cos` 的极限几乎不随 $\Delta t$ 变化，
说明其界面不稳定主要由曲线界面那一行的离散格式决定，而不是 CFL。

### 8.4 与 SPECFEM2D 的对比（`double_cos`，29 道界面点，0.04–1.96 s）

| $\sigma$ | vz 平均 NCC | vx 平均 NCC | 最差道 | vz 最大残余/峰值 |
|---|---|---|---|---|
| 0.55 | 0.9911 | 0.9925 | vz@50 m 0.9662，vx@250 m 0.9649 | 15.1 % / 11.7 % |
| 0.53 | 0.9917 | 0.9932 | vz@50 m 0.9677，vx@250 m 0.9694 | 14.8 % / 10.8 % |

即 $\sigma$ 从 0.55 降到 0.53，最差道只提高 +0.0015（vz）/ +0.0045（vx），改善主要出现在
1.5–1.96 s 窗口（虚假反射到达之后），且个别道反而略降。FDM 与 SEM 的差异主要不是
滤波强度造成的，不建议为这点收益把 $\sigma$ 压到接近稳定极限。

### 8.5 开关与推荐配置

`include/elements.h`：

- `MPML_PMAX = 0.3f`：辅助方向阻尼系数，$d_2=p_{12}d_1$。
- `MPML_P12_CONST = false`：$p_{12}=(x/L)^2p_{\max}$（论文式 64）；置 true 为常数 $p_{\max}$。
- `MPML_BETA2 = 1.0f`：辅助方向缩放因子（论文取 1）。
- `MPML_A2_CFS = true`：辅助方向是否使用 CFS 频移（见 8.2）。

推荐：$\beta_2=1$、$p_{12}=(x/L)^2\cdot0.3$、辅助方向带 CFS 频移；辅助阻尼在离自由表面
/ 流固界面 3 行内不施加；界面滤波保持 $\sigma=0.55$。

### 8.6 MPML 的精度代价（`ortho_classic_scholte`，正交网格实测）

正交网格上单轴 ADE CFS-PML 本来稳定，辅助阻尼并非必需；打开后它带来的代价可以量化
（物理域内，10 s 记录与 2 s 内快照）：

| 指标 | 单轴 | MPML | 相对变化 |
|---|---|---|---|
| 晚时（2–10 s）$p$ 记录 max | 8.21e-08（0.33 % 峰值） | 9.77e-08（0.39 %） | +19 % |
| 晚时（2–10 s）$p$ 记录 rms | 3.99e-09 | 4.55e-09 | +14 % |
| 与单轴波场之差（$t=1.0$ s / 2.0 s） | 0 | 0.94 % / 0.32 % 峰值 | — |
| 差值最大位置 | — | 紧邻 PML 内边界的那一列 | — |
| 1.5–5 s 的 $p$ 记录尾波 | 基准 | 约 10 倍 | — |

原因：辅助方向（平行于 PML 面）不是严格匹配方向，多加的阻尼必然带来额外反射；论文也
明确指出多轴阻尼使 rms 误差比 CFS-PML 略大（"makes the rms error a little larger"）。
因此 **MPML 是稳定性手段，不是精度手段**：

- 只有单轴 CFS-PML 会失稳的强变形网格（如 60° 平行四边形、无正交 pad）才值得付出这个代价；
- 正交或弱变形模型（本项目现有算例都属于这一类）应保持单轴 ADE CFS-PML；
- 本算例中差异全部来自侧边 PML 的平行方向（ζ 族），底边 PML 的平行方向（ξ 族）无可测影响，
  可用 `MPML_AUX_ZE` / `MPML_AUX_XI` 分别关闭验证。
