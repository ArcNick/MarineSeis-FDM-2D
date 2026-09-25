# 1. 几何基底与方向导数算子

## 1.1 坐标变换与几何度规

设物理坐标 $(x,z)$ 与计算坐标 $(\xi,\zeta)$ 之间的映射为 $x = x(\xi,\zeta),\, z = z(\xi,\zeta)$，雅可比行列式满足 $J = x_{,\xi} z_{,\zeta} - x_{,\zeta} z_{,\xi} \ne 0$。

* **逆变基矢量：**

$$
\boldsymbol e^{\xi} = \begin{pmatrix} \xi_{,x} \\ \xi_{,z} \end{pmatrix} = \frac{1}{J} \begin{pmatrix} z_{,\zeta} \\ -x_{,\zeta} \end{pmatrix}, \qquad
\boldsymbol e^{\zeta} = \begin{pmatrix} \zeta_{,x} \\ \zeta_{,z} \end{pmatrix} = \frac{1}{J} \begin{pmatrix} -z_{,\xi} \\ x_{,\xi} \end{pmatrix}
$$

贴体起伏网格（矩形物理域，只让某条 $\zeta=\mathrm{const}$ 网格线起伏）：

$$x(\xi,\zeta)=h\xi,\qquad
z(\xi,\zeta)=h\zeta+A_i\sin\!\frac{2\pi x}{\lambda_i}\,w_i(\zeta)
+A_f\sin\!\frac{2\pi x}{\lambda_f}\,w_f(\zeta)$$

$$w_i(\zeta)=\begin{cases}
\dfrac{\sin^2\!\big[\pi(\zeta-a_i)/(b-a_i)\big]}{\sin^2\!\big[\pi(\zeta_0-a_i)/(b-a_i)\big]}, & a_i\le\zeta\le b\\[2mm]
0, & \zeta>b
\end{cases}
\qquad
w_f(\zeta)=\begin{cases}
\cos^2\!\big[\tfrac{\pi}{2}(\zeta-3)/(b-3)\big], & 3\le\zeta\le b\\[2mm]
0, & \zeta>b
\end{cases}$$

$a_i=3$（自由表面行），$b=nz-4-\mathrm{npd}$（吸收带内边界），$\zeta_0$：起伏线所在行。
包络在吸收带内恒为 0、导数也恒为 0 ⇒ 带内 $\partial z/\partial\zeta=h$。


* **度量张量逆变分量：**

$$
g^{\xi\xi} = \xi_{,x}^{2} + \xi_{,z}^{2}, \qquad
g^{\xi\zeta} = g^{\zeta\xi} = \xi_{,x}\zeta_{,x} + \xi_{,z}\zeta_{,z}, \qquad
g^{\zeta\zeta} = \zeta_{,x}^{2} + \zeta_{,z}^{2}
$$

---

## 1.2 标量方向导数算子

定义沿计算网格坐标轴的标量方向导数算子 $D_1^{\xi}$ 与 $D_1^{\zeta}$：

$$
D_1^{\xi} = \frac{1}{g^{\xi\xi}} \left( g^{\xi\xi} \partial_{\xi} + g^{\xi\zeta} \partial_{\zeta} \right), \qquad
D_1^{\zeta} = \frac{1}{g^{\zeta\zeta}} \left( g^{\zeta\zeta} \partial_{\zeta} + g^{\zeta\xi} \partial_{\xi} \right)
$$

作用于任意标量函数 $f$ 得到：

$$
D_1^{\xi} f = \frac{1}{g^{\xi\xi}} \left( g^{\xi\xi} f_{,\xi} + g^{\xi\zeta} f_{,\zeta} \right), \qquad
D_1^{\zeta} f = \frac{1}{g^{\zeta\zeta}} \left( g^{\zeta\zeta} f_{,\zeta} + g^{\zeta\xi} f_{,\xi} \right)
$$

---

## 1.3 面力通量（二维）

$$
\boxed{
\begin{aligned}
T_{xx} &= J\bigl(\xi_{,x}\sigma_{xx} + \xi_{,z}\sigma_{xz}\bigr),
&
T_{xz} &= J\bigl(\zeta_{,x}\sigma_{xx} + \zeta_{,z}\sigma_{xz}\bigr),\\[4pt]
T_{zx} &= J\bigl(\xi_{,x}\sigma_{xz} + \xi_{,z}\sigma_{zz}\bigr),
&
T_{zz} &= J\bigl(\zeta_{,x}\sigma_{xz} + \zeta_{,z}\sigma_{zz}\bigr).
\end{aligned}
}
\tag{T-def}
$$

---

## 1.4 投影方向散度（矢量，守恒形式）

$$
\boxed{
\begin{aligned}
\bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_x
&= \frac{\xi_{,x}}{g^{\xi\xi}J}
\left[
\xi_{,x}\left( \frac{\partial T_{xx}}{\partial \xi} + \frac{\partial T_{xz}}{\partial \zeta} \right)
+ \xi_{,z}\left( \frac{\partial T_{zx}}{\partial \xi} + \frac{\partial T_{zz}}{\partial \zeta} \right)
\right],\\[8pt]
\bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_z
&= \frac{\xi_{,z}}{g^{\xi\xi}J}
\left[
\xi_{,x}\left( \frac{\partial T_{xx}}{\partial \xi} + \frac{\partial T_{xz}}{\partial \zeta} \right)
+ \xi_{,z}\left( \frac{\partial T_{zx}}{\partial \xi} + \frac{\partial T_{zz}}{\partial \zeta} \right)
\right].
\end{aligned}
}
\tag{DIV-$\xi$-cons}
$$

$$
\boxed{
\begin{aligned}
\bigl[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}\bigr]_x
&= \frac{\zeta_{,x}}{g^{\zeta\zeta}J}
\left[
\zeta_{,x}\left( \frac{\partial T_{xx}}{\partial \xi} + \frac{\partial T_{xz}}{\partial \zeta} \right)
+ \zeta_{,z}\left( \frac{\partial T_{zx}}{\partial \xi} + \frac{\partial T_{zz}}{\partial \zeta} \right)
\right],\\[8pt]
\bigl[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}\bigr]_z
&= \frac{\zeta_{,z}}{g^{\zeta\zeta}J}
\left[
\zeta_{,x}\left( \frac{\partial T_{xx}}{\partial \xi} + \frac{\partial T_{xz}}{\partial \zeta} \right)
+ \zeta_{,z}\left( \frac{\partial T_{zx}}{\partial \xi} + \frac{\partial T_{zz}}{\partial \zeta} \right)
\right].
\end{aligned}
}
\tag{DIV-$\zeta$-cons}
$$

---

## 1.5 投影方向散度（矢量，非守恒形式）

$$
\boxed{
\begin{aligned}
\bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_x
&= \xi_{,x} D_1^{\xi}\sigma_{xx} + \xi_{,z} D_1^{\xi}\sigma_{xz},\\[4pt]
\bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_z
&= \xi_{,x} D_1^{\xi}\sigma_{xz} + \xi_{,z} D_1^{\xi}\sigma_{zz}.
\end{aligned}
}
\tag{DIV-$\xi$-nc}
$$

$$
\boxed{
\begin{aligned}
\bigl[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}\bigr]_x
&= \zeta_{,x} D_1^{\zeta}\sigma_{xx} + \zeta_{,z} D_1^{\zeta}\sigma_{xz},\\[4pt]
\bigl[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}\bigr]_z
&= \zeta_{,x} D_1^{\zeta}\sigma_{xz} + \zeta_{,z} D_1^{\zeta}\sigma_{zz}.
\end{aligned}
}
\tag{DIV-$\zeta$-nc}
$$

---

## 1.6 物理散度的守恒形式

$$
\boxed{
\sigma_{xx,x} + \sigma_{xz,z}
= \frac{1}{J}\left( \frac{\partial T_{xx}}{\partial \xi} + \frac{\partial T_{xz}}{\partial \zeta} \right),
\qquad
\sigma_{xz,x} + \sigma_{zz,z}
= \frac{1}{J}\left( \frac{\partial T_{zx}}{\partial \xi} + \frac{\partial T_{zz}}{\partial \zeta} \right).
}
\tag{div-cons}
$$

---

# 2. 原始波动方程（无 PML）

## 2.1 非守恒形式

### 2.1.1 动量方程

$$
\boxed{
\begin{aligned}
\rho v_{x,t} &= \sigma_{xx,x} + \sigma_{xz,z},\\[4pt]
\rho v_{z,t} &= \sigma_{xz,x} + \sigma_{zz,z}.
\end{aligned}
}
\tag{M0-nc}
$$

### 2.1.2 本构方程

$$
\boxed{
\begin{aligned}
\sigma_{xx,t} &= C_{11} v_{x,x} + C_{13} v_{z,z},\\[4pt]
\sigma_{zz,t} &= C_{13} v_{x,x} + C_{33} v_{z,z},\\[4pt]
\sigma_{xz,t} &= C_{55}\bigl(v_{x,z} + v_{z,x}\bigr).
\end{aligned}
}
\tag{C0-nc}
$$

---

## 2.2 守恒形式

### 2.2.1 动量方程

$$
\boxed{
\begin{aligned}
\rho v_{x,t}
&= \frac{1}{J}\left( \frac{\partial T_{xx}}{\partial \xi} + \frac{\partial T_{xz}}{\partial \zeta} \right),\\[6pt]
\rho v_{z,t}
&= \frac{1}{J}\left( \frac{\partial T_{zx}}{\partial \xi} + \frac{\partial T_{zz}}{\partial \zeta} \right).
\end{aligned}
}
\tag{M0-cons}
$$

### 2.2.2 本构方程

本构方程无守恒形式，保持 (C0-nc) 不变。

---

# 3. 耦合 ADE CFS-PML 的波动方程（非守恒形式）

## 3.1 $\xi$ 方向 PML

复频移拉伸因子：

$$
s_{\xi} = \beta_{\xi} + \frac{d_{\xi}}{\alpha_{\xi} + i\omega}
$$

### 3.1.1 动量方程（矢量记忆变量）

$$
\boxed{
\begin{aligned}
\rho v_{x,t}
&= \sigma_{xx,x} + \sigma_{xz,z}
+ \left(\frac{1}{\beta_{\xi}}-1\right)
\bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_x
- \frac{1}{\beta_{\xi}} V_x^{\xi},
\\[6pt]
\rho v_{z,t}
&= \sigma_{xz,x} + \sigma_{zz,z}
+ \left(\frac{1}{\beta_{\xi}}-1\right)
\bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_z
- \frac{1}{\beta_{\xi}} V_z^{\xi}.
\end{aligned}
}
\tag{M-$\xi$-nc}
$$

其中 $[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}]$ 由式 (DIV-$\xi$-nc) 给出。

矢量记忆变量：

$$
\boxed{
\begin{aligned}
V_{x,t}^{\xi}
+ \left(\alpha_{\xi} + \frac{d_{\xi}}{\beta_{\xi}}\right) V_x^{\xi}
&= \frac{d_{\xi}}{\beta_{\xi}} \bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_x,
\\[4pt]
V_{z,t}^{\xi}
+ \left(\alpha_{\xi} + \frac{d_{\xi}}{\beta_{\xi}}\right) V_z^{\xi}
&= \frac{d_{\xi}}{\beta_{\xi}} \bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_z.
\end{aligned}
}
\tag{MV-$\xi$-nc}
$$

### 3.1.2 本构方程（标量记忆变量）

$$
\boxed{
\begin{aligned}
\sigma_{xx,t}
&= C_{11} v_{x,x} + C_{13} v_{z,z}
+ C_{11}\left(\frac{1}{\beta_{\xi}}-1\right)\xi_{,x} D_1^{\xi} v_x
+ C_{13}\left(\frac{1}{\beta_{\xi}}-1\right)\xi_{,z} D_1^{\xi} v_z \\
&\quad - \frac{C_{11}\xi_{,x}}{\beta_{\xi}} E_x^{\xi}
- \frac{C_{13}\xi_{,z}}{\beta_{\xi}} E_z^{\xi},
\\[6pt]
\sigma_{zz,t}
&= C_{13} v_{x,x} + C_{33} v_{z,z}
+ C_{13}\left(\frac{1}{\beta_{\xi}}-1\right)\xi_{,x} D_1^{\xi} v_x
+ C_{33}\left(\frac{1}{\beta_{\xi}}-1\right)\xi_{,z} D_1^{\xi} v_z \\
&\quad - \frac{C_{13}\xi_{,x}}{\beta_{\xi}} E_x^{\xi}
- \frac{C_{33}\xi_{,z}}{\beta_{\xi}} E_z^{\xi},
\\[6pt]
\sigma_{xz,t}
&= C_{55}\bigl(v_{x,z} + v_{z,x}\bigr)
+ C_{55}\left(\frac{1}{\beta_{\xi}}-1\right)
\Bigl( \xi_{,z} D_1^{\xi} v_x + \xi_{,x} D_1^{\xi} v_z \Bigr) \\
&\quad - \frac{C_{55}}{\beta_{\xi}}
\Bigl( \xi_{,z} E_x^{\xi} + \xi_{,x} E_z^{\xi} \Bigr).
\end{aligned}
}
\tag{C-$\xi$-nc}
$$

$$
\boxed{
\begin{aligned}
E_{x,t}^{\xi}
+ \left(\alpha_{\xi}+\frac{d_{\xi}}{\beta_{\xi}}\right) E_x^{\xi}
&= \frac{d_{\xi}}{\beta_{\xi}} D_1^{\xi} v_x, \\[4pt]
E_{z,t}^{\xi}
+ \left(\alpha_{\xi}+\frac{d_{\xi}}{\beta_{\xi}}\right) E_z^{\xi}
&= \frac{d_{\xi}}{\beta_{\xi}} D_1^{\xi} v_z.
\end{aligned}
}
\tag{CV-$\xi$-nc}
$$

---

## 3.2 $\zeta$ 方向 PML

复频移拉伸因子：

$$
s_{\zeta} = \beta_{\zeta} + \frac{d_{\zeta}}{\alpha_{\zeta} + i\omega}
$$

### 3.2.1 动量方程（矢量记忆变量）

$$
\boxed{
\begin{aligned}
\rho v_{x,t}
&= \sigma_{xx,x} + \sigma_{xz,z}
+ \left(\frac{1}{\beta_{\zeta}}-1\right)
\bigl[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}\bigr]_x
- \frac{1}{\beta_{\zeta}} V_x^{\zeta},
\\[6pt]
\rho v_{z,t}
&= \sigma_{xz,x} + \sigma_{zz,z}
+ \left(\frac{1}{\beta_{\zeta}}-1\right)
\bigl[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}\bigr]_z
- \frac{1}{\beta_{\zeta}} V_z^{\zeta}.
\end{aligned}
}
\tag{M-$\zeta$-nc}
$$

其中 $[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}]$ 由式 (DIV-$\zeta$-nc) 给出。

矢量记忆变量：

$$
\boxed{
\begin{aligned}
V_{x,t}^{\zeta}
+ \left(\alpha_{\zeta} + \frac{d_{\zeta}}{\beta_{\zeta}}\right) V_x^{\zeta}
&= \frac{d_{\zeta}}{\beta_{\zeta}} \bigl[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}\bigr]_x,
\\[4pt]
V_{z,t}^{\zeta}
+ \left(\alpha_{\zeta} + \frac{d_{\zeta}}{\beta_{\zeta}}\right) V_z^{\zeta}
&= \frac{d_{\zeta}}{\beta_{\zeta}} \bigl[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}\bigr]_z.
\end{aligned}
}
\tag{MV-$\zeta$-nc}
$$

### 3.2.2 本构方程（标量记忆变量）

$$
\boxed{
\begin{aligned}
\sigma_{xx,t}
&= C_{11} v_{x,x} + C_{13} v_{z,z}
+ C_{11}\left(\frac{1}{\beta_{\zeta}}-1\right)\zeta_{,x} D_1^{\zeta} v_x
+ C_{13}\left(\frac{1}{\beta_{\zeta}}-1\right)\zeta_{,z} D_1^{\zeta} v_z \\
&\quad - \frac{C_{11}\zeta_{,x}}{\beta_{\zeta}} E_x^{\zeta}
- \frac{C_{13}\zeta_{,z}}{\beta_{\zeta}} E_z^{\zeta},
\\[6pt]
\sigma_{zz,t}
&= C_{13} v_{x,x} + C_{33} v_{z,z}
+ C_{13}\left(\frac{1}{\beta_{\zeta}}-1\right)\zeta_{,x} D_1^{\zeta} v_x
+ C_{33}\left(\frac{1}{\beta_{\zeta}}-1\right)\zeta_{,z} D_1^{\zeta} v_z \\
&\quad - \frac{C_{13}\zeta_{,x}}{\beta_{\zeta}} E_x^{\zeta}
- \frac{C_{33}\zeta_{,z}}{\beta_{\zeta}} E_z^{\zeta},
\\[6pt]
\sigma_{xz,t}
&= C_{55}\bigl(v_{x,z} + v_{z,x}\bigr)
+ C_{55}\left(\frac{1}{\beta_{\zeta}}-1\right)
\Bigl( \zeta_{,z} D_1^{\zeta} v_x + \zeta_{,x} D_1^{\zeta} v_z \Bigr) \\
&\quad - \frac{C_{55}}{\beta_{\zeta}}
\Bigl( \zeta_{,z} E_x^{\zeta} + \zeta_{,x} E_z^{\zeta} \Bigr).
\end{aligned}
}
\tag{C-$\zeta$-nc}
$$

$$
\boxed{
\begin{aligned}
E_{x,t}^{\zeta}
+ \left(\alpha_{\zeta}+\frac{d_{\zeta}}{\beta_{\zeta}}\right) E_x^{\zeta}
&= \frac{d_{\zeta}}{\beta_{\zeta}} D_1^{\zeta} v_x, \\[4pt]
E_{z,t}^{\zeta}
+ \left(\alpha_{\zeta}+\frac{d_{\zeta}}{\beta_{\zeta}}\right) E_z^{\zeta}
&= \frac{d_{\zeta}}{\beta_{\zeta}} D_1^{\zeta} v_z.
\end{aligned}
}
\tag{CV-$\zeta$-nc}
$$

---

# 4. 耦合 ADE CFS-PML 的波动方程（守恒形式）

## 4.1 $\xi$ 方向 PML

### 4.1.1 动量方程（守恒形式，矢量记忆变量）

$$
\boxed{
\begin{aligned}
\rho v_{x,t}
&= \frac{1}{J}\left( \frac{\partial T_{xx}}{\partial \xi} + \frac{\partial T_{xz}}{\partial \zeta} \right)
+ \left(\frac{1}{\beta_{\xi}}-1\right)
\bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_x
- \frac{1}{\beta_{\xi}} V_x^{\xi},
\\[8pt]
\rho v_{z,t}
&= \frac{1}{J}\left( \frac{\partial T_{zx}}{\partial \xi} + \frac{\partial T_{zz}}{\partial \zeta} \right)
+ \left(\frac{1}{\beta_{\xi}}-1\right)
\bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_z
- \frac{1}{\beta_{\xi}} V_z^{\xi}.
\end{aligned}
}
\tag{M-$\xi$-cons}
$$

其中 $[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}]$ 可由守恒形式 (DIV-$\xi$-cons) 给出。

矢量记忆变量：

$$
\boxed{
\begin{aligned}
V_{x,t}^{\xi}
+ \left(\alpha_{\xi} + \frac{d_{\xi}}{\beta_{\xi}}\right) V_x^{\xi}
&= \frac{d_{\xi}}{\beta_{\xi}} \bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_x,
\\[4pt]
V_{z,t}^{\xi}
+ \left(\alpha_{\xi} + \frac{d_{\xi}}{\beta_{\xi}}\right) V_z^{\xi}
&= \frac{d_{\xi}}{\beta_{\xi}} \bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_z.
\end{aligned}
}
\tag{MV-$\xi$-cons}
$$

---

## 4.2 $\zeta$ 方向 PML

### 4.2.1 动量方程（守恒形式，矢量记忆变量）

$$
\boxed{
\begin{aligned}
\rho v_{x,t}
&= \frac{1}{J}\left( \frac{\partial T_{xx}}{\partial \xi} + \frac{\partial T_{xz}}{\partial \zeta} \right)
+ \left(\frac{1}{\beta_{\zeta}}-1\right)
\bigl[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}\bigr]_x
- \frac{1}{\beta_{\zeta}} V_x^{\zeta},
\\[8pt]
\rho v_{z,t}
&= \frac{1}{J}\left( \frac{\partial T_{zx}}{\partial \xi} + \frac{\partial T_{zz}}{\partial \zeta} \right)
+ \left(\frac{1}{\beta_{\zeta}}-1\right)
\bigl[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}\bigr]_z
- \frac{1}{\beta_{\zeta}} V_z^{\zeta}.
\end{aligned}
}
\tag{M-$\zeta$-cons}
$$

其中 $[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}]$ 可由守恒形式 (DIV-$\zeta$-cons) 给出。

矢量记忆变量：

$$
\boxed{
\begin{aligned}
V_{x,t}^{\zeta}
+ \left(\alpha_{\zeta} + \frac{d_{\zeta}}{\beta_{\zeta}}\right) V_x^{\zeta}
&= \frac{d_{\zeta}}{\beta_{\zeta}} \bigl[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}\bigr]_x,
\\[4pt]
V_{z,t}^{\zeta}
+ \left(\alpha_{\zeta} + \frac{d_{\zeta}}{\beta_{\zeta}}\right) V_z^{\zeta}
&= \frac{d_{\zeta}}{\beta_{\zeta}} \bigl[\nabla_1^{\zeta}\cdot\boldsymbol{\sigma}\bigr]_z.
\end{aligned}
}
\tag{MV-$\zeta$-cons}
$$

---

# 5. 自由表面修正的波动方程

## 5.1 自由表面条件与速度导数关系

自由表面为 $\zeta = 0$ 面，面力为零：

$$
\sigma_{xx}\zeta_{,x} + \sigma_{xz}\zeta_{,z} = 0,
\qquad
\sigma_{xz}\zeta_{,x} + \sigma_{zz}\zeta_{,z} = 0 .
\tag{FS-1}
$$

将二维 VTI/HTI 本构关系代入并整理，得到：

$$
\begin{pmatrix} v_{x,\zeta}\\[2pt] v_{z,\zeta} \end{pmatrix}
= -\, M
\begin{pmatrix} v_{x,\xi}\\[2pt] v_{z,\xi} \end{pmatrix},
\qquad
M = D_\zeta^{-1} D_\xi .
\tag{FS-2}
$$

$$
D_\zeta =
\begin{pmatrix}
C_{11}\zeta_{,x}^2 + C_{55}\zeta_{,z}^2 &
(C_{13}+C_{55})\zeta_{,x}\zeta_{,z}\\[2pt]
(C_{13}+C_{55})\zeta_{,x}\zeta_{,z} &
C_{55}\zeta_{,x}^2 + C_{33}\zeta_{,z}^2
\end{pmatrix},
\tag{FS-3}
$$

$$
D_\xi =
\begin{pmatrix}
C_{11}\zeta_{,x}\xi_{,x} + C_{55}\zeta_{,z}\xi_{,z} &
C_{13}\zeta_{,x}\xi_{,z} + C_{55}\zeta_{,z}\xi_{,x}\\[2pt]
C_{13}\zeta_{,z}\xi_{,x} + C_{55}\zeta_{,x}\xi_{,z} &
C_{55}\zeta_{,x}\xi_{,x} + C_{33}\zeta_{,z}\xi_{,z}
\end{pmatrix}.
\tag{FS-4}
$$

$$
M =
\begin{pmatrix}
m_{11} & m_{12}\\[2pt]
m_{21} & m_{22}
\end{pmatrix}.
\tag{FS-5}
$$

---

## 5.2 本构方程（自由表面修正，非守恒形式，消去 $\zeta$ 导数）

### 5.2.1 原始项系数

$$
v_{x,x} = \alpha_{11}^{xx} v_{x,\xi} + \alpha_{12}^{xx} v_{z,\xi},
\qquad
v_{z,z} = \alpha_{11}^{zz} v_{x,\xi} + \alpha_{12}^{zz} v_{z,\xi},
$$

$$
v_{x,z}+v_{z,x}
= \alpha_{11}^{xz} v_{x,\xi} + \alpha_{12}^{xz} v_{z,\xi}.
\tag{orig-elim}
$$

$$
\boxed{
\begin{aligned}
\alpha_{11}^{xx} &= \xi_{,x} - \zeta_{,x} m_{11},
&
\alpha_{12}^{xx} &= - \zeta_{,x} m_{12},
\\[4pt]
\alpha_{11}^{zz} &= - \zeta_{,z} m_{21},
&
\alpha_{12}^{zz} &= \xi_{,z} - \zeta_{,z} m_{22},
\\[4pt]
\alpha_{11}^{xz} &= \xi_{,z} - \zeta_{,z} m_{11} - \zeta_{,x} m_{21},
&
\alpha_{12}^{xz} &= \xi_{,x} - \zeta_{,z} m_{12} - \zeta_{,x} m_{22}.
\end{aligned}
}
\tag{alpha-coeff}
$$

### 5.2.2 PML 方向导数消去 $\zeta$ 导数

$$
D_1^{\xi} v_x = A_{xx} v_{x,\xi} + A_{xz} v_{z,\xi},
\qquad
D_1^{\xi} v_z = A_{zx} v_{x,\xi} + A_{zz} v_{z,\xi}.
\tag{D-elim}
$$

$$
\boxed{
\begin{aligned}
A_{xx} &= \frac{g^{\xi\xi} - g^{\xi\zeta} m_{11}}{g^{\xi\xi}},
&
A_{xz} &= - \frac{g^{\xi\zeta} m_{12}}{g^{\xi\xi}},
\\[4pt]
A_{zx} &= - \frac{g^{\xi\zeta} m_{21}}{g^{\xi\xi}},
&
A_{zz} &= \frac{g^{\xi\xi} - g^{\xi\zeta} m_{22}}{g^{\xi\xi}}.
\end{aligned}
}
\tag{A-coeff}
$$

### 5.2.3 自由表面处最终应力方程

$$
\boxed{
\begin{aligned}
\sigma_{xx,t}
&= \Bigl[ C_{11}\alpha_{11}^{xx} + C_{13}\alpha_{11}^{zz} \Bigr] v_{x,\xi}
+ \Bigl[ C_{11}\alpha_{12}^{xx} + C_{13}\alpha_{12}^{zz} \Bigr] v_{z,\xi} \\[2pt]
&\quad + \Bigl(\frac{1}{\beta_{\xi}}-1\Bigr)
\Bigl[ C_{11}\xi_{,x} A_{xx} + C_{13}\xi_{,z} A_{zx} \Bigr] v_{x,\xi} \\[2pt]
&\quad + \Bigl(\frac{1}{\beta_{\xi}}-1\Bigr)
\Bigl[ C_{11}\xi_{,x} A_{xz} + C_{13}\xi_{,z} A_{zz} \Bigr] v_{z,\xi} \\[2pt]
&\quad - \frac{C_{11}\xi_{,x}}{\beta_{\xi}} E_x^{\xi}
- \frac{C_{13}\xi_{,z}}{\beta_{\xi}} E_z^{\xi}.
\end{aligned}
}
\tag{Sxx-fs-final}
$$

$$
\boxed{
\begin{aligned}
\sigma_{zz,t}
&= \Bigl[ C_{13}\alpha_{11}^{xx} + C_{33}\alpha_{11}^{zz} \Bigr] v_{x,\xi}
+ \Bigl[ C_{13}\alpha_{12}^{xx} + C_{33}\alpha_{12}^{zz} \Bigr] v_{z,\xi} \\[2pt]
&\quad + \Bigl(\frac{1}{\beta_{\xi}}-1\Bigr)
\Bigl[ C_{13}\xi_{,x} A_{xx} + C_{33}\xi_{,z} A_{zx} \Bigr] v_{x,\xi} \\[2pt]
&\quad + \Bigl(\frac{1}{\beta_{\xi}}-1\Bigr)
\Bigl[ C_{13}\xi_{,x} A_{xz} + C_{33}\xi_{,z} A_{zz} \Bigr] v_{z,\xi} \\[2pt]
&\quad - \frac{C_{13}\xi_{,x}}{\beta_{\xi}} E_x^{\xi}
- \frac{C_{33}\xi_{,z}}{\beta_{\xi}} E_z^{\xi}.
\end{aligned}
}
\tag{Szz-fs-final}
$$

$$
\boxed{
\begin{aligned}
\sigma_{xz,t}
&= C_{55} \Bigl[ \alpha_{11}^{xz} v_{x,\xi} + \alpha_{12}^{xz} v_{z,\xi} \Bigr] \\[2pt]
&\quad + C_{55}\Bigl(\frac{1}{\beta_{\xi}}-1\Bigr)
\Bigl[ \xi_{,z} \bigl( A_{xx} v_{x,\xi} + A_{xz} v_{z,\xi} \bigr)
+ \xi_{,x} \bigl( A_{zx} v_{x,\xi} + A_{zz} v_{z,\xi} \bigr) \Bigr] \\[2pt]
&\quad - \frac{C_{55}}{\beta_{\xi}}
\Bigl[ \xi_{,z} E_x^{\xi} + \xi_{,x} E_z^{\xi} \Bigr].
\end{aligned}
}
\tag{Sxz-fs-final}
$$

### 5.2.4 自由表面处记忆变量 $E$ 更新方程

$$
\boxed{
\begin{aligned}
E_{x,t}^{\xi}
+ \Bigl(\alpha_{\xi}+\frac{d_{\xi}}{\beta_{\xi}}\Bigr) E_x^{\xi}
&= \frac{d_{\xi}}{\beta_{\xi}}
\bigl( A_{xx} v_{x,\xi} + A_{xz} v_{z,\xi} \bigr),
\\[4pt]
E_{z,t}^{\xi}
+ \Bigl(\alpha_{\xi}+\frac{d_{\xi}}{\beta_{\xi}}\Bigr) E_z^{\xi}
&= \frac{d_{\xi}}{\beta_{\xi}}
\bigl( A_{zx} v_{x,\xi} + A_{zz} v_{z,\xi} \bigr).
\end{aligned}
}
\tag{E-fs-final}
$$

---

## 5.3 动量方程（自由表面修正，混合形式：物理散度守恒 + PML 方向项非守恒）

自由表面为 $\zeta = 0$，零面力条件为

$$
T_{xz}\big|_{\zeta=0} = 0,\qquad
T_{zz}\big|_{\zeta=0} = 0 .
\tag{FS-traction}
$$

动量方程：

$$
\boxed{
\begin{aligned}
\rho v_{x,t}
&= \frac{1}{J}\left( \frac{\partial T_{xx}}{\partial \xi} + \frac{\partial T_{xz}}{\partial \zeta} \right)
+ \left(\frac{1}{\beta_{\xi}}-1\right)
\bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_x
- \frac{1}{\beta_{\xi}} V_x^{\xi},
\\[8pt]
\rho v_{z,t}
&= \frac{1}{J}\left( \frac{\partial T_{zx}}{\partial \xi} + \frac{\partial T_{zz}}{\partial \zeta} \right)
+ \left(\frac{1}{\beta_{\xi}}-1\right)
\bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_z
- \frac{1}{\beta_{\xi}} V_z^{\xi}.
\end{aligned}
}
\tag{M-fs}
$$

其中 $[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}]$ 由非守恒形式 (DIV-$\xi$-nc) 给出。

矢量记忆变量：

$$
\boxed{
\begin{aligned}
V_{x,t}^{\xi}
+ \left(\alpha_{\xi} + \frac{d_{\xi}}{\beta_{\xi}}\right) V_x^{\xi}
&= \frac{d_{\xi}}{\beta_{\xi}} \bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_x,
\\[4pt]
V_{z,t}^{\xi}
+ \left(\alpha_{\xi} + \frac{d_{\xi}}{\beta_{\xi}}\right) V_z^{\xi}
&= \frac{d_{\xi}}{\beta_{\xi}} \bigl[\nabla_1^{\xi}\cdot\boldsymbol{\sigma}\bigr]_z.
\end{aligned}
}
\tag{MV-fs}
$$

---

# 6. 流固耦合边界

## 6.1 流固边界条件

### 6.1.1 应力条件

$$
\boldsymbol n^T \boldsymbol \sigma^s \boldsymbol n=-p, \quad
\boldsymbol \tau^T\boldsymbol \sigma^s \boldsymbol n=0
$$

由面力通量定义 (T-def) 可知：

$$
\boxed{
\begin{aligned}
T^s_{xz}=T^f_{xz}&=-Jp\zeta_{,x},\\
T^s_{zz}=T^f_{zz}&=-Jp\zeta_{,z}.
\end{aligned}
}
\tag{FSI-stress}
$$
### 6.1.2 速度条件

$$
\boxed{
\zeta_{,x}v_x^s+\zeta_{,z}v_z^s
=
\zeta_{,x}v_x^f+\zeta_{,z}v_z^f.
}
\tag{FSI-velocity}
$$

---
# 7. 全局总结

## 7.1 方程形式清单

| 编号 | 方程 | 形式 | 记忆变量 |
|------|------|------|----------|
| M0-nc | 原始动量方程 | 非守恒 | 无 |
| C0-nc | 原始本构方程 | 非守恒 | 无 |
| M0-cons | 原始动量方程 | 守恒 | 无 |
| M-$\xi$-nc | $\xi$ 方向 PML 动量方程 | 非守恒 | 矢量 $\boldsymbol{V}^{\xi}$ |
| MV-$\xi$-nc | $\xi$ 方向 PML 动量记忆变量 | 非守恒 | 矢量 |
| C-$\xi$-nc | $\xi$ 方向 PML 本构方程 | 非守恒 | 标量 $E_x^{\xi},E_z^{\xi}$ |
| CV-$\xi$-nc | $\xi$ 方向 PML 本构记忆变量 | 非守恒 | 标量 |
| M-$\zeta$-nc | $\zeta$ 方向 PML 动量方程 | 非守恒 | 矢量 $\boldsymbol{V}^{\zeta}$ |
| MV-$\zeta$-nc | $\zeta$ 方向 PML 动量记忆变量 | 非守恒 | 矢量 |
| C-$\zeta$-nc | $\zeta$ 方向 PML 本构方程 | 非守恒 | 标量 $E_x^{\zeta},E_z^{\zeta}$ |
| CV-$\zeta$-nc | $\zeta$ 方向 PML 本构记忆变量 | 非守恒 | 标量 |
| M-$\xi$-cons | $\xi$ 方向 PML 动量方程 | 守恒 | 矢量 $\boldsymbol{V}^{\xi}$ |
| MV-$\xi$-cons | $\xi$ 方向 PML 动量记忆变量 | 守恒 | 矢量 |
| M-$\zeta$-cons | $\zeta$ 方向 PML 动量方程 | 守恒 | 矢量 $\boldsymbol{V}^{\zeta}$ |
| MV-$\zeta$-cons | $\zeta$ 方向 PML 动量记忆变量 | 守恒 | 矢量 |
| Sxx-fs-final | 自由表面 $\sigma_{xx}$ 方程 | 非守恒（消去 $\zeta$ 导数） | 标量 |
| Szz-fs-final | 自由表面 $\sigma_{zz}$ 方程 | 非守恒（消去 $\zeta$ 导数） | 标量 |
| Sxz-fs-final | 自由表面 $\sigma_{xz}$ 方程 | 非守恒（消去 $\zeta$ 导数） | 标量 |
| E-fs-final | 自由表面本构记忆变量 | 非守恒（消去 $\zeta$ 导数） | 标量 |
| M-fs | 自由表面动量方程 | 混合（物理散度守恒 + PML 项非守恒） | 矢量 |
| MV-fs | 自由表面动量记忆变量 | 混合 | 矢量 |

## 7.2 记忆变量体系

| 方程 | 记忆变量 | 形式 |
|------|----------|------|
| 动量方程 | $\boldsymbol{V}^{\xi}=(V_x^{\xi},V_z^{\xi})$，$\boldsymbol{V}^{\zeta}=(V_x^{\zeta},V_z^{\zeta})$ | 矢量 |
| 本构方程 | $E_x^{\xi},E_z^{\xi}$，$E_x^{\zeta},E_z^{\zeta}$ | 标量 |

---

# 8. CFS-PML 参数与剖面公式

来源：Zhang & Shen (2010) 式(12)(34)–(37)(41)；Sun et al. (2017) 式(22)–(24)。

## 8.1 复频移拉伸函数

$$s_x=\beta_x+\frac{d_x}{\alpha_x+i\omega}\qquad(12)$$

## 8.2 剖面与阻尼幅值

$$d_x=d_0\left(\frac{x}{L}\right)^{p_d},\qquad
\beta_x=1+(\beta_0-1)\left(\frac{x}{L}\right)^{p_\beta},\qquad
\alpha_x=\alpha_0\left[1-\left(\frac{x}{L}\right)^{p_\alpha}\right]\qquad(34\text{–}36)$$

$$d_0=-\frac{(p_d+1)\,c_P}{2L}\ln R,\qquad
\alpha_0=\pi f_c,\qquad p_d=2,\quad p_\beta=2,\quad p_\alpha=1\qquad(37)$$

$x$：到 PML–内部界面的距离；$L$：PML 厚度；$R$：给定反射系数；
$c_P$：PML 所在介质的 P 波速度。

## 8.3 缩放因子 $\beta_0$

$$\beta_0=\frac{C}{0.5\,\mathrm{PPW}_0\,\Delta h\,f_c}\qquad(41),
\qquad C=\min\!\big(c_S^{\text{solid}},\ c_P^{\text{fluid}}\big)$$

$\mathrm{PPW}_0$：格式的最小每波长点数（4 阶格式取 6）；
$\Delta h$：网格间距；$f_c$：震源主频。

## 8.4 界面高斯滤波

$$w_k=A\cdot\frac{\alpha}{\sqrt{\pi}\,2n}e^{-\left(\frac{\alpha k}{2n}\right)^2},
\quad k=-2n,\dots,2n,\qquad \sum_{k=-2n}^{2n}w_k=1,\qquad \alpha=4.2,\ n=1\qquad(22\text{–}23)$$

$$\tilde\varphi(i_0)=\sum_{i=-2}^{2}w_{i+i_0}\,\varphi(i+i_0)\qquad(24)$$

## 8.5 雅可比与度量约定

$$J=\det\frac{\partial(x,z)}{\partial(\xi,\zeta)}=x_{,\xi}z_{,\zeta}-x_{,\zeta}z_{,\xi}$$

$$T^\xi=J\big(\xi_{,x}\sigma_{xx}+\xi_{,z}\sigma_{xz},\ \xi_{,x}\sigma_{xz}+\xi_{,z}\sigma_{zz}\big),\qquad
T^\zeta=J\big(\zeta_{,x}\sigma_{xx}+\zeta_{,z}\sigma_{xz},\ \zeta_{,x}\sigma_{xz}+\zeta_{,z}\sigma_{zz}\big)$$

$$\rho\,\boldsymbol v_{,t}=\frac1J\big[\partial_\xi T^\xi+\partial_\zeta T^\zeta\big]$$

流体（$\sigma_{xx}=\sigma_{zz}=-p,\ \sigma_{xz}=0$）：

$$T^\xi_f=-Jp\,(\xi_{,x},\ \xi_{,z}),\qquad T^\zeta_f=-Jp\,(\zeta_{,x},\ \zeta_{,z})$$

FANDS 时固体数组第 $i$ 行对应全局 $\zeta$ 行 $i+(nz_f-1)$，度量（$J,\xi_{,x},\zeta_{,z}$ 等）
一律按全局行取值。
