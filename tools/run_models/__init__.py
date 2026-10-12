"""模型脚本接口字典集中出口，供 plot_wavefield.py import。

每个模型脚本里都定义了 case 和 interface 两个模块级变量（interface 是
"界面名 -> 全局 ζ 行号"的字典）。这里把它们汇总成 INTERFACES，缺脚本就跳过。
"""
import importlib

#==============================
# 需要汇总的模型脚本
#==============================
_MODULES = [
    "model_SOLID_PANDF_ortho",
    "model_FANDS_PANDF_wavy",
    "model_SOLID_PANDF_wavy_if",
    "model_FANDS_PANDF_wavy_padmove",
    "model_FANDS_PANDF_wavy_padmove_center",
    "model_FANDS_PANDF_double_cos_padmove",
    "model_ortho_classic_scholte",
    "model_ortho_freesurface_scholte",
    "model_Zhao_scholte",
]

#==============================
# 汇总
#==============================
INTERFACES = {}
for _name in _MODULES:
    try:
        _model = importlib.import_module("." + _name, __package__)
    except ImportError:              # 脚本不存在/依赖缺失就跳过，不影响出图
        continue
    if hasattr(_model, "case") and hasattr(_model, "interface"):
        INTERFACES[_model.case] = _model.interface
