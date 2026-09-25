"""模型脚本接口字典集中出口，供 plot_wavefield.py import。"""
from . import model_iso_SOLID_PANDF_ortho as _solid_ortho
from . import model_iso_FANDS_PANDF_wavy as _fands_wavy
from . import model_iso_SOLID_PANDF_wavy_if as _solid_wavy

INTERFACES = {
    _solid_ortho.case: _solid_ortho.interface,
    _fands_wavy.case: _fands_wavy.interface,
    _solid_wavy.case: _solid_wavy.interface,
}
