"""Same cold direct-energy DQ experiment, with bounded carbon-line work."""

from .dq_hornkohl_cell_energy import DirectEnergyCellDQMaterial
from .dq_bounded_carbon_lines import BoundedCarbonLines
from .dq_exact_opacity_cache import ExactLayerOpacityCache


class BoundedDirectEnergyDQMaterial(DirectEnergyCellDQMaterial):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.bounded_carbon=BoundedCarbonLines(self.atomic,self.atomic_keys,self.atomic_anchors)
        self.maximum_carbon_opacity_bound=0.

    def prepare_opacity_state(self,a,carbon,c2):
        super().prepare_opacity_state(a,carbon,c2)
        if self.bounded_carbon.prepare(a,carbon):
            # An unchanged layer can acquire different wings when another
            # layer switches the full-column resonance-support decision.
            self.opacity_cache=ExactLayerOpacityCache()

    def carbon_line_opacity(self,a,carbon,wavelength,background):
        value,info=self.bounded_carbon.evaluate(a,carbon,wavelength,background)
        self.maximum_carbon_opacity_bound=max(self.maximum_carbon_opacity_bound,info['relative_bound'])
        return value

