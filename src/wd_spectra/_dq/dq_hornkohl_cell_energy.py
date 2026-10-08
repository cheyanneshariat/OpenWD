"""Cold DQ test of direct conservative energy equations after gray seeding.

Keep the existing initial ML2 projection and bolometric seed correction,
but use physical cell-energy equations from the first nonlinear iteration.
The separate approximate-gradient and flux-only iteration phases are not
needed as convergence authorities. No external atmosphere is supplied.
The function override is restricted to this isolated research process.
"""
from unittest.mock import patch

from . import base as dq
from .dq_hornkohl_cell import CellIntegratedDQMaterial


class DirectEnergyCellDQMaterial(CellIntegratedDQMaterial):
    def solve(self,*args,**kwargs):
        original=dq.solve_adaptive_lte_structure
        def physical_equations(*positional,**options):
            options['use_convective_gradient_preconditioner']=False
            return original(*positional,**options)
        with patch.object(dq,'solve_adaptive_lte_structure',physical_equations):
            return super().solve(*args,**kwargs)


