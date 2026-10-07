"""
elements -- fea_engine's element library.

Originally one 1322-line element.py; split during the fea_engine
restructuring into one module per element family (base engine, solids,
plates, beams, trusses, contact) purely for navigability -- no element's
logic, docstring, or numerical behavior changed. Every class is
re-exported here so ``from fea_engine.elements import Beam2DCorotational``
(or ``from fea_engine import elements; elements.Beam2DCorotational``)
keeps working exactly like the old flat ``element.py`` did, and
ELEMENT_REGISTRY (used by fea_engine.geometry's dim-driven mesh/FESystem
builder) is assembled here from all six submodules.

To add a new element type: subclass Element in the appropriate submodule
(or add a new submodule for a new family), implement shape_and_derivs()
(+ B_matrix() for the generic Gauss loop, or override stiffness()/mass()
directly), and add one line to ELEMENT_REGISTRY below.
"""
from .base import (
    Element,
    gauss_legendre,
    gauss_product,
    jacobian,
    spurious_zero_energy_modes,
    tet_quadrature_4pt,
    tri_quadrature_3pt,
)
from .solids import (
    Quad4PlaneStress, Hex8Solid3D, Tri3PlaneStress, Tet4Solid3D,
    Quad8PlaneStress, Hex20Solid3D, Tet10Solid3D, Tri6PlaneStress,
    Hex8SolidBbar,
)
from .plates import Quad4MindlinPlate
from .beams import Beam2DEulerBernoulli, Beam2DCorotational, Beam2DReissner
from .beams3d import Beam3DEulerBernoulli, Beam3DCorotational
from .trusses import TrussTL2D, TrussPlastic2D
from .contact import (GapContactPenalty, GapContactCurvedFriction,
                       NodeToSegmentContact2D, NodeToSegmentContact2DFriction,
                       closest_point_on_segment_2d, find_contact_pairs_2d)
from .nonlinear_solids import (Hex8PlasticJ2, Tet4NeoHookean, Tet10SolidTL,
                                Quad4PlasticJ2PlaneStress, Hex8PlasticJ2Kinematic)
from .shells import Shell4MITC, Shell4MITCCorotational
from .shells_director import Shell4Director

ELEMENT_REGISTRY = {
    "quad4_plane_stress": Quad4PlaneStress,
    "hex8_solid3d": Hex8Solid3D,
    "quad4_mindlin_plate": Quad4MindlinPlate,
    "beam2d_euler_bernoulli": Beam2DEulerBernoulli,
    "truss2d_tl": TrussTL2D,
    "truss2d_plastic": TrussPlastic2D,
    "beam2d_corotational": Beam2DCorotational,
    "beam2d_reissner": Beam2DReissner,
    "gap_contact_penalty": GapContactPenalty,
    "tri3_plane_stress": Tri3PlaneStress,
    "tet4_solid3d": Tet4Solid3D,
    "gap_contact_curved_friction": GapContactCurvedFriction,
    "quad8_plane_stress": Quad8PlaneStress,
    "hex20_solid3d": Hex20Solid3D,
    "tet10_solid3d": Tet10Solid3D,
    "beam3d_euler_bernoulli": Beam3DEulerBernoulli,
    "beam3d_corotational": Beam3DCorotational,
    "hex8_plastic_j2": Hex8PlasticJ2,
    "tet4_neo_hookean": Tet4NeoHookean,
    "tet10_solid_tl": Tet10SolidTL,
    "shell4_mitc": Shell4MITC,
    "shell4_mitc_corotational": Shell4MITCCorotational,
    "shell4_director": Shell4Director,
    "tri6_plane_stress": Tri6PlaneStress,
    "hex8_solid_bbar": Hex8SolidBbar,
    "quad4_plastic_j2_plane_stress": Quad4PlasticJ2PlaneStress,
    "hex8_plastic_j2_kinematic": Hex8PlasticJ2Kinematic,
    "node_to_segment_contact_2d": NodeToSegmentContact2D,
    "node_to_segment_contact_2d_friction": NodeToSegmentContact2DFriction,
}

__all__ = [
    "Element", "gauss_legendre", "gauss_product", "jacobian",
    "spurious_zero_energy_modes", "tet_quadrature_4pt", "tri_quadrature_3pt",
    "Quad4PlaneStress", "Hex8Solid3D", "Tri3PlaneStress", "Tet4Solid3D",
    "Quad4MindlinPlate",
    "Beam2DEulerBernoulli", "Beam2DCorotational", "Beam2DReissner", "Beam3DEulerBernoulli",
    "Beam3DCorotational",
    "TrussTL2D", "TrussPlastic2D",
    "GapContactPenalty", "GapContactCurvedFriction",
    "NodeToSegmentContact2D", "NodeToSegmentContact2DFriction",
    "closest_point_on_segment_2d", "find_contact_pairs_2d",
    "Quad8PlaneStress", "Hex20Solid3D", "Tet10Solid3D",
    "Hex8PlasticJ2", "Tet4NeoHookean", "Tet10SolidTL",
    "Shell4MITC", "Shell4MITCCorotational", "Shell4Director",
    "Tri6PlaneStress",
    "Hex8SolidBbar",
    "Quad4PlasticJ2PlaneStress",
    "Hex8PlasticJ2Kinematic",
    "ELEMENT_REGISTRY",
]
