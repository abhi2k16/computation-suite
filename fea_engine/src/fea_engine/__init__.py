# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
fea_engine -- a small, additive from-scratch FEA package.

This is a restructured, pip-installable version of the original
fea_package (16 development modules, all validated). No solver logic
changed during the restructuring -- every class/function was moved
verbatim (byte-for-byte diffed against the original during migration);
only file organization and import paths changed, plus the addition of
proper packaging metadata and a NumPy-style docstring layer for Sphinx.

Modules
-------
material         : materials, section properties, constitutive (D) matrices
damping          : Rayleigh and modal damping models
mesh             : mesh generation (structured/mapped) and boundary-node selection
grading          : geometry-agnostic mesh-grading/sizing-function math shared by
                    mesh.py (see docs/generalized_mesh_grading_roadmap.md)
build_mesh       : the grading-aware mesh dispatcher (build_mesh()) -- decides
                    whether/how to grade from a feature list and refuses
                    rotational-DOF elements on unverified curved/unstructured
                    topology (docs/generalized_mesh_grading_roadmap.md, Phase 4)
geometry         : dimension-driven front end (dim -> mesh -> default element ->
                    FESystem); Gmsh-backed geometry support (arbitrary CAD-like
                    shapes, STEP/IGES/BREP import) has been removed -- see
                    geometry/__init__.py
elements         : Gauss-Legendre integration engine + element formulations,
                    organized by family (base, solids, plates, beams, trusses,
                    contact)
loads            : load definitions (static, time-history, harmonic, PSD)
solver           : global assembly, boundary conditions, static/modal/dynamic
                    solve + generic nonlinear assembly + contact elements
nonlinear_solver : incremental Newton-Raphson drivers (geometric, material,
                    and contact/boundary nonlinearity)
postprocess      : derived response quantities (modal participation, PSD stats)
fields           : FEField -- the ndarray-subclass solution vector returned by the solvers, with
                    named access (U.component("uy", nodes="tip"), U.nodal, U.magnitude())
newton_options   : NewtonOptions -- shared tol/max_iter/line_search settings for the nonlinear drivers
batch            : batch.map(fn, items, n_jobs=...) for parameter sweeps and load cases
series           : FieldSeries -- multi-step results (load path, time history, modes) with one access style
linear_system    : ReducedSystem -- FESystem.form_linear_system(): constrained K, F plus recover()
export           : export.write_vtu / write_series -- ParaView (.vtu, .pvd) output of meshes and results
iterative_solvers: Wave 8 items 36-40 (docs/consolidated_future_roadmap.md) --
                    fill-reducing reordering, preconditioners, conjugate
                    gradients, classical stationary iterations, and geometric
                    multigrid for structured Quad4 grids. Independent of
                    FESystem's own direct-solve path -- opt-in, not a
                    replacement.
adaptivity       : Wave 8 items 41-45 -- a posteriori error estimation,
                    marking strategies, conforming (Rivara longest-edge)
                    local refinement, mesh-hierarchy tracking, and the
                    solution-driven adaptive refinement loop. Scoped to
                    Tri3PlaneStress -- see the module's own docstring.
visualization    : plotting helpers; the Gmsh-native geometry/mesh/result
                    screenshot renderer (visualization.gmsh_plot) has been
                    removed -- see visualization/__init__.py; mesh.py's
                    matplotlib plot_mesh_2d/plot_mesh_3d/plot_mesh_annotated
                    remain the supported path

Design principle: every axis of variation (new material law, new
geometry, new element type, new load type, new solution strategy) lives
behind a small registry or a subclass, so extending the package means
ADDING a function/class, not editing existing ones. See each module's
docstring for the specific extension point.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
from . import (material, damping, mesh, grading, build_mesh, geometry, elements, loads, solver,
               postprocess, nonlinear_solver, iterative_solvers, adaptivity)

# Flat re-exports: every script in this project imports its handful of
# needed names directly off `fea_engine` (`from fea_engine import
# Material, Quad4PlaneStress, rectangle_mesh, FESystem, ...`) rather
# than off each submodule -- this is that flat namespace.
# Gmsh-backed geometry/visualization support (formerly geometry.gmsh_engine
# and visualization.gmsh_plot) has been removed from this package entirely
# -- see geometry/__init__.py and visualization/__init__.py.
from .material import (Material, Section, Section3D, PlasticMaterial1D, D_plane_stress,
                        D_plane_strain, D_solid3d, D_mindlin_plate, EI_beam,
                        beam3d_rigidities, beam3d_mass_props, CONSTITUTIVE_REGISTRY,
                        PlasticMaterialJ2, NeoHookeanMaterial, j2_radial_return_3d,
                        j2_radial_return_plane_stress,
                        PlasticMaterialJ2Kinematic, j2_radial_return_3d_kinematic,
                        neo_hookean_pk2_stress, D_shell, shell_rho_matrix)
from .damping import RayleighDamping, ModalDamping, FieldDamping
from .mesh import (Mesh, MultiBlockMesh, line_mesh, rectangle_mesh,
                    rectangle_with_hole_mesh_quarter, rectangle_with_hole_mesh_quarter_full,
                    graded_partition, rectangle_mesh_from_partitions, weld_meshes,
                    mirror_mesh, extrude_mesh,
                    hole_in_rectangle_mesh, hole_in_rectangle_mesh_graded,
                    box_mesh, box_with_hole_mesh, plot_mesh_2d, plot_mesh_3d,
                    plot_mesh_annotated)
from .grading import (Hole, Fillet, EdgeBias, Notch, GradingPlan, MeshGradingError,
                       plan as grading_plan, graded_partition as grading_graded_partition,
                       growth_ratio_of_partition, grade_for_growth_ratio,
                       threshold_field_size, dist_max_for_growth_ratio,
                       DEFAULT_GROWTH_RATIO)
# NOTE: build_mesh() the function is deliberately NOT flattened to
# `fea_engine.build_mesh` -- that name is already the SUBMODULE (see the
# "Modules" list above), and rebinding it here would shadow one with the
# other depending on import order. Call it as `fea_engine.build_mesh.
# build_mesh(...)` or `from fea_engine.build_mesh import build_mesh`.
from .build_mesh import RectangleWithHole, ROTATIONAL_DOF_ELEMENTS
from .elements import (Element, gauss_product, Quad4PlaneStress, Hex8Solid3D,
                        Quad4MindlinPlate, Beam2DEulerBernoulli, TrussTL2D,
                        TrussPlastic2D, Beam2DCorotational, Beam2DReissner, GapContactPenalty,
                        Tri3PlaneStress, Tet4Solid3D, GapContactCurvedFriction,
                        Quad8PlaneStress, Hex20Solid3D, Tet10Solid3D,
                        Beam3DEulerBernoulli, Beam3DCorotational, Hex8PlasticJ2, Tet4NeoHookean,
                        Tet10SolidTL, Shell4MITC, Shell4MITCCorotational, Shell4Director,
                        Tri6PlaneStress, Hex8SolidBbar, Quad4PlasticJ2PlaneStress,
                        Hex8PlasticJ2Kinematic,
                        NodeToSegmentContact2D, NodeToSegmentContact2DFriction,
                        closest_point_on_segment_2d, find_contact_pairs_2d,
                        ELEMENT_REGISTRY)
from .solver import FESystem
from .fields import FEField
from .series import FieldSeries
from .newton_options import NewtonOptions
from . import batch, units, export, coefficients, recovery, boundary, convergence
from .linear_system import ReducedSystem
from .iterative_solvers import (
    reverse_cuthill_mckee, fill_in_count, permuted_solve,
    jacobi_preconditioner, ssor_preconditioner, incomplete_cholesky0,
    preconditioned_cg, jacobi_solve, gauss_seidel_solve, sor_solve,
    structured_quad_hierarchy, node_prolongation_matrix, dof_prolongation_matrix,
    restrict_prolongation_to_free_dofs, galerkin_coarse_operator,
    v_cycle, multigrid_solve,
)
from .adaptivity import (
    element_stresses, zz_recovery_estimator, jump_residual_estimator,
    fixed_fraction_marking, threshold_marking, equidistribution_marking,
    RefinementRecord, refine_triangle_mesh_longest_edge, adaptive_refine_solve,
)

__all__ = [
    "material", "damping", "mesh", "grading", "build_mesh", "geometry", "elements",
    "loads", "solver", "postprocess", "nonlinear_solver", "iterative_solvers", "adaptivity",
    "Material", "Section", "Section3D", "PlasticMaterial1D", "D_plane_stress", "D_plane_strain",
    "D_solid3d", "D_mindlin_plate", "EI_beam", "beam3d_rigidities", "beam3d_mass_props",
    "PlasticMaterialJ2", "NeoHookeanMaterial", "j2_radial_return_3d",
    "j2_radial_return_plane_stress", "neo_hookean_pk2_stress",
    "PlasticMaterialJ2Kinematic", "j2_radial_return_3d_kinematic",
    "D_shell", "shell_rho_matrix",
    "RayleighDamping", "ModalDamping", "FieldDamping",
    "CONSTITUTIVE_REGISTRY",
    "Mesh", "MultiBlockMesh", "line_mesh", "rectangle_mesh",
    "rectangle_with_hole_mesh_quarter", "rectangle_with_hole_mesh_quarter_full",
    "graded_partition", "rectangle_mesh_from_partitions", "weld_meshes",
    "mirror_mesh", "extrude_mesh",
    "hole_in_rectangle_mesh", "hole_in_rectangle_mesh_graded",
    "box_mesh", "box_with_hole_mesh", "plot_mesh_2d", "plot_mesh_3d", "plot_mesh_annotated",
    "Hole", "Fillet", "EdgeBias", "Notch", "GradingPlan", "MeshGradingError",
    "grading_plan", "grading_graded_partition", "growth_ratio_of_partition",
    "grade_for_growth_ratio", "threshold_field_size", "dist_max_for_growth_ratio",
    "DEFAULT_GROWTH_RATIO",
    "RectangleWithHole", "ROTATIONAL_DOF_ELEMENTS",
    "Element", "gauss_product", "Quad4PlaneStress", "Hex8Solid3D",
    "Quad4MindlinPlate", "Beam2DEulerBernoulli", "TrussTL2D", "TrussPlastic2D",
    "Beam2DCorotational", "Beam2DReissner", "GapContactPenalty", "Tri3PlaneStress", "Tet4Solid3D",
    "GapContactCurvedFriction", "Quad8PlaneStress", "Hex20Solid3D", "Tet10Solid3D",
    "Beam3DEulerBernoulli", "Beam3DCorotational", "Hex8PlasticJ2", "Tet4NeoHookean", "Tet10SolidTL",
    "Shell4MITC", "Shell4MITCCorotational", "Shell4Director",
    "Tri6PlaneStress", "Hex8SolidBbar", "Quad4PlasticJ2PlaneStress",
    "Hex8PlasticJ2Kinematic",
    "NodeToSegmentContact2D", "NodeToSegmentContact2DFriction",
    "closest_point_on_segment_2d", "find_contact_pairs_2d",
    "ELEMENT_REGISTRY",
    "FESystem", "FEField", "FieldSeries", "NewtonOptions", "batch", "units",
    "export", "ReducedSystem", "coefficients", "recovery", "boundary", "convergence",
    "reverse_cuthill_mckee", "fill_in_count", "permuted_solve",
    "jacobi_preconditioner", "ssor_preconditioner", "incomplete_cholesky0",
    "preconditioned_cg", "jacobi_solve", "gauss_seidel_solve", "sor_solve",
    "structured_quad_hierarchy", "node_prolongation_matrix", "dof_prolongation_matrix",
    "restrict_prolongation_to_free_dofs", "galerkin_coarse_operator",
    "v_cycle", "multigrid_solve",
    "element_stresses", "zz_recovery_estimator", "jump_residual_estimator",
    "fixed_fraction_marking", "threshold_marking", "equidistribution_marking",
    "RefinementRecord", "refine_triangle_mesh_longest_edge", "adaptive_refine_solve",
]

__version__ = "1.0.1"
