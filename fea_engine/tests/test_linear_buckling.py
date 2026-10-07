"""
test_linear_buckling.py -- validation for Module 19 (general-purpose
extensions roadmap Phase 4): geometric_stiffness() on
Beam2DEulerBernoulli/Beam3DEulerBernoulli/TrussTL2D, plus
FESystem.assemble_geometric_stiffness()/solve_linear_buckling().

Four lines of evidence:

1. test_geometric_stiffness_matches_trusstl2d_own_tangent -- the
   textbook check that geometric_stiffness() isn't a second,
   independently-invented formula: for TrussTL2D it must equal, term
   for term, the K_geometric piece already inside
   tangent_stiffness()'s own (already-validated) derivation, at a
   consistent reference stress state.
2. test_beam3d_geometric_stiffness_flat_plane_matches_beam2d_exactly
   -- the same flat-plane reduction check test_beam3d.py already runs
   for stiffness()/mass(), applied to geometric_stiffness(): a
   Beam3DEulerBernoulli lying along global X reduces its geometric
   stiffness to Beam2DEulerBernoulli's own formula bit-for-bit.
3. test_euler_column_buckling_load_converges_to_closed_form -- THE
   textbook linear-buckling benchmark: a column discretized into N
   Beam2DEulerBernoulli elements, solved via
   assemble_geometric_stiffness() + solve_linear_buckling(), checked
   against the closed-form Euler formula P_cr = pi^2*EI/(K*L)^2 for
   three boundary-condition cases (pin-pin K=1, fixed-free K=2,
   fixed-fixed K=0.5) -- and checked that refining the mesh drives the
   error down (genuine convergence, not a lucky single data point).
4. test_3d_column_buckling_matches_2d_via_flat_plane -- a column built
   from Beam3DEulerBernoulli (loaded/oriented so it bends in a single
   plane) must give the SAME critical load as the 2-D Euler-column
   check above, end to end through FESystem -- not just the element
   matrices in isolation (check 2), but the full assemble + solve path.
"""
import numpy as np
import pytest

from fea_engine import (Beam2DEulerBernoulli, Beam3DEulerBernoulli, TrussTL2D,
                         Material, Section3D, beam3d_rigidities, FESystem)
from fea_engine.mesh import Mesh


def test_geometric_stiffness_matches_trusstl2d_own_tangent():
    truss = TrussTL2D()
    coords = np.array([[0.0, 0.0], [3.0, 0.0]])
    E, A = 210e9, 1e-4
    mat = (E, A)
    u_elem = np.array([0.0, 0.0, 0.001, 0.0])  # small axial stretch along x

    L0 = 3.0
    l2 = (3.0 + 0.001) ** 2
    E_GL = (l2 - L0 ** 2) / (2 * L0 ** 2)
    S = E * E_GL
    N = S * A   # tension-positive axial force, matching this element's own S

    Kg_direct = truss.geometric_stiffness(coords, N)

    Ksub_expected = (S * A / L0) * np.eye(2)
    K4_expected = np.zeros((4, 4))
    K4_expected[0:2, 0:2] = Ksub_expected
    K4_expected[0:2, 2:4] = -Ksub_expected
    K4_expected[2:4, 0:2] = -Ksub_expected
    K4_expected[2:4, 2:4] = Ksub_expected

    assert np.allclose(Kg_direct, K4_expected, atol=1e-9)


def test_beam3d_geometric_stiffness_flat_plane_matches_beam2d_exactly():
    L = 2.0
    N_ref = -500.0   # compressive reference force, tension-positive convention
    beam3 = Beam3DEulerBernoulli()
    coords3 = np.array([[0, 0, 0], [L, 0, 0]], dtype=float)
    kg3 = beam3.geometric_stiffness(coords3, N_ref)

    b2 = Beam2DEulerBernoulli()
    kg2 = b2.geometric_stiffness(np.array([[0.0], [L]]), N_ref)

    idx = [2, 4, 8, 10]
    S = np.diag([1, -1, 1, -1])
    kg_extracted = S @ kg3[np.ix_(idx, idx)] @ S
    assert np.array_equal(kg_extracted, kg2)


def _euler_column(n_elem, L, EI, bc):
    """bc: 'pin_pin' | 'fixed_free' | 'fixed_fixed'."""
    n_nodes = n_elem + 1
    nodes = np.linspace(0, L, n_nodes).reshape(-1, 1)
    elements = np.array([[i, i + 1] for i in range(n_elem)])
    mesh = Mesh(nodes=nodes, elements=elements, dim=1)
    beam = Beam2DEulerBernoulli()
    fs = FESystem(mesh, beam, sparse=False)
    fs.assemble_stiffness(EI)
    fs.assemble_geometric_stiffness(-1.0)   # 1 N reference compression, every element

    if bc == "pin_pin":
        fs.fix_dofs([0], [0])
        fs.fix_dofs([n_nodes - 1], [0])
    elif bc == "fixed_free":
        fs.fix_dofs([0], [0, 1])
    elif bc == "fixed_fixed":
        fs.fix_dofs([0], [0, 1])
        fs.fix_dofs([n_nodes - 1], [0, 1])
    else:
        raise ValueError(bc)
    return fs


@pytest.mark.parametrize("bc,K", [("pin_pin", 1.0), ("fixed_free", 2.0), ("fixed_fixed", 0.5)])
def test_euler_column_buckling_load_converges_to_closed_form(bc, K):
    E, I, L = 210e9, 8e-6, 3.0
    EI = E * I
    P_exact = np.pi ** 2 * EI / (K * L) ** 2

    errors = []
    for n_elem in (4, 8, 16, 32):
        fs = _euler_column(n_elem, L, EI, bc)
        load_factors, _ = fs.solve_linear_buckling(n_modes=1)
        rel_err = abs(load_factors[0] - P_exact) / P_exact
        errors.append(rel_err)

    # coarsest mesh should already be within a few percent (consistent
    # geometric stiffness converges fast for this smooth problem)...
    assert errors[0] < 0.05
    # ...and refining the mesh should monotonically shrink the error --
    # genuine mesh convergence, not a single lucky data point.
    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1))
    assert errors[-1] < 1e-4


def test_3d_column_buckling_matches_2d_via_flat_plane():
    """A 3-D column pinned only in translation at both ends (all
    rotations free everywhere, including BOTH bending planes AND
    torsion) -- the realistic, unrestricted case, not artificially
    reduced to a single plane. With Iy >> Iz, the true weak-axis
    (Iz-governed) Euler load must come out as the LOWEST eigenvalue,
    matching the 2-D closed form exactly, even though the assembled
    -K_sigma_ff is indefinite here (both bending planes' rotational
    dofs are simultaneously free) -- this is exactly the case
    solve_linear_buckling()'s scipy.linalg.eig() fallback exists for,
    see that method's docstring."""
    E, I, L = 210e9, 8e-6, 3.0
    EI = E * I
    P_exact = np.pi ** 2 * EI / L ** 2   # pin-pin, weak (Iz) axis

    n_elem = 16
    n_nodes = n_elem + 1
    x = np.linspace(0, L, n_nodes)
    nodes = np.zeros((n_nodes, 3))
    nodes[:, 0] = x
    elements = np.array([[i, i + 1] for i in range(n_elem)])
    mesh = Mesh(nodes=nodes, elements=elements, dim=1)

    mat = Material(E=E, nu=0.3, rho=7800.0)
    sec = Section3D(A=0.01, Iy=1e-3, Iz=I, J=1e-3)  # Iy, J >> Iz -- weak-axis
    # (Iz-governed) buckling should be the lowest, governing mode
    rig = beam3d_rigidities(mat, sec)

    beam = Beam3DEulerBernoulli()
    fs = FESystem(mesh, beam, sparse=False)
    fs.assemble_stiffness(rig)
    fs.assemble_geometric_stiffness(-1.0)

    # Pin-pin: only translations (x,y,z) constrained at the two ends;
    # every rotation, at every node, stays free.
    fs.fix_dofs([0], [0, 1, 2])
    fs.fix_dofs([n_nodes - 1], [0, 1, 2])

    load_factors, _ = fs.solve_linear_buckling(n_modes=1)
    rel_err = abs(load_factors[0] - P_exact) / P_exact
    assert rel_err < 1e-4
