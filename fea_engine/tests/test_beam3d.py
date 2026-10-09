"""
test_beam3d.py -- validation for Beam3DEulerBernoulli (Module 18,
general-purpose extensions roadmap Phase 3).

Beam3DEulerBernoulli is built entirely from closed-form blocks (no
Gauss quadrature to converge), so every check here is an EXACT
comparison against an independent closed-form or an independently
re-derived reference implementation -- no discretization-error
tolerance is needed, only floating-point round-off (atol/rtol chosen
accordingly, typically 1e-9 or tighter).

Five independent lines of evidence, each targeting a different part of
the element (axis-aligned closed form / skew orientation & the general
rotation matrix / mass matrix physical totals / the flat-plane
reduction back to the already-validated Beam2DEulerBernoulli / a full
FESystem-assembled multi-element 3-D frame against an independently
written reference assembler):

1. test_axis_aligned_cantilever_matches_1d_closed_form -- axial,
   torsional, and both bending directions of a beam along global X,
   each checked against the elementary 1-D cantilever formula
   (PL/EA, TL/GJ, PL^3/3EI) for that loading alone.
2. test_skew_orientation_bending_matches_closed_form -- the same
   bending check repeated for a beam along a skew (non-axis-aligned)
   direction, load applied along one local principal axis and
   deflection decomposed back into the local frame -- this is the
   check that actually exercises _local_axes()'s general rotation
   construction (the axis-aligned case above has e1=global X, e2=Z,
   e3=-Y, i.e. R is just a permutation, not a general rotation).
3. test_mass_matrix_reproduces_physical_totals -- u^T M u for a rigid
   translation (no rotation) must equal rho*A*L exactly, and for a
   rigid twist about the beam axis must equal rho*Ip*L exactly; a
   consistent mass matrix does NOT have this property automatically
   (it holds only because the element is exactly rigid-body-exact by
   construction), so this is a genuine check, not a tautology.
4. test_flat_plane_reduction_matches_beam2d_exactly -- with the
   element's local (v, theta_z) block correctly extracted from global
   dof indices (accounting for the local-to-global axis mapping), it
   must equal Beam2DEulerBernoulli's stiffness/mass EXACTLY (0.0 diff,
   not just close) for the same EIz/rho*A, since both are literally
   the same closed-form Hermite formula.
5. test_full_3d_frame_matches_independent_assembler -- a 2-element
   L-shaped space frame assembled through FESystem is checked against
   a from-scratch, independently written reference element-stiffness
   function + manual scatter-assembly (not calling
   Beam3DEulerBernoulli or FESystem at all) -- this is the end-to-end
   check that the registry/FESystem wiring, not just the element in
   isolation, is correct.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from fea_engine import (Beam3DEulerBernoulli, Beam2DEulerBernoulli, Material,
                         Section3D, beam3d_rigidities, beam3d_mass_props, FESystem)
from fea_engine.elements.beams3d import _local_axes
from fea_engine.mesh import Mesh


@pytest.fixture
def props():
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    sec = Section3D(A=0.01, Iy=8e-6, Iz=6e-6, J=5e-6)
    rig = beam3d_rigidities(mat, sec)
    mprops = beam3d_mass_props(mat, sec)
    return mat, sec, rig, mprops


def _cantilever_free_block(beam, coords, rig):
    ke = beam.stiffness(coords, rig)
    free = list(range(6, 12))
    return ke[np.ix_(free, free)]


def test_axis_aligned_cantilever_matches_1d_closed_form(props):
    mat, sec, rig, mprops = props
    EA, GJ, EIy, EIz = rig
    L = 2.0
    beam = Beam3DEulerBernoulli()
    coords = np.array([[0, 0, 0], [L, 0, 0]], dtype=float)
    Kff = _cantilever_free_block(beam, coords, rig)

    def solve(F6):
        return np.linalg.solve(Kff, np.asarray(F6, dtype=float))

    u = solve([1000.0, 0, 0, 0, 0, 0])
    assert u[0] == pytest.approx(1000.0 * L / EA, rel=1e-12)

    u = solve([0, 0, 0, 500.0, 0, 0])
    assert u[3] == pytest.approx(500.0 * L / GJ, rel=1e-12)

    # global Y load: bends about local axis using EIy for this frame orientation
    u = solve([0, 1000.0, 0, 0, 0, 0])
    assert np.hypot(u[1], u[2]) == pytest.approx(1000.0 * L**3 / (3 * EIy), rel=1e-10)

    # global Z load: bends using EIz
    u = solve([0, 0, 1000.0, 0, 0, 0])
    assert np.hypot(u[1], u[2]) == pytest.approx(1000.0 * L**3 / (3 * EIz), rel=1e-10)


def test_skew_orientation_bending_matches_closed_form(props):
    mat, sec, rig, mprops = props
    EA, GJ, EIy, EIz = rig
    beam = Beam3DEulerBernoulli()
    X1 = np.array([0.0, 0.0, 0.0])
    X2 = np.array([1.0, 1.0, 1.0])   # skew, exercises the general rotation matrix
    L, e1, e2, e3 = _local_axes(X1, X2)
    coords = np.array([X1, X2])
    Kff = _cantilever_free_block(beam, coords, rig)

    F_global = 1000.0 * e2   # pure bending about local z -> governed by EIz
    F = np.zeros(6)
    F[0:3] = F_global
    u = np.linalg.solve(Kff, F)
    disp = u[0:3]

    exact = 1000.0 * L**3 / (3 * EIz)
    assert abs(disp @ e1) < 1e-9 * exact       # no axial component
    assert disp @ e2 == pytest.approx(exact, rel=1e-10)
    assert abs(disp @ e3) < 1e-9 * exact       # no out-of-plane component


def test_mass_matrix_reproduces_physical_totals(props):
    mat, sec, rig, mprops = props
    rho_A, rho_Ip = mprops
    L = 2.0
    beam = Beam3DEulerBernoulli()
    coords = np.array([[0, 0, 0], [L, 0, 0]], dtype=float)
    me = beam.mass(coords, mprops)

    for v in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
        u = np.zeros(12)
        u[0:3] = v
        u[6:9] = v   # both nodes translate rigidly by v, no rotation
        assert u @ me @ u == pytest.approx(rho_A * L, rel=1e-12)

    u = np.zeros(12)
    u[3] = 1.0
    u[9] = 1.0   # rigid twist about the beam axis
    assert u @ me @ u == pytest.approx(rho_Ip * L, rel=1e-12)


def test_flat_plane_reduction_matches_beam2d_exactly(props):
    mat, sec, rig, mprops = props
    EA, GJ, EIy, EIz = rig
    rho_A, rho_Ip = mprops
    L = 2.0
    beam = Beam3DEulerBernoulli()
    coords = np.array([[0, 0, 0], [L, 0, 0]], dtype=float)
    ke = beam.stiffness(coords, rig)
    me = beam.mass(coords, mprops)

    b2 = Beam2DEulerBernoulli()
    coords_1d = np.array([[0.0], [L]])
    k2 = b2.stiffness(coords_1d, EIz)
    m2 = b2.mass(coords_1d, rho_A)

    # For this beam (along global X, default ref_up), _local_axes gives
    # e1=X, e2=Z, e3=-Y -- so global (w, theta_y) at indices [2,4,8,10]
    # is the local (v, theta_z) block with a sign flip on the rotation
    # dofs (dw/dx = -theta_y is the local convention; theta_y_global =
    # -theta_z_local here since e3=-Y).
    idx = [2, 4, 8, 10]
    S = np.diag([1, -1, 1, -1])
    k_extracted = S @ ke[np.ix_(idx, idx)] @ S
    m_extracted = S @ me[np.ix_(idx, idx)] @ S

    assert np.array_equal(k_extracted, k2)  # bit-for-bit: same closed-form call
    assert np.array_equal(m_extracted, m2)


def test_full_3d_frame_matches_independent_assembler(props):
    mat, sec, rig, mprops = props
    EA, GJ, EIy, EIz = rig

    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1]], dtype=float)
    elements = np.array([[0, 1], [1, 2]])
    mesh = Mesh(nodes=nodes, elements=elements, dim=1)
    beam = Beam3DEulerBernoulli()
    fs = FESystem(mesh, beam, sparse=False)
    fs.assemble_stiffness(rig)

    ndof = nodes.shape[0] * 6
    fixed = list(range(0, 6))
    free_dofs = [i for i in range(ndof) if i not in fixed]
    F = np.zeros(ndof)
    F[6 * 2 + 1] = 1000.0
    Kff = fs.K[np.ix_(free_dofs, free_dofs)]
    uf = np.linalg.solve(Kff, F[free_dofs])
    u_full = np.zeros(ndof)
    u_full[free_dofs] = uf

    # Independent reference: a from-scratch element-stiffness function
    # and manual scatter-assembly, not calling Beam3DEulerBernoulli or
    # FESystem at all -- an end-to-end check on the registry/assembly
    # wiring, not just the element formulation in isolation.
    def local_beam_matrices(L, EA, GJ, EIy, EIz):
        Mloc = np.zeros((12, 12))
        Mloc[np.ix_([0, 6], [0, 6])] = (EA / L) * np.array([[1, -1], [-1, 1]])
        Mloc[np.ix_([3, 9], [3, 9])] = (GJ / L) * np.array([[1, -1], [-1, 1]])
        kb = lambda EI: (EI / L**3) * np.array([
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L**2, -6 * L, 2 * L**2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L**2, -6 * L, 4 * L**2]])
        Mloc[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] = kb(EIz)
        FLIP = np.diag([1, -1, 1, -1])
        Mloc[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] = FLIP @ kb(EIy) @ FLIP
        return Mloc

    def elem_stiffness(X1, X2, EA, GJ, EIy, EIz):
        dX = X2 - X1
        L = np.linalg.norm(dX)
        e1 = dX / L
        ref = np.array([0, 0, 1.0])
        if np.linalg.norm(np.cross(e1, ref)) < 1e-6:
            ref = np.array([0, 1.0, 0])
        e3 = np.cross(e1, ref)
        e3 /= np.linalg.norm(e3)
        e2 = np.cross(e3, e1)
        Mloc = local_beam_matrices(L, EA, GJ, EIy, EIz)
        R = np.vstack([e1, e2, e3])
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return T.T @ Mloc @ T

    K2 = np.zeros((18, 18))
    for (a, b) in [(0, 1), (1, 2)]:
        ke = elem_stiffness(nodes[a], nodes[b], EA, GJ, EIy, EIz)
        dofs = list(range(6 * a, 6 * a + 6)) + list(range(6 * b, 6 * b + 6))
        K2[np.ix_(dofs, dofs)] += ke

    assert np.array_equal(fs.K, K2)

    Kff2 = K2[np.ix_(free_dofs, free_dofs)]
    uf2 = np.linalg.solve(Kff2, F[free_dofs])
    u_full2 = np.zeros(ndof)
    u_full2[free_dofs] = uf2
    assert np.allclose(u_full, u_full2, atol=1e-12)


def test_rigid_body_modes_and_matrix_properties(props):
    mat, sec, rig, mprops = props
    beam = Beam3DEulerBernoulli()
    coords = np.array([[0, 0, 0], [2.0, 0, 0]], dtype=float)
    ke = beam.stiffness(coords, rig)
    me = beam.mass(coords, mprops)

    assert np.allclose(ke, ke.T)
    assert np.allclose(me, me.T)

    eig_k = np.linalg.eigvalsh(ke)
    n_zero = np.sum(np.abs(eig_k) < 1e-6 * np.abs(eig_k).max())
    assert n_zero == 6   # free 3-D beam: 3 translations + 3 rotations

    eig_m = np.linalg.eigvalsh(me)
    assert np.all(eig_m > 0)   # mass matrix positive-definite
