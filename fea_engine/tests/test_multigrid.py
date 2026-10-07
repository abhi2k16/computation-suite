"""
test_multigrid.py -- Wave 8 item 40 (docs/consolidated_future_roadmap.
md): geometric multigrid for structured Quad4 rectangle grids
(iterative_solvers.py). Scoped, as the module docstring states, to
mesh.rectangle_mesh()'s plain nx-by-ny grids -- see that docstring
for exactly why this pair of grids admits an EXACT nested hierarchy.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from fea_engine import FESystem, Quad4PlaneStress, D_plane_stress, Material
from fea_engine.iterative_solvers import (
    structured_quad_hierarchy, node_prolongation_matrix, dof_prolongation_matrix,
    galerkin_coarse_operator, restrict_prolongation_to_free_dofs,
    v_cycle, multigrid_solve, gauss_seidel_solve,
)


def _build_hierarchy(Lx, Ly, nx0, ny0, n_levels, E=1000.0, nu=0.3, t=1.0):
    mat = Material(E=E, nu=nu)
    D = D_plane_stress(mat)
    meshes = structured_quad_hierarchy(Lx, Ly, nx0, ny0, n_levels)
    npn = 2
    Ks, frees, fss = [], [], []
    for m in meshes:
        fs = FESystem(m, Quad4PlaneStress(), thickness=t, sparse=True)
        fs.assemble_stiffness(D)
        fixed_nodes = m.nodes_on_line(0, 0.0)
        fs.fix_dofs(fixed_nodes, [0, 1])
        Ks.append(sp.csr_matrix(fs.K))
        frees.append(fs.free_dofs)
        fss.append(fs)

    P_levels_free = []
    for k in range(n_levels - 1):
        nxc, nyc = nx0 * 2 ** k, ny0 * 2 ** k
        Pn = node_prolongation_matrix(nxc, nyc)
        Pd = dof_prolongation_matrix(Pn, npn)
        P_levels_free.append(restrict_prolongation_to_free_dofs(Pd, frees[k + 1], frees[k]))

    A_levels_free = [Ks[k][np.ix_(frees[k], frees[k])] for k in range(n_levels)]
    return meshes, Ks, frees, fss, P_levels_free, A_levels_free, npn, D


class TestGalerkinIdentity:
    def test_prolongation_reproduces_coarse_stiffness(self):
        """fem_implementation_lessons.md's own cited structural fact:
        I_{h,2h} K_h I_{2h,h} = K_{2h} exactly, for the right
        prolongation/restriction pair. Checked against an
        INDEPENDENTLY, directly-assembled coarse-mesh stiffness
        matrix -- not merely asserted."""
        meshes, Ks, frees, fss, P_levels_free, A_levels_free, npn, D = \
            _build_hierarchy(2.0, 1.0, 4, 2, 2)
        Pn = node_prolongation_matrix(4, 2)
        Pd = dof_prolongation_matrix(Pn, npn)
        A_coarse_galerkin = galerkin_coarse_operator(Ks[1], Pd)
        diff = abs(A_coarse_galerkin - Ks[0])
        scale = abs(Ks[0]).max()
        assert diff.max() < 1e-9 * scale


class TestVCycleAndMultigridSolve:
    def test_multigrid_matches_direct_solve(self):
        meshes, Ks, frees, fss, P_levels_free, A_levels_free, npn, D = \
            _build_hierarchy(2.0, 1.0, 4, 2, 4)
        fs_fine = fss[-1]
        tip_nodes = meshes[-1].nodes_on_line(0, 2.0)
        fs_fine.add_nodal_force(tip_nodes, 1, -100.0)
        b_free = fs_fine.F[frees[-1]]

        x_direct = sp.linalg.spsolve(A_levels_free[-1].tocsc(), b_free)
        x_mg, n_iter = multigrid_solve(A_levels_free, P_levels_free, b_free,
                                        tol=1e-10, maxiter=100)
        assert np.allclose(x_mg, x_direct, atol=1e-8)
        assert n_iter < 50

    def test_multigrid_beats_plain_gauss_seidel_on_the_fine_system(self):
        """The actual payoff item 40 exists for: "defeating the
        h-dependence" (fem_implementation_lessons.md Chapter 13's own
        framing) -- checked by putting plain Gauss-Seidel (item 39) up
        against multigrid on the IDENTICAL fine-level system. Plain
        Gauss-Seidel should need drastically more iterations (or fail
        to converge at all within a generous cap), while multigrid
        converges in a small, roughly mesh-size-INDEPENDENT number of
        V-cycles."""
        meshes, Ks, frees, fss, P_levels_free, A_levels_free, npn, D = \
            _build_hierarchy(2.0, 1.0, 4, 2, 4)
        fs_fine = fss[-1]
        tip_nodes = meshes[-1].nodes_on_line(0, 2.0)
        fs_fine.add_nodal_force(tip_nodes, 1, -100.0)
        b_free = fs_fine.F[frees[-1]]

        _, n_mg = multigrid_solve(A_levels_free, P_levels_free, b_free,
                                   tol=1e-8, maxiter=100)
        with pytest.raises(RuntimeError):
            gauss_seidel_solve(A_levels_free[-1], b_free, tol=1e-8, maxiter=3000)
        assert n_mg < 50

    def test_three_level_v_cycle_runs_and_converges(self):
        meshes, Ks, frees, fss, P_levels_free, A_levels_free, npn, D = \
            _build_hierarchy(1.0, 1.0, 2, 2, 3)
        fs_fine = fss[-1]
        top_nodes = meshes[-1].nodes_on_line(1, 1.0)
        fs_fine.add_nodal_force(top_nodes, 0, 10.0)
        b_free = fs_fine.F[frees[-1]]
        x_direct = sp.linalg.spsolve(A_levels_free[-1].tocsc(), b_free)
        x_mg, n_iter = multigrid_solve(A_levels_free, P_levels_free, b_free,
                                        tol=1e-9, maxiter=100)
        assert np.allclose(x_mg, x_direct, atol=1e-7)
