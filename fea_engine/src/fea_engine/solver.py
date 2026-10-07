"""
solver.py -- Module 4: global assembly, boundary conditions, and the
linear/eigenvalue solve.

FESystem is deliberately element-agnostic: it calls element.stiffness()
/ element.mass() and scatters the result using element.dofs_per_node --
it never asks "is this a Quad4 or a Hex8 or a beam?". That means any
element registered in element.ELEMENT_REGISTRY (including ones added
later) works here with zero changes to this file. This is the payoff
of the design in element.py: the "additive" property the whole package
is built around lands here, in the one place that would otherwise need
an if/elif per element type.

Module 14: mixed-element-type assembly. Every assemble_*/init_state/
commit_all_states method below loops over self._blocks, a list of
(block_name, element_formulation, connectivity_array) triples built
once in __init__(). The ORIGINAL single-element-type usage --
FESystem(mesh, elem_formulation) with a plain mesh.Mesh -- normalizes
to exactly one block (block_name=None, connectivity=mesh.elements),
so every method below is unchanged in BEHAVIOR for that case; it is
now just "the len(self._blocks)==1 special case" of a general loop,
not a separate code path. Passing a mesh.MultiBlockMesh together with
a DICT of {block_name: element_formulation} (matching mesh.blocks'
keys) instead builds one block per entry -- letting a mesh with, say,
Tri3PlaneStress AND Quad4PlaneStress elements (the case a Gmsh
"recombine" pass typically produces: quads where element pairing
succeeded, leftover triangles where it didn't -- see
geometry_engine.py) assemble through the exact same K/M/F_int/K_T as
every uniform mesh always has.

Scope, stated explicitly: every block sharing one FESystem must use
the SAME dofs_per_node (global DOFs are numbered per NODE, not per
block -- see _global_dofs()). This covers "mixed element TOPOLOGY,
same physics" (Tri3+Quad4 plane stress, Tet4+Hex8 solid, ...), which is
what an unstructured mesher's "leftover element" problem actually looks
like. It does NOT support mixing element FAMILIES with different
dofs_per_node in one mesh (e.g. a 2-dof/node plane-stress block with a
3-dof/node plate block) -- that is a genuinely different (multi-field)
problem, not a generalization of this mechanism. D/mat/rho arguments to
assemble_stiffness()/assemble_mass()/assemble_internal_force()/etc. may
themselves be a dict keyed the same way as the blocks (per-block
material/physics, e.g. different D matrices per block) or a single
plain value reused for every block (the common case: one physics, just
more than one element topology) -- see _per_block_arg().
"""
import numpy as np
from scipy.linalg import eigh


class FESystem:
    def __init__(self, mesh, elem_formulation, thickness=1.0, sparse=False,
                 backend="scipy", device="cpu"):
        """elem_formulation: either a single element.Element instance
        (the original usage -- mesh must be a plain mesh.Mesh), or a
        dict {block_name: element.Element instance} matching
        mesh.blocks' keys exactly (mesh must be a mesh.MultiBlockMesh).
        See this module's docstring for the mixed-element-type case.

        sparse=True (Module 18, general-purpose extensions Phase 2):
        store K/M/M_lumped as scipy.sparse.lil_matrix (efficient
        incremental element-by-element scatter-add, the SAME `+=`
        syntax the dense np.zeros() path already uses -- see
        docs/general_purpose_extensions_roadmap.md Section 7) instead
        of dense numpy arrays. This is a STORAGE-FORMAT change only:
        assemble_stiffness()/assemble_mass()/assemble_lumped_mass()
        are otherwise unchanged, and solve_static()/solve_modal()
        internally convert to CSR (scipy.sparse.linalg's preferred
        format for spsolve()/eigsh()) at solve time. Default is False
        (dense) -- every existing model/test in this package keeps
        using the original path unchanged; sparse=True is opt-in,
        for problems large enough that dense O(n_dof^2) storage/
        O(n_dof^3) solves become the bottleneck. NOT yet wired into
        the nonlinear (assemble_internal_force/assemble_tangent_
        stiffness) or transient-dynamics paths -- scoped to the
        static/modal case for this first pass; see the roadmap doc's
        Section 7 validation plan for why an additive, opt-in rollout
        was chosen over converting every existing path at once.

        backend="scipy" (default) or "torch" -- selects which linear-
        algebra engine solve_static() dispatches to. This is an
        explicit, user-facing CHOICE between two coexisting, fully
        maintained implementations, not a replacement of one by the
        other -- see torch_sparse_solver.py's module docstring and
        docs/consolidated_future_roadmap.md for why: SciPy's Cholesky/
        eigen-decomposition path (backend="scipy") is the default,
        battle-tested, CPU-only path with no extra dependency; the
        PyTorch path (backend="torch") reuses the exact solve logic
        already validated standalone in torch_sparse_solver.py's
        fesystem_solve_static_torch() (dense torch.linalg.solve() when
        sparse=False, Jacobi-preconditioned sparse CG when sparse=True),
        and additionally opens a GPU path via device="cuda" on a
        machine with a working CUDA-enabled torch install. Choosing
        backend="torch" requires torch to actually import successfully
        (see _HAS_TORCH in torch_sparse_solver.py) -- this is checked
        HERE, at construction time, not deferred to solve_static(),
        so a missing/broken torch install fails fast with a clear
        message rather than after assembly work has already been done
        (the same fail-fast-at-construction convention already used by
        geometry/gmsh_engine.py's UnifiedGeometryEngine for _HAS_GMSH).
        backend="torch" currently only affects solve_static() -- every
        other solve method (solve_modal(), solve_linear_buckling(),
        transient-dynamics drivers, ...) remains SciPy/NumPy-only
        regardless of this setting, since no torch implementation of
        those exists yet; this is intentional, not an oversight, and
        matches the additive/opt-in rollout philosophy noted above for
        sparse=True. device is passed straight through to torch (e.g.
        "cpu" or "cuda") and is ignored when backend="scipy".

        This backend= mechanism is the general pattern this package
        now follows whenever more than one valid implementation exists
        for the same capability: keep every implementation available
        and give the caller an explicit switch, rather than silently
        preferring or hiding one -- apply this same pattern to future
        additions, not just this solver.

        backend="auto" (Wave 11 item 108, docs/consolidated_future_
        roadmap.md): a THIRD option, alongside the two explicit choices
        above -- lets solve_static() pick "scipy" or "torch" itself,
        per call, based on Kff's own size and SPD-ness (see
        backend_dispatch.py's own module docstring for the exact rule
        and why it can never silently route a non-SPD system to CG).
        Unlike backend="torch", this does NOT fail fast here at
        construction -- whether torch ends up being needed depends on
        Kff, which doesn't exist until solve_static() assembles the
        system, so the torch-availability check (if "auto" ever does
        pick "torch") happens there instead. Pass verbose=True to
        solve_static() to see which backend "auto" picked and why."""
        if backend not in ("scipy", "torch", "auto"):
            raise ValueError(
                f"FESystem: unknown backend={backend!r} -- expected 'scipy' "
                f"(default, SciPy-based direct/eigen solve), 'torch' "
                f"(PyTorch-based dense/CG solve, see torch_sparse_solver.py), "
                f"or 'auto' (Wave 11 item 108 -- picks between the two at "
                f"solve_static() time, see backend_dispatch.py).")
        if backend == "torch":
            from .torch_sparse_solver import _require_torch
            _require_torch()
        self.backend = backend
        self.device = device
        self.mesh = mesh
        self.thickness = thickness
        self.sparse = sparse
        self.multi_block = isinstance(elem_formulation, dict)

        if self.multi_block:
            if not hasattr(mesh, "blocks"):
                raise TypeError(
                    "FESystem: elem_formulation is a dict (mixed-element-type mode) "
                    "but mesh has no .blocks -- pass a mesh.MultiBlockMesh, not a "
                    "plain mesh.Mesh, when using more than one element formulation.")
            missing = set(mesh.blocks) - set(elem_formulation)
            extra = set(elem_formulation) - set(mesh.blocks)
            if missing or extra:
                raise ValueError(
                    f"FESystem: elem_formulation keys {sorted(elem_formulation)} don't "
                    f"match mesh.blocks keys {sorted(mesh.blocks)} "
                    f"(missing={sorted(missing)}, extra={sorted(extra)})")
            npn_values = {name: e.dofs_per_node for name, e in elem_formulation.items()}
            if len(set(npn_values.values())) != 1:
                raise ValueError(
                    f"FESystem: every block must share the same dofs_per_node (global "
                    f"DOFs are numbered per NODE, not per block) -- got {npn_values}. "
                    f"Mixing element FAMILIES with different dofs_per_node (e.g. a "
                    f"plate's [w,betax,betay] with a plane-stress element's [u,v]) in "
                    f"one mesh needs a fundamentally different (multi-field) DOF "
                    f"numbering scheme, not this mechanism -- see the module docstring.")
            self.elem = elem_formulation   # dict, kept for inspection/parity
            self.npn = next(iter(npn_values.values()))
            self._blocks = [(name, elem_formulation[name], mesh.blocks[name])
                             for name in mesh.blocks]
        else:
            self.elem = elem_formulation
            self.npn = elem_formulation.dofs_per_node
            self._blocks = [(None, elem_formulation, mesh.elements)]

        self.n_dof = len(mesh.nodes) * self.npn
        self.K = self._zeros_matrix()
        self.M = None
        self.M_lumped = None
        self.K_sigma = None  # geometric ("stress stiffness") matrix, Module 19
                              # (general-purpose extensions Phase 4) -- None
                              # until assemble_geometric_stiffness() is called,
                              # same "None means not built yet" convention as
                              # self.M above; only solve_linear_buckling() reads it.
        self.C = None
        self.F = np.zeros(self.n_dof)
        self.fixed_dofs = set()
        self.fixed_dof_values = {}   # Wave 16 item 131 -- dof -> prescribed
                                      # value, populated by fix_dofs(value=...).
                                      # Missing entries mean 0.0 (the original,
                                      # homogeneous-only behavior). See fix_dofs()
                                      # and solve_static() docstrings for the
                                      # exact, deliberately narrow scope of what
                                      # honors this dict today.
        self.state = None   # per-element path-dependent state (plasticity,
                             # viscoelasticity, ...) -- None until init_state()
                             # is called; every method below treats None as
                             # "this system has no path-dependent elements",
                             # so nothing about the linear or geometrically-
                             # nonlinear-only usage above changes. Structured as
                             # {block_name: [state_or_None, ...]} once populated
                             # (block_name is always None in the single-block
                             # case) -- see init_state().
        self.iter_state = None   # per-element MIXED-FORMULATION internal
                             # unknown (Module 23 -- e.g. Shell4MITCCorotational's
                             # von Karman coupling stress resultant) -- a
                             # DELIBERATELY SEPARATE mechanism from self.state
                             # above, not a variant of it, because the two have
                             # genuinely different update timing: self.state
                             # only ever advances on a CONVERGED load step
                             # (commit_all_states(), called once per step); this
                             # advances on EVERY accepted Newton CORRECTION
                             # (update_iter_states(), called once per iteration
                             # by nonlinear_solver.py's drivers) -- an element
                             # using self.state for real path-dependent physics
                             # (TrussPlastic2D, Hex8PlasticJ2) is completely
                             # unaffected by this ever existing, since nothing
                             # here touches self.state. None until
                             # init_iter_state() is called; every method below
                             # treats None as "no element here uses this",
                             # unchanged from every existing usage. Structured
                             # exactly like self.state:
                             # {block_name: [iter_state_or_None, ...]}.
        self.contact_elements = []   # auxiliary (elem, node_ids, mat) triples,
                                      # see add_contact_element() -- lets a
                                      # contact/gap element (Module 10) act on
                                      # nodes of the MAIN mesh without the mesh
                                      # itself needing a second element type.
        self.contact_state = []      # per-contact-element path-dependent state
                                      # (Module 13: GapContactCurvedFriction needs
                                      # to remember its stick anchor between
                                      # iterations, the same way self.state does
                                      # for the main mesh -- see add_contact_element()
                                      # and commit_all_states(). A contact element
                                      # without init_state() (e.g. GapContactPenalty,
                                      # Module 10, which is stateless) simply gets
                                      # None here, silently ignored everywhere below,
                                      # exactly like self.state's per-element None.

    # -----------------------------------------------------------------
    def _zeros_matrix(self):
        """A fresh, empty (n_dof, n_dof) global matrix in whichever
        storage format this system uses -- scipy.sparse.lil_matrix
        (sparse=True) or a dense numpy array (sparse=False, the
        original/default behavior). LIL supports the same fancy-index
        `M[np.ix_(g, g)] += ke` scatter-add every assemble_*() method
        already uses for the dense case, so no assembly loop needs an
        if/else -- only this allocation call does."""
        if self.sparse:
            from scipy.sparse import lil_matrix
            return lil_matrix((self.n_dof, self.n_dof))
        return np.zeros((self.n_dof, self.n_dof))

    def _as_solve_matrix(self, M):
        """CSR is scipy.sparse.linalg's preferred format for
        spsolve()/eigsh() (fast matrix-vector products, efficient
        slicing) -- LIL is efficient to build (assembly) but slow to
        solve with, so every solve path converts once, right before
        the linear-algebra call, never during assembly."""
        return M.tocsr() if self.sparse else M

    def _global_dofs(self, elem_conn):
        npn = self.npn
        return np.array([npn * n + k for n in elem_conn for k in range(npn)])

    @staticmethod
    def _per_block_arg(arg, name):
        """If arg is a dict keyed by block name (per-block D/mat/rho --
        e.g. different constitutive matrices for different blocks),
        look up this block's own value; otherwise use arg UNCHANGED for
        every block -- the common case, one physics/material shared by
        every block regardless of how many element topologies are
        mixed in. D/mat/rho are always tuples/arrays/dataclasses in
        this package, never plain dicts, so isinstance(arg, dict) is an
        unambiguous discriminator."""
        if isinstance(arg, dict) and name in arg:
            return arg[name]
        return arg

    # -----------------------------------------------------------------
    def assemble_stiffness(self, D, method="full", vectorized=False, **kwargs):
        """kwargs are passed straight through to the chosen element
        method -- e.g. thickness=t for Quad4PlaneStress, or gauss_order=
        for a one-off integration-order override. Any element-specific
        keyword the formulation needs can be added without touching
        this method. D may be a single value shared by every block, or
        a dict keyed by block name for per-block materials/physics --
        see _per_block_arg().

        method (Wave 2 item 12, docs/consolidated_future_roadmap.md):
        which of Element's stiffness variants (elements/base.py) to
        call for EVERY element in this system -- "full" (default,
        ORIGINAL behavior: formulation.stiffness(elem_coords, D,
        **kwargs), so passing gauss_order= explicitly still works
        exactly as it always has), "reduced" (formulation.
        reduced_stiffness(...) -- relieves locking, but see that
        method's own docstring for why it can be rank-deficient), or
        "hourglass_stabilized" (formulation.hourglass_stabilized_
        stiffness(...) -- reduced integration PLUS a rank-restoring
        perturbation stiffness, safe to actually deploy; see elements/
        base.py's own extensive docstring for the derivation). All
        three remain available side by side -- this is an explicit,
        opt-in CHOICE, matching the same principle FESystem's own
        backend= parameter established for the SciPy-vs-PyTorch solve
        choice: never silently prefer or replace one implementation.

        vectorized (Wave 11 item 107, docs/consolidated_future_roadmap.md):
        when True, ADDS this call's stiffness contribution via
        vectorized_assembly.assemble_stiffness_vectorized() instead of
        the per-element Python loop below -- a tensorized, no-Python-
        loop-over-elements path built on item 106's MeshTransformation,
        for the homogeneous continuum-solid/plane-stress element types
        that module supports (see its own Scope docstring; anything
        else raises ValueError rather than silently falling back).
        Requires method="full" (the only variant vectorized_assembly
        implements) and passes only D/gauss_order/thickness through --
        other **kwargs are not supported on this path and raise
        TypeError, the same way an unrecognized kwarg would on the
        plain per-element path. Numerically identical (to floating-
        point solve-vs-solve agreement, not merely "close") to the
        default per-element loop -- see tests/test_vectorized_assembly.py.

        chunk_size (Wave 16 item 129, docs/consolidated_future_roadmap.md):
        only meaningful with vectorized=True -- an optional memory-
        chunking knob threaded straight through to
        vectorized_assembly.assemble_stiffness_vectorized() (see that
        function's own docstring). None (default) is the original,
        unchunked single-pass behavior; a positive int caps how many
        elements' worth of precomputed tensors are held in memory at
        once, with no change to the assembled result.

        backend ("numpy" default, or "torch"), device ("cpu" default,
        or "cuda") -- Wave 9 addendum item 136, docs/consolidated_
        future_roadmap.md: only meaningful with vectorized=True,
        threaded straight through to vectorized_assembly.assemble_
        stiffness_vectorized()'s own backend=/device= (which in turn
        thread through to MeshTransformation, item 135). Note this is
        INDEPENDENT of this FESystem's own constructor-time backend=
        (solver.py __init__, Wave 0 item 3) -- that one selects scipy
        vs. torch for the LINEAR SOLVE step (solve_static()); this one
        selects numpy vs. torch for the ASSEMBLY step. The two may be
        set differently (e.g. assemble on torch/GPU, solve on scipy)
        since assembly always writes a plain numpy/scipy self.K
        regardless of which backend built it (see scatter_global_
        stiffness()'s own docstring)."""
        if method not in ("full", "reduced", "hourglass_stabilized"):
            raise ValueError(
                f"assemble_stiffness: unknown method={method!r} -- expected "
                f"'full' (default), 'reduced', or 'hourglass_stabilized'.")
        if vectorized:
            if method != "full":
                raise ValueError(
                    f"assemble_stiffness: vectorized=True only supports "
                    f"method='full' (got method={method!r}) -- see "
                    f"vectorized_assembly.py's own Scope docstring.")
            from .vectorized_assembly import assemble_stiffness_vectorized
            gauss_order = kwargs.pop("gauss_order", None)
            thickness = kwargs.pop("thickness", 1.0)
            chunk_size = kwargs.pop("chunk_size", None)
            assembly_backend = kwargs.pop("backend", "numpy")
            assembly_device = kwargs.pop("device", "cpu")
            if kwargs:
                raise TypeError(
                    f"assemble_stiffness: vectorized=True does not support "
                    f"kwargs {list(kwargs)} -- only gauss_order=/thickness=/"
                    f"chunk_size=/backend=/device= are threaded through this path.")
            assemble_stiffness_vectorized(self, D, gauss_order=gauss_order, thickness=thickness,
                                           chunk_size=chunk_size, backend=assembly_backend,
                                           device=assembly_device)
            return
        for name, formulation, connectivity in self._blocks:
            block_D = self._per_block_arg(D, name)
            for elem_conn in connectivity:
                elem_coords = self.mesh.nodes[elem_conn]
                if method == "full":
                    ke = formulation.stiffness(elem_coords, block_D, **kwargs)
                elif method == "reduced":
                    ke = formulation.reduced_stiffness(elem_coords, block_D, **kwargs)
                else:
                    ke = formulation.hourglass_stabilized_stiffness(elem_coords, block_D, **kwargs)
                g = self._global_dofs(elem_conn)
                self.K[np.ix_(g, g)] += ke

    def assemble_mass(self, rho_or_matrix, **kwargs):
        if self.M is None:
            self.M = self._zeros_matrix()
        for name, formulation, connectivity in self._blocks:
            block_rho = self._per_block_arg(rho_or_matrix, name)
            for elem_conn in connectivity:
                elem_coords = self.mesh.nodes[elem_conn]
                me = formulation.mass(elem_coords, block_rho, **kwargs)
                g = self._global_dofs(elem_conn)
                self.M[np.ix_(g, g)] += me

    def assemble_lumped_mass(self, rho_or_matrix, **kwargs):
        """Element-by-element HRZ lumping (element.Element.lumped_mass())
        -- required before solve_transient_explicit(). Kept as a
        separate assembly from assemble_mass() (rather than derived
        from the already-assembled consistent self.M) because HRZ needs
        each element's OWN total mass and translational/rotational DOF
        split, which isn't recoverable from the global matrix alone."""
        if self.M_lumped is None:
            self.M_lumped = self._zeros_matrix()
        for name, formulation, connectivity in self._blocks:
            block_rho = self._per_block_arg(rho_or_matrix, name)
            for elem_conn in connectivity:
                elem_coords = self.mesh.nodes[elem_conn]
                me = formulation.lumped_mass(elem_coords, block_rho, **kwargs)
                g = self._global_dofs(elem_conn)
                self.M_lumped[np.ix_(g, g)] += me

    @staticmethod
    def _per_element_arg(arg, i):
        """Like _per_block_arg() above, but one level finer: within a
        block, the reference axial force N that
        assemble_geometric_stiffness() needs generally varies element
        by element (e.g. a frame's members don't all carry the same
        axial force under one reference load, even though they all
        share one material/section). A plain scalar (int/float) is
        broadcast to every element in the block -- the common case for
        a single straight column, where equilibrium makes N the SAME
        in every element -- while an array-like of length
        len(connectivity) supplies each element's own value, indexed
        by its position i in that block's connectivity (the same
        per-element indexing assemble_internal_force()/assemble_
        tangent_stiffness() already use for block_states[i])."""
        if np.isscalar(arg):
            return arg
        return arg[i]

    def assemble_geometric_stiffness(self, N, **kwargs):
        """Module 19 (general-purpose extensions Phase 4): global
        geometric ("stress stiffness") matrix K_sigma, built from a
        REFERENCE axial force state N (TENSION-POSITIVE, see Element.
        geometric_stiffness's docstring) rather than from mass or a
        constitutive matrix -- otherwise the SAME per-block, per-
        element scatter-assembly loop assemble_stiffness()/
        assemble_mass() already use. N may be a single scalar (shared
        by every element in every block -- the common case: a
        statically determinate column under a pure reference axial
        load has the SAME axial force in every element by
        equilibrium), a dict keyed by block name (per-block, itself
        scalar-or-array, see _per_block_arg()), or a plain array-like
        of length len(connectivity) (per-element within a single-
        block system) -- see _per_element_arg(). Elements with no
        geometric_stiffness() override (see Element's default) raise
        NotImplementedError here, surfacing which element type is
        missing it rather than silently contributing zero."""
        self.K_sigma = self._zeros_matrix()
        for name, formulation, connectivity in self._blocks:
            block_N = self._per_block_arg(N, name)
            for i, elem_conn in enumerate(connectivity):
                elem_coords = self.mesh.nodes[elem_conn]
                N_elem = self._per_element_arg(block_N, i)
                kg = formulation.geometric_stiffness(elem_coords, N_elem, **kwargs)
                g = self._global_dofs(elem_conn)
                self.K_sigma[np.ix_(g, g)] += kg

    # -----------------------------------------------------------------
    # Nonlinear assembly (Module 8: geometric nonlinearity). These are
    # the nonlinear counterparts of assemble_stiffness() above: instead
    # of building ONE global K, they build the internal-force vector
    # and tangent stiffness AT A GIVEN DISPLACEMENT STATE u_global,
    # calling element.internal_force()/tangent_stiffness() per element
    # exactly the way assemble_stiffness() calls element.stiffness() --
    # same element-agnostic loop, same "FESystem never asks what kind
    # of element this is" property. nonlinear_solver.py calls these
    # once per Newton-Raphson iteration; they do not touch self.K/self.F,
    # so a linear analysis on the same mesh/FESystem is unaffected.
    # -----------------------------------------------------------------
    def assemble_internal_force(self, u_global, mat, **kwargs):
        """Global internal-force vector F_int(u_global). mat is passed
        straight through to element.internal_force() (e.g. (E, A) for
        TrussTL2D) -- same homogeneous-material convention as D in
        assemble_stiffness() (may be a dict per-block, see
        _per_block_arg()). If init_state() has been called, each
        element also receives its own CONVERGED (last-committed) state
        as a state= kwarg; a path-dependent element (e.g.
        TrussPlastic2D) uses it to compute a TRIAL response for the
        current, not-yet-converged u_global without mutating anything
        -- see commit_all_states(). Also sums in every contact element
        added via add_contact_element(), each with its OWN mat (not
        the mat argument here, which is the MAIN mesh's material) --
        see that method's docstring."""
        F_int = np.zeros(self.n_dof)
        for name, formulation, connectivity in self._blocks:
            block_mat = self._per_block_arg(mat, name)
            block_states = self.state[name] if self.state is not None else None
            block_iter_states = self.iter_state[name] if self.iter_state is not None else None
            for i, elem_conn in enumerate(connectivity):
                elem_coords = self.mesh.nodes[elem_conn]
                g = self._global_dofs(elem_conn)
                u_elem = u_global[g]
                call_kwargs = dict(kwargs)
                if block_states is not None:
                    call_kwargs["state"] = block_states[i]
                if block_iter_states is not None:
                    call_kwargs["iter_state"] = block_iter_states[i]
                f_elem = formulation.internal_force(elem_coords, u_elem, block_mat, **call_kwargs)
                F_int[g] += f_elem
        for i, (celem, node_ids, cmat) in enumerate(self.contact_elements):
            elem_coords = self.mesh.nodes[node_ids]
            g = self._global_dofs(node_ids)
            u_elem = u_global[g]
            c_kwargs = dict(kwargs)
            c_kwargs["state"] = self.contact_state[i]
            F_int[g] += celem.internal_force(elem_coords, u_elem, cmat, **c_kwargs)
        return F_int

    def assemble_tangent_stiffness(self, u_global, mat, **kwargs):
        """Global tangent stiffness K_T(u_global) = d(F_int)/d(u), at
        the current displacement state -- must be reassembled every
        Newton-Raphson iteration (unlike the linear self.K, which is
        assembled once). Same state= passthrough, per-block mat, and
        contact-element summation as assemble_internal_force()."""
        K_T = np.zeros((self.n_dof, self.n_dof))
        for name, formulation, connectivity in self._blocks:
            block_mat = self._per_block_arg(mat, name)
            block_states = self.state[name] if self.state is not None else None
            block_iter_states = self.iter_state[name] if self.iter_state is not None else None
            for i, elem_conn in enumerate(connectivity):
                elem_coords = self.mesh.nodes[elem_conn]
                g = self._global_dofs(elem_conn)
                u_elem = u_global[g]
                call_kwargs = dict(kwargs)
                if block_states is not None:
                    call_kwargs["state"] = block_states[i]
                if block_iter_states is not None:
                    call_kwargs["iter_state"] = block_iter_states[i]
                k_elem = formulation.tangent_stiffness(elem_coords, u_elem, block_mat, **call_kwargs)
                K_T[np.ix_(g, g)] += k_elem
        for i, (celem, node_ids, cmat) in enumerate(self.contact_elements):
            elem_coords = self.mesh.nodes[node_ids]
            g = self._global_dofs(node_ids)
            u_elem = u_global[g]
            c_kwargs = dict(kwargs)
            c_kwargs["state"] = self.contact_state[i]
            K_T[np.ix_(g, g)] += celem.tangent_stiffness(elem_coords, u_elem, cmat, **c_kwargs)
        return K_T

    def add_contact_element(self, elem, node_ids, mat):
        """Registers a contact/constraint element (e.g.
        element.GapContactPenalty) acting on node_ids of the MAIN mesh,
        with its own mat (e.g. (k_p, g0, n_hat)) -- separate from the
        structural mat passed into assemble_internal_force()/
        solve_nonlinear_static(), since a real problem may have several
        contact pairs with different gaps/directions/stiffnesses.
        Nothing else needs to change: nonlinear_solver.py's existing
        drivers already sum contact contributions automatically (via
        assemble_internal_force()/assemble_tangent_stiffness() above),
        the same additive payoff Modules 8-9 already demonstrated.

        If elem defines init_state() (Module 13:
        GapContactCurvedFriction, which must remember its stick anchor
        between iterations -- Module 10's GapContactPenalty is
        stateless and doesn't), a fresh state slot is created in
        self.contact_state, parallel to self.contact_elements, and
        advanced by commit_all_states() exactly like the main mesh's
        self.state."""
        self.contact_elements.append((elem, list(node_ids), mat))
        self.contact_state.append(elem.init_state() if hasattr(elem, "init_state") else None)

    # -----------------------------------------------------------------
    # Path-dependent element state (Module 9: material nonlinearity --
    # plasticity, and later viscoelasticity/creep, both need to
    # remember history, not just the current total strain). The
    # convention: internal_force()/tangent_stiffness() above ALWAYS
    # compute a TRIAL response from the last-COMMITTED state plus the
    # current (possibly not-yet-equilibrated) u_global -- they never
    # mutate self.state themselves. Only once a Newton-Raphson load
    # step has actually converged does commit_all_states() advance
    # self.state to that trial response, permanently. This mirrors
    # standard FE plasticity practice (return-map every iteration,
    # commit only at converged load steps) and is why
    # nonlinear_solver.py calls commit_all_states() once per step, not
    # once per iteration.
    # -----------------------------------------------------------------
    def init_state(self):
        """Call once, before the first nonlinear solve, for any system
        that uses a path-dependent element. Elements without state
        (everything before Module 9, e.g. TrussTL2D) simply don't
        define init_state(), so they get None -- harmless, and
        assemble_internal_force()/assemble_tangent_stiffness() above
        still pass state=None through to them (silently ignored, since
        every element's signature accepts **kwargs). self.state is
        {block_name: [state_or_None, ...]} -- block_name is always
        None in the ordinary single-element-type case, so this is a
        one-entry dict there, not a behavior change from before this
        module existed (nothing outside this file indexes self.state
        directly -- see the mixed-element-assembly audit that preceded
        this feature)."""
        self.state = {
            name: [formulation.init_state() if hasattr(formulation, "init_state") else None
                   for _ in connectivity]
            for name, formulation, connectivity in self._blocks
        }

    def commit_all_states(self, u_global, mat, **kwargs):
        """Advance self.state to the converged response at u_global --
        call this once per load/displacement step, AFTER Newton-Raphson
        has converged, never mid-iteration. No-op (does nothing) if
        init_state() was never called, or for any element that doesn't
        define commit_state() (i.e. every stateless element). mat may
        be a dict per-block, see _per_block_arg()."""
        if self.state is None:
            return
        for name, formulation, connectivity in self._blocks:
            if not hasattr(formulation, "commit_state"):
                continue
            block_mat = self._per_block_arg(mat, name)
            block_states = self.state[name]
            for i, elem_conn in enumerate(connectivity):
                if block_states[i] is None:
                    continue
                elem_coords = self.mesh.nodes[elem_conn]
                g = self._global_dofs(elem_conn)
                u_elem = u_global[g]
                block_states[i] = formulation.commit_state(elem_coords, u_elem, block_mat,
                                                             block_states[i], **kwargs)

    # -----------------------------------------------------------------
    # Mixed-formulation internal state (Module 23). A DELIBERATELY
    # SEPARATE mechanism from init_state()/commit_all_states() above --
    # see self.iter_state's own docstring (in __init__) for why the two
    # are not variants of each other. Motivated directly by
    # Shell4MITCCorotational's von Karman bending-membrane coupling term
    # (docs/shells.md Section 4.3, dead end 4): a
    # DISPLACEMENT-ONLY quartic coupling energy makes Newton's tangent go
    # locally indefinite once the membrane/bending stiffness ratio is
    # large (documented, reproduced, and traced to its root cause in
    # shells.py's own "Design history" comment and a matching literature
    # result, Magisano/Leonetti/Garcea IJNME 2017); the fix is a MIXED
    # (Hellinger-Reissner-style) reformulation where the coupling term's
    # own stress resultant becomes an independent per-element unknown,
    # corrected by only its OWN SHARE of each Newton iteration's
    # linearized step -- not resolved exactly from the current
    # displacement every call, which was verified (both algebraically
    # and by direct experiment) to collapse back to the ORIGINAL
    # ill-conditioned formulation with zero benefit. This needs the
    # ACTUAL Newton correction delta_u a driver's own linear solve just
    # produced, which no earlier extension point in this file exposes.
    # -----------------------------------------------------------------
    def init_iter_state(self):
        """Call once, before the first nonlinear solve, for any system
        that uses a mixed-formulation element. Elements without this
        need (everything that doesn't define init_iter_state()) simply
        get None -- harmless, and assemble_internal_force()/
        assemble_tangent_stiffness() above only add the iter_state=
        kwarg when self.iter_state is not None, so calling this is a
        genuine opt-in with zero effect on any element that doesn't ask
        for it. Structured exactly like init_state()'s self.state:
        {block_name: [iter_state_or_None, ...]}."""
        self.iter_state = {
            name: [formulation.init_iter_state() if hasattr(formulation, "init_iter_state") else None
                   for _ in connectivity]
            for name, formulation, connectivity in self._blocks
        }

    def update_iter_states(self, u_global, delta_u_global, mat, **kwargs):
        """Advance self.iter_state using the REALIZED Newton correction
        delta_u_global a driver's own linear solve just produced --
        call this EVERY Newton iteration (not just on convergence, the
        opposite timing from commit_all_states()), right after updating
        u_global with delta_u_global and BEFORE the next residual/
        tangent assembly. No-op if init_iter_state() was never called,
        or for any element that doesn't define update_iter_state() (the
        overwhelming majority -- this only matters to elements that
        opted in). mat may be a dict per-block, see _per_block_arg()."""
        if self.iter_state is None:
            return
        for name, formulation, connectivity in self._blocks:
            if not hasattr(formulation, "update_iter_state"):
                continue
            block_mat = self._per_block_arg(mat, name)
            block_iter_states = self.iter_state[name]
            for i, elem_conn in enumerate(connectivity):
                g = self._global_dofs(elem_conn)
                elem_coords = self.mesh.nodes[elem_conn]
                u_elem = u_global[g]
                delta_u_elem = delta_u_global[g]
                block_iter_states[i] = formulation.update_iter_state(
                    elem_coords, u_elem, delta_u_elem, block_mat,
                    block_iter_states[i], **kwargs)

    def assemble_damping(self, damping, **kwargs):
        """damping: a damping.RayleighDamping instance, a
        damping.FieldDamping instance (Wave 17 item 141, docs/
        consolidated_future_roadmap.md -- Georgiou 2005's single-
        scalar-per-field model, NOT mass/stiffness-proportional), or a
        raw damping matrix (plain ndarray, or scipy.sparse matrix when
        self.sparse) added to self.C directly -- the "today fesystem.C
        can only be set by hand" escape hatch the roadmap explicitly
        asks to keep available alongside the two named damping types.
        Requires assemble_mass() to have been called first for
        RayleighDamping/FieldDamping (and, implicitly, assemble_
        stiffness() for RayleighDamping's beta*K term). damping.
        ModalDamping doesn't go through here -- it's used directly by
        solve_modal_superposition(), which never needs a global C
        matrix at all.

        ADDITIVE, not replacing (Wave 17 item 141's own explicit
        instruction, matching this project's standing "never silently
        prefer or replace one implementation" principle, applied here
        to damping SOURCES rather than backends): each call ADDS its
        contribution to self.C rather than overwriting it, so e.g.
        RayleighDamping (mass/stiffness-proportional numerical
        damping) and FieldDamping (a genuine physical per-field
        damping mechanism) can both be included by calling this twice.
        A single call reproduces the original (pre-Wave-17) overwrite
        behavior exactly, since self.C starts at None -> zeros.
        kwargs are passed through to FieldDamping.assemble() (which
        forwards them to each element's own mass()), unused for the
        other two damping types."""
        from .damping import RayleighDamping, FieldDamping
        if self.C is None:
            self.C = self._zeros_matrix()
        if isinstance(damping, RayleighDamping):
            assert self.M is not None, "call assemble_mass() before assemble_damping()"
            assert self.K is not None, "call assemble_stiffness() before assemble_damping() with RayleighDamping"
            self.C = self.C + damping.alpha * self.M + damping.beta * self.K
        elif isinstance(damping, FieldDamping):
            assert self.M is not None, "call assemble_mass() before assemble_damping()"
            self.C = self.C + damping.assemble(self, **kwargs)
        else:
            # Raw matrix escape hatch: add it in directly.
            self.C = self.C + damping

    # -----------------------------------------------------------------
    def fix_dofs(self, node_ids, dof_indices, value=0.0):
        """dof_indices: which local DOF(s) at each node to constrain
        (0-based, element-formulation-specific, e.g. 0,1 for u,v).

        value (Wave 16 item 131, docs/consolidated_future_roadmap.md,
        source: TensorMesh's `Boundary Conditions` page's non-
        homogeneous `Condenser(mask, values)` case): the prescribed
        displacement at every (node, dof_index) pair this call fixes.
        Default 0.0 reproduces the original, homogeneous-only
        behavior exactly. A single scalar applies to every (node,
        dof_index) pair in this call; call fix_dofs() again (it's
        additive, like every other call here) if different nodes/DOFs
        need different nonzero values.

        IMPORTANT SCOPE NOTE: a nonzero `value` is currently honored
        ONLY by `solve_static()`'s default (`backend="scipy"`, the
        default) and `backend="auto"` paths, via the standard static-
        condensation RHS correction `F_free -= K[free, fixed] @
        u_fixed` -- the direct analog of TensorMesh's own `Condenser`
        RHS term. `solve_static(backend="torch")` raises a specific,
        actionable error if any nonzero value is present (rather than
        silently solving the homogeneous system and returning a wrong
        answer) since that path's own correction isn't implemented
        yet. EVERY OTHER consumer of `fixed_dofs` in this codebase --
        every nonlinear driver (`solve_nonlinear_static()` and its
        siblings in nonlinear_solver.py), every transient driver, and
        `mixed_assembly.py`'s own `fixed_u_dofs` convention -- does
        NOT read `fixed_dof_values` at all and will silently treat a
        DOF fixed via `value=nonzero` as fixed at ZERO, exactly as
        before this item existed. This is a DELIBERATELY narrow first
        step (the plain linear static path only, matching what the
        TensorMesh comparison this item is sourced from actually
        looked at) -- see this method's caller-facing warning repeated
        in solve_static()'s own docstring, and the module docstring's
        own precedent (Wave 4/Wave 11's "reuses two already-validated
        pieces rather than touching everything at once") for why a
        narrower, honestly-scoped addition was chosen over a riskier
        codebase-wide sweep. A genuinely non-zero prescribed
        displacement in a NONLINEAR model should still go through
        `solve_nonlinear_displacement_control()`, which predates this
        item and remains the supported mechanism there."""
        for n in node_ids:
            for k in dof_indices:
                dof = self.npn * int(n) + k
                self.fixed_dofs.add(dof)
                self.fixed_dof_values[dof] = float(value)

    def add_nodal_force(self, node_ids, dof_index, total_force):
        """Splits total_force evenly across node_ids at local DOF
        dof_index -- the same simplified load-lumping convention used
        throughout this project's earlier scripts."""
        share = total_force / len(node_ids)
        for n in node_ids:
            self.F[self.npn * int(n) + dof_index] += share

    def add_consistent_edge_load(self, node_pairs, dof_index, traction, thickness=1.0):
        """Consistent nodal load for a uniform traction along a chain of
        2-node edge segments (node_pairs = [(n0,n1), (n1,n2), ...]),
        weighted by each segment's actual length -- more accurate than
        add_nodal_force() when node spacing is non-uniform (e.g. a
        graded mesh).

        Straight, LINEAR (2-node) segments only -- see add_consistent_
        facet_load() below (Wave 15 item 127) for curved/quadratic
        edges (Tri6/Quad8) and 3-D element faces (Tet4/Tet10/Hex8/
        Hex20), via real Gauss quadrature instead of this method's own
        closed-form 2-node formula."""
        for n0, n1 in node_pairs:
            length = np.linalg.norm(self.mesh.nodes[n1] - self.mesh.nodes[n0])
            share = traction * thickness * length / 2.0
            self.F[self.npn * int(n0) + dof_index] += share
            self.F[self.npn * int(n1) + dof_index] += share

    def add_consistent_facet_load(self, formulation, facets, dof_index, traction,
                                   quad_order=2, thickness=1.0):
        """Wave 15 item 127 (docs/consolidated_future_roadmap.md,
        source: TensorMesh's "Elements and Quadrature" documentation
        page's facet_quadrature()/nanson_scale() machinery): the
        generalization of add_consistent_edge_load() above to curved/
        quadratic edges (Tri6PlaneStress/Quad8PlaneStress) and 3-D
        element faces (Tet4Solid3D/Tet10Solid3D/Hex8Solid3D/
        Hex20Solid3D), via facet_loads.py's real Gauss-quadrature
        integration restricted to each facet, instead of this method's
        sibling's own closed-form straight-2-node-edge formula.

        formulation: the Element instance whose facet topology
        (facet_loads.list_facets(formulation)) defines the valid
        facet node orderings for THIS element type.
        facets: list of GLOBAL node-id tuples, one per loaded facet
        occurrence, each already in the corner-then-midside order
        facet_loads.list_facets(formulation) documents -- build these
        directly from a mesh element's own connectivity row, e.g.:

            for local_idx, family, _ in facet_loads.list_facets(formulation):
                facets.append(tuple(elem_conn[i] for i in local_idx))

        for every boundary element/facet you want loaded (a natural
        pairing with Wave 14 item 123's compute_boundary_mask(), which
        identifies which faces/edges of a mesh are on the boundary in
        the first place).
        dof_index, traction, thickness: same convention as
        add_consistent_edge_load() -- a single global DOF direction
        and a uniform scalar traction magnitude (not a directional/
        pressure-normal-to-facet convention -- see facet_loads.py's own
        docstring for why that is explicitly out of scope here)."""
        from . import facet_loads
        for facet_global_nodes in facets:
            facet_coords = self.mesh.nodes[list(facet_global_nodes)]
            share = facet_loads.consistent_facet_load_shares(
                formulation.dim, facet_coords, traction * thickness, quad_order=quad_order)
            for local_i, node_id in enumerate(facet_global_nodes):
                self.F[self.npn * int(node_id) + dof_index] += share[local_i]

    @property
    def free_dofs(self):
        return np.array([d for d in range(self.n_dof) if d not in self.fixed_dofs])

    @property
    def fixed_dofs_array(self):
        """Wave 16 item 131 -- `fixed_dofs` (a set) in a fixed, sorted
        order, matching `free_dofs`'s own convention -- so a caller
        pairing this against `_fixed_dof_values_array()` below gets a
        consistent, reproducible ordering."""
        return np.array(sorted(self.fixed_dofs), dtype=int)

    def _fixed_dof_values_array(self, fixed):
        """`fixed`: an int array of DOF indices (e.g. `fixed_dofs_array`).
        Returns the prescribed value at each, defaulting to 0.0 for any
        DOF in `fixed_dofs` that was fixed via the original value-less
        `fix_dofs()` call (or a call with the default `value=0.0`)."""
        return np.array([self.fixed_dof_values.get(int(d), 0.0) for d in fixed])

    def _dirichlet_rhs_correction(self, Kmat, free, fixed, u_fixed):
        """The `-Kio @ uo` term of non-homogeneous static condensation
        (Wave 16 item 131, source: TensorMesh's `Boundary Conditions`
        page) -- `Ff = F[free] - K[free, fixed] @ u_fixed`. Returns the
        zero vector (cheaply, without even slicing `Kmat`) when every
        fixed DOF's prescribed value is 0.0 -- the original,
        homogeneous-only case -- so this is a no-op in both RESULT and
        (near enough) COST whenever `value=` was never used, matching
        this project's established "identical cost when unused"
        convention for optional, backward-compatible parameters."""
        if len(fixed) == 0 or not np.any(u_fixed):
            return np.zeros(len(free))
        Kio = Kmat[np.ix_(free, fixed)]
        return Kio @ u_fixed

    # -----------------------------------------------------------------
    @staticmethod
    def _dense_spd_solve(Kff, Ff, tol=1e-6):
        """Wave 0 item 5 (docs/consolidated_future_roadmap.md, source
        fem_implementation_lessons.md): route the dense static solve
        through a Cholesky factorization when Kff is SPD, instead of
        always paying for a general LU (np.linalg.solve's dense
        path). Kff is symmetric for every element formulation this
        package assembles (no non-symmetric constitutive operator
        exists anywhere in elements/), so it IS SPD whenever the model
        is well enough constrained to remove all rigid-body motion --
        exactly the common case. Cholesky itself is roughly half the
        flops of a general LU because it exploits the symmetry a
        generic solver has to spend work discovering (and then
        discarding).

        IMPORTANT, found empirically (not assumed): scipy.linalg.
        cho_factor()'s OWN success/failure (whether it raises
        LinAlgError) is NOT by itself a reliable SPD test for a matrix
        whose near-zero eigenvalues are many orders of magnitude
        smaller than its largest entries -- exactly the pure-Neumann/
        floating-structure case Wave 0 item 6 targets. Measured
        directly on a real free-floating plane-stress model: the true
        rigid-body eigenvalues were ~1e-5 while the matrix's own
        entries were ~1e9-1e12 -- a 14-17 order-of-magnitude gap, far
        below the floating-point noise floor of the elimination itself,
        so LAPACK's dpotrf silently "succeeded" (no exception) with a
        tiny-but-technically-positive computed pivot even though the
        TRUE matrix has negative eigenvalues and is not actually SPD.
        Catching only the exception let a garbage, rigid-body-drifted
        "solution" slip through undetected (see _eigen_solve()'s
        docstring for the concrete before/after investigation, and
        tests/test_singular_system.py for the regression coverage).

        The fix: also inspect the Cholesky factor's own diagonal
        (its pivots) directly, the same idea as _sparse_lu_solve()'s
        SuperLU-pivot check below -- a healthy, well-constrained model
        was measured to keep this ratio around 1e-2 (small models) down
        to no worse than roughly 1e-3-1e-4 for badly-conditioned but
        genuinely nonsingular ones, while the near-singular case above
        measured ~1e-8 -- several orders of magnitude apart, an
        unambiguous signal once actually checked instead of trusted by
        exception alone. Below `tol` (default 1e-6, comfortably between
        those two measured regimes) this raises LinAlgError itself,
        exactly like a genuine cho_factor failure, so solve_static()'s
        single except-block below handles both uniformly.

        Deliberately does NOT fall back to np.linalg.solve() on
        failure (an earlier version of this method did, and that was
        ALSO found to be unsafe for the same reason -- LAPACK's dense
        LU has the identical silent-tiny-pivot blind spot). Instead
        this raises on any non-SPD (or near-singular) Kff, so
        solve_static() can route EVERY such case through the single
        eigendecomposition-based path (_eigen_solve()) that handles
        both "indefinite but nonsingular" and "genuinely singular"
        correctly and uniformly."""
        from scipy.linalg import cho_factor, cho_solve
        c, low = cho_factor(Kff, lower=True)
        diag = np.abs(np.diag(c))
        scale = float(diag.max()) if diag.size else 1.0
        if diag.size and diag.min() < tol * scale:
            raise np.linalg.LinAlgError(
                "_dense_spd_solve: Cholesky factorization completed "
                "without raising, but its smallest pivot is suspiciously "
                f"tiny relative to its largest (ratio {diag.min() / scale:.3e} "
                f"< tol={tol:.1e}) -- Kff is very likely singular or "
                "near-singular (e.g. an unconstrained/pure-Neumann "
                "model), not genuinely SPD; routing to the "
                "eigenvalue-based fallback instead of trusting this "
                "factorization.")
        return cho_solve((c, low), Ff)

    @staticmethod
    def _eigen_solve(Kff, Ff, tol=1e-8):
        """Wave 0 items 5 and 6's shared fallback for a dense Kff that
        _dense_spd_solve() rejected as non-SPD (docs/consolidated_
        future_roadmap.md, source fem_implementation_lessons.md).
        Symmetrically eigendecomposes Kff (valid -- Kff is always
        symmetric in this package) and reconstructs the solution as
        x = V @ diag(1/eigvals) @ V.T @ F. For a NONSINGULAR indefinite
        Kff (no near-zero eigenvalues -- a case Wave 0 item 5's
        Cholesky-first dispatch can reach even though no linear-elastic
        static K in this package is expected to actually be indefinite
        in practice) this is an exact, numerically robust general
        solve, just via eigenvectors instead of LU -- correct, if not
        the cheapest possible route for that particular sub-case.

        For a SINGULAR Kff (Wave 0 item 6's actual target: a
        pure-Neumann/floating-structure model, e.g. a free-floating
        part loaded only by a self-equilibrated pressure/traction, with
        no Dirichlet BC removing rigid-body motion) one rigid-body mode
        sits in the null space per unconstrained rigid-body DOF, and
        np.linalg.solve()/scipy.sparse.linalg.spsolve() on a genuinely
        singular matrix were BOTH found (empirically, comparing against
        an independent scipy.linalg.pinv() reference -- see
        tests/test_singular_system.py) to silently return a finite,
        LOW-RESIDUAL, but physically WRONG answer -- an arbitrary
        amount of rigid-body drift baked in by catastrophic
        cancellation in the near-zero pivot(s), invisible to a
        residual check (Kff@x-F stays small for ANY x that differs
        from the true minimum-norm answer by a null-space vector,
        since Kff annihilates exactly that difference). That is why
        this method is reached via _dense_spd_solve()'s up-front
        Cholesky-failure signal -- proactively, before ever trusting a
        general solver's numerical output -- rather than via a
        post-hoc check on that output.

        This method's own eigenvalue-magnitude threshold (the SAME
        detector, reused) is what distinguishes the two sub-cases: any
        eigenvalue with |eigval| < tol*max(|eigvals|) is treated as a
        rigid-body/null direction and regularized -- rank 1 (one
        missing pin) up to rank 6 (a totally unsupported 3-D body) are
        all handled by the same loop, since the true rigid-body rank
        isn't known in advance. This is the null-space-projection
        reading of the roadmap item's "K_rho = K + rho*E*E^T",
        generalized beyond a single hand-picked E. rho is chosen
        relative to Kff's own largest eigenvalue so the regularization
        is negligible next to the real stiffness -- as rho -> 0 this
        converges to the Moore-Penrose minimum-norm solution of the
        original singular system, the standard well-posed answer for a
        self-equilibrated Neumann problem (load resisted entirely by
        internal elasticity, zero arbitrary drift along the null
        space) -- verified in tests/test_singular_system.py to agree
        with scipy.linalg.pinv() to near machine precision.

        Before regularizing, checks that Ff is actually
        self-equilibrated against the detected rigid-body modes (their
        projection of the load should be ~0) -- a load that does NOT
        satisfy this is not a well-posed free-floating problem, it is a
        genuinely under-constrained model (a missing support the user
        forgot), and silently "solving" it would hide that modeling
        error rather than produce a meaningful displacement field, so
        this raises LinAlgError with a specific, actionable message
        instead."""
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(Kff)
        scale = float(np.max(np.abs(eigvals))) if eigvals.size else 1.0
        threshold = tol * max(scale, 1.0)
        is_rigid = np.abs(eigvals) < threshold

        if is_rigid.any():
            rigid_modes = eigvecs[:, is_rigid]
            residual = rigid_modes.T @ Ff
            max_load = float(np.max(np.abs(Ff))) if Ff.size else 0.0
            if max_load > 0 and np.max(np.abs(residual)) > 1e-6 * max_load:
                raise np.linalg.LinAlgError(
                    "solve_static: the free-DOF stiffness matrix is "
                    f"singular ({int(is_rigid.sum())} rigid-body mode(s) "
                    "detected) and the applied load is NOT "
                    "self-equilibrated against those modes -- this is a "
                    "genuinely under-constrained model (likely a missing "
                    "boundary condition), not a well-posed free-floating "
                    "Neumann problem. Add boundary conditions to remove "
                    "the rigid-body motion, or confirm the load truly "
                    "balances to zero net force/moment if an unsupported "
                    "solve was intended."
                )
            rho = max(scale, 1.0)
        else:
            rho = 1.0   # unused -- np.where below never selects it

        reg_eigvals = np.where(is_rigid, rho, eigvals)
        # x = V @ diag(1/reg_eigvals) @ V.T @ F -- the rigid-body
        # component of F is ~0 (checked above), so its contribution is
        # ~0 regardless of rho; rho only needs to be nonzero so the
        # division is well defined.
        coeffs = (eigvecs.T @ Ff) / reg_eigvals
        return eigvecs @ coeffs

    @staticmethod
    def _sparse_lu_solve(Kff_csc, Ff, tol=1e-10):
        """Sparse counterpart of the _dense_spd_solve()/_eigen_solve()
        dispatch above (Wave 0 items 5 and 6). No sparse-native
        Cholesky is used here (true sparse Cholesky needs CHOLMOD/
        scikit-sparse, an optional binary dependency this package does
        not currently require -- see docs/consolidated_future_roadmap.
        md backlog item 36 for the related fill-reducing-reordering
        follow-on); instead this reuses the SAME SuperLU sparse LU
        factorization scipy.sparse.linalg.spsolve() already performs
        internally, via splu() directly, so detecting near-singularity
        costs nothing extra on top of the solve itself.

        The detector: SuperLU's own U-factor diagonal (its pivots) --
        for a healthy, well-constrained model these stay within a few
        orders of magnitude of Kff's largest entries (empirically
        ~1e-3 relative, measured on the models in
        tests/test_singular_system.py); for a genuinely singular Kff
        (e.g. a pure-Neumann/floating-structure model, Wave 0 item 6),
        the rigid-body null space collapses one or more pivots to
        floating-point noise near zero, empirically ~1e-16 relative --
        twelve-plus orders of magnitude apart, an unambiguous signal.
        This is the sparse analogue of _dense_spd_solve()'s Cholesky
        pivot check, checked proactively BEFORE trusting the LU
        solution, for exactly the same reason: scipy.sparse.linalg.
        spsolve() on a near-singular matrix was found (empirically,
        see _eigen_solve()'s docstring) to silently return a finite,
        low-residual, but physically wrong (rigid-body-drifted) answer
        that a residual check alone cannot catch.

        On a near-singular signal, falls back to the SAME dense,
        eigenvalue-based _eigen_solve() above, densifying only the
        free-dof block -- which is exactly what finding a null space
        already needs to do. This fallback path is for genuinely rare
        models, so trading sparsity for one dense eigendecomposition
        ONLY once SuperLU's own pivots have already signaled trouble
        is the right tradeoff, not a general-purpose sparse Cholesky
        reimplementation."""
        from scipy.sparse.linalg import splu
        lu = splu(Kff_csc)
        diagU = np.abs(lu.U.diagonal())
        scale = float(diagU.max()) if diagU.size else 1.0
        if diagU.size and diagU.min() < tol * scale:
            return FESystem._eigen_solve(Kff_csc.toarray(), Ff)
        return lu.solve(Ff)

    def solve_static(self, verbose=False, method=None, **method_kwargs):
        """Dense: SPD-aware Cholesky solve when Kff is SPD (the common,
        well-constrained case -- see _dense_spd_solve(), Wave 0 item
        5), falling back to the shared eigenvalue-based solve
        (_eigen_solve()) for a non-SPD Kff -- which includes the
        genuinely SINGULAR case Wave 0 item 6 targets: an
        unconstrained/pure-Neumann model.

        sparse=True: the sparse counterpart, _sparse_lu_solve() --
        SuperLU direct sparse LU (spsolve()'s own factorization,
        reused via splu() so the near-singularity check is free),
        falling back to the SAME dense eigenvalue-based solve only
        when SuperLU's own pivots signal near-singularity. See
        _dense_spd_solve()'s docstring for why neither path attempts a
        plain np.linalg.solve()/spsolve() call as a "general solver"
        fallback anymore -- that was tried and found, empirically, to
        silently return wrong (rigid-body-drifted) answers for a
        singular Kff rather than raising, which a residual check alone
        cannot detect (see _eigen_solve()'s docstring for the concrete
        investigation).

        Validated in tests/test_sparse_assembly.py to agree with the
        dense path to near machine precision on every (well-posed)
        model checked; see tests/test_singular_system.py for the new
        regularized-solve coverage (Wave 0 item 6, including agreement
        with an independent scipy.linalg.pinv() reference) and
        tests/test_spd_solve.py for the Cholesky-path coverage (Wave 0
        item 5).

        backend="torch" (set at construction, see __init__'s docstring):
        dispatches instead to torch_sparse_solver.fesystem_solve_static_
        torch() -- dense torch.linalg.solve() when sparse=False, Jacobi-
        preconditioned sparse CG when sparse=True (raises RuntimeError
        on CG non-convergence rather than the SciPy path's automatic
        SPD/eigen fallback; the two backends are validated to agree
        with each other in tests/test_torch_sparse_solver.py, not
        merged into one fallback chain -- see __init__'s docstring for
        why backend is an explicit choice, not an automatic upgrade).

        backend="auto" (Wave 11 item 108): decided HERE, once Kff
        exists -- see backend_dispatch.select_backend() for the exact
        SPD/size rule. verbose=True prints the one-line diagnostic
        backend_dispatch.format_diagnostic_line() builds (n, nnz,
        dtype, device, symmetric, spd, backend, method chosen).

        Non-homogeneous Dirichlet values (Wave 16 item 131, source:
        TensorMesh's `Boundary Conditions` page): if any DOF was fixed
        via `fix_dofs(..., value=nonzero)`, the default (`backend=
        "scipy"`) and `backend="auto"` paths below apply the standard
        static-condensation RHS correction `Ff -= K[free, fixed] @
        u_fixed` and fill `U[fixed] = u_fixed` in the returned vector
        -- see `fix_dofs()`'s own docstring for the exact, deliberately
        narrow scope of what honors this (this method only).
        `backend="torch"` does NOT implement this correction yet and
        raises a specific `NotImplementedError` if any nonzero value
        is present, rather than silently solving the homogeneous
        system and returning a wrong answer.

        method= (Wave 16 item 132, source: TensorMesh's `Sparse
        Solvers` page): None (default) is the direct-solve behavior
        described above, completely unchanged -- IDENTICAL cost when
        unused, same convention as every other optional parameter in
        this package since Wave 1's du_tol/energy_tol. Passing
        method="cg"/"pcg" instead routes the free-DOF solve through
        Wave 8's own hand-written (preconditioned) conjugate-gradient
        solver (iterative_solvers.py, items 37/38) -- see
        _solve_static_iterative()'s own docstring for the exact
        dispatch, method_kwargs (tol=, maxiter=, preconditioner= for
        "pcg"), and why Jacobi/Gauss-Seidel/SOR (item 39) are
        deliberately NOT exposed as a third method= choice here.

        Valid with backend="scipy" (the default) OR backend="torch"
        (Wave 9 addendum item 137, docs/consolidated_future_roadmap.md
        -- CLOSES a gap this item's own docstring used to describe:
        method= combined with backend="torch" now routes through
        iterative_solvers.preconditioned_cg(backend="torch", device=
        self.device) instead of raising, reusing torch_sparse_solver.
        py's own generic CG loop rather than the separate dense/sparse-
        CG dispatch backend="torch" ALREADY has for method=None -- an
        explicit, caller-typed method="cg"/"pcg" AND backend="torch" is
        unambiguous, unlike backend="auto" below). Still raises
        ValueError if combined with backend="auto" specifically: "auto"
        decides scipy-vs-torch dynamically per system, and combining
        that runtime choice with an explicitly-typed method= would be
        the exact "silently reinterpret" ambiguity this project's
        "never silently reinterpret, fail loudly instead" precedent
        (Wave 0 item 3 onward) exists to avoid -- pass backend="scipy"
        or backend="torch" explicitly instead. The -Kio*uo
        non-homogeneous-Dirichlet RHS correction above is honored
        identically regardless of method -- it's a property of the
        linear system being solved, not of how it's solved.
        NOT wired here: Wave 8's own multigrid_solve() (item 40) -- it
        needs a structured Quad4 mesh HIERARCHY
        (iterative_solvers.structured_quad_hierarchy()), which isn't
        derivable from an arbitrary already-assembled FESystem without
        the caller's own mesh-generation parameters (Lx, Ly, nx0, ny0);
        a caller who wants multigrid still builds that hierarchy by
        hand and calls iterative_solvers.multigrid_solve() directly, as
        before this item."""
        fixed = self.fixed_dofs_array
        u_fixed = self._fixed_dof_values_array(fixed)
        has_nonzero_dirichlet = bool(np.any(u_fixed))

        if method is not None:
            if self.backend not in ("scipy", "torch"):
                raise ValueError(
                    f"solve_static(method={method!r}): method= is only "
                    f"supported with backend='scipy' or backend='torch' "
                    f"(this FESystem has backend={self.backend!r}) -- "
                    f"backend='auto' decides scipy-vs-torch dynamically per "
                    f"system, and combining that runtime choice with an "
                    f"explicitly-typed method= would be ambiguous; construct "
                    f"with backend='scipy' (the default) or backend='torch' "
                    f"explicitly to use method=.")
            return self._solve_static_iterative(method, fixed, u_fixed, method_kwargs)

        if self.backend == "torch":
            if has_nonzero_dirichlet:
                raise NotImplementedError(
                    "solve_static(backend='torch'): non-homogeneous Dirichlet "
                    "values (fix_dofs(..., value=nonzero)) are not yet "
                    "supported on the torch backend (Wave 16 item 131 only "
                    "implemented the scipy/auto paths) -- use backend='scipy' "
                    "or backend='auto', or fix this DOF at value=0.0 and "
                    "apply the offset some other way.")
            from .torch_sparse_solver import fesystem_solve_static_torch
            method = "cg" if self.sparse else "dense"
            return fesystem_solve_static_torch(self, method=method, device=self.device)

        if self.backend == "auto":
            from .backend_dispatch import select_backend, format_diagnostic_line
            free = self.free_dofs
            Kmat = self._as_solve_matrix(self.K)
            Kff_probe = Kmat[np.ix_(free, free)]
            chosen, diag = select_backend(Kff_probe, device=self.device)
            if verbose:
                print(format_diagnostic_line(diag))
            if chosen == "torch":
                if has_nonzero_dirichlet:
                    raise NotImplementedError(
                        "solve_static(backend='auto'): backend_dispatch chose "
                        "'torch' for this system, but non-homogeneous Dirichlet "
                        "values (fix_dofs(..., value=nonzero)) are not yet "
                        "supported on the torch backend (Wave 16 item 131) -- "
                        "pass backend='scipy' explicitly to force the scipy "
                        "path, which does support them.")
                from .torch_sparse_solver import fesystem_solve_static_torch, _require_torch
                _require_torch()
                method = "cg" if self.sparse else "dense"
                return fesystem_solve_static_torch(self, method=method, device=self.device)
            # else: fall through to the plain scipy path below, reusing
            # Kff_probe instead of re-slicing self.K a second time.
            U = np.zeros(self.n_dof)
            U[fixed] = u_fixed
            Ff = self.F[free] - self._dirichlet_rhs_correction(Kmat, free, fixed, u_fixed)
            if self.sparse:
                Uf = self._sparse_lu_solve(Kff_probe.tocsc(), Ff)
            else:
                try:
                    Uf = self._dense_spd_solve(Kff_probe, Ff)
                except np.linalg.LinAlgError:
                    Uf = self._eigen_solve(Kff_probe, Ff)
            U[free] = Uf
            return U

        free = self.free_dofs
        U = np.zeros(self.n_dof)
        U[fixed] = u_fixed
        Kmat = self._as_solve_matrix(self.K)
        Kff = Kmat[np.ix_(free, free)]
        Ff = self.F[free] - self._dirichlet_rhs_correction(Kmat, free, fixed, u_fixed)
        if verbose:
            from .backend_dispatch import format_diagnostic_line
            n = Kff.shape[0]
            nnz = int(Kff.nnz) if hasattr(Kff, "nnz") else int(np.count_nonzero(Kff))
            print(format_diagnostic_line({
                "n": n, "nnz": nnz, "dtype": str(getattr(Kff, "dtype", Kff.dtype)),
                "device": self.device, "symmetric": "n/a", "spd": "n/a",
                "backend": self.backend, "method": "sparse_lu" if self.sparse else "cholesky_or_eigen",
            }))
        if self.sparse:
            Uf = self._sparse_lu_solve(Kff.tocsc(), Ff)
        else:
            try:
                Uf = self._dense_spd_solve(Kff, Ff)
            except np.linalg.LinAlgError:
                Uf = self._eigen_solve(Kff, Ff)
        U[free] = Uf
        return U

    def _solve_static_iterative(self, method, fixed, u_fixed, method_kwargs):
        """Wave 16 item 132: the method= branch of solve_static() above,
        split out for readability. Builds Kff/Ff exactly like the plain
        scipy path (same -Kio*uo correction, same free/fixed split),
        then dispatches to Wave 8's preconditioned_cg() (items 37/38)
        instead of a direct solve.

        Deliberately narrower than "wire up all of iterative_solvers.py":
        only "cg"/"pcg" are exposed here, matching the Krylov-method
        vocabulary TensorMesh's own `Sparse Solvers` page actually
        offers through `.solve(method=...)` (cg, bicgstab, gmres,
        minres, lsqr, lsmr) -- Jacobi/Gauss-Seidel/SOR (item 39) are
        deliberately NOT exposed as a solve_static() method= choice:
        this was tried first and confirmed, directly, to diverge on an
        ordinary Quad4 plane-stress stiffness matrix (not diagonally
        dominant), exactly matching iterative_solvers.py's own module
        docstring ("building blocks, not competitive solvers... their
        real role is as multigrid smoothers, not meant to be reached
        for as a standalone large-problem solver") -- wiring them in as
        a top-level solve method would silently hand a caller a solver
        that's likely to fail on their actual FE problem. A caller who
        specifically wants Jacobi/Gauss-Seidel/SOR (e.g. as hand-rolled
        multigrid smoothers) still imports iterative_solvers.py
        directly, unchanged by this item.

        method_kwargs:
          "cg"  -- tol=, maxiter=, x0= (unpreconditioned CG, requires
                   SPD Kff -- preconditioned_cg() itself raises
                   LinAlgError if it isn't).
          "pcg" -- same as "cg" plus preconditioner=
                   "jacobi"/"ssor"/"ic0" (default "jacobi"), and
                   omega= (only used by "ssor").

        Both return (x, n_iter) from preconditioned_cg() -- n_iter is
        discarded here (solve_static()'s own return contract is just
        U, matching every other branch); a caller who wants the
        iteration count calls iterative_solvers.preconditioned_cg()
        directly instead of going through solve_static().

        self.backend (Wave 9 addendum item 137, docs/consolidated_
        future_roadmap.md): "scipy" (default) builds Kff/Ff as plain
        numpy/scipy and calls preconditioned_cg(backend="scipy")
        (unchanged). "torch" builds the SAME Kff/Ff (still numpy/scipy
        -- self.K's own contract, see vectorized_assembly.py's own
        scatter_global_stiffness() docstring for the identical point
        made about assembly) and calls preconditioned_cg(backend=
        "torch", device=self.device) instead, which converts internally.
        For "pcg" specifically, preconditioner="jacobi"/"ssor" route to
        jacobi_preconditioner_torch()/ssor_preconditioner_torch()
        instead of the numpy versions; preconditioner="ic0" raises
        NotImplementedError on backend="torch" (incomplete_cholesky0()
        has no torch port -- see iterative_solvers.py's own module
        docstring for why)."""
        from . import iterative_solvers as itsolv
        free = self.free_dofs
        Kmat = self._as_solve_matrix(self.K)
        Kff = Kmat[np.ix_(free, free)]
        Ff = self.F[free] - self._dirichlet_rhs_correction(Kmat, free, fixed, u_fixed)

        cg_backend_kwargs = {"backend": "torch", "device": self.device} if self.backend == "torch" else {}

        kwargs = dict(method_kwargs)
        if method == "cg":
            Uf, _n_iter = itsolv.preconditioned_cg(Kff, Ff, M=None, **kwargs, **cg_backend_kwargs)
        elif method == "pcg":
            precond = kwargs.pop("preconditioner", "jacobi")
            omega = kwargs.pop("omega", 1.0)
            if self.backend == "torch":
                if precond == "jacobi":
                    M = itsolv.jacobi_preconditioner_torch(Kff, device=self.device)
                elif precond == "ssor":
                    M = itsolv.ssor_preconditioner_torch(Kff, omega=omega, device=self.device)
                elif precond == "ic0":
                    raise NotImplementedError(
                        "solve_static(method='pcg', preconditioner='ic0') is "
                        "not supported with backend='torch' -- incomplete_"
                        "cholesky0() has no torch port (see iterative_"
                        "solvers.py's own module docstring, 'GPU/torch "
                        "side-by-side path'); use preconditioner='jacobi'/"
                        "'ssor', or backend='scipy'.")
                else:
                    raise ValueError(
                        f"solve_static(method='pcg'): unknown preconditioner="
                        f"{precond!r} -- choose 'jacobi', 'ssor', or 'ic0'.")
            else:
                if precond == "jacobi":
                    M = itsolv.jacobi_preconditioner(Kff)
                elif precond == "ssor":
                    M = itsolv.ssor_preconditioner(Kff, omega=omega)
                elif precond == "ic0":
                    M = itsolv.incomplete_cholesky0(Kff)
                else:
                    raise ValueError(
                        f"solve_static(method='pcg'): unknown preconditioner="
                        f"{precond!r} -- choose 'jacobi', 'ssor', or 'ic0'.")
            Uf, _n_iter = itsolv.preconditioned_cg(Kff, Ff, M=M, **kwargs, **cg_backend_kwargs)
        else:
            raise ValueError(
                f"solve_static: unknown method={method!r} -- choose from "
                f"'cg', 'pcg' (or method=None, the default, for the "
                f"existing direct solve; Jacobi/Gauss-Seidel/SOR are "
                f"intentionally not exposed here -- see "
                f"_solve_static_iterative()'s own docstring for why -- "
                f"import iterative_solvers.py directly for those).")
        U = np.zeros(self.n_dof)
        U[fixed] = u_fixed
        U[free] = Uf
        return U

    def solve_modal(self, n_modes=4):
        """Generalized eigenproblem K*phi = omega^2*M*phi on the free DOFs.
        Returns (freq_hz (n_modes,), mode_shapes (n_dof, n_modes)).

        sparse=True: scipy.sparse.linalg.eigsh() with shift-invert
        (sigma=0) instead of scipy.linalg.eigh()'s full dense
        decomposition -- the standard Lanczos-based technique for the
        SMALLEST eigenvalues of a large sparse generalized eigenproblem
        (a full dense eigh() computes EVERY eigenvalue, O(n_dof^3),
        wasteful when only the first few modes are wanted; eigsh's
        shift-invert targets the eigenvalues nearest sigma=0 --
        exactly the lowest vibration modes -- without ever forming a
        dense matrix). Only k=n_modes eigenpairs are computed, so
        n_modes must be < the number of free DOFs (an eigsh
        requirement, not a dense-solve one -- see its docstring)."""
        assert self.M is not None, "call assemble_mass() first"
        free = self.free_dofs
        Kff = self._as_solve_matrix(self.K)[np.ix_(free, free)]
        Mff = self._as_solve_matrix(self.M)[np.ix_(free, free)]
        if self.sparse:
            from scipy.sparse.linalg import eigsh
            eigvals, eigvecs = eigsh(Kff, k=n_modes, M=Mff, sigma=0, which="LM")
        else:
            eigvals, eigvecs = eigh(Kff, Mff)
        eigvals = np.clip(eigvals, 0, None)
        omega = np.sqrt(eigvals)
        freq_hz = omega / (2 * np.pi)

        mode_shapes = np.zeros((self.n_dof, n_modes))
        for m in range(n_modes):
            full = np.zeros(self.n_dof)
            full[free] = eigvecs[:, m]
            mode_shapes[:, m] = full
        return freq_hz[:n_modes], mode_shapes

    def solve_linear_buckling(self, n_modes=4):
        """Module 19 (general-purpose extensions Phase 4): the linear
        (eigenvalue) buckling problem (K + lambda*K_sigma)*phi = 0 on
        the free DOFs, i.e. K*phi = -lambda*K_sigma*phi -- the exact
        same GENERALIZED EIGENPROBLEM PATTERN solve_modal() above
        solves (K*phi = omega^2*M*phi), with K_sigma (built from a
        REFERENCE axial-force state via assemble_geometric_stiffness(),
        tension-positive) standing in for M, and -K_sigma_ff in place
        of Mff -- the sign flip is exactly what turns a TENSION-
        positive N (this package's one consistent sign convention,
        shared with TrussTL2D's S) into the physically-positive-
        definite-under-compression matrix eigh() needs for its
        standard Cholesky-based algorithm to apply (a purely
        compressive reference state gives K_sigma_ff itself negative
        in this convention, so -K_sigma_ff is the positive one).

        Unlike M (always positive definite for any real structure),
        -K_sigma_ff is NOT guaranteed positive definite in general --
        e.g. a 3-D frame member with BOTH bending planes simultaneously
        free carries bending-rotation cross-coupling in K_sigma that
        can make the assembled -K_sigma_ff indefinite even though the
        underlying physics (a genuinely compressive reference state)
        is perfectly well posed. eigh()'s Cholesky-based algorithm
        requires strict positive definiteness and raises
        numpy.linalg.LinAlgError on that case; this method catches
        exactly that and falls back to scipy.linalg.eig() (the general
        non-Hermitian solver, no positive-definiteness requirement),
        keeping only the real part of each eigenvalue/eigenvector
        (buckling eigenvalues for a well-posed reference state are
        real; a large imaginary part would indicate a genuinely
        ill-posed problem, not something this method tries to detect
        -- see test_linear_buckling.py for a worked case that needs
        this fallback).

        Returns (load_factors (n_modes,), mode_shapes (n_dof, n_modes)).
        `load_factors` are the multipliers on whatever reference axial
        force state N was used to build K_sigma -- e.g. if N came from
        a 1 N reference compressive load, load_factors[0] IS the
        critical buckling load in Newtons directly. Only the lowest
        POSITIVE load factors are returned (ascending) -- a negative
        eigenvalue here is mathematically valid but means "buckles
        under a reference load reversed in direction", not physically
        meaningful for the compressive reference state this method is
        meant to be used with; see docs/general_purpose_extensions_
        roadmap.md Section 6 for the validation this was checked
        against (the Euler column closed form). Dense-only for this
        first pass (mirrors solve_modal()'s sparse=True path being
        reserved for future work if a large-buckling use case
        actually needs it -- see that method's docstring for the
        general sparse-eigensolver reasoning, not yet extended here)."""
        assert self.K_sigma is not None, "call assemble_geometric_stiffness() first"
        free = self.free_dofs
        Kff = self._as_solve_matrix(self.K)[np.ix_(free, free)]
        Ksigff = self._as_solve_matrix(self.K_sigma)[np.ix_(free, free)]
        try:
            eigvals, eigvecs = eigh(Kff, -Ksigff)
        except np.linalg.LinAlgError:
            from scipy.linalg import eig
            eigvals, eigvecs = eig(Kff, -Ksigff)
            eigvals = eigvals.real
            eigvecs = eigvecs.real

        positive = eigvals > 0
        order = np.argsort(eigvals[positive])
        load_factors = eigvals[positive][order][:n_modes]
        vecs = eigvecs[:, positive][:, order][:, :n_modes]

        n_found = load_factors.shape[0]
        mode_shapes = np.zeros((self.n_dof, n_found))
        for m in range(n_found):
            full = np.zeros(self.n_dof)
            full[free] = vecs[:, m]
            mode_shapes[:, m] = full
        return load_factors, mode_shapes

    # -----------------------------------------------------------------
    # Dynamics: direct time integration, modal superposition,
    # frequency-domain (harmonic) response, and random vibration (PSD).
    # All four reuse the K/M/C already assembled above -- none of them
    # need config.py, mesh.py, or element.py to change.
    # -----------------------------------------------------------------
    def solve_transient_implicit(self, load, T_total, dt, beta=0.25, gamma=0.5,
                                  u0=None, v0=None):
        """Newmark-beta direct time integration (default: average-
        acceleration, unconditionally stable for this linear,
        time-invariant system). `load` is any object exposing
        force_at(t, n_dof, npn) -- see loads.TimeHistoryLoad. K_eff is
        factored ONCE before the time loop (not every step): for a
        linear system it doesn't change, so there's no reason to pay
        for it more than once. Returns (t, U_hist) with U_hist shape
        (n_steps+1, n_dof).

        sparse=True (Wave 16 item 134, docs/consolidated_future_
        roadmap.md, source: TensorMesh's `Time Integration` page --
        its worked heat-equation example spends a whole section on
        "factorize the time-stepped operator once, reuse it every step
        via back-substitution," the SAME idiom this method already
        used independently for the dense case, well before that
        comparison): before this item, this method unconditionally
        called `np.linalg.inv()` on `Kff + a0c*Mff + a1c*Cff`, which
        silently assumed a dense `Kff`/`Mff`/`Cff` -- for
        `sparse=True`, those are `scipy.sparse` matrices, and
        `np.linalg.inv()` does not accept one; the method crashed with
        a confusing, low-level `LinAlgError: 0-dimensional array
        given` rather than either working or failing with a clear
        message. Now dispatches on `self.sparse`: the sparse path
        factors `K_eff` ONCE via `scipy.sparse.linalg.splu()` (the
        same SuperLU factorization `_sparse_lu_solve()` already uses
        elsewhere in this class, just held onto across the whole time
        loop instead of being discarded after one `.solve()` call)
        and calls `.solve(rhs)` every step; the dense path now uses
        `scipy.linalg.lu_factor()`/`lu_solve()` (a proper one-time
        factorization + repeated back-substitution) instead of
        explicitly forming the inverse -- the textbook-preferred
        idiom (Higham, "Accuracy and Stability of Numerical
        Algorithms": prefer factor+solve over an explicit inverse),
        and consistent with how every OTHER solve path in this class
        already avoids forming one (`_dense_spd_solve()`,
        `_sparse_lu_solve()`, `_eigen_solve()`). Checked directly, not
        assumed: on a well-conditioned free-dof-sized system, the
        precision difference between the old `inv()`-based path and
        the new `lu_factor()` path is negligible (both land within a
        few multiples of machine epsilon of an independent reference
        solve) -- the decisive win here is the sparse crash fix and
        internal consistency, not a large accuracy gap. No behavior
        change whatsoever for the (already-only-supported)
        `sparse=False` case beyond this internal solve-mechanism
        swap: same K_eff, same predictor/corrector recursion, same
        `tests/test_nonlinear_transient.py` machine-precision
        cross-check this method is validated against."""
        assert self.M is not None and self.C is not None, \
            "call assemble_mass() and assemble_damping() first"
        free = self.free_dofs
        n_steps = int(round(T_total / dt))
        t = np.arange(n_steps + 1) * dt

        Mmat = self._as_solve_matrix(self.M)
        Cmat = self._as_solve_matrix(self.C)
        Kmat = self._as_solve_matrix(self.K)
        Mff = Mmat[np.ix_(free, free)]
        Cff = Cmat[np.ix_(free, free)]
        Kff = Kmat[np.ix_(free, free)]

        a0c = 1 / (beta * dt**2); a1c = gamma / (beta * dt); a2c = 1 / (beta * dt)
        a3c = 1 / (2 * beta) - 1; a4c = gamma / beta - 1; a5c = dt / 2 * (gamma / beta - 2)
        a6c = dt * (1 - gamma); a7c = dt * gamma

        Keff = Kff + a0c * Mff + a1c * Cff
        if self.sparse:
            from scipy.sparse.linalg import splu
            keff_lu = splu(Keff.tocsc())
            def keff_solve(rhs):
                return keff_lu.solve(rhs)
        else:
            from scipy.linalg import lu_factor, lu_solve
            keff_lu = lu_factor(Keff)
            def keff_solve(rhs):
                return lu_solve(keff_lu, rhs)

        d = np.zeros(len(free)) if u0 is None else np.asarray(u0)[free].copy()
        v = np.zeros(len(free)) if v0 is None else np.asarray(v0)[free].copy()

        def F_free(tt):
            return load.force_at(tt, self.n_dof, self.npn)[free]

        rhs0 = F_free(0.0) - Cff @ v - Kff @ d
        if self.sparse:
            from scipy.sparse.linalg import spsolve
            a = spsolve(Mff.tocsc(), rhs0)
        else:
            a = np.linalg.solve(Mff, rhs0)

        U_hist = np.zeros((n_steps + 1, self.n_dof))
        U_hist[0, free] = d

        for step in range(n_steps):
            rhs = (F_free(t[step + 1]) + Mff @ (a0c * d + a2c * v + a3c * a)
                   + Cff @ (a1c * d + a4c * v + a5c * a))
            d_new = keff_solve(rhs)
            a_new = a0c * (d_new - d) - a2c * v - a3c * a
            v_new = v + a6c * a + a7c * a_new
            d, v, a = d_new, v_new, a_new
            U_hist[step + 1, free] = d

        return t, U_hist

    def critical_timestep(self):
        """dt_crit = 2/omega_max, the central-difference stability limit,
        from the FULL eigenspectrum of the free system (not just the
        first few modes retained by solve_modal() -- the highest-
        frequency mode of the discretized mesh is what governs
        stability, and it is typically far above the lowest few
        'physical' modes). Exact but O(n_dof^3) via a dense eigh, so for
        very large systems estimate dt_crit from a single representative
        element instead -- see critical_timestep_local() (Wave 6 item
        32) for the cheap O(n_elements) local alternative this method's
        own gap used to flag with no fix; fine for the mesh sizes this
        package targets."""
        assert self.M is not None, "call assemble_mass() first"
        free = self.free_dofs
        eigvals = eigh(self.K[np.ix_(free, free)], self.M[np.ix_(free, free)],
                        eigvals_only=True)
        omega_max = np.sqrt(max(eigvals[-1], 0.0))
        return 2.0 / omega_max

    def critical_timestep_local(self, wave_speed, alpha=0.9):
        """Item 32 (Wave 6, docs/consolidated_future_roadmap.md, source
        nonlinear_fem_lessons.md's appendix): the book's cheap LOCAL
        per-element estimate dt_crit = alpha * min_e(l_e / c_e), an
        O(n_elements) alternative to critical_timestep()'s exact but
        O(n_dof^3) global spectral estimate (a dense eigh of the FULL
        assembled K/M) -- for a mesh large enough that repeated dense
        eigendecompositions become the real bottleneck (the exact gap
        critical_timestep()'s own docstring already flagged, unaddressed
        until this item), this trades a modest amount of conservatism
        for a per-element loop that costs about the same as one
        assemble_mass() call, and needs neither K nor M assembled first.

        l_e (characteristic element length): the MINIMUM pairwise
        distance between this element's own nodes -- deliberately
        geometry-only, with no element-formulation-specific
        "characteristic length" convention assumed (e.g. volume/max-
        face-area for a solid), since this package's element library
        spans 1-D beams, 2-D shells, and 3-D solids with very different
        natural length measures and no single formula fits all of them.
        Minimum pairwise node distance is a safe (if occasionally
        pessimistic -- e.g. for a many-node quadratic element it under-
        counts the true integration-point spacing) lower bound on every
        one of those conventions: erring toward a SMALLER l_e only ever
        makes dt_crit smaller (more conservative, i.e. still stable),
        never larger, so this stays safe even where it isn't tight.

        c_e (per-element wave speed): supplied by the CALLER, not
        derived here -- this package's material representations vary
        enormously across element families (a raw scalar E for a
        truss, an (E, A, I) tuple for a beam, a full elasticity tensor
        D for a continuum element, a Material dataclass elsewhere), so
        there is no single formula this method could apply uniformly
        without silently guessing wrong for some formulation. Instead
        wave_speed follows the SAME convention as D/mat/rho throughout
        this class (see _per_block_arg()): either one scalar shared by
        every block (the common case), or a dict keyed by block name
        for a per-block wave speed. The caller computes c_e from
        whatever elastic constants and density are appropriate for
        that block's own material model -- typical choices: sqrt(E/rho)
        for a 1-D bar/beam's axial wave speed, sqrt((lambda + 2*mu) /
        rho) for a 3-D solid's dilatational wave speed, sqrt(E / (rho *
        (1 - nu**2))) for a plane-stress/membrane wave speed.

        alpha: the Courant safety factor (default 0.9, a standard
        choice in explicit-dynamics practice, e.g. LS-DYNA's own
        default) -- this LOCAL estimate ignores the element-to-element
        coupling the global eigenvalue estimate captures exactly, so a
        strict alpha < 1 margin below the nominal single-element CFL
        limit is standard, expected practice here, not a hedge specific
        to this implementation.

        Returns a single float: alpha times the SMALLEST l_e/c_e ratio
        over every element in the mesh (every block, if more than
        one)."""
        dt_min = np.inf
        for name, formulation, connectivity in self._blocks:
            c_e = self._per_block_arg(wave_speed, name)
            for elem_conn in connectivity:
                elem_coords = self.mesh.nodes[elem_conn]
                n_nodes = elem_coords.shape[0]
                l_e = np.inf
                for i in range(n_nodes):
                    for j in range(i + 1, n_nodes):
                        d = np.linalg.norm(elem_coords[i] - elem_coords[j])
                        if d < l_e:
                            l_e = d
                dt_e = l_e / c_e
                if dt_e < dt_min:
                    dt_min = dt_e
        return alpha * dt_min

    def solve_transient_explicit(self, load, T_total, dt, u0=None, v0=None):
        """Central-difference explicit time integration on a LUMPED mass
        matrix. If self.C is None, or is diagonal (pure mass-proportional
        Rayleigh damping, beta=0), every step is a simple elementwise
        divide -- no matrix solve at all, which is the entire point of
        an explicit method. A general (non-diagonal) C still gives the
        correct central-difference formulation, but falls back to a
        per-step linear solve and loses that speed advantage -- this is
        detected automatically, not something the caller has to specify.
        Conditionally stable: dt must be below the mesh's critical time
        step (see critical_timestep()) or the solution will blow up --
        no automatic check is done here, since forcing one on every call
        would be expensive for large systems; call critical_timestep()
        once yourself and pick dt as a safe fraction of it (e.g. 0.5x)."""
        assert self.M_lumped is not None, "call assemble_lumped_mass() first"
        free = self.free_dofs
        n_steps = int(round(T_total / dt))
        t = np.arange(n_steps + 1) * dt

        Mff = self.M_lumped[np.ix_(free, free)]
        Kff = self.K[np.ix_(free, free)]
        Cff = None if self.C is None else self.C[np.ix_(free, free)]
        diag_fast = (Cff is None) or np.allclose(Cff, np.diag(np.diag(Cff)))

        d = np.zeros(len(free)) if u0 is None else np.asarray(u0)[free].copy()
        v = np.zeros(len(free)) if v0 is None else np.asarray(v0)[free].copy()

        def F_free(tt):
            return load.force_at(tt, self.n_dof, self.npn)[free]

        m_diag = np.diag(Mff)
        Cv0 = (Cff @ v) if Cff is not None else 0.0
        a = (F_free(0.0) - Cv0 - Kff @ d) / m_diag
        d_prev = d - dt * v + 0.5 * dt**2 * a

        U_hist = np.zeros((n_steps + 1, self.n_dof))
        U_hist[0, free] = d

        if diag_fast:
            c_diag = np.zeros(len(free)) if Cff is None else np.diag(Cff)
            keff_diag = m_diag / dt**2 + c_diag / (2 * dt)
            for step in range(n_steps):
                Fn = F_free(t[step])
                rhs = (Fn - Kff @ d + (2 * m_diag / dt**2) * d
                       - (m_diag / dt**2 - c_diag / (2 * dt)) * d_prev)
                d_new = rhs / keff_diag
                d_prev, d = d, d_new
                U_hist[step + 1, free] = d
        else:
            Keff = Mff / dt**2 + Cff / (2 * dt)
            A2 = 2 * Mff / dt**2
            A3 = Mff / dt**2 - Cff / (2 * dt)
            for step in range(n_steps):
                Fn = F_free(t[step])
                rhs = Fn - Kff @ d + A2 @ d - A3 @ d_prev
                d_new = np.linalg.solve(Keff, rhs)
                d_prev, d = d, d_new
                U_hist[step + 1, free] = d

        return t, U_hist

    def solve_modal_superposition(self, load, T_total, dt, n_modes, zeta):
        """Reduces the transient problem to n_modes decoupled SDOF
        equations via the mass-normalized mode shapes from solve_modal()
        (scipy.linalg.eigh already returns M-orthonormal eigenvectors
        for a generalized eigenproblem, so no separate normalization
        step is needed), integrates each with the same Newmark-beta
        scheme used above -- trivially, since each mode's "matrix" is
        just a scalar -- then reconstructs U(t) = Phi @ q(t). `zeta` may
        be a single value (applied to every retained mode) or an array
        of length n_modes. This is the method that makes modal
        superposition cheap: no n_dof x n_dof solve ever happens here."""
        assert self.M is not None, "call assemble_mass() first"
        freq_hz, mode_shapes = self.solve_modal(n_modes=n_modes)
        omega = 2 * np.pi * freq_hz
        free = self.free_dofs
        Mff = self.M[np.ix_(free, free)]
        Phi = mode_shapes[free, :]

        # sanity check: mass-orthonormality (Phi.T @ M @ Phi == I)
        ortho_err = np.max(np.abs(Phi.T @ Mff @ Phi - np.eye(n_modes)))
        assert ortho_err < 1e-6, f"modes not mass-orthonormal (err={ortho_err:.2e})"

        zeta_arr = np.full(n_modes, zeta) if np.isscalar(zeta) else np.asarray(zeta)

        n_steps = int(round(T_total / dt))
        t = np.arange(n_steps + 1) * dt
        beta, gamma = 0.25, 0.5
        a0c = 1 / (beta * dt**2); a1c = gamma / (beta * dt); a2c = 1 / (beta * dt)
        a3c = 1 / (2 * beta) - 1; a4c = gamma / beta - 1; a5c = dt / 2 * (gamma / beta - 2)
        a6c = dt * (1 - gamma); a7c = dt * gamma
        c_modal = 2 * zeta_arr * omega            # modal damping coefficient (modal mass = 1)
        keff_modal = omega**2 + a0c + a1c * c_modal

        def p_of_t(tt):
            F = load.force_at(tt, self.n_dof, self.npn)[free]
            return Phi.T @ F

        q = np.zeros((n_steps + 1, n_modes))
        qd = np.zeros(n_modes)
        qdd = p_of_t(0.0) - c_modal * qd - omega**2 * q[0]

        for step in range(n_steps):
            rhs = (p_of_t(t[step + 1]) + (a0c * q[step] + a2c * qd + a3c * qdd)
                   + c_modal * (a1c * q[step] + a4c * qd + a5c * qdd))
            q_new = rhs / keff_modal
            qdd_new = a0c * (q_new - q[step]) - a2c * qd - a3c * qdd
            qd_new = qd + a6c * qdd + a7c * qdd_new
            q[step + 1] = q_new
            qd, qdd = qd_new, qdd_new

        U_hist = np.zeros((n_steps + 1, self.n_dof))
        U_hist[:, free] = q @ Phi.T
        return t, U_hist, freq_hz

    def solve_harmonic(self, Omega, F0_vector):
        """Direct complex solve of (-Omega^2*M + i*Omega*C + K) U0 = F0
        at a single driving frequency Omega (rad/s) -- no time marching.
        At Omega=0 this reduces exactly to the static solve, a useful
        sanity check. Requires assemble_mass() and assemble_damping()."""
        assert self.M is not None and self.C is not None, \
            "call assemble_mass() and assemble_damping() first"
        free = self.free_dofs
        Z = (-Omega**2 * self.M + 1j * Omega * self.C + self.K)[np.ix_(free, free)]
        U0 = np.zeros(self.n_dof, dtype=complex)
        U0[free] = np.linalg.solve(Z, F0_vector[free])
        return U0

    def solve_frequency_sweep(self, Omega_array, F0_vector):
        """solve_harmonic() at every frequency in Omega_array (rad/s).
        Returns a complex array of shape (len(Omega_array), n_dof)."""
        U = np.zeros((len(Omega_array), self.n_dof), dtype=complex)
        for i, Om in enumerate(Omega_array):
            U[i] = self.solve_harmonic(Om, F0_vector)
        return U

    def solve_random_vibration(self, freqs_hz, psd_input, F0_pattern, output_dof):
        """PSD (random vibration) formulation: S_out(f) = |H(f)|^2 *
        S_in(f), where H(f) is the transfer function from the
        unit-magnitude load F0_pattern to output_dof, built from
        solve_frequency_sweep() -- i.e. random vibration is implemented
        as a thin, general layer on top of the harmonic solver, valid
        for any MDOF system (no SDOF closed-form assumed). Returns
        (S_out array, sigma_out) where sigma_out is the response RMS,
        sqrt of the trapezoidal-integrated output PSD."""
        Omega_array = 2 * np.pi * np.asarray(freqs_hz)
        U = self.solve_frequency_sweep(Omega_array, F0_pattern)
        H = U[:, output_dof]
        S_out = np.abs(H)**2 * np.asarray(psd_input)
        sigma_out = np.sqrt(np.trapz(S_out, freqs_hz))
        return S_out, sigma_out
