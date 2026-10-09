"""
cms.py -- component mode synthesis: Guyan (static) condensation and Craig-Bampton substructure
reduction, plus coupling of reduced substructures into one reduced model.

WHY. Classic structural-dynamics reduction for assemblies. Each substructure is reduced on its own
(offline, possibly by different teams or from different FE models) while its INTERFACE degrees of
freedom stay physical, so substructures can be joined, replaced or modified without re-reducing the
rest. The package's other methods reduce a whole model at once; this one is about reducing parts.

METHODS.
  * `guyan(K, M, master)`           static condensation: keep `master` DOFs, slave DOFs follow
                                     statically. Exact for static loads on the master DOFs; the mass
                                     is approximate, so frequencies are accurate only for modes much
                                     lower than the first slave-DOF mode.
  * `craig_bampton(K, M, interface, n_modes)`
                                     Craig-Bampton (Hurty/Craig-Bampton fixed-interface method): the
                                     basis is  [ constraint modes | fixed-interface normal modes ].
                                     Reduced coordinates are [interface displacements; modal
                                     amplitudes]. K_r is block-diagonal in the modal part (the
                                     eigenvalues) and M_r has identity there (mass-normalised modes).
  * `couple(subs, interface_maps)`  joins several Craig-Bampton substructures by identifying their
                                     interface DOFs (displacement compatibility), giving one reduced
                                     model with `solve_modal()`.

Everything works on plain arrays; the package's usual rule (never imports fea_engine). Interface DOFs
must be listed in the same physical order in every substructure that shares them (the
`interface_maps` give each substructure's interface DOF a global interface index).

CHECKED IN THE TESTS against full-order eigenfrequencies of a beam split into two substructures,
including the exactness limit (keeping all fixed-interface modes reproduces the full model) and the
expected monotone improvement with the number of retained modes.
"""
__author__ = "Abhijeet"
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh


def _arr(x):
    return np.asarray(x, dtype=float)


def _split(n, keep):
    keep = np.asarray(keep, dtype=int)
    if len(set(keep.tolist())) != len(keep) or keep.min(initial=0) < 0 or keep.max(initial=-1) >= n:
        raise ValueError("DOF list must contain distinct indices inside 0..n-1")
    rest = np.setdiff1d(np.arange(n), keep)
    return keep, rest


# ----------------------------------------------------------------------------- Guyan
@dataclass
class GuyanModel:
    T: np.ndarray            # (n, n_master) static transformation  u = T u_master
    K: np.ndarray
    M: np.ndarray
    master: np.ndarray
    slave: np.ndarray

    def expand(self, u_master):
        return self.T @ _arr(u_master)


def guyan(K, M, master):
    """Static (Guyan-Irons) condensation onto the `master` DOFs.

    u = T u_m with T = [I; -K_ss^-1 K_sm] (rows ordered like the original DOFs)."""
    K, M = _arr(K), _arr(M)
    n = K.shape[0]
    master, slave = _split(n, master)
    Ksm = K[np.ix_(slave, master)]
    Psi = -np.linalg.solve(K[np.ix_(slave, slave)], Ksm)
    T = np.zeros((n, len(master)))
    T[master, :] = np.eye(len(master))
    T[slave, :] = Psi
    return GuyanModel(T, T.T @ K @ T, T.T @ M @ T, master, slave)


# ----------------------------------------------------------------------------- Craig-Bampton
@dataclass
class CraigBamptonModel:
    """Reduced substructure. Coordinates: [u_interface (n_b); modal amplitudes eta (n_m)]."""
    T: np.ndarray                 # (n, n_b + n_m) basis: constraint modes then fixed-interface modes
    K: np.ndarray                 # (n_b+n_m)^2
    M: np.ndarray
    interface: np.ndarray         # original DOF indices of the interface (n_b)
    internal: np.ndarray          # original DOF indices of the internal DOFs
    omega_fixed: np.ndarray       # fixed-interface natural frequencies [rad/s] kept
    n_interface: int
    n_modes: int

    def expand(self, coords):
        """Full-order displacement from reduced coordinates."""
        return self.T @ _arr(coords)

    @property
    def freq_fixed_hz(self):
        return self.omega_fixed / (2 * np.pi)


def craig_bampton(K, M, interface, n_modes=None):
    """Craig-Bampton reduction of one substructure.

    Parameters
    ----------
    K, M : (n, n) full-order stiffness and mass (symmetric; K may be singular only through the
        interface, i.e. K_ii must be positive definite).
    interface : indices of boundary (interface) DOFs kept physical.
    n_modes : number of fixed-interface normal modes to keep (default: all).
    """
    K, M = _arr(K), _arr(M)
    n = K.shape[0]
    b, i = _split(n, interface)
    Kii, Kib = K[np.ix_(i, i)], K[np.ix_(i, b)]
    Mii = M[np.ix_(i, i)]
    nm = len(i) if n_modes is None else int(min(n_modes, len(i)))
    w2, Phi = eigh(Kii, Mii)
    if w2[0] <= 0:
        raise ValueError("K_ii is not positive definite: the internal DOFs are not constrained by the "
                         "interface (rigid-body motion left over)")
    Phi = Phi[:, :nm]                         # mass-normalised: Phi^T M_ii Phi = I
    Psi = -np.linalg.solve(Kii, Kib)          # constraint modes
    T = np.zeros((n, len(b) + nm))
    T[b, :len(b)] = np.eye(len(b))
    T[i, :len(b)] = Psi
    T[i, len(b):] = Phi
    Kr = T.T @ K @ T
    Mr = T.T @ M @ T
    # these blocks are known analytically; set them exactly instead of keeping round-off noise
    nb = len(b)
    Kr[:nb, nb:] = 0.0
    Kr[nb:, :nb] = 0.0
    Kr[nb:, nb:] = np.diag(w2[:nm])
    Mr[nb:, nb:] = np.eye(nm)
    return CraigBamptonModel(T, Kr, Mr, b, i, np.sqrt(w2[:nm]), len(b), nm)


# ----------------------------------------------------------------------------- coupling
class CoupledModel:
    """Several Craig-Bampton substructures joined at shared interface DOFs.

    Global coordinates: [global interface DOFs (n_g); modes of substructure 0; modes of 1; ...].
    """

    def __init__(self, subs, interface_maps):
        if len(subs) != len(interface_maps):
            raise ValueError("one interface map per substructure")
        maps = [np.asarray(m, dtype=int) for m in interface_maps]
        for s, m in zip(subs, maps):
            if len(m) != s.n_interface:
                raise ValueError("interface map length must equal the substructure's n_interface")
        self.subs, self.maps = list(subs), maps
        self.n_g = int(max(m.max() for m in maps)) + 1
        self.offsets = np.cumsum([0] + [s.n_modes for s in subs]) + self.n_g
        n_tot = self.n_g + sum(s.n_modes for s in subs)
        K = np.zeros((n_tot, n_tot))
        M = np.zeros((n_tot, n_tot))
        for k, (s, m) in enumerate(zip(subs, maps)):
            # local -> global coordinate index of each reduced coordinate of this substructure
            gidx = np.concatenate([m, self.offsets[k] + np.arange(s.n_modes)])
            K[np.ix_(gidx, gidx)] += s.K
            M[np.ix_(gidx, gidx)] += s.M
        self.K, self.M = K, M

    @property
    def n(self):
        return self.K.shape[0]

    def solve_modal(self, n_modes=6):
        """Lowest natural frequencies [Hz] and shapes in reduced coordinates."""
        w2, X = eigh(self.K, self.M)
        w2 = np.clip(w2, 0.0, None)
        k = min(n_modes, len(w2))
        return np.sqrt(w2[:k]) / (2 * np.pi), X[:, :k]

    def local_coords(self, k, x):
        """Reduced coordinates of substructure k extracted from a global reduced vector x."""
        s = self.subs[k]
        gidx = np.concatenate([self.maps[k], self.offsets[k] + np.arange(s.n_modes)])
        return x[gidx]

    def expand(self, k, x):
        """Full-order displacement of substructure k for a global reduced vector x."""
        return self.subs[k].expand(self.local_coords(k, x))


def couple(subs, interface_maps):
    """Join Craig-Bampton substructures; see `CoupledModel`."""
    return CoupledModel(subs, interface_maps)
