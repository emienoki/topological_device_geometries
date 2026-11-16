import numpy as np
from poisoning import (xi_analytic, bulk_gap, kT_over_t, code_time_to_ns, OMEGA_A_PER_NS, poisoning_rate)
from device_builder import DeviceParams, build_t_junction, bdg_hamiltonian
from tjunction_exchange import site_mu, walls_at
from holonomy import Model, BRAID, path_fractions
from virtual_trijunction import build_virtual_trijunction_device

T_GRID = [100, 200, 400, 800, 1600, 3200]        


def tj_phases(s, dynamic):
    if not dynamic:
        return (np.pi, 0.0, 0.0)
    p = s * 6
    f2 = _ramp(p - 0)          # leg 1a: phi2 0 -> pi  (arm2 retracted)
    f0 = _ramp(p - 2)          # leg 2a: phi0 pi -> 0  (arm0 retracted)
    f1 = _ramp(p - 4)          # leg 3a: phi1 0 -> pi  (arm1 retracted)
    return (np.pi * (1 - f0), np.pi * f1, np.pi * f2)


class TJDesign:
    def __init__(self, N=12, delta=0.6, k=6, c=3, width=2.0, dynamic=True):
        self.N, self.delta, self.k, self.c, self.width, self.dynamic = N, delta, k, c, width, dynamic

    def device_at(self, s):
        dev = build_t_junction(DeviceParams(N_per_arm=self.N, t=1.0, delta=self.delta, mu=0.0,
                                            phases=tj_phases(s, self.dynamic)))
        dev.site_mu = list(site_mu(self.N, walls_at(s, self.k, self.c), self.width))
        return dev

    def _parts(self):
        if hasattr(self, "_S"):
            return
        N, n = self.N, 3 * self.N + 1
        self._hop = np.zeros((n, n)); self._S = [np.zeros((n, n)) for _ in range(3)]
        for a in range(3):
            st = 1 + a * N
            for (i, j) in [(0, st)] + [(st + q, st + q + 1) for q in range(N - 1)]:
                self._hop[i, j] = self._hop[j, i] = -1.0
                self._S[a][i, j], self._S[a][j, i] = self.delta, -self.delta

    def H_at(self, s):
        self._parts()
        xi = self._hop - np.diag(site_mu(self.N, walls_at(s, self.k, self.c), self.width))
        D = sum(np.exp(1j * ph) * S for ph, S in zip(tj_phases(s, self.dynamic), self._S))
        return 0.5 * np.block([[xi, D], [-D.conj(), -xi]])

    def qubit_sites(self, s):
        w = walls_at(s, self.k, self.c)
        ends = [1 + a * self.N + int(round(wa)) for a, wa in enumerate(w) if wa >= 0.5]   # site at distance w (off-by-one fixed)
        return ends if len(ends) >= 2 else ends + [0]


class AncillaDesign:
    def __init__(self, N=12, delta=0.6, g=0.6):
        self.N, self.delta, self.g = N, delta, g
        self.m = Model(N=N, t=1.0, delta=delta, g_max=g)

    def device_at(self, s):
        e = self.m.r + (1 - self.m.r) * path_fractions(s, BRAID)
        return build_virtual_trijunction_device(self.N, 1.0, self.delta, self.m.mu_max * e[0],
                                                self.g * e[1], self.g * e[2], np.pi)

    def H_at(self, s):
        return self.m.H_at(s, BRAID)

    def qubit_sites(self, s):
        return None   # exact 4-fold zero manifold throughout


def manifold(H, cut=0.1):
    w, v = np.linalg.eigh(H)
    return v[:, np.abs(2 * w) < cut]


def evaluate(design, T_mK=30.0, dt=0.25, n_path=72):
    xi, gap = xi_analytic(1.0, design.delta, 0.0), bulk_gap(1.0, design.delta, 0.0)
    kT = kT_over_t(T_mK)
    ss = (np.arange(n_path) + 0.5) / n_path
    Gbar = float(np.mean([poisoning_rate(design.device_at(s), xi, kT, gap, qubit_sites=design.qubit_sites(s))
                          for s in ss]))
    V0, V1 = manifold(design.H_at(0.0)), manifold(design.H_at(1.0))
    assert V0.shape[1] == V1.shape[1], (V0.shape, V1.shape)
    rows = []
    for Tc in T_GRID:
        n = max(200, int(Tc / dt)); U = np.eye(V0.shape[0], dtype=complex)
        for k in range(n):
            ww, vv = np.linalg.eigh(design.H_at((k + 0.5) / n))
            U = (vv * np.exp(-1j * ww * Tc / n)) @ vv.conj().T @ U
        leak = float(1 - np.linalg.svd(V1.conj().T @ U @ V0, compute_uv=False).min() ** 2)
        T_ns = code_time_to_ns(Tc)
        rows.append(dict(T_ns=T_ns, leakage=leak, poisoning=Gbar * OMEGA_A_PER_NS * T_ns))
    tot = [r["leakage"] + r["poisoning"] for r in rows]
    b = int(np.argmin(tot))
    return dict(J=tot[b], T_opt_ns=rows[b]["T_ns"], leak_at_opt=rows[b]["leakage"],
                pois_at_opt=rows[b]["poisoning"], mean_rate=Gbar, dim=V0.shape[1], rows=rows)
