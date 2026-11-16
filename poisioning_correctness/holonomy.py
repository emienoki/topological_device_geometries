import numpy as np
from basic_physics.device_builder import bdg_hamiltonian
from basic_physics.virtual_trijunction import build_virtual_trijunction_device

# (mu_D, Gamma_L, Gamma_R) states. C1 L up, C2 D down, C3 R up, C4 L down, C5 D up, C6 R down
BRAID = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 0, 0)]
BRAID_REVERSED = BRAID[::-1]
BACKTRACK = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # retraces itself, encloses nothing


def ramp(x):
    return 0.5 * (1 - np.cos(np.pi * np.clip(x, 0.0, 1.0)))


def path_fractions(s, states):
    n = len(states) - 1
    leg = min(int(s * n), n - 1)
    a, b = np.array(states[leg], float), np.array(states[leg + 1], float)
    return a + (b - a) * ramp(s * n - leg)


class Model:
    def __init__(self, N=4, t=1.0, delta=1.0, g_max=1.0, phase_R=np.pi, r_D=0.0, r_L=0.0, r_R=0.0):
        self.N = N
        mk = lambda mu, gl, gr: bdg_hamiltonian(build_virtual_trijunction_device(N, t, delta, mu, gl, gr, phase_R))
        self.H0 = mk(0, 0, 0)
        self.HD, self.HL, self.HR = mk(1, 0, 0) - self.H0, mk(0, 1, 0) - self.H0, mk(0, 0, 1) - self.H0
        assert np.allclose(mk(0.7, 0.3, 0.2), self.H0 + 0.7 * self.HD + 0.3 * self.HL + 0.2 * self.HR)
        self.mu_max, self.g_max = 2.0 * g_max, g_max   # equal Majorana couplings (calibrated: mu_D enters as mu_D/2)
        self.r = np.array([r_D, r_L, r_R])             # residual "off" value as fraction of max

    def H_at(self, s, states):
        eff = self.r + (1 - self.r) * path_fractions(s, states)
        return self.H0 + self.mu_max * eff[0] * self.HD + self.g_max * eff[1] * self.HL + self.g_max * eff[2] * self.HR


def zero_manifold(H, tol=1e-8):
    w, v = np.linalg.eigh(H)
    return v[:, np.abs(w) < tol]


def polar(M):
    u, _, vh = np.linalg.svd(M)
    return u @ vh


def propagator(model, states, T, dt=0.1):
    n = max(80, int(np.ceil(T / dt)))
    U = np.eye(model.H0.shape[0], dtype=complex)
    for k in range(n):
        w, v = np.linalg.eigh(model.H_at((k + 0.5) / n, states))
        U = (v * np.exp(-1j * w * T / n)) @ v.conj().T @ U
    return U


def read_angles(W):
    phases = np.angle(np.linalg.eigvals(W))
    c = np.clip((np.real(np.trace(W)) - 2.0) / 2.0, -1.0, 1.0)
    return float(np.max(np.abs(phases))), float(np.arccos(c)), np.sort(phases)


def holonomy_dynamic(model, states, T=None, dt=0.1, U=None):
    V0 = zero_manifold(model.H_at(0.0, states))
    if U is None:
        U = propagator(model, states, T, dt)
    W = V0.conj().T @ U @ V0
    sv = np.linalg.svd(W, compute_uv=False)
    Wp = polar(W)
    te, tt, ph = read_angles(Wp)
    return dict(theta_eig=te, theta_tr=tt, phases=ph, leakage=float(1 - sv.min() ** 2), W=Wp, V0=V0, U=U)


def holonomy_wilson(model, states, M=3000, random_gauge=False, seed=0):
    rng = np.random.default_rng(seed)
    Vs = []
    for k in range(M + 1):
        V = zero_manifold(model.H_at(k / M, states))
        if random_gauge and 0 < k < M:
            d = V.shape[1]
            Q, _ = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
            V = V @ Q
        Vs.append(V)
    W = np.eye(Vs[0].shape[1], dtype=complex)
    for k in range(M):
        W = polar(Vs[k + 1].conj().T @ Vs[k]) @ W
    W = polar(Vs[0].conj().T @ Vs[M]) @ W
    te, tt, ph = read_angles(W)
    return dict(theta_eig=te, theta_tr=tt, phases=ph, W=W, V0=Vs[0], dims=sorted({V.shape[1] for V in Vs}))


def which_modes(res):
    w, vecs = np.linalg.eig(res["W"])
    order = np.argsort(np.abs(np.angle(w)))
    n = res["V0"].shape[0] // 2
    out = {}
    for label, idx in [("fixed", order[:2]), ("rotated", order[2:])]:
        dens = np.zeros(n)
        for i in idx:
            psi = res["V0"] @ vecs[:, i]
            dens += np.abs(psi[:n]) ** 2 + np.abs(psi[n:]) ** 2
        top = np.argsort(dens)[::-1][:3]
        out[label] = [(int(s), round(float(dens[s]) / 2, 3)) for s in top]
    return out
