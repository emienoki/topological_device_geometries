import numpy as np  # type: ignore[reportMissingImports]
from collections import deque
from basic_physics.device_builder import bdg_hamiltonian
from basic_physics.braiding_dynamics import plain_chain_device

T_UEV = 12.0                      #hopping scale t in micro-eV
KB_UEV_PER_MK = 0.08617333262     #Boltzmann constant, micro-eV per millikelvin
HBAR_UEV_NS = 0.6582119569        # hbar in micro-eV * ns
OMEGA_A_PER_NS = 1.0 / 50.0       # attempt frequency


def kT_over_t(T_mK, t_ueV=T_UEV):
    return KB_UEV_PER_MK * T_mK / t_ueV


def code_time_to_ns(tau_code, t_ueV=T_UEV):
    """exp(-i H_code tau) with H_code = H_true/2 in units of t (hbar = 1),
    so physical time = (tau/2) * hbar/t."""
    return 0.5 * tau_code * HBAR_UEV_NS / t_ueV


# distances
def graph_distances(device):
    n = device.n_sites
    adj = [[] for _ in range(n)]
    for (i, j, t_ij, d_ij) in device.bonds:
        if abs(t_ij) < 1e-12 and abs(d_ij) < 1e-12:
            continue                      # a switched-off coupling is not a tunneling path
        adj[i].append(j)
        adj[j].append(i)
    D = np.full((n, n), np.inf)
    for s in range(n):
        D[s, s] = 0
        q = deque([s])
        while q:
            a = q.popleft()
            for b in adj[a]:
                if D[s, b] == np.inf:
                    D[s, b] = D[s, a] + 1
                    q.append(b)
    return D


# coherence length
def xi_analytic(t, delta, mu):
    best = None
    for a, c in [(t + delta, t - delta), (t - delta, t + delta)]:
        if abs(a) < 1e-14:
            continue
        r = np.roots([a, mu, c])
        m = np.max(np.abs(r))
        if m < 1 and (best is None or m < best):
            best = m
    if best is None:
        return np.nan
    return 0.0 if best < 1e-12 else -1.0 / np.log(best)


def xi_numeric(t, delta, mu, N=160):
    H = bdg_hamiltonian(plain_chain_device(N, t, delta, [mu] * N))
    w, v = np.linalg.eigh(H)
    V = v[:, np.argsort(np.abs(w))[:2]]
    P = np.zeros(2 * N)
    P[:N // 2] = 1
    P[N:N + N // 2] = 1
    pw, pv = np.linalg.eigh((V.conj().T * P) @ V)
    psi = V @ pv[:, -1]                       # the combination living on the left half
    rho = (np.abs(psi[:N]) ** 2 + np.abs(psi[N:]) ** 2)[:N // 2]
    if rho[1:].sum() < 1e-24:
        return 0.0
    W = np.cumsum(rho[::-1])[::-1]            # tail weight beyond site j (smooths oscillations)
    j = np.arange(len(W))
    ok = (W < 1e-3) & (W > 1e-18)
    if ok.sum() < 4:
        return np.nan
    slope = np.polyfit(j[ok], np.log(W[ok]), 1)[0]
    return -2.0 / slope


def bulk_gap(t, delta, mu, nk=20001):
    """Bulk gap of the uniform infinite Kitaev chain (true energy units):
    E_k = sqrt((2t cos k + mu)^2 + (2 Delta sin k)^2); sweet spot gives 2t."""
    k = np.linspace(0, np.pi, nk)
    return float(np.min(np.sqrt((2 * t * np.cos(k) + mu) ** 2 + (2 * delta * np.sin(k)) ** 2)))


# model
def decompose(device, gap, qubit_sites=None, radius=2, zero_tol=1e-3, subgap_frac=0.98):
    """split subgap section"""
    n = device.n_sites
    H = bdg_hamiltonian(device)
    w, v = np.linalg.eigh(H)
    E = 2.0 * w                                            # true energies
    dens = lambda M: np.abs(M[:n]) ** 2 + np.abs(M[n:]) ** 2
    S = v[:, np.abs(E) < subgap_frac * gap]
    if qubit_sites is None:
        qmask = np.abs(E[np.abs(E) < subgap_frac * gap]) < zero_tol
        Q, R = S[:, qmask], S[:, ~qmask]
    else:
        D = graph_distances(device)
        near = np.any(D[:, list(qubit_sites)] <= radius, axis=1).astype(float)
        P = np.concatenate([near, near])
        pw, pv = np.linalg.eigh((S.conj().T * P) @ S)
        U = S @ pv
        Q, R = U[:, pw > 0.5], U[:, pw <= 0.5]
    if R.shape[1]:
        eR, cR = np.linalg.eigh(R.conj().T @ H @ R)
        R, ER = R @ cR, 2.0 * eR
    else:
        ER = np.zeros(0)
    return dict(rho_q=dens(Q).sum(axis=1), n_qubit=Q.shape[1],
                rho_src=dens(R), E_src=ER, E_all=E)


def poisoning_rate(device, xi, kT, gap, qubit_sites=None, radius=2, zero_tol=1e-3,
                   subgap_frac=0.98, breakdown=False):
    """total subgap poisoning rate of the qubit Majoranas"""
    d = decompose(device, gap, qubit_sites, radius, zero_tol, subgap_frac)
    D = graph_distances(device)
    K = (D == 0).astype(float) if xi == 0 else np.exp(-2.0 * D / xi)
    kq = d["rho_q"] @ K
    contrib = 0.5 * (kq @ d["rho_src"]) * np.exp(-np.abs(d["E_src"]) / kT)
    total = float(contrib.sum())
    if not breakdown:
        return total
    near_zero = np.abs(d["E_src"]) < 10 * kT * 1e-3
    order = np.argsort(contrib)[::-1][:4]
    return dict(total=total, n_qubit=d["n_qubit"], n_source_directions=len(d["E_src"]),
                from_near_zero_sources=float(contrib[near_zero].sum()),
                top_sources=[(round(float(abs(d["E_src"][k])), 4), float(contrib[k])) for k in order if contrib[k] > 0])


# full model (no gap cutoff)
def decompose_full(device, gap, qubit_sites=None, radius=2, zero_tol=0.1):
    """Like decompose(), but the sources are every non-qubit BdG state, including
    the chain's own above-gap continuum"""
    n = device.n_sites
    H = bdg_hamiltonian(device)
    w, v = np.linalg.eigh(H)
    E = 2.0 * w
    dens = lambda M: np.abs(M[:n]) ** 2 + np.abs(M[n:]) ** 2
    sub = np.abs(E) < gap
    S = v[:, sub]
    if qubit_sites is None:
        Q = S[:, np.abs(E[sub]) < zero_tol]
    else:
        D = graph_distances(device)
        near = np.any(D[:, list(qubit_sites)] <= radius, axis=1).astype(float)
        P = np.concatenate([near, near])
        pw, pv = np.linalg.eigh((S.conj().T * P) @ S)
        Q = (S @ pv)[:, pw > 0.5]
    # complement of Q in the full space
    Pq = Q @ Q.conj().T
    cw, cv = np.linalg.eigh(np.eye(2 * n) - Pq)
    R = cv[:, cw > 0.5]
    eR, cR = np.linalg.eigh(R.conj().T @ H @ R)
    return dict(rho_q=dens(Q).sum(axis=1), n_qubit=Q.shape[1], rho_src=dens(R @ cR), E_src=2.0 * eR)


def poisoning_rate_full(device, xi, kT, gap, qubit_sites=None, radius=2, zero_tol=0.1, split=False):
    """Thermal poisoning from all non-qubit states, in units of omega_a"""
    d = decompose_full(device, gap, qubit_sites, radius, zero_tol)
    D = graph_distances(device)
    K = (D == 0).astype(float) if xi == 0 else np.exp(-2.0 * D / xi)
    c = 0.5 * ((d["rho_q"] @ K) @ d["rho_src"]) * np.exp(-np.abs(d["E_src"]) / kT)
    if not split:
        return float(c.sum())
    above = np.abs(d["E_src"]) >= gap
    return dict(total=float(c.sum()), below=float(c[~above].sum()), above=float(c[above].sum()),
                n_qubit=d["n_qubit"])


# ------------------------------------- parent-superconductor (non-equilibrium) channel
def parent_sc_error(rate_per_majorana_per_s, n_majorana, T_ns):
    return n_majorana * rate_per_majorana_per_s * T_ns * 1e-9
