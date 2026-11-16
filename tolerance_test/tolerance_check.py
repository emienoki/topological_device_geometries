import numpy as np, json, sys
from tolerance import make, BASE, NS, perturbation, perturbed, braid_matrix, gate_error
from sweep_lib import manifold

def polar(M):
    u, _, vh = np.linalg.svd(M); return u @ vh

def own_states_score(arch, kind, sigma, seed, dt=0.25):
    d = make(arch); Tc = BASE[arch]["T_opt_full"] / NS; n = d.H_at(0).shape[0] // 2
    V0, V1 = manifold(d.H_at(0.0)), manifold(d.H_at(1.0))
    M0 = braid_matrix(d, Tc); Q0 = polar(M0)
    p = perturbation(arch, kind, sigma, n, np.random.default_rng(seed))
    H0p, H1p = perturbed(d.H_at(0.0), *p), perturbed(d.H_at(1.0), *p)
    W0, W1 = manifold(H0p), manifold(H1p)
    split = float(np.max(np.abs(2 * np.linalg.eigvalsh(W0.conj().T @ H0p @ W0))))
    steps = max(200, int(Tc / dt)); Psi = W0.astype(complex)
    for k in range(steps):
        w, v = np.linalg.eigh(perturbed(d.H_at((k + 0.5) / steps), *p))
        Psi = v @ (np.exp(-1j * w * Tc / steps)[:, None] * (v.conj().T @ Psi))
    Mown = W1.conj().T @ Psi
    A0, A1 = polar(V0.conj().T @ W0), polar(V1.conj().T @ W1)
    Mal = A1 @ Mown @ A0.conj().T
    Mideal_basis = braid_matrix(d, Tc, p)
    return dict(ideal_basis=gate_error(Mideal_basis, Q0), own_basis=gate_error(Mal, Q0),
                start_overlap_loss=float(1 - np.linalg.svd(V0.conj().T @ W0, compute_uv=False).min() ** 2),
                splitting=split)

for arch, kind, sigma in [("ancilla", "phase", 0.01), ("ancilla", "phase", 0.03), ("ancilla", "site", 0.03),
                          ("tjunction", "phase", 0.1), ("tjunction", "balance", 0.03)]:
    rs = [own_states_score(arch, kind, sigma, s) for s in (1, 2, 3)]
    print(arch, kind, sigma, {k: f"{np.mean([r[k] for r in rs]):.2e}" for k in rs[0]})
