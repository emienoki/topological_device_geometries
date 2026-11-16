
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from device_builder import Device, bdg_hamiltonian


def plain_chain_device(N, t, delta, site_mu):
    bonds = [(i, i + 1, t, delta) for i in range(N - 1)]
    return Device(n_sites=N, bonds=bonds, site_mu=list(site_mu),
                  arm_ends=[0, N - 1], junction_site=-1, lead_site=0, trap_site=N - 1,
                  params=None)


def wall_mu_profile(N, wall_position, width, mu_topo, mu_triv):
    sites = np.arange(N)
    frac = 0.5 * (1 + np.tanh((sites - wall_position) / width))
    return mu_topo + (mu_triv - mu_topo) * frac


def localized_zero_mode(H, N, target_site, n_search=6, degeneracy_tol=1e-6):
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argsort(np.abs(eigvals))[:n_search]
    e0 = eigvals[idx[0]]
    degenerate_idx = [k for k in idx if abs(eigvals[k] - e0) < degeneracy_tol or abs(eigvals[k]) < degeneracy_tol]
    if not degenerate_idx:
        degenerate_idx = [idx[0]]

    best_psi, best_norm = None, -1.0
    for v_sign in (1.0, -1.0):
        trial = np.zeros(2 * N, dtype=complex)
        trial[target_site] = 1.0
        trial[N + target_site] = v_sign
        psi = np.zeros(2 * N, dtype=complex)
        for k in degenerate_idx:
            psi += eigvecs[:, k] * np.vdot(eigvecs[:, k], trial)
        norm = np.linalg.norm(psi)
        if norm > best_norm:
            best_norm, best_psi = norm, psi

    if best_norm < 1e-10:
        raise RuntimeError(f"Near-zero eigenspace has ~no overlap with either Majorana "
                            f"handedness at site {target_site}")
    best_psi /= best_norm
    mean_E = np.mean([eigvals[k] for k in degenerate_idx])
    return mean_E, best_psi


def min_gap_along_sweep(N, t, delta, mu_topo, mu_triv, width, k0, k1, n_check=30):
    gaps = []
    for s in np.linspace(0, 1, n_check):
        wall = k0 + (k1 - k0) * s
        mu = wall_mu_profile(N, wall, width, mu_topo, mu_triv)
        dev = plain_chain_device(N, t, delta, mu)
        eigvals = np.linalg.eigvalsh(bdg_hamiltonian(dev))
        sorted_abs = np.sort(np.abs(eigvals))
        gaps.append(sorted_abs[2])  # [0],[1] = the two near-zero modes; [2] = first excited
    return min(gaps)


def simulate_sweep(N, t, delta, mu_topo, mu_triv, width, k0, k1, T, n_steps):
    dt = T / n_steps

    mu0 = wall_mu_profile(N, k0, width, mu_topo, mu_triv)
    H0 = bdg_hamiltonian(plain_chain_device(N, t, delta, mu0))
    _, psi = localized_zero_mode(H0, N, target_site=k0)
    psi = psi.astype(complex)

    for step in range(n_steps):
        s = (step + 0.5) / n_steps  # Hamiltonian at the midpoint of this interval
        wall = k0 + (k1 - k0) * s
        mu = wall_mu_profile(N, wall, width, mu_topo, mu_triv)
        H = bdg_hamiltonian(plain_chain_device(N, t, delta, mu))
        psi = expm(-1j * H * dt) @ psi
        psi = psi / np.linalg.norm(psi)  # guard against numerical drift

    mu1 = wall_mu_profile(N, k1, width, mu_topo, mu_triv)
    H1 = bdg_hamiltonian(plain_chain_device(N, t, delta, mu1))
    _, psi_target = localized_zero_mode(H1, N, target_site=k1)

    fidelity = float(np.abs(np.vdot(psi_target, psi)) ** 2)
    return fidelity, psi, psi_target


if __name__ == "__main__":
    N, t, delta = 40, 1.0, 1.0
    mu_topo, mu_triv = 0.0, 4.0
    width = 2.0
    k0, k1 = 8, 25

    gap_min = min_gap_along_sweep(N, t, delta, mu_topo, mu_triv, width, k0, k1)
    print(f"Minimum gap encountered during the sweep: {gap_min:.4f} (units of t)")
    print(f"Naive adiabatic-theorem estimate: T_needed ~ 1/gap_min^2 = {1/gap_min**2:.2f}\n")

    T_values = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    fidelities = []
    for T in T_values:
        n_steps = max(80, int(T * 15))
        fid, _, _ = simulate_sweep(N, t, delta, mu_topo, mu_triv, width, k0, k1, T, n_steps)
        fidelities.append(fid)
        print(f"  T={T:6.1f}  n_steps={n_steps:4d}  fidelity={fid:.6f}  infidelity={1-fid:.2e}")

    plt.figure(figsize=(7, 5))
    plt.semilogx(T_values, fidelities, marker="o")
    plt.axhline(0.99, color="gray", ls="--", lw=1, label="99% fidelity")
    plt.axvline(1/gap_min**2, color="coral", ls=":", lw=1.5, label=r"naive $1/{\rm gap}_{min}^2$")
    plt.xlabel("Sweep duration T (1/t)")
    plt.ylabel("Fidelity")
    plt.title("Adiabaticity of a Single Domain-Wall Sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig("braid_fidelity_vs_time.png", dpi=150)
    plt.close()
    print("\nSaved braid_fidelity_vs_time.png")
