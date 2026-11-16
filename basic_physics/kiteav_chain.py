import numpy as np
import matplotlib.pyplot as plt


def kitaev_chain_bdg(N, t, mu, delta):
    xi = np.zeros((N, N))
    for i in range(N):
        xi[i, i] = -mu
    for i in range(N - 1):
        xi[i, i + 1] = -t
        xi[i + 1, i] = -t

    Delta = np.zeros((N, N))
    for i in range(N - 1):
        Delta[i, i + 1] = delta
        Delta[i + 1, i] = -delta

    H_BdG = np.block([
        [xi,            Delta],
        [-Delta.conj(), -xi.conj()]
    ]) / 2.0

    return H_BdG

"""diagonalize the bdg"""
def spectrum_vs_mu(N, t, delta, mu_range):
    energies = np.zeros((len(mu_range), 2 * N))
    for k, mu in enumerate(mu_range):
        H = kitaev_chain_bdg(N, t, mu, delta)
        energies[k] = np.linalg.eigvalsh(H)
    return energies

"""return nmodes eigenvalues/eigenvectors closest to zero energy""" 
def majorana_wavefunctions(N, t, mu, delta, n_modes=2):
    H = kitaev_chain_bdg(N, t, mu, delta)
    eigvals, eigvecs = np.linalg.eigh(H)
    idx_sorted = np.argsort(np.abs(eigvals))
    idx_zero = idx_sorted[:n_modes]
    return eigvals[idx_zero], eigvecs[:, idx_zero]


if __name__ == "__main__":
    N = 50        # chain length (sites)
    t = 1.0       # hopping, sets energy scale
    delta = 1.0   # pairing amplitude (delta = t is the "sweet spot" Kitaev point)

# topoligcal phase transititon   
    energies = spectrum_vs_mu(N, t, delta, mu_range)

    plt.figure(figsize=(7, 5))
    for band in range(energies.shape[1]):
        plt.plot(mu_range / t, energies[:, band], color="steelblue", lw=0.6)
    plt.axvline(-2, color="gray", ls="--", lw=1, label=r"$\mu/t = \pm 2$ (phase boundary)")
    plt.axvline(2, color="gray", ls="--", lw=1)
    plt.axhline(0, color="black", lw=0.5)
    plt.xlabel(r"$\mu / t$")
    plt.ylabel("Energy / t")
    plt.title("Kitaev Chain: BdG Spectrum vs Chemical Potential")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spectrum_vs_mu.png", dpi=150)
    plt.close()
    print("Saved spectrum_vs_mu.png")

#wave function localization 
    mu_topo = 0.0  
    eigvals_zero, eigvecs_zero = majorana_wavefunctions(N, t, mu_topo, delta)

    plt.figure(figsize=(7, 5))
    sites = np.arange(N)
    for k in range(2):
        u = eigvecs_zero[:N, k]
        v = eigvecs_zero[N:, k]
        density = np.abs(u) ** 2 + np.abs(v) ** 2
        plt.plot(sites, density, marker="o", ms=3,
                 label=f"mode {k+1}, E = {eigvals_zero[k]:.2e} t")
    plt.xlabel("Site index")
    plt.ylabel(r"$|u_i|^2 + |v_i|^2$")
    plt.title(f"Near-Zero Mode Localization (topological phase, $\\mu$ = {mu_topo})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("majorana_localization.png", dpi=150)
    plt.close()
    print("Saved majorana_localization.png")

#trivial phase comparison 
    mu_trivial = 3.5 * t
    eigvals_triv, eigvecs_triv = majorana_wavefunctions(N, t, mu_trivial, delta)

    plt.figure(figsize=(7, 5))
    for k in range(2):
        u = eigvecs_triv[:N, k]
        v = eigvecs_triv[N:, k]
        density = np.abs(u) ** 2 + np.abs(v) ** 2
        plt.plot(sites, density, marker="o", ms=3,
                 label=f"mode {k+1}, E = {eigvals_triv[k]:.2f} t")
    plt.xlabel("Site index")
    plt.ylabel(r"$|u_i|^2 + |v_i|^2$")
    plt.title(f"Same Check in the TRIVIAL Phase ($\\mu$ = {mu_trivial} t) — No Edge Localization")
    plt.legend()
    plt.tight_layout()
    plt.savefig("trivial_phase_comparison.png", dpi=150)
    plt.close()
    print("Saved trivial_phase_comparison.png")

    print(f"\nLowest |energy| pair in topological phase (mu=0): {eigvals_zero}")
    print(f"Lowest |energy| pair in trivial phase (mu={mu_trivial}): {eigvals_triv}")
