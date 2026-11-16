import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DeviceParams:
    N_per_arm: int = 15          #dots per arm 
    a: float = 10.0              #inter-dot spacing, nm 
    t: float = 1.0               #hopping amp
    delta: float = 1.0           #pairing amp 
    mu: float = 0.0              #chemical potential
    branch_width: float = 50.0   #w, nm 
                                  #carried as metadata for the diffusion model D(w)
    phases: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    lead_arm: int = 0            #which arm (0, 1, or 2) hosts the lead
    lead_offset: int = 0         #site index within that arm
    trap_arm: int = 1            #which arm hosts the quasiparticle trap
    trap_offset: int = -1        #site index within that arm


@dataclass
class Device:
    """A concrete, resolved device: sites, bonds, and site-role metadata."""
    n_sites: int
    bonds: List[Tuple[int, int, float, float]]  # (i, j, t_ij, Delta_ij)
    site_mu: List[float]     #per site chem potential 
    arm_ends: List[int]     #the 3 outermost site indices, potential mzm locaitons 
    junction_site: int
    lead_site: int
    trap_site: int
    params: object


def build_t_junction(theta: DeviceParams) -> Device:
    """
    Site layout:
      site 0              = the shared junction site
      sites 1 .. N          = arm A, ordered outward from the junction
      sites N+1 .. 2N         = arm B
      sites 2N+1 .. 3N          = arm C
    """
    N = theta.N_per_arm
    n_sites = 3 * N + 1
    junction = 0
    arm_starts = [1, N + 1, 2 * N + 1]
    bonds = []

    for arm_idx, start in enumerate(arm_starts):
        delta_k = theta.delta * np.exp(1j * theta.phases[arm_idx])
        bonds.append((junction, start, theta.t, delta_k))  # junction -> first site of arm
        for k in range(N - 1):
            i, j = start + k, start + k + 1
            bonds.append((i, j, theta.t, delta_k))

    arm_ends = [start + N - 1 for start in arm_starts]

    def resolve(arm, offset):
        start = arm_starts[arm]
        idx = offset if offset >= 0 else N + offset  # -1 -> last site of the arm
        return start + idx

    lead_site = resolve(theta.lead_arm, theta.lead_offset)
    trap_site = resolve(theta.trap_arm, theta.trap_offset)

    return Device(
        n_sites=n_sites, bonds=bonds, site_mu=[theta.mu] * n_sites, arm_ends=arm_ends,
        junction_site=junction, lead_site=lead_site, trap_site=trap_site,
        params=theta,
    )

def bdg_hamiltonian(device: Device) -> np.ndarray:

    N = device.n_sites

    xi = np.zeros((N, N), dtype=complex)
    for i in range(N):
        xi[i, i] = -device.site_mu[i]
    for (i, j, t_ij, _delta_ij) in device.bonds:
        xi[i, j] = -t_ij
        xi[j, i] = -t_ij

    Delta = np.zeros((N, N), dtype=complex)
    for (i, j, _t_ij, delta_ij) in device.bonds:
        Delta[i, j] = delta_ij
        Delta[j, i] = -delta_ij  # antisymmetric p-wave bond

    H_BdG = np.block([
        [xi, Delta],
        [-Delta.conj(), -xi.conj()]
    ]) / 2.0
    return H_BdG


@dataclass
class AncillaDotParams:
   
    N_per_arm: int = 15
    a: float = 10.0
    t: float = 1.0
    delta: float = 1.0
    mu: float = 0.0
    branch_width: float = 50.0
    phases: Tuple[float, float] = (0.0, 0.0)   # one phase per arm (2 arms, not 3)
    mu_ancilla: float = 0.0      # ancilla's own on-site energy -- independently tunable
    t_ancilla: float = 1.0       # hopping, ancilla <-> each arm's inner end
    delta_ancilla: float = 1.0   # pairing, ancilla <-> each arm's inner end
    lead_arm: int = 0            # 0 or 1
    lead_offset: int = 0
    trap_arm: int = 1
    trap_offset: int = -1


def build_ancilla_dot(theta: AncillaDotParams) -> Device:

    N = theta.N_per_arm
    n_sites = 2 * N + 1
    ancilla = 0
    arm_starts = [1, N + 1]
    bonds = []

    for arm_idx, start in enumerate(arm_starts):
        delta_k = theta.delta * np.exp(1j * theta.phases[arm_idx])
        delta_anc_k = theta.delta_ancilla * np.exp(1j * theta.phases[arm_idx])
        sign = -1.0 if arm_idx == 0 else 1.0
        bonds.append((ancilla, start, theta.t_ancilla, sign * delta_anc_k))
        for k in range(N - 1):
            i, j = start + k, start + k + 1
            bonds.append((i, j, theta.t, sign * delta_k))

    arm_ends = [start + N - 1 for start in arm_starts]

    def resolve(arm, offset):
        start = arm_starts[arm]
        idx = offset if offset >= 0 else N + offset
        return start + idx

    lead_site = resolve(theta.lead_arm, theta.lead_offset)
    trap_site = resolve(theta.trap_arm, theta.trap_offset)

    site_mu = [theta.mu] * n_sites
    site_mu[ancilla] = theta.mu_ancilla

    return Device(
        n_sites=n_sites, bonds=bonds, site_mu=site_mu, arm_ends=arm_ends,
        junction_site=ancilla, lead_site=lead_site, trap_site=trap_site,
        params=theta,
    )


def near_zero_modes(device: Device, n_modes: int = 6):
    """Return the n_modes eigenvalues/eigenvectors closest to zero energy."""
    H = bdg_hamiltonian(device)
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argsort(np.abs(eigvals))[:n_modes]
    return eigvals[idx], eigvecs[:, idx]


if __name__ == "__main__":
    theta = DeviceParams()  # defaults: N_per_arm=15, t=delta=1.0, mu=0.0 (Kitaev sweet point)
    device = build_t_junction(theta)
    print(f"Built T-junction: {device.n_sites} sites, {len(device.bonds)} bonds")
    print(f"Junction site: {device.junction_site}, arm ends: {device.arm_ends}")
    print(f"Lead site: {device.lead_site}, trap site: {device.trap_site}")

    eigvals, eigvecs = near_zero_modes(device, n_modes=8)
    print(f"\n8 eigenvalues closest to zero:\n{eigvals}")

    N = device.n_sites
    sites = np.arange(N)
    plt.figure(figsize=(9, 5))
    for k in range(4):  # the 4 closest-to-zero modes
        u = eigvecs[:N, k]
        v = eigvecs[N:, k]
        density = np.abs(u) ** 2 + np.abs(v) ** 2
        plt.plot(sites, density, marker="o", ms=3, label=f"mode {k+1}, E={eigvals[k]:.2e}")
    for end in device.arm_ends:
        plt.axvline(end, color="gray", ls=":", lw=1)
    plt.axvline(device.junction_site, color="red", ls="--", lw=1, label="junction site")
    plt.xlabel("Site index")
    plt.ylabel(r"$|u_i|^2 + |v_i|^2$")
    plt.title("T-Junction: Localization of the 4 Near-Zero Modes")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("t_junction_localization.png", dpi=150)
    plt.close()
    print("\nSaved t_junction_localization.png")
