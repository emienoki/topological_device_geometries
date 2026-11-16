import numpy as np
from scipy.linalg import expm
from device_builder import Device, bdg_hamiltonian
from braiding_dynamics import localized_zero_mode
from geometric_phase import bargmann_geometric_phase, gauge_invariance_check

"""builds ancillia dot device with indepdendent, asymmetric ancilla coupling plus the 
 ancillia's own on site term, phase_R (supercondcting phase on arm1's pairing required to be pi)"""
def build_virtual_trijunction_device(N, t, delta, mu_D, gamma_L, gamma_R, phase_R=np.pi):
    bonds = []
    for k in range(N - 1):
        i, j = 1 + k, 2 + k
        bonds.append((i, j, t, -delta))
    #arm1 internal bonds (sites N+1..2N), with the required phase_R
    delta_R = delta * np.exp(1j * phase_R)
    for k in range(N - 1):
        i, j = N + 1 + k, N + 2 + k
        bonds.append((i, j, t, delta_R))
    #ancilla <-> arm0 (Gamma_L) and ancilla <-> arm1 (Gamma_R)
    bonds.append((0, 1, gamma_L, -gamma_L))       # matches arm0's sign convention
    bonds.append((0, N + 1, gamma_R, gamma_R * np.exp(1j * phase_R)))

    site_mu = np.zeros(2 * N + 1)
    site_mu[0] = mu_D  #ancilla's own on-site term

    return Device(n_sites=2 * N + 1, bonds=bonds, site_mu=list(site_mu),
                  arm_ends=[N, 2 * N], junction_site=0, lead_site=1, trap_site=N + 1,
                  params=None)


def coupling_schedule(s, mu_D_max, gamma_max, width=0.05):
    """(mu_D off, Gamma_L on) -> (Gamma_L off,
    Gamma_R on) -> (Gamma_R off, mu_D back on). width sets the smoothing
    (as a fraction of each 1/3 phase) between steps."""
    def step_up(x, center, w):
        return 0.5 * (1 + np.tanh((x - center) / w))

    def step_down(x, center, w):
        return 1 - step_up(x, center, w)

    # mu_D: on for s in [0,1/3]-ish transition out, off in the middle, back on at the end
    mu_D = mu_D_max * (step_down(s, 1/6, width) + step_up(s, 5/6, width))
    gamma_L = gamma_max * (step_up(s, 1/6, width) * step_down(s, 3/6, width))
    gamma_R = gamma_max * (step_up(s, 3/6, width) * step_down(s, 5/6, width))
    return mu_D, gamma_L, gamma_R


def simulate_exchange(N, t, delta, mu_D_max, gamma_max, T, n_steps, width=0.05, phase_R=np.pi):
    """Single exchange (3-step protocol). Returns fidelity (windowed, at
    the swapped-to site), trajectory, and the device at the final config."""
    from t_junction_braid import windowed_fidelity

    mu_D0, gL0, gR0 = coupling_schedule(0.0, mu_D_max, gamma_max, width)
    device = build_virtual_trijunction_device(N, t, delta, mu_D0, gL0, gR0, phase_R)
    H0 = bdg_hamiltonian(device)
    start_site = 1  # arm0's near-ancilla end (gamma_2 = gamma_{L2B})
    _, psi = localized_zero_mode(H0, device.n_sites, target_site=start_site, n_search=8)
    psi = psi.astype(complex)
    trajectory = [psi.copy()]

    dt = T / n_steps
    for step in range(n_steps):
        s = (step + 0.5) / n_steps
        mu_D, gL, gR = coupling_schedule(s, mu_D_max, gamma_max, width)
        device = build_virtual_trijunction_device(N, t, delta, mu_D, gL, gR, phase_R)
        H = bdg_hamiltonian(device)
        psi = expm(-1j * H * dt) @ psi
        psi = psi / np.linalg.norm(psi)
        trajectory.append(psi.copy())

    end_site = N + 1  # arm1's near-ancilla end (gamma_3 = gamma_{R1A}) -- the exchange target
    fidelity = windowed_fidelity(psi, device.n_sites, end_site, radius=2)
    return fidelity, trajectory, device
