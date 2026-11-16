import numpy as np
from device_builder import DeviceParams, build_t_junction, bdg_hamiltonian
from t_junction_braid import arm_mu_profile

def site_mu(N, walls, width=2.0, mu_topo=0.0, mu_triv=4.0):
    mu = np.zeros(3 * N + 1)
    mu[0] = mu_topo
    for a, w in enumerate(walls):
        mu[1 + a * N: 1 + (a + 1) * N] = arm_mu_profile(N, w, width, mu_topo, mu_triv)
    return mu

def walls_at(s, k, c):
    lo = -c
    ramp = lambda x: 0.5 * (1 - np.cos(np.pi * np.clip(x, 0, 1)))   # smooth, zero-slope joins
    p = s * 6
    leg = min(int(p), 5)
    f = ramp(p - leg)
    start = [(k, k, lo), (lo, k, lo), (lo, k, k), (lo, lo, k), (k, lo, k), (k, lo, lo)]
    end = start[1:] + [(k, k, lo)]
    a, b = np.array(start[leg], float), np.array(end[leg], float)
    return a + (b - a) * f


class TJ:
    def __init__(self, N=12, t=1.0, delta=0.6, k=6, c=3, width=2.0):
        self.N, self.k, self.c, self.width = N, k, c, width
        self.dev = build_t_junction(DeviceParams(N_per_arm=N, t=t, delta=delta, mu=0.0))

    def device_at(self, s):
        self.dev.site_mu = list(site_mu(self.N, walls_at(s, self.k, self.c), self.width))
        return self.dev

    def H_at(self, s):
        return bdg_hamiltonian(self.device_at(s))

    def end_sites(self, s):
        w = walls_at(s, self.k, self.c)
        ends = [1 + a * self.N + int(round(wa)) - 1 for a, wa in enumerate(w) if wa >= 0.5]
        if len(ends) < 2:
            ends.append(0)
        return ends
