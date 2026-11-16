import numpy as np
from csd_sim import NPIX, VWIN

KERNEL_PTS = np.linspace(-8, 8, 33)            # thermal kernel sample points, units of kT


def lowest_states_U(e1, e2, t, D, U):
    s, d = e1 + e2, e1 - e2
    su = s + U
    Ee = 0.5 * (su - np.sqrt(su ** 2 + 4 * D ** 2))
    a00, a11 = D, Ee
    n = np.sqrt(a00 ** 2 + a11 ** 2) + 1e-15
    a00, a11 = a00 / n, a11 / n
    Eo = 0.5 * (s - np.sqrt(d ** 2 + 4 * t ** 2))
    b10, b01 = t, Eo - e1
    n = np.sqrt(b10 ** 2 + b01 ** 2) + 1e-15
    return Ee, Eo, a00, a11, b10 / n, b01 / n


def line_shape(dE, gamma, kT):
    """Lorentzian of width gamma convolved with the thermal window at kT."""
    if kT <= 1e-9:
        return 1.0 / (1.0 + (dE / gamma) ** 2)
    e = KERNEL_PTS * kT
    w = 1.0 / np.cosh(KERNEL_PTS / 2) ** 2
    w = w / w.sum()
    x = (dE[..., None] - e) / gamma
    out = (w / (1.0 + x ** 2)).sum(-1)
    return out / out.max() if out.max() > 0 else out


def csd_real(t, D, U, rng, a, npix=NPIX):
    v = np.linspace(-VWIN, VWIN, npix)
    V1, V2 = np.meshgrid(v, v, indexing="ij")          # axis 0: dot-1 gate (fast), axis 1: dot-2 gate (slow)
    ct1, ct2 = a["crosstalk"]
    off1, off2 = a["offset"]
    # slow drift along the slow axis (index j): random walk + telegraph switch
    dr1 = np.cumsum(rng.normal(0, a["drift"], npix)); dr2 = np.cumsum(rng.normal(0, a["drift"], npix))
    tel = np.zeros(npix); state = 0
    for j in range(npix):
        if rng.random() < a["tel_rate"]:
            state = 1 - state
        tel[j] = state * a["tel_amp"]
    e1 = a["lever"] * (V1 + ct1 * V2) + off1 + (dr1 + tel)[None, :]
    e2 = a["lever"] * (V2 + ct2 * V1) + off2 + dr2[None, :]
    Ee, Eo, a00, a11, b10, b01 = lowest_states_U(e1, e2, t, D, U)
    line = line_shape(Eo - Ee, a["gamma"], a["kT"])
    G = np.stack([((b10 * a00) ** 2 + (b01 * a11) ** 2) * line,
                  ((b01 * a00) ** 2 + (b10 * a11) ** 2) * line])
    G = G * a["amp"] + a["bg"] * (1 + 0.5 * V1 / VWIN)[None]
    G = G + a["noise"] * rng.standard_normal(G.shape)
    G = G - G.mean()
    return (G / (G.std() + 1e-12)).astype(np.float32)


def random_aug_real(rng, kind="train"):
    if kind == "train":
        return dict(lever=rng.uniform(0.85, 1.15), crosstalk=tuple(rng.uniform(0, 0.12, 2)),
                    offset=tuple(rng.uniform(-0.3, 0.3, 2)), gamma=rng.uniform(0.03, 0.15),
                    kT=rng.uniform(0.0, 0.08), drift=rng.uniform(0, 0.008),
                    tel_rate=rng.uniform(0, 0.04), tel_amp=rng.uniform(0, 0.12),
                    amp=rng.uniform(0.5, 2), bg=rng.uniform(0, 0.1), noise=rng.uniform(0, 0.08))
    return dict(lever=rng.uniform(0.8, 1.2), crosstalk=tuple(rng.uniform(0.1, 0.2, 2)),
                offset=tuple(rng.uniform(-0.35, 0.35, 2)), gamma=rng.uniform(0.08, 0.2),
                kT=rng.uniform(0.06, 0.14), drift=rng.uniform(0.006, 0.014),
                tel_rate=rng.uniform(0.02, 0.08), tel_amp=rng.uniform(0.08, 0.2),
                amp=rng.uniform(0.5, 2), bg=rng.uniform(0.05, 0.2), noise=rng.uniform(0.05, 0.15))


def random_couplings_real(rng, kind="train"):
    r = rng.uniform(-0.8, 0.8)
    big = rng.uniform(0.4, 1.0)
    U = rng.uniform(0, 0.5) if kind == "train" else rng.uniform(0.3, 0.7)
    if r >= 0:
        t = big; De = t * (1 - r) / (1 + r)
    else:
        De = big; t = De * (1 + r) / (1 - r)
    return t, De + U / 2, U, r


def dataset_real(n, seed, kind="train"):
    rng = np.random.default_rng(seed)
    X = np.empty((n, 2, NPIX, NPIX), np.float32); y = np.empty(n, np.float32)
    for k in range(n):
        t, D, U, r = random_couplings_real(rng, kind)
        X[k] = csd_real(t, D, U, rng, random_aug_real(rng, kind)); y[k] = r
    return X, y
