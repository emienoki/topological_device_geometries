import numpy as np

NPIX = 32
VWIN = 2.5            # plunger window in units of the coupling scale


def lowest_states(e1, e2, t, D):
    s, d = e1 + e2, e1 - e2
    # even block lowest eigenpair
    re = np.sqrt(s ** 2 + 4 * D ** 2)
    Ee = 0.5 * (s - re)
    a00, a11 = D, Ee                                   # unnormalized: [[0,D],[D,s]] v = Ee v
    n = np.sqrt(a00 ** 2 + a11 ** 2) + 1e-15
    a00, a11 = a00 / n, a11 / n
    # odd block lowest eigenpair
    ro = np.sqrt(d ** 2 + 4 * t ** 2)
    Eo = 0.5 * (s - ro)
    b10, b01 = t, Eo - e1
    n = np.sqrt(b10 ** 2 + b01 ** 2) + 1e-15
    b10, b01 = b10 / n, b01 / n
    return Ee, Eo, a00, a11, b10, b01


def csd(t, D, rng=None, aug=None, npix=NPIX):
    v = np.linspace(-VWIN, VWIN, npix)
    V1, V2 = np.meshgrid(v, v, indexing="ij")
    a = aug or {}
    lev = a.get("lever", 1.0)
    ct1, ct2 = a.get("crosstalk", (0.0, 0.0))
    off1, off2 = a.get("offset", (0.0, 0.0))
    e1 = lev * (V1 + ct1 * V2) + off1
    e2 = lev * (V2 + ct2 * V1) + off2
    Ee, Eo, a00, a11, b10, b01 = lowest_states(e1, e2, t, D)
    gam = a.get("width", 0.12)
    x = (Eo - Ee) / gam
    line = np.exp(-x ** 2) if a.get("shape", "lorentz") == "gauss" else 1.0 / (1.0 + x ** 2)
    G = np.stack([((b10 * a00) ** 2 + (b01 * a11) ** 2) * line,
                  ((b01 * a00) ** 2 + (b10 * a11) ** 2) * line])
    if a:
        G = G * a.get("amp", 1.0)
        G = G + a.get("bg", 0.0) * (1 + 0.5 * V1 / VWIN)[None]           # sloped background
        if a.get("jumps", 0) and rng is not None:                         # charge jumps: shift some rows
            for _ in range(a["jumps"]):
                r0 = rng.integers(npix)
                G[:, r0:] = np.roll(G[:, r0:], rng.integers(-2, 3), axis=2)
        if rng is not None:
            G = G + a.get("noise", 0.0) * rng.standard_normal(G.shape)
    G = G - G.mean()
    return (G / (G.std() + 1e-12)).astype(np.float32)


def random_aug(rng, shifted=False):
    if not shifted:
        return dict(lever=rng.uniform(0.85, 1.15), crosstalk=tuple(rng.uniform(0, 0.12, 2)),
                    offset=tuple(rng.uniform(-0.25, 0.25, 2)), width=rng.uniform(0.06, 0.25),
                    shape="lorentz", amp=rng.uniform(0.5, 2), bg=rng.uniform(0, 0.1),
                    noise=rng.uniform(0, 0.08), jumps=0)
    return dict(lever=rng.uniform(0.8, 1.2), crosstalk=tuple(rng.uniform(0.1, 0.22, 2)),
                offset=tuple(rng.uniform(-0.35, 0.35, 2)), width=rng.uniform(0.1, 0.35),
                shape="gauss", amp=rng.uniform(0.5, 2), bg=rng.uniform(0.05, 0.2),
                noise=rng.uniform(0.05, 0.15), jumps=int(rng.integers(0, 3)))


def random_couplings(rng):
    r = rng.uniform(-0.8, 0.8)
    big = rng.uniform(0.4, 1.0)
    if r >= 0:
        t = big; D = t * (1 - r) / (1 + r)
    else:
        D = big; t = D * (1 + r) / (1 - r)
    return t, D, r


def dataset(n, seed, shifted=False):
    rng = np.random.default_rng(seed)
    X = np.empty((n, 2, NPIX, NPIX), np.float32); y = np.empty(n, np.float32)
    for k in range(n):
        t, D, r = random_couplings(rng)
        X[k] = csd(t, D, rng, random_aug(rng, shifted)); y[k] = r
    return X, y
