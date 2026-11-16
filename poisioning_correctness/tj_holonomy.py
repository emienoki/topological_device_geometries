import numpy as np
from sweep_lib import TJDesign, AncillaDesign, manifold


def polar(M):
    u, _, vh = np.linalg.svd(M); return u @ vh


def wilson(design, M, G=None, random_gauge=False, seed=0):
    rng = np.random.default_rng(seed)
    V0 = manifold(design.H_at(0.0)); Vp = V0; P = np.eye(V0.shape[1], dtype=complex)
    for k in range(1, M + 1):
        V = manifold(design.H_at(k / M))
        assert V.shape[1] == V0.shape[1], (k, V.shape)
        if random_gauge and k < M:
            d = V.shape[1]
            Q, _ = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))); V = V @ Q
        P = polar(V.conj().T @ Vp) @ P
        Vp = V
    ref = V0 if G is None else G @ V0
    W = polar(ref.conj().T @ Vp @ P)
    ph = np.sort(np.angle(np.linalg.eigvals(W)))
    return ph


if __name__ == "__main__":
    tj = TJDesign(); n = tj.H_at(0).shape[0] // 2
    G = np.diag(np.r_[np.full(n, 1j), np.full(n, -1j)])
    print("T-junction (dynamic phases), qubit space dim", manifold(tj.H_at(0)).shape[1])
    for M in (500, 1000, 2000, 4000):
        ph = wilson(tj, M, G)
        print(f"  steps {M:5d}: eigenphases {np.round(ph, 5)}  -> theta = {np.max(np.abs(ph)):.5f}  (pi/2 = {np.pi/2:.5f})")
    for s in (1, 2):
        ph = wilson(tj, 2000, G, random_gauge=True, seed=s)
        print(f"  random gauge seed {s}: theta = {np.max(np.abs(ph)):.5f}")
    an = AncillaDesign()
    ph = wilson(an, 2000)
    print("ancilla-dot reference (same code), eigenphases", np.round(ph, 5))
