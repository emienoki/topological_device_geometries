
import numpy as np
from multiprocessing import Pool
from sweep_lib import TJDesign, AncillaDesign, manifold
from poisoning import code_time_to_ns

TC = [100 * 2 ** (k / 6) for k in range(31)]          # code units, 100 ... 3200


def leakage(design, Tc, dt=0.25):
    V0, V1 = manifold(design.H_at(0.0)), manifold(design.H_at(1.0))
    assert V0.shape[1] == V1.shape[1]
    n = max(200, int(Tc / dt))
    Psi = V0.astype(complex)
    for k in range(n):
        w, v = np.linalg.eigh(design.H_at((k + 0.5) / n))
        Psi = v @ (np.exp(-1j * w * Tc / n)[:, None] * (v.conj().T @ Psi))
    return float(1 - np.linalg.svd(V1.conj().T @ Psi, compute_uv=False).min() ** 2)


GEOMS = [("ancilla", dict(g=g)) for g in (0.3, 0.45, 0.6, 0.8, 1.0)] + \
        [("ancilla", dict(N=N)) for N in (6, 9, 16)] + \
        [("ancilla", dict(delta=D)) for D in (0.4, 0.8)] + \
        [("tjunction", dict())] + [("tjunction", dict(k=k, N=k + 6)) for k in (4, 8)] + \
        [("tjunction", dict(width=w)) for w in (1.0, 3.0)] + \
        [("tjunction", dict(c=c)) for c in (2, 5)] + [("tjunction", dict(delta=D)) for D in (0.4, 0.8)]


def make(arch, p):
    return AncillaDesign(**p) if arch == "ancilla" else TJDesign(**p)


def job(item):
    arch, p = item
    t0 = time.time()
    d = make(arch, p)
    rows = [dict(Tc=Tc, T_ns=code_time_to_ns(Tc), leakage=leakage(d, Tc)) for Tc in TC]
    r = dict(arch=arch, params=p, rows=rows, seconds=round(time.time() - t0, 1))
    with open("fine_grid.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    return arch, p, r["seconds"]


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        # consistency with the coarse sweep (full propagator) + dt convergence
        for arch, p in [("ancilla", {}), ("tjunction", {})]:
            d = make(arch, p)
            for Tc in (200, 800):
                print(arch, Tc, f"dt=0.25 {leakage(d, Tc):.4e}  dt=0.125 {leakage(d, Tc, 0.125):.4e}", flush=True)
    else:
        with Pool(2) as pool:
            for arch, p, s in pool.imap_unordered(job, GEOMS):
                print(arch, p, f"{s}s", flush=True)
