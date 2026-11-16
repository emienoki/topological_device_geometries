import json, sys, time
import numpy as np
from multiprocessing import Pool
from sweep_lib import TJDesign, AncillaDesign, manifold
from poisoning import code_time_to_ns

NS = code_time_to_ns(1.0)
FINE = {r["arch"] + "_" + r["varied"] + "_" + str(r["value"]): r for r in json.load(open("fine_results.json"))}
BASE = {"ancilla": FINE["ancilla_g_0.6"], "tjunction": FINE["tjunction_dynamic_True"]}
SIGMAS = {"site": [0.003, 0.01, 0.03, 0.1], "balance": [0.003, 0.01, 0.03, 0.1], "phase": [0.01, 0.03, 0.1, 0.3]}
NSAMP = 12


def make(arch):
    return AncillaDesign() if arch == "ancilla" else TJDesign()


def arm_of_sites(arch, n):
    if arch == "ancilla":
        N = (n - 1) // 2
        g = np.zeros(n, int); g[N + 1:] = 1; g[0] = -1   # arm1 carries phase_R; site 0 = ancilla (junction)
        return g
    N = (n - 1) // 3
    g = np.full(n, -1); g[0] = -1
    for a in range(3):
        g[1 + a * N:1 + (a + 1) * N] = a
    return g


def perturbation(arch, kind, sigma, n, rng):
    dmu = np.zeros(n); EA = np.ones((n, n)); EB = np.ones((n, n)); PH = np.ones((n, n), complex)
    if kind == "site":
        dmu = rng.normal(0, sigma, n)
    elif kind == "balance":
        ea = rng.normal(0, sigma, (n, n)); eb = rng.normal(0, sigma, (n, n))
        EA = 1 + np.triu(ea, 1) + np.triu(ea, 1).T
        EB = 1 + np.triu(eb, 1) + np.triu(eb, 1).T
    elif kind == "phase":
        g = arm_of_sites(arch, n)
        if arch == "ancilla":
            dphi = {1: rng.normal(0, sigma)}
        else:
            dphi = {a: rng.normal(0, sigma) for a in range(3)}
        for i in range(n):
            for j in range(n):
                # a bond's pairing carries the phase of the arm it lies in (junction bonds: the arm site)
                a = g[j] if g[i] < 0 else g[i] if g[j] < 0 or g[j] == g[i] else None
                if a is None:
                    continue
                if a in dphi:
                    PH[i, j] = np.exp(1j * dphi[a])
    return dmu, EA, EB, PH


def perturbed(H, dmu, EA, EB, PH):
    n = H.shape[0] // 2
    A = 2 * H[:n, :n] * EA - np.diag(dmu)
    B = 2 * H[:n, n:] * EB * PH
    return 0.5 * np.block([[A, B], [-B.conj(), -A.conj()]])


def _polar(M):
    u, _, vh = np.linalg.svd(M); return u @ vh


def braid_matrix(design, Tc, pert=None, dt=0.25, own=True):
    V0, V1 = manifold(design.H_at(0.0)), manifold(design.H_at(1.0))
    W0, W1 = V0, V1
    if pert is not None and own:
        W0, W1 = manifold(perturbed(design.H_at(0.0), *pert)), manifold(perturbed(design.H_at(1.0), *pert))
    n = max(200, int(Tc / dt)); Psi = W0.astype(complex)
    for k in range(n):
        H = design.H_at((k + 0.5) / n)
        if pert is not None:
            H = perturbed(H, *pert)
        w, v = np.linalg.eigh(H)
        Psi = v @ (np.exp(-1j * w * Tc / n)[:, None] * (v.conj().T @ Psi))
    M = W1.conj().T @ Psi
    if pert is not None and own:
        M = _polar(V1.conj().T @ W1) @ M @ _polar(V0.conj().T @ W0).conj().T
    return M


def gate_error(M, Q0):
    d = M.shape[0]
    return float(1 - abs(np.trace(Q0.conj().T @ M) / d) ** 2)


_IDEAL = {}


def job(args):
    arch, kind, sigma, seed = args
    d = make(arch); T_ns = BASE[arch]["T_opt_full"]; Tc = T_ns / NS
    if arch not in _IDEAL:                      # ideal braid computed once per worker
        M0 = braid_matrix(d, Tc); u, _, vh = np.linalg.svd(M0); _IDEAL[arch] = (M0, u @ vh)
    M0, Q0 = _IDEAL[arch]
    n = d.H_at(0.0).shape[0] // 2
    rng = np.random.default_rng(seed)
    M = braid_matrix(d, Tc, perturbation(arch, kind, sigma, n, rng))
    return arch, kind, sigma, seed, gate_error(M, Q0), gate_error(M0, Q0)


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(a, k, s, 1000 * i + j) for a in ("ancilla", "tjunction") for k, ss in SIGMAS.items()
            for i, s in enumerate(ss) for j in range(NSAMP)]
    out = {a: {"T_opt_ns": BASE[a]["T_opt_full"], "J_ideal": BASE[a]["J_full"],
               "poisoning_at_opt": BASE[a]["pois_full"], "runs": []} for a in BASE}
    with Pool(2) as pool:
        for arch, kind, sigma, seed, err, err0 in pool.imap_unordered(job, jobs):
            out[arch]["runs"].append(dict(kind=kind, sigma=sigma, seed=seed, gate_error=err))
            out[arch]["gate_error_ideal"] = err0
    for a in out:
        summ = {}
        for k, ss in SIGMAS.items():
            for s in ss:
                e = np.array([r["gate_error"] for r in out[a]["runs"] if r["kind"] == k and r["sigma"] == s])
                summ[f"{k}_{s}"] = dict(median=float(np.median(e)), mean=float(e.mean()), p90=float(np.quantile(e, 0.9)),
                                        total_mean=float(e.mean() + out[a]["poisoning_at_opt"]))
        out[a]["summary"] = summ
        print(a, "ideal gate error", out[a]["gate_error_ideal"], "poisoning", out[a]["poisoning_at_opt"])
        for k, v in summ.items():
            print(f"   {k:14s} mean {v['mean']:.2e}  median {v['median']:.2e}  p90 {v['p90']:.2e}  total {v['total_mean']:.2e}")
    out["seconds"] = round(time.time() - t0)
    json.dump(out, open("tolerance_results.json", "w"), indent=1)
