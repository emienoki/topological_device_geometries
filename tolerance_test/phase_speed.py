import json, numpy as np
from multiprocessing import Pool
from sweep_lib import AncillaDesign
from tolerance import braid_matrix, gate_error, arm_of_sites, NS
from poisoning import OMEGA_A_PER_NS
import tolerance

F = {r["arch"] + "_" + r["varied"] + "_" + str(r["value"]): r for r in json.load(open("fine_results.json"))}
RATE = F["ancilla_g_0.6"]["rate_total"] * OMEGA_A_PER_NS          # poisoning per ns
TNS = [2.7 * 2 ** (k / 3) for k in range(16)]                     # 2.7 ... 88 ns
DPHI = [0.0, 0.003, 0.01, 0.03]

def phase_pert(n, dphi):
    g = arm_of_sites("ancilla", n); PH = np.ones((n, n), complex)
    for i in range(n):
        for j in range(n):
            a = g[j] if g[i] < 0 else g[i] if g[j] < 0 or g[j] == g[i] else None
            if a == 1: PH[i, j] = np.exp(1j * dphi)
    return (np.zeros(n), np.ones((n, n)), np.ones((n, n)), PH)

def job(args):
    T_ns, dphi = args
    d = AncillaDesign(); n = d.H_at(0).shape[0] // 2; Tc = T_ns / NS
    M0 = braid_matrix(d, Tc); u, _, vh = np.linalg.svd(M0); Q0 = u @ vh
    # reference = the perfectly tuned braid at a slow duration (88 ns), whose rotation is pi/2 to 1e-6
    return T_ns, dphi, M0, Q0

def run(args):
    T_ns, dphi = args
    d = AncillaDesign(); n = d.H_at(0).shape[0] // 2; Tc = T_ns / NS
    M0 = braid_matrix(d, Tc); u, _, vh = np.linalg.svd(M0); Q0 = u @ vh      # ideal braid, SAME duration
    M = braid_matrix(d, Tc, phase_pert(n, dphi)) if dphi else M0
    return T_ns, dphi, gate_error(M, Q0)

if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(run, [(T, f) for f in DPHI for T in TNS])
    out = {}
    for f in DPHI:
        rows = sorted((T, e, e + RATE * T) for T, ff, e in res if ff == f)
        k = int(np.argmin([r[2] for r in rows]))
        out[str(f)] = dict(rows=[dict(T_ns=r[0], gate_error=r[1], total=r[2]) for r in rows],
                           best_total=rows[k][2], best_T_ns=rows[k][0])
        print(f"dphi {f:5.3f} rad ({np.degrees(f):.2f} deg): best total {rows[k][2]:.2e} at {rows[k][0]:.1f} ns   "
              + "  ".join(f"{r[0]:.0f}ns:{r[2]:.1e}" for r in rows[::3]))
    json.dump(out, open("phase_speed.json", "w"), indent=1)
