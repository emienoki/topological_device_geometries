
import json
import numpy as np
from sweep_lib import TJDesign, AncillaDesign
from poisoning import xi_analytic, bulk_gap, kT_over_t, poisoning_rate, OMEGA_A_PER_NS

def path_rate(d, T_mK, n=144, frac=1.0):
    xi, gap, kT = xi_analytic(1, d.delta, 0), bulk_gap(1, d.delta, 0), kT_over_t(T_mK)
    expected = 4 if isinstance(d, AncillaDesign) else 2
    G, dims, next_level, spatial = [], set(), 9.0, 0
    for s in (np.arange(n) + 0.5) / n:
        E = np.sort(np.abs(2 * np.linalg.eigvalsh(d.H_at(s))))
        k = int(np.sum(E < 0.1)); dims.add(k); next_level = min(next_level, E[k])
        if k == expected:
            G.append(poisoning_rate(d.device_at(s), xi, kT, gap, qubit_sites=None, zero_tol=0.1, subgap_frac=frac))
        else:
            spatial += 1
            G.append(poisoning_rate(d.device_at(s), xi, kT, gap, qubit_sites=d.qubit_sites(s), subgap_frac=frac))
    return float(np.mean(G)), sorted(dims), float(next_level), spatial / n

if __name__ == "__main__":
    rows = [json.loads(l) for l in open("sweep_results.jsonl")]
    out = []
    for r in rows:
        p = {k: v for k, v in r["params"].items() if k != "T_mK"}
        d = AncillaDesign(**p) if r["arch"] == "ancilla" else TJDesign(**p)
        G, dims, nxt, sp = path_rate(d, r["params"]["T_mK"])
        tot = [x["leakage"] + G * OMEGA_A_PER_NS * x["T_ns"] for x in r["rows"]]
        b = int(np.argmin(tot))
        r2 = dict(arch=r["arch"], params=r["params"], varied=r["varied"], mean_rate=G, J=tot[b],
                  T_opt_ns=r["rows"][b]["T_ns"], leak_at_opt=r["rows"][b]["leakage"],
                  pois_at_opt=G * OMEGA_A_PER_NS * r["rows"][b]["T_ns"], cluster_dims=dims,
                  next_level=nxt, spatial_fraction=sp, rows=r["rows"], old_rate=r["mean_rate"])
        out.append(r2)
        print(f"{r['arch']:9s} {r['varied']:7s}={str(r['params'][r['varied']]):6s} rate {r['mean_rate']:.2e} -> {G:.2e}   "
              f"J={tot[b]:.2e} @ {r2['T_opt_ns']:.1f} ns   cluster dims {dims}  next level >= {nxt:.2f} t  spatial {sp:.0%}", flush=True)
    json.dump(out, open("sweep_results_corrected.json", "w"), indent=1)
