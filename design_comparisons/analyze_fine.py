import json
import numpy as np
from poisoning import OMEGA_A_PER_NS

DEF = {"ancilla": dict(N=12, delta=0.6, g=0.6), "tjunction": dict(N=12, delta=0.6, k=6, c=3, width=2.0, dynamic=True)}
fine = [json.loads(l) for l in open("fine_grid.jsonl")]
full = json.load(open("sweep_results_full.json"))

def geom(arch, p):
    g = dict(DEF[arch]); g.update(p); return tuple(sorted(g.items()))

leak = {(f["arch"], geom(f["arch"], f["params"])): f["rows"] for f in fine}
out = []
for r in full:
    p = {k: v for k, v in r["params"].items() if k != "T_mK"}
    key = (r["arch"], geom(r["arch"], p))
    if key not in leak:
        continue
    rows = leak[key]
    T = np.array([x["T_ns"] for x in rows]); L = np.array([x["leakage"] for x in rows])
    res = {}
    for name, rate in [("full", r["rate_total"]), ("subgap", r["rate_below"])]:
        tot = L + rate * OMEGA_A_PER_NS * T
        b = int(np.argmin(tot))
        Lr = np.maximum(L, np.append(L[1:], L[-1]))          # robust: no single lucky dip
        totr = Lr + rate * OMEGA_A_PER_NS * T; br = int(np.argmin(totr))
        res[name] = dict(J=float(tot[b]), T_opt=float(T[b]), leak=float(L[b]), pois=float(tot[b] - L[b]),
                         J_robust=float(totr[br]), T_opt_robust=float(T[br]))
    out.append(dict(arch=r["arch"], varied=r["varied"], value=r["params"][r["varied"]], params=r["params"],
                    J_coarse=r["J"], T_coarse=r["T_opt_ns"], **{f"{k}_{n}": v for n, d in res.items() for k, v in d.items()},
                    T_ns=T.tolist(), leakage=L.tolist(), rate_total=r["rate_total"], rate_below=r["rate_below"]))
    f = res["full"]
    print(f"{r['arch']:9s} {r['varied']:7s}={str(r['params'][r['varied']]):6s} coarse {r['J']:.2e}@{r['T_opt_ns']:5.1f}  "
          f"fine {f['J']:.2e}@{f['T_opt']:5.1f} (leak {f['leak']:.1e}, pois {f['pois']:.1e})  robust {f['J_robust']:.2e}@{f['T_opt_robust']:.1f}  "
          f"| subgap-only fine {res['subgap']['J']:.2e}")
json.dump(out, open("fine_results.json", "w"), indent=1)
