
import json
import numpy as np
from sweep_lib import TJDesign, AncillaDesign
from recompute_full import path_rate_full, spectator_rate
from poisoning import OMEGA_A_PER_NS

full = json.load(open("sweep_results_full.json"))
def rows_of(arch, varied, val):
    return next(r for r in full if r["arch"] == arch and r["varied"] == varied and str(r["params"][varied]) == str(val))["rows"]

designs = {"ancilla_g0.6": (AncillaDesign(g=0.6), rows_of("ancilla", "g", 0.6)),
           "ancilla_g1.0": (AncillaDesign(g=1.0), rows_of("ancilla", "g", 1.0)),
           "ancilla_D0.8": (AncillaDesign(delta=0.8), rows_of("ancilla", "delta", 0.8)),
           "tj_dynamic":   (TJDesign(dynamic=True), rows_of("tjunction", "dynamic", True)),
           "tj_D0.8":      (TJDesign(delta=0.8), rows_of("tjunction", "delta", 0.8))}

def J_of(rows, rate_per_ns):
    tot = [x["leakage"] + rate_per_ns * x["T_ns"] for x in rows]
    b = int(np.argmin(tot)); return tot[b], rows[b]["T_ns"]

out = {"T_mK": [15, 20, 25, 30, 40, 50, 60], "temp": {}, "parent": {}}
for name, (d, rows) in designs.items():
    res = []
    for T in out["T_mK"]:
        b, a = path_rate_full(d, T, n=96)
        s = 0.0 if name.startswith("anc") else 2 * spectator_rate(d.delta, T)
        J, Topt = J_of(rows, (b + a + s) * OMEGA_A_PER_NS)
        Jsub, Tsub = J_of(rows, b * OMEGA_A_PER_NS)
        res.append(dict(T_mK=T, below=b, above=a, spect=s, J=J, T_opt=Topt, J_subgap_only=Jsub))
    out["temp"][name] = res
    print(name, " ".join(f"{r['T_mK']}mK:{r['J']:.1e}" for r in res), flush=True)

Gp = np.logspace(-5, 7, 49)                     # per-Majorana parent-SC rate, 1/s
for name, (d, rows) in designs.items():
    r30 = next(r for r in out["temp"][name] if r["T_mK"] == 30)
    th = (r30["below"] + r30["above"] + r30["spect"]) * OMEGA_A_PER_NS     # per ns
    J, Topt = zip(*[J_of(rows, th + 4 * g * 1e-9) for g in Gp])
    out["parent"][name] = dict(Gamma_p=Gp.tolist(), J=list(J), T_opt=list(Topt),
                               thermal_rate_per_majorana_s=th * 1e9 / 4)
    print(name, f"thermal rate per Majorana {th*1e9/4:.2e} /s", " ".join(f"{g:.0e}:{j:.1e}" for g, j in list(zip(Gp, J))[::8]))
json.dump(out, open("bulk_channel_scan.json", "w"), indent=1)
