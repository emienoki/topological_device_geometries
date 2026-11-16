
import json
import numpy as np
from sweep_lib import TJDesign, AncillaDesign
from braiding_dynamics import plain_chain_device
from poisoning import xi_analytic, bulk_gap, kT_over_t, poisoning_rate_full, OMEGA_A_PER_NS


def spectator_rate(delta, T_mK, N=12):
    xi, gap, kT = xi_analytic(1, delta, 0), bulk_gap(1, delta, 0), kT_over_t(T_mK)
    r = poisoning_rate_full(plain_chain_device(N, 1.0, delta, [0.0] * N), xi, kT, gap, split=True)
    assert r["n_qubit"] == 2 and r["below"] == 0
    return r["total"] / 2


def path_rate_full(d, T_mK, n=144):
    xi, gap, kT = xi_analytic(1, d.delta, 0), bulk_gap(1, d.delta, 0), kT_over_t(T_mK)
    expected = 4 if isinstance(d, AncillaDesign) else 2
    B, A = [], []
    for s in (np.arange(n) + 0.5) / n:
        E = np.sort(np.abs(2 * np.linalg.eigvalsh(d.H_at(s))))
        qs = None if int(np.sum(E < 0.1)) == expected else d.qubit_sites(s)
        r = poisoning_rate_full(d.device_at(s), xi, kT, gap, qubit_sites=qs, split=True)
        B.append(r["below"]); A.append(r["above"])
    return float(np.mean(B)), float(np.mean(A))


if __name__ == "__main__":
    rows = [json.loads(l) for l in open("sweep_results.jsonl")]
    old = {(r["arch"], r["varied"], str(r["params"][r["varied"]])): r for r in json.load(open("sweep_results_corrected.json"))}
    out = []
    for r in rows:
        p = {k: v for k, v in r["params"].items() if k != "T_mK"}
        T = r["params"]["T_mK"]
        d = AncillaDesign(**p) if r["arch"] == "ancilla" else TJDesign(**p)
        below, above = path_rate_full(d, T)
        spec = 0.0 if r["arch"] == "ancilla" else 2 * spectator_rate(d.delta, T)
        G = below + above + spec
        tot = [x["leakage"] + G * OMEGA_A_PER_NS * x["T_ns"] for x in r["rows"]]
        b = int(np.argmin(tot))
        o = old[(r["arch"], r["varied"], str(r["params"][r["varied"]]))]
        out.append(dict(arch=r["arch"], params=r["params"], varied=r["varied"], rate_below=below, rate_above=above,
                        rate_spectators=spec, rate_total=G, J=tot[b], T_opt_ns=r["rows"][b]["T_ns"],
                        leak_at_opt=r["rows"][b]["leakage"], J_subgap_only=o["J"], T_opt_subgap_only=o["T_opt_ns"],
                        rows=r["rows"]))
        print(f"{r['arch']:9s} {r['varied']:7s}={str(r['params'][r['varied']]):6s} below {below:.2e} above {above:.2e} "
              f"spect {spec:.2e} | J {o['J']:.2e}@{o['T_opt_ns']:.0f}ns -> {tot[b]:.2e}@{r['rows'][b]['T_ns']:.0f}ns", flush=True)
    json.dump(out, open("sweep_results_full.json", "w"), indent=1)
