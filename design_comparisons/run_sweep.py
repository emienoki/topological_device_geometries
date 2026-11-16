import json, time, traceback
from multiprocessing import Pool
from sweep_lib import TJDesign, AncillaDesign, evaluate

AB = dict(N=12, delta=0.6, g=0.6, T_mK=30.0)
TB = dict(N=12, delta=0.6, k=6, c=3, width=2.0, dynamic=True, T_mK=30.0)
cfgs = []
def add(arch, base, key, vals):
    for v in vals:
        c = dict(base); c[key] = v
        if key == "k": c["N"] = v + 6
        tag = (arch, tuple(sorted(c.items())))
        if tag not in [(a, tuple(sorted(x.items()))) for a, x, _ in cfgs]:
            cfgs.append((arch, c, key))
add("ancilla", AB, "g", [0.3, 0.45, 0.6, 0.8, 1.0])
add("ancilla", AB, "N", [6, 9, 16])
add("ancilla", AB, "delta", [0.4, 0.8])
add("ancilla", AB, "T_mK", [20.0, 50.0])
add("tjunction", TB, "dynamic", [True, False])
add("tjunction", TB, "k", [4, 8])
add("tjunction", TB, "width", [1.0, 3.0])
add("tjunction", TB, "c", [2, 5])
add("tjunction", TB, "delta", [0.4, 0.8])
add("tjunction", TB, "T_mK", [20.0, 50.0])

def job(item):
    arch, c, key = item
    t0 = time.time()
    try:
        p = {k: v for k, v in c.items() if k != "T_mK"}
        d = AncillaDesign(**p) if arch == "ancilla" else TJDesign(**p)
        r = evaluate(d, T_mK=c["T_mK"])
    except Exception as e:
        r = dict(error=repr(e), tb=traceback.format_exc()[-400:])
    r.update(arch=arch, params=c, varied=key, seconds=round(time.time() - t0, 1))
    with open("sweep_results.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    return arch, key

if __name__ == "__main__":
    open("sweep_results.jsonl", "w").close()
    print(len(cfgs), "designs", flush=True)
    with Pool(2) as pool:
        for i, (a, k) in enumerate(pool.imap_unordered(job, cfgs), 1):
            print(f"{i}/{len(cfgs)} done ({a}, {k})", flush=True)
    print("SWEEP COMPLETE", flush=True)
