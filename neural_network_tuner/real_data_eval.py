import sys, os, glob, json, re
sys.path.insert(0, "/home/claude")
import numpy as np
import torch, torch.nn.functional as Fn
import xarray as xr
from scipy.stats import spearmanr
from train_csd import CSDNet
from train_real import WideNet, swap

torch.set_num_threads(2)
W = "/home/claude"
SWEET_B = 979.6


def nets():
    v1 = CSDNet(); v1.load_state_dict(torch.load(f"{W}/csd_cnn.pt")); v1.eval()
    v2 = CSDNet(); v2.load_state_dict(torch.load(f"{W}/csd_v2.pt")); v2.eval()
    v3 = WideNet(); v3.load_state_dict(torch.load(f"{W}/csd_v3.pt")); v3.eval()
    return {"v1": (v1, False), "v2": (v2, False), "v3": (v3, True)}


def prep(LL, RR, frac):
    def one(m, ci, cj):
        n0, n1 = m.shape
        h0, h1 = max(4, int(frac * n0 / 2)), max(4, int(frac * n1 / 2))
        i0 = int(np.clip(ci - h0, 0, n0 - 2 * h0)); j0 = int(np.clip(cj - h1, 0, n1 - 2 * h1))
        c = m[i0:i0 + 2 * h0, j0:j0 + 2 * h1]
        t = torch.tensor(c, dtype=torch.float32)[None, None]
        return Fn.interpolate(t, size=(32, 32), mode="area")[0, 0].numpy()
    LL = np.nan_to_num(LL); RR = np.nan_to_num(RR)
    w = np.clip(LL - np.median(LL), 0, None) + np.clip(RR - np.median(RR), 0, None) + 1e-12
    ii, jj = np.indices(w.shape)
    ci, cj = (w * ii).sum() / w.sum(), (w * jj).sum() / w.sum()
    G = np.stack([one(LL, ci, cj), one(RR, ci, cj)]); G = G - G.mean()
    return (G / (G.std() + 1e-12)).astype(np.float32)


def predict(models, LL, RR):
    out = {}
    with torch.no_grad():
        for frac in (1.0, 0.75, 0.5):
            x = torch.from_numpy(prep(LL, RR, frac))[None]
            for k, (net, tta) in models.items():
                p = net(x)
                if tta:
                    p = 0.5 * (p + net(swap(x)))
                out[f"{k}@{frac}"] = float(p)
    return out


if __name__ == "__main__" and len(sys.argv) == 2:
    D = sys.argv[1]
    M = nets()
    res = {"labeled6": [], "sweep61": []}
    for f, dev, lab in [("map_t_delta_t", "A", +1), ("map_t_delta_equal", "A", 0), ("map_t_delta_delta", "A", -1),
                        ("devB_CSD_t", "B", +1), ("devB_CSD_equal", "B", 0), ("devB_CSD_Delta", "B", -1)]:
        ds = xr.open_dataset(os.path.join(D, f + ".nc"))
        p = predict(M, ds.lockinLL_G.values, ds.lockinRR_G.values)
        res["labeled6"].append(dict(file=f, device=dev, expected_sign=lab, **p))
    for path in sorted(glob.glob(os.path.join(D, "devB_all_CSDs", "*.nc"))):
        pg = float(re.search(r"PG=([\d.]+)mV", path).group(1))
        ds = xr.open_dataset(path)
        p = predict(M, ds.lockinLL_G.values, ds.lockinRR_G.values)
        res["sweep61"].append(dict(PG_mV=pg, **p))
    keys = [k for k in res["labeled6"][0] if "@" in k]
    summary = {}
    for k in keys:
        lab = res["labeled6"]
        signs_ok = sum(1 for r in lab if r["expected_sign"] != 0 and np.sign(r[k]) == r["expected_sign"])
        order_ok = sum(1 for d in "AB" if [r[k] for r in lab if r["device"] == d] ==
                       sorted([r[k] for r in lab if r["device"] == d], reverse=True))
        sw = res["sweep61"]; pg = np.array([r["PG_mV"] for r in sw]); pr = np.array([r[k] for r in sw])
        far = np.abs(pg - SWEET_B) > 2
        summary[k] = dict(labeled_signs_correct=f"{signs_ok}/4", labeled_order_correct=f"{order_ok}/2",
                          labeled_sweet_spot_values=[round(r[k], 3) for r in lab if r["expected_sign"] == 0],
                          sweep_rank_correlation=round(float(spearmanr(pg, pr)[0]), 3),
                          sweep_sign_accuracy_far=round(float(np.mean(np.sign(pr[far]) == np.sign(pg[far] - SWEET_B))), 3),
                          sweep_zero_crossing_mV=None)
        # where does the prediction cross zero? (smoothed over 5 points)
        sm = np.convolve(pr, np.ones(5) / 5, mode="same")
        cross = [float((pg[i] + pg[i + 1]) / 2) for i in range(2, len(pg) - 3) if np.sign(sm[i]) != np.sign(sm[i + 1])]
        summary[k]["sweep_zero_crossing_mV"] = cross
    res["summary"] = summary
    json.dump(res, open(f"{W}/real_data_results.json", "w"), indent=1)
    for k, v in summary.items():
        print(k, v)

def geometric_label(LL, RR, top=0.04):
    G = np.nan_to_num(LL) + np.nan_to_num(RR)
    G = G - np.median(G)
    thr = np.quantile(G, 1 - top)
    w = np.where(G >= thr, G, 0.0)
    ii, jj = np.indices(G.shape)
    ci, cj = (w * ii).sum() / w.sum(), (w * jj).sum() / w.sum()
    x, y = ii - ci, jj - cj
    return float((w * x * y).sum() / np.sqrt((w * x * x).sum() * (w * y * y).sum()))


def score_sweep(D):
    res = json.load(open(f"{W}/real_data_results.json"))
    for r in res["sweep61"]:
        ds = xr.open_dataset(os.path.join(D, "devB_all_CSDs", f"devB_all_CSD_PG={r['PG_mV']:.1f}mV.nc"))
        r["geometric_c"] = round(geometric_label(ds.lockinLL_G.values, ds.lockinRR_G.values), 3)
    lab_check = []
    for f, s in [("map_t_delta_t", 1), ("map_t_delta_equal", 0), ("map_t_delta_delta", -1),
                 ("devB_CSD_t", 1), ("devB_CSD_equal", 0), ("devB_CSD_Delta", -1)]:
        ds = xr.open_dataset(os.path.join(D, f + ".nc"))
        lab_check.append((f, s, round(geometric_label(ds.lockinLL_G.values, ds.lockinRR_G.values), 3)))
    sw = res["sweep61"]
    c = np.array([r["geometric_c"] for r in sw]); keep = np.abs(c) >= C_SWEET
    out = {"labeler_on_paper_labels": lab_check, "n_scored": int(keep.sum()), "n_near_sweet_spot": int((~keep).sum())}
    for k in [k for k in sw[0] if "@" in k]:
        p = np.array([r[k] for r in sw])
        out[k] = dict(sign_accuracy=round(float(np.mean(np.sign(p[keep]) == np.sign(c[keep]))), 3),
                      rank_corr_with_geometry=round(float(spearmanr(c, p)[0]), 3))
    res["sweep_geometric_scoring"] = out
    json.dump(res, open(f"{W}/real_data_results.json", "w"), indent=1)
    print("labeler on the paper's 6 labeled maps (file, expected sign, c):")
    for row in lab_check: print("  ", row)
    print(f"sweep: {out['n_scored']} maps scored, {out['n_near_sweet_spot']} near a sweet spot")
    for k, v in out.items():
        if "@" in k: print(" ", k, v)
    print("PG, c, v1, v3:", [(r["PG_mV"], r["geometric_c"], round(r["v1@1.0"], 2), round(r["v3@1.0"], 2)) for r in sw[::4]])


if len(sys.argv) > 2 and sys.argv[2] == "geometric":
    score_sweep(sys.argv[1])
