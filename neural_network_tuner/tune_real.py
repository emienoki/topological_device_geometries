import json, sys
import numpy as np
import torch
from sklearn.linear_model import Ridge
from csd_real import csd_real, random_aug_real, dataset_real
from train_csd import CSDNet, features
from train_real import WideNet, swap
from tune_loop import tune

torch.set_num_threads(2)


class RealDevice:
    def __init__(self, rng, kind):
        self.rng = rng
        self.Vstar = rng.uniform(-25, 25)
        self.s = rng.uniform(0.5, 2.0)
        self.big = rng.uniform(0.4, 1.0)
        self.U = rng.uniform(0, 0.5) if kind == "train" else rng.uniform(0.3, 0.7)
        self.aug = random_aug_real(rng, kind)
        self.off = np.array(self.aug["offset"], float)
        self.n = 0

    def measure(self, V):
        self.n += 1
        self.off += self.rng.normal(0, 0.03, 2)                       # drift between measurements
        if self.rng.random() < 0.1:
            self.off[0] += self.rng.choice([-1, 1]) * self.aug["tel_amp"]
        L = self.s * (V - self.Vstar) / 20.0
        t, De = (self.big, self.big * np.exp(-L)) if L >= 0 else (self.big * np.exp(L), self.big)
        a = dict(self.aug); a["offset"] = tuple(np.clip(self.off, -0.8, 0.8))
        return csd_real(t, De + self.U / 2, self.U, self.rng, a)


def run(predict, kind, n=300, seed=30):
    rng = np.random.default_rng(seed)
    err, meas = [], []
    for _ in range(n):
        dev = RealDevice(rng, kind)
        V, m = tune(dev, predict, V0=dev.Vstar + rng.uniform(-40, 40))
        err.append(abs(V - dev.Vstar)); meas.append(m)
    err = np.array(err)
    return dict(within_1p5mV=float(np.mean(err <= 1.5)), within_4p5mV=float(np.mean(err <= 4.5)),
                median_error_mV=float(np.median(err)), mean_measurements=float(np.mean(meas)))


if __name__ == "__main__":
    nets = {}
    v1 = CSDNet(); v1.load_state_dict(torch.load("csd_cnn.pt")); v1.eval(); nets["v1_simple_trained"] = (v1, False)
    v2 = CSDNet(); v2.load_state_dict(torch.load("csd_v2.pt")); v2.eval(); nets["v2_realistic_24k"] = (v2, False)
    try:
        v3 = WideNet(); v3.load_state_dict(torch.load("csd_v3.pt")); v3.eval(); nets["v3_wide"] = (v3, True)
    except FileNotFoundError:
        print("no v3 yet")
    X, y = dataset_real(24000, seed=11)
    rb = Ridge(alpha=1e-3).fit(features(X), y)

    def cnn(net, tta):
        def p(G):
            x = torch.from_numpy(G[None])
            out = net(x)
            if tta:
                out = 0.5 * (out + net(swap(x)))
            return float(out.item())
        return p
    out = {}
    with torch.no_grad():
        for kind in ("train", "shifted"):
            for name, (net, tta) in nets.items():
                out[f"{name}__{kind}"] = run(cnn(net, tta), kind)
                print(name, kind, out[f"{name}__{kind}"], flush=True)
            out[f"ridge__{kind}"] = run(lambda G: float(np.clip(rb.predict(features(G[None]))[0], -1, 1)), kind)
            print("ridge", kind, out[f"ridge__{kind}"], flush=True)
    json.dump(out, open("real_tuning_results.json", "w"), indent=1)
