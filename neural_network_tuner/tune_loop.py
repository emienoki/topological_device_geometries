import json
import numpy as np
import torch
from csd_sim import csd, random_aug, dataset
from train_csd import CSDNet, features, metrics
from sklearn.linear_model import Ridge

torch.manual_seed(0); torch.set_num_threads(2)


def load(path="csd_cnn.pt"):
    n = CSDNet(); n.load_state_dict(torch.load(path)); n.eval(); return n


def finetune(net, X, y, epochs=40, lr=2e-4):
    opt = torch.optim.Adam(net.parameters(), lr)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 16):
            b = perm[i:i + 16]
            loss = ((net(Xt[b]) - yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net


class FakeDevice:
    def __init__(self, rng, shifted):
        self.rng = rng
        self.Vstar = rng.uniform(-25, 25)
        self.s = rng.uniform(0.5, 2.0)
        self.big = rng.uniform(0.4, 1.0)
        self.aug = random_aug(rng, shifted)            # fixed personality
        self.n = 0

    def r_true(self, V):
        return np.tanh(self.s * (V - self.Vstar) / 40.0)

    def measure(self, V):
        self.n += 1
        L = self.s * (V - self.Vstar) / 20.0           # log(t/D)
        t, D = (self.big, self.big * np.exp(-L)) if L >= 0 else (self.big * np.exp(L), self.big)
        a = dict(self.aug)                             # same device; csd() draws fresh noise each call
        return csd(t, D, self.rng, a)


def tune(dev, predict, V0, max_meas=25, tol=0.02):
    V_prev, r_prev = V0, predict(dev.measure(V0))
    V = V0 + 5.0; r = predict(dev.measure(V))
    slope = (r - r_prev) / (V - V_prev)
    slope = slope if slope > 1e-3 else 0.02            # the ratio rises with V on this gate; guard bad estimates
    while abs(r) > tol and dev.n < max_meas:
        step = float(np.clip(-r / slope, -15, 15))
        V_prev, r_prev = V, r
        V = V + step; r = predict(dev.measure(V))
        if abs(V - V_prev) > 0.3:
            new = (r - r_prev) / (V - V_prev)
            if new > 1e-3:
                slope = new
    return V, dev.n


def run(predict, shifted, n=300, seed=10):
    rng = np.random.default_rng(seed)
    err, meas = [], []
    for _ in range(n):
        dev = FakeDevice(rng, shifted)
        V, m = tune(dev, predict, V0=dev.Vstar + rng.uniform(-40, 40))
        err.append(abs(V - dev.Vstar)); meas.append(m)
    err = np.array(err)
    return dict(within_1p5mV=float(np.mean(err <= 1.5)), within_4p5mV=float(np.mean(err <= 4.5)),
                median_error_mV=float(np.median(err)), mean_measurements=float(np.mean(meas))), err.tolist()


if __name__ == "__main__":
    base_net = load()
    Xsh, ysh = dataset(3000, seed=4, shifted=True)                  # same shifted test set as train_csd.py
    Xft, yft = dataset(51, seed=99, shifted=True)                   # 51 labeled pictures from the new device
    ft_net = finetune(load(), Xft, yft)
    with torch.no_grad():
        ft_metrics = metrics(ft_net(torch.from_numpy(Xsh)).numpy(), ysh)
    print("fine-tuned on 51 shifted pictures:", ft_metrics, flush=True)
    torch.save(ft_net.state_dict(), "csd_cnn_finetuned.pt")

    Xtr, ytr = dataset(24000, seed=1)
    rb = Ridge(alpha=1e-3).fit(features(Xtr), ytr)

    def cnn(net):
        return lambda G: float(net(torch.from_numpy(G[None])).item())
    predictors = {"cnn_training_style_device": (cnn(base_net), False),
                  "cnn_shifted_device": (cnn(base_net), True),
                  "cnn_finetuned_shifted_device": (cnn(ft_net), True),
                  "baseline_shifted_device": (lambda G: float(np.clip(rb.predict(features(G[None]))[0], -1, 1)), True)}
    out = {"finetune_51_shifted_metrics": ft_metrics, "tuning": {}, "errors_mV": {}}
    with torch.no_grad():
        for name, (p, sh) in predictors.items():
            out["tuning"][name], out["errors_mV"][name] = run(p, sh)
            print(name, out["tuning"][name], flush=True)
    json.dump(out, open("csd_tuning_results.json", "w"), indent=1)
