import json, time
import numpy as np
import torch, torch.nn as nn
from sklearn.linear_model import Ridge
from csd_sim import dataset
from csd_real import dataset_real
from train_csd import CSDNet, features, metrics

torch.set_num_threads(2)


class WideNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                    nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
        self.f = nn.Sequential(*block(2, 32), *block(32, 64), *block(64, 128),
                               nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Dropout(0.2),
                               nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):
        return self.f(x).squeeze(1)


def swap(x):
    return x.flip(1).transpose(2, 3)


def train(net, X, y, Xva, yva, epochs, lr, augment=False, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * ((len(X) + 127) // 128))
    Xt, yt, Xv, yv = map(torch.from_numpy, (X, y, Xva, yva))
    best, state = 9.0, None
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]; xb = Xt[b]
            if augment:
                m = torch.rand(len(b)) < 0.5
                xb = torch.where(m[:, None, None, None], swap(xb), xb)
            loss = ((net(xb) - yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            va = float((predict(net, Xv, augment) - yv).abs().mean())
        if va < best:
            best, state = va, {k: v.clone() for k, v in net.state_dict().items()}
        print(f"  epoch {ep + 1:2d}  val MAE {va:.4f}", flush=True)
    net.load_state_dict(state); net.eval()
    return net


def predict(net, X, tta=False):
    X = torch.as_tensor(X)
    with torch.no_grad():
        out = []
        for i in range(0, len(X), 1000):
            xb = X[i:i + 1000]; p = net(xb)
            if tta:
                p = 0.5 * (p + net(swap(xb)))
            out.append(p)
        return torch.cat(out)


if __name__ == "__main__":
    t0 = time.time()
    tests = {"realistic": dataset_real(3000, seed=13), "realistic_shift": dataset_real(3000, seed=14, kind="shifted"),
             "simple_shift": dataset(3000, seed=4, shifted=True)}
    Xva, yva = dataset_real(3000, seed=12)
    print(f"test data {time.time() - t0:.0f}s", flush=True)
    res, preds = {}, {}

    def evaluate(name, fn):
        res[name] = {}
        for k, (X, y) in tests.items():
            p = fn(X); preds[f"{name}_{k}"] = p
            res[name][k] = metrics(p, y)
        print(name, json.dumps(res[name]), flush=True)

    v1 = CSDNet(); v1.load_state_dict(torch.load("csd_cnn.pt")); v1.eval()
    evaluate("v1_simple_trained", lambda X: predict(v1, X).numpy())

    X24, y24 = dataset_real(24000, seed=11)
    rb = Ridge(alpha=1e-3).fit(features(X24), y24)
    evaluate("ridge_realistic", lambda X: np.clip(rb.predict(features(X)), -1, 1))

    print("training v2 (CSDNet on 24k realistic)", flush=True)
    v2 = train(CSDNet(), X24, y24, Xva, yva, epochs=20, lr=2e-3)
    torch.save(v2.state_dict(), "csd_v2.pt")
    evaluate("v2_realistic_24k", lambda X: predict(v2, X).numpy())
    del X24

    X60, y60 = dataset_real(60000, seed=21)
    print(f"training v3 (WideNet on 60k realistic)  [{time.time() - t0:.0f}s]", flush=True)
    v3 = train(WideNet(), X60, y60, Xva, yva, epochs=24, lr=3e-3, augment=True)
    torch.save(v3.state_dict(), "csd_v3.pt")
    evaluate("v3_wide_60k", lambda X: predict(v3, X, tta=True).numpy())

    for k, (X, y) in tests.items():
        preds[f"y_{k}"] = y
    np.savez("real_preds.npz", **preds)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open("real_results.json", "w"), indent=1)
    print("done", res["seconds"], "s")
