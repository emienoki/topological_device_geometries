
import json, time
import numpy as np
import torch, torch.nn as nn
from sklearn.linear_model import Ridge
from csd_sim import dataset, NPIX

torch.manual_seed(0); torch.set_num_threads(2)


class CSDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 32 -> 16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 16 -> 8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
            nn.Flatten(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        return self.f(x).squeeze(1)


def features(X):
    """Baseline features from the summed, positive part of each image."""
    w = np.clip(X.sum(1), 0, None) + 1e-9
    c = np.linspace(-1, 1, NPIX)
    x, y = np.meshgrid(c, c, indexing="ij")
    W = w.sum((1, 2))
    mx, my = (w * x).sum((1, 2)) / W, (w * y).sum((1, 2)) / W
    cxy = (w * (x - mx[:, None, None]) * (y - my[:, None, None])).sum((1, 2)) / W
    vx = (w * (x - mx[:, None, None]) ** 2).sum((1, 2)) / W
    vy = (w * (y - my[:, None, None]) ** 2).sum((1, 2)) / W
    q13 = (w * ((x * y) > 0)).sum((1, 2)) / W                 # weight in quadrants 1 and 3
    return np.stack([cxy, vx, vy, q13, cxy / np.sqrt(vx * vy)], 1)


def metrics(pred, y):
    e = np.abs(pred - y); near = np.abs(y) < 0.1
    return dict(mae=float(e.mean()), mae_near_sweet_spot=float(e[near].mean()),
                sign_correct=float(np.mean(np.sign(pred) == np.sign(y))))


if __name__ == "__main__":
    t0 = time.time()
    Xtr, ytr = dataset(24000, seed=1)
    Xva, yva = dataset(3000, seed=2)
    Xin, yin = dataset(3000, seed=3)
    Xsh, ysh = dataset(3000, seed=4, shifted=True)
    print(f"data {time.time() - t0:.0f}s", flush=True)

    # ---- baseline
    rb = Ridge(alpha=1e-3).fit(features(Xtr), ytr)
    base = {k: metrics(np.clip(rb.predict(features(X)), -1, 1), y) for k, X, y in
            [("in_distribution", Xin, yin), ("shifted_device", Xsh, ysh)]}

    # ---- CNN
    net = CSDNet(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    best, best_state = 9.0, None
    for ep in range(20):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]
            loss = ((net(Xt[b]) - yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad():
            va = float(((net(torch.from_numpy(Xva)) - torch.from_numpy(yva)).abs()).mean())
        if va < best:
            best, best_state = va, {k: v.clone() for k, v in net.state_dict().items()}
        print(f"epoch {ep + 1:2d}  val MAE {va:.4f}", flush=True)
    net.load_state_dict(best_state); net.eval()
    torch.save(best_state, "csd_cnn.pt")
    with torch.no_grad():
        pin, psh = (net(torch.from_numpy(X)).numpy() for X in (Xin, Xsh))
    cnn = {"in_distribution": metrics(pin, yin), "shifted_device": metrics(psh, ysh)}
    np.savez("csd_preds.npz", yin=yin, pin=pin, ysh=ysh, psh=psh,
             bin=np.clip(rb.predict(features(Xin)), -1, 1), bsh=np.clip(rb.predict(features(Xsh)), -1, 1))
    out = dict(cnn=cnn, baseline=base, n_train=len(ytr), seconds=round(time.time() - t0))
    json.dump(out, open("csd_results.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
