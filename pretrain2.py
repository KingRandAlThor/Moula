"""Modèle 1X2 par match : marché + Elo + forme + buts + repos + arbitre. Comparé au consensus hors échantillon.
Réglage sur 2324 uniquement ; 2425-2526 = test final, regardé une seule fois.
Usage : python pretrain2.py
"""
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import backtest

warnings.filterwarnings("ignore")


def features():
    df = backtest.load()
    elo = defaultdict(lambda: 1500.0)
    hist = defaultdict(list)  # équipe -> [(date, points, buts pour, buts contre)]
    ref = defaultdict(lambda: [0, 0, 0])  # arbitre -> [n, home wins, draws]
    rows = []
    for r in df.itertuples():
        p, w = backtest.consensus(r._asdict() if False else pd.Series(r._asdict()))
        h, a = r.HomeTeam, r.AwayTeam
        res = {"H": 0, "D": 1, "A": 2}[r.FTR]
        if p is not None:
            def form(t, n=5):
                x = hist[t][-n:]
                if not x:
                    return 1.4, 1.3, 1.3, 7
                return (np.mean([i[1] for i in x]), np.mean([i[2] for i in x]), np.mean([i[3] for i in x]),
                        (r.date - x[-1][0]).days)
            fh, fa = form(h), form(a)
            rr = ref[getattr(r, "Referee", None)]
            rows.append({
                "idx": len(rows), "date": r.date, "season": r.season, "res": res,
                "pH": p[0], "pD": p[1], "pA": p[2],
                "elo": elo[h] - elo[a], "fpH": fh[0], "fpA": fa[0], "gfH": fh[1], "gaH": fh[2],
                "gfA": fa[1], "gaA": fa[2], "restH": min(fh[3], 30), "restA": min(fa[3], 30),
                "nH": len(hist[h]), "nA": len(hist[a]),
                "refHome": (rr[1] + 4) / (rr[0] + 10), "refDraw": (rr[2] + 2.5) / (rr[0] + 10),
                "div": r.Div,
            })
        # mise à jour après le match
        hg, ag = r.FTHG, r.FTAG
        exp = 1 / (1 + 10 ** ((elo[a] - elo[h] - 60) / 400))
        sc = 1.0 if res == 0 else 0.5 if res == 1 else 0.0
        gd = max(1, abs(hg - ag)) ** 0.5
        elo[h] += 20 * gd * (sc - exp)
        elo[a] -= 20 * gd * (sc - exp)
        pts = lambda x, y: 3 if x > y else 1 if x == y else 0
        hist[h].append((r.date, pts(hg, ag), hg, ag))
        hist[a].append((r.date, pts(ag, hg), ag, hg))
        k = getattr(r, "Referee", None)
        if isinstance(k, str):
            ref[k][0] += 1
            ref[k][1] += res == 0
            ref[k][2] += res == 1
    return pd.DataFrame(rows)


def lg(p):
    p = np.clip(p, 1e-6, 1)
    return np.log(p / p.sum(1, keepdims=True))


if __name__ == "__main__":
    d = features()
    d = pd.concat([d, pd.get_dummies(d["div"]).astype(float)], axis=1)
    cols = [c for c in d.columns if c not in ("idx", "date", "season", "res", "div", "pH", "pD", "pA")]
    mk = d[["pH", "pD", "pA"]].values
    d[[f"l{i}" for i in range(3)]] = lg(mk)
    cols += ["l0", "l1", "l2"]
    d = d[d.nH >= 5].reset_index(drop=True)
    mk = d[["pH", "pD", "pA"]].values
    tr = d.season.isin(["1819", "1920", "2021", "2122", "2223"]).values
    va = (d.season == "2324").values
    te = d.season.isin(["2425", "2526"]).values
    y = d.res.values
    print(f"matchs train {tr.sum()} valid {va.sum()} test {te.sum()}")
    print(f"marché  valid {log_loss(y[va], mk[va]):.5f}  test {log_loss(y[te], mk[te]):.5f}")
    best = None
    for lr, depth, l2 in ((0.03, 2, 10), (0.03, 3, 30), (0.05, 2, 30)):
        m = HistGradientBoostingClassifier(learning_rate=lr, max_depth=depth, l2_regularization=l2, max_iter=150,
                                           min_samples_leaf=200, random_state=0)
        m.fit(d.loc[tr, cols], y[tr])
        ll = log_loss(y[va], m.predict_proba(d.loc[va, cols]))
        print(f"  GBM lr={lr} depth={depth} l2={l2}: valid {ll:.5f}")
        if best is None or ll < best[0]:
            best = (ll, m, (lr, depth, l2))
    lr_ = LogisticRegression(C=1.0, max_iter=2000)
    lr_.fit(d.loc[tr, cols].fillna(0), y[tr])
    print(f"  Logistique valid {log_loss(y[va], lr_.predict_proba(d.loc[va, cols].fillna(0))):.5f}")
    m = best[1]
    pt = m.predict_proba(d.loc[te, cols])
    print(f"TEST  GBM {log_loss(y[te], pt):.5f} vs marché {log_loss(y[te], mk[te]):.5f}")
    bl = 0.5 * pt + 0.5 * mk[te]
    print(f"TEST  mélange 50/50 {log_loss(y[te], bl):.5f}")
    # ROI : parier là où le modèle voit un avantage vs cotes Bet365 (cote cible)
    t = d[te].copy()
    odds = []
    raw = backtest.load()
    key = {(r.date, r.HomeTeam, r.AwayTeam): r for r in raw.itertuples()}
    print("(ROI sur Bet365 : voir backtest.simulate, non reproduit ici sans cotes alignées)")
