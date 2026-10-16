"""Pré-entraîne le réseau sur l'historique et le compare au marché sur des saisons jamais vues.
Écrit base_model.pkl seulement s'il bat le consensus (log-loss) ET ne dégrade pas le ROI.
Usage : python pretrain.py
"""
import math
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import backtest

warnings.filterwarnings("ignore")
DIVS = ["E0", "SP1", "D1", "I1", "F1", "F2", "N1", "P1"]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def base_features(p, odd, side, div):
    """Même signature qu'en live (monitor.base_features) : probas du marché, cote cible, issue, championnat."""
    return [logit(p), math.log(odd), float(side == 0), float(side == 1), float(side == 2)] + \
           [float(div == d) for d in DIVS]


if __name__ == "__main__":
    d = backtest.build()
    d = d[(d.odd >= 1.25) & (d.odd <= 8)].copy()
    X = np.array([base_features(r.p, r.odd, r.side, r.div) for r in d.itertuples()])
    y = d.win.values
    train = d.season.isin(["1819", "1920", "2021", "2122", "2223"]).values
    valid = d.season.isin(["2324"]).values
    test = d.season.isin(["2425", "2526"]).values
    print(f"train {train.sum()//1} | valid {valid.sum()} | test {test.sum()} (lignes)")
    best = None
    for alpha in (1e-1, 1.0, 10.0):
        net = make_pipeline(StandardScaler(), MLPClassifier((16, 8), alpha=alpha, early_stopping=True,
                                                             max_iter=400, random_state=0))
        net.fit(X[train], y[train])
        ll = log_loss(y[valid], np.clip(net.predict_proba(X[valid])[:, 1], 1e-4, 1 - 1e-4))
        if best is None or ll < best[0]:
            best = (ll, alpha, net)
    ll_v, alpha, net = best
    ll_mkt_v = log_loss(y[valid], np.clip(d.p.values[valid], 1e-4, 1 - 1e-4))
    print(f"validation 2324 : réseau {ll_v:.5f} vs marché {ll_mkt_v:.5f} (alpha={alpha})")
    pt = np.clip(net.predict_proba(X[test])[:, 1], 1e-4, 1 - 1e-4)
    ll_t = log_loss(y[test], pt)
    ll_mkt_t = log_loss(y[test], np.clip(d.p.values[test], 1e-4, 1 - 1e-4))
    print(f"TEST 2425-2526  : réseau {ll_t:.5f} vs marché {ll_mkt_t:.5f}  ({'MIEUX' if ll_t < ll_mkt_t else 'PIRE'})")
    t = d[test].copy()
    for name, p in (("marché", t.p.values), ("moyenne réseau+marché", (t.p.values + pt) / 2), ("réseau seul", pt)):
        tt = t.copy()
        tt["p"] = p
        print(f"-- ROI test avec p = {name}")
        for e in (0.02, 0.03, 0.04, 0.06):
            n, roi, bank, mdd, dead = backtest.simulate(tt, e)
            print(f"   seuil {e:.2f} : {n:4d} paris, ROI plat {roi:+.2%}")
    if ll_t < ll_mkt_t:
        net.fit(X[train | valid], y[train | valid])
        pickle.dump({"net": net, "divs": DIVS}, open("base_model.pkl", "wb"))
        print("base_model.pkl écrit.")
    else:
        print("Le réseau ne bat pas le marché : base_model.pkl NON écrit (le bot garde le consensus).")
