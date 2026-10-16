"""Modèle 100 % maison (aucune cote en entrée) : Elo, forme, buts, repos, arbitre, championnat.
Il parie contre Bet365 là où il voit un avantage. Entraîné <=2223, réglé 2324, test 2425-2526."""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

import pretrain2

warnings.filterwarnings("ignore")
d = pretrain2.features()
d = d.dropna(subset=["oH", "oD", "oA"])
d = d[d.nH >= 5].reset_index(drop=True)
d = pd.concat([d, pd.get_dummies(d["div"]).astype(float)], axis=1)
own = [c for c in d.columns if c not in ("idx", "date", "season", "res", "div", "pH", "pD", "pA", "oH", "oD", "oA")]
y = d.res.values
tr = d.season.isin(["1819", "1920", "2021", "2122", "2223"]).values
va = (d.season == "2324").values
te = d.season.isin(["2425", "2526"]).values
mk = d[["pH", "pD", "pA"]].values
print(f"train {tr.sum()} valid {va.sum()} test {te.sum()}")
print(f"marché test {log_loss(y[te], mk[te]):.4f}")
sc = StandardScaler().fit(d.loc[tr, own])
lr = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(d.loc[tr, own]), y[tr])
gb = HistGradientBoostingClassifier(learning_rate=0.03, max_depth=3, l2_regularization=30, max_iter=150,
                                    min_samples_leaf=200, random_state=0).fit(d.loc[tr, own], y[tr])
models = {"Logistique maison": lambda X: lr.predict_proba(sc.transform(X[own])),
          "Boosting maison": lambda X: gb.predict_proba(X[own])}
for n, f in models.items():
    print(f"{n}: valid {log_loss(y[va], f(d[va])):.4f} | TEST {log_loss(y[te], f(d[te])):.4f}")
    P = f(d[te])
    O = d.loc[te, ["oH", "oD", "oA"]].values
    yy = y[te]
    for edge in (0.0, 0.05, 0.10, 0.15):
        ev = P * O - 1
        k = ev.argmax(1)
        best = ev[np.arange(len(k)), k]
        m = best >= edge
        win = (k == yy)[m]
        roi = np.where(win, O[np.arange(len(k)), k][m] - 1, -1)
        print(f"   edge>={edge:.2f}: {m.sum():4d} paris, ROI {roi.mean():+.2%}" +
              (f" (p={stats.ttest_1samp(roi, 0)[1]:.2f})" if m.sum() > 30 else ""))
# référence : parier au hasard sur Bet365
O = d.loc[te, ["oH", "oD", "oA"]].values
print(f"Référence : tout parier = {np.mean([(O[i, y[te][i]] - 1) if True else 0 for i in range(len(O))]) * 0 + np.mean(np.eye(3)[y[te]] * O - 1):+.2%}")
