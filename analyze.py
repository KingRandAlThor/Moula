"""Courbes modèle vs marché + recherche de facteurs/événements (avec correction pour tests multiples).
Usage : python analyze.py   -> dossier out/
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

import pretrain2

warnings.filterwarnings("ignore")
os.makedirs("out", exist_ok=True)

d = pretrain2.features()
d = pd.concat([d, pd.get_dummies(d["div"]).astype(float)], axis=1)
mk = d[["pH", "pD", "pA"]].values
for i in range(3):
    d[f"l{i}"] = np.log(np.clip(mk[:, i], 1e-6, 1))
cols = [c for c in d.columns if c not in ("idx", "date", "season", "res", "div", "pH", "pD", "pA")]
d = d[d.nH >= 5].reset_index(drop=True)
mk = d[["pH", "pD", "pA"]].values
y = d.res.values
tr = d.season.isin(["1819", "1920", "2021", "2122", "2223"]).values
te = d.season.isin(["2425", "2526"]).values
m = HistGradientBoostingClassifier(learning_rate=0.03, max_depth=3, l2_regularization=30, max_iter=150,
                                   min_samples_leaf=200, random_state=0).fit(d.loc[tr, cols], y[tr])
pm = m.predict_proba(d[cols])
d["mH"], d["mD"], d["mA"] = pm[:, 0], pm[:, 1], pm[:, 2]

# 1) calibration + log-loss cumulé sur le test
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
t = te
for name, P, c in (("Marché", mk, "tab:blue"), ("Modèle", pm, "tab:red")):
    p = P[t].ravel()
    o = np.eye(3)[y[t]].ravel()
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1
    xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() > 30]
    ys = [o[idx == b].mean() for b in range(10) if (idx == b).sum() > 30]
    ax[0].plot(xs, ys, "o-", label=name, color=c)
ax[0].plot([0, 1], [0, 1], "k--", lw=1)
ax[0].set(title="Calibration (test 2425-2526)", xlabel="proba prédite", ylabel="fréquence réelle")
ax[0].legend()
tt = d[t].sort_values("date")
ll_m = -np.log(np.clip(tt[["pH", "pD", "pA"]].values[np.arange(len(tt)), tt.res.values], 1e-6, 1))
ll_p = -np.log(np.clip(tt[["mH", "mD", "mA"]].values[np.arange(len(tt)), tt.res.values], 1e-6, 1))
ax[1].plot(tt.date.values, np.cumsum(ll_p - ll_m), color="tab:red")
ax[1].axhline(0, color="k", lw=1)
ax[1].set(title="Perte cumulée modèle − marché (>0 = modèle pire)", xlabel="date")
ax[1].tick_params(axis="x", rotation=30)
w = tt.copy()
w["wk"] = w.date.dt.to_period("W").dt.start_time
g = w.groupby("wk").apply(lambda z: pd.Series({"mark": z.pH.mean(), "reel": (z.res == 0).mean(), "mod": z.mH.mean()}))
ax[2].plot(g.index, g.reel.rolling(4).mean(), label="% victoires domicile réel (4 sem.)", color="k")
ax[2].plot(g.index, g.mark.rolling(4).mean(), label="prédit marché", color="tab:blue")
ax[2].plot(g.index, g["mod"].rolling(4).mean(), label="prédit modèle", color="tab:red")
ax[2].legend(fontsize=8)
ax[2].set(title="Victoires à domicile par semaine")
ax[2].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("out/courbes.png", dpi=110)

# 2) chasse aux facteurs : résidus (réel − proba du marché) par segment, toutes saisons
d["resH"] = (d.res == 0) - d.pH
d["resD"] = (d.res == 1) - d.pD
d["resA"] = (d.res == 2) - d.pA
d["weekday"] = d.date.dt.day_name()
d["month"] = d.date.dt.month
d["week"] = d.date.dt.to_period("W").astype(str)
d["fav"] = pd.cut(d.pH - d.pA, [-1, -0.3, -0.1, 0.1, 0.3, 1], labels=["gros outsider dom", "outsider dom", "équilibré", "fav dom", "gros fav dom"])
d["rest"] = pd.cut(d.restH - d.restA, [-31, -3, 3, 31], labels=["dom moins reposé", "égal", "dom plus reposé"])
d["elo_b"] = pd.cut(d.elo, [-1000, -150, 0, 150, 1000], labels=["elo--", "elo-", "elo+", "elo++"])
rows = []
for seg in ["div", "weekday", "month", "fav", "rest", "elo_b", "season"]:
    for val, z in d.groupby(seg):
        if len(z) < 300:
            continue
        for lab, col in (("domicile", "resH"), ("nul", "resD"), ("extérieur", "resA")):
            tstat, pv = stats.ttest_1samp(z[col], 0)
            rows.append({"facteur": f"{seg}={val}", "issue": lab, "n": len(z), "écart": z[col].mean(), "p": pv})
# événement « semaine » : semaines où les domiciles écrasent
wk = d.groupby(["div", "week"]).agg(n=("res", "size"), home=("resH", "mean"), dom=("res", lambda s: (s == 0).mean())).reset_index()
wk = wk[wk.n >= 6]
nxt = wk.sort_values("week").groupby("div").apply(lambda z: z.assign(next_home=z.home.shift(-1))).dropna()
r, pv = stats.pearsonr(nxt.home, nxt.next_home)
hot = nxt[nxt.home > nxt.home.quantile(0.9)]
cold = nxt[nxt.home < nxt.home.quantile(0.1)]
ev = {"corr semaine→semaine suivante": (r, pv), "après semaine « domiciles » (top10%) : écart suivant": (hot.next_home.mean(), stats.ttest_1samp(hot.next_home, 0)[1]),
      "après semaine « extérieurs » (bas10%) : écart suivant": (cold.next_home.mean(), stats.ttest_1samp(cold.next_home, 0)[1])}
res = pd.DataFrame(rows)
res["p_bonf"] = np.minimum(1, res.p * len(res))
res = res.sort_values("p")
print(f"{len(res)} segments testés. Seuil Bonferroni à 5 % : p < {0.05 / len(res):.5f}")
print(res.head(12).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nÉVÉNEMENT « SEMAINE DOMICILE » :")
for k, (a, b) in ev.items():
    print(f"  {k}: {a:+.4f} (p={b:.3f})")
print(f"  semaines étudiées : {len(nxt)} ; semaines 'domiciles' : {len(hot)}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
top = res.head(15).iloc[::-1]
ax[0].barh(top.facteur + " / " + top.issue, top.écart * 100, color=["tab:green" if p < 0.05 / len(res) else "tab:gray" for p in top.p])
ax[0].set(title="Plus gros écarts réel − marché (points de %) ; vert = significatif après correction", xlabel="points de %")
ax[0].tick_params(axis="y", labelsize=7)
ax[1].scatter(nxt.home * 100, nxt.next_home * 100, s=6, alpha=0.4)
ax[1].set(title=f"Écart domicile semaine S vs S+1 (r={r:+.3f})", xlabel="semaine S (pts %)", ylabel="semaine S+1 (pts %)")
plt.tight_layout()
plt.savefig("out/facteurs.png", dpi=110)
