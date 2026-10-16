"""Recherche exhaustive de stratégies (grille de paramètres). Sélection sur 1819-2324, vérification sur 2425-2526.
Critère demandé : meilleur winrate (avec minimum de paris) ; on affiche aussi le ROI, seul vrai juge."""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import backtest

warnings.filterwarnings("ignore")
d = backtest.build()
d = d[(d.odd >= 1.2) & (d.odd <= 10)].copy()
d = d.dropna(subset=["div"]).copy()
d["edge"] = d.p * d.odd - 1
d["ret"] = d.win * d.odd - 1
sel_mask = ~d.season.isin(["2425", "2526"])
A, T = d[sel_mask], d[~sel_mask]

edges = [0.0, 0.02, 0.03, 0.05, 0.08]
omin = [1.2, 1.5, 2.0, 3.0]
omax = [2.0, 3.0, 5.0, 10.0]
pmin = [0.0, 0.3, 0.5, 0.65]
sides = [(0, 1, 2), (0,), (1,), (2,), (0, 2)]
divs = [None] + [(x,) for x in sorted(d["div"].unique())]
MIN_N = 300
rows = []


def stat(z):
    return len(z), z.win.mean(), z.ret.mean()


for e, lo, hi, pm, sd, dv in itertools.product(edges, omin, omax, pmin, sides, divs):
    if hi <= lo:
        continue
    f = lambda x, e=e, lo=lo, hi=hi, pm=pm, sd=sd, dv=dv: x[(x.edge >= e) & (x.edge <= 0.25) & x.odd.between(lo, hi) & (x.p >= pm) & x.side.isin(sd) &
                    (True if dv is None else x["div"].isin(dv))]
    za = f(A)
    if len(za) < MIN_N:
        continue
    n, wr, roi = stat(za)
    rows.append((e, lo, hi, pm, sd, dv, n, wr, roi, f))
print(f"{len(rows)} combinaisons testées (>= {MIN_N} paris en sélection)")

res = pd.DataFrame([r[:9] for r in rows], columns=["edge", "odd_min", "odd_max", "p_min", "issues", "ligue", "n", "winrate", "roi"])
fn = [r[9] for r in rows]


def show(title, order):
    print(f"\n=== {title} ===")
    for i in order[:5]:
        r = res.loc[i]
        zt = fn[i](T)
        nt, wt, rt = stat(zt) if len(zt) else (0, float("nan"), float("nan"))
        pv = stats.ttest_1samp(zt.ret, 0)[1] if len(zt) > 30 else float("nan")
        print(f"edge>={r.edge:.2f} cote {r.odd_min}-{r.odd_max} p>={r.p_min} issues={r.issues} ligue={r.ligue} | "
              f"SÉLECTION n={r.n} win {r.winrate:.1%} ROI {r.roi:+.1%}  ->  TEST n={nt} win {wt:.1%} ROI {rt:+.1%} (p={pv:.2f})")


show("MEILLEUR WINRATE (ce que tu as demandé)", list(res.winrate.sort_values(ascending=False).index))
show("MEILLEUR ROI en sélection", list(res.roi.sort_values(ascending=False).index))
# vérité globale : sur les stratégies, le ROI test est-il corrélé au ROI sélection ?
tr_roi = [stat(f(T))[2] if len(f(T)) > 30 else np.nan for f in fn]
res["roi_test"] = tr_roi
ok = res.dropna(subset=["roi_test"])
print(f"\nCorrélation ROI sélection -> ROI test sur {len(ok)} stratégies : {np.corrcoef(ok.roi, ok.roi_test)[0, 1]:+.3f}")
print(f"ROI test moyen de toutes les stratégies : {ok.roi_test.mean():+.2%} ; part des stratégies rentables en test : {(ok.roi_test > 0).mean():.0%}")
print(f"ROI test moyen du top 20 par ROI sélection : {ok.sort_values('roi', ascending=False).head(20).roi_test.mean():+.2%}")
