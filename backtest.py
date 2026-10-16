"""Backtest de la stratégie sur l'historique football-data.co.uk (dossier hist/).
Cible « molle » = Bet365 (proxy d'Unibet) ; probabilité de référence = consensus des AUTRES books (Pinnacle x3), comme en live.
Usage : python backtest.py
"""
import glob
import math
import sys

import numpy as np
import pandas as pd

BOOKS = {"PS": 3.0, "BW": 1.0, "WH": 1.0, "1XB": 1.0, "BF": 1.0, "VC": 1.0, "IW": 1.0, "LB": 1.0}
TARGET = "B365"


def load():
    frames = []
    for f in sorted(glob.glob("hist/*.csv")):
        try:
            d = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")
        except Exception:
            continue
        d["season"] = f.split("\\")[-1].split("/")[-1].split("_")[0]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["FTR"].isin(["H", "D", "A"])].copy()
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def consensus(row):
    acc = np.zeros(3)
    w = 0.0
    for b, wt in BOOKS.items():
        try:
            o = np.array([row[f"{b}H"], row[f"{b}D"], row[f"{b}A"]], dtype=float)
        except KeyError:
            continue
        if np.any(np.isnan(o)) or np.any(o < 1.01):
            continue
        p = (1 / o) / (1 / o).sum()
        acc += p * wt
        w += wt
    return (acc / w, w) if w >= 4 else (None, 0)


def build():
    df = load()
    rows = []
    for _, r in df.iterrows():
        p, w = consensus(r)
        try:
            odds = np.array([r[f"{TARGET}H"], r[f"{TARGET}D"], r[f"{TARGET}A"]], dtype=float)
        except KeyError:
            continue
        if p is None or np.any(np.isnan(odds)):
            continue
        res = {"H": 0, "D": 1, "A": 2}[r["FTR"]]
        for k in range(3):
            rows.append({"date": r["date"], "season": r["season"], "div": r["Div"], "side": k, "p": p[k],
                         "odd": odds[k], "win": int(k == res)})
    return pd.DataFrame(rows)


def simulate(d, min_edge, max_edge=0.25, kelly=0.25, cap=0.06, start=10.0):
    d = d[(d.odd >= 1.25) & (d.odd <= 8)].copy()
    d["edge"] = d.p * d.odd - 1
    sel = d[(d.edge >= min_edge) & (d.edge <= max_edge)]
    flat = (sel.win * sel.odd - 1)
    bank, peak, mdd, dead = start, start, 0.0, False
    for _, r in sel.sort_values("date").iterrows():
        stake = min(bank * r.edge / (r.odd - 1) * kelly, bank * cap)
        if stake < 0.10:
            continue
        bank += stake * (r.win * r.odd - 1)
        peak = max(peak, bank)
        mdd = max(mdd, 1 - bank / peak)
        if bank <= 0.05:
            dead = True
            break
    return len(sel), flat.mean() if len(sel) else float("nan"), bank, mdd, dead


if __name__ == "__main__":
    d = build()
    print(f"{len(d) // 3} matchs, {d.date.min().date()} -> {d.date.max().date()}")
    base = d.win.mean()
    print("ROI brut (parier tout) :", round((d.win * d.odd - 1).mean(), 4))
    print(f"{'seuil':>6} {'paris':>7} {'ROI plat':>9} {'solde fin':>10} {'drawdown':>9} mort")
    for e in (0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        n, roi, bank, mdd, dead = simulate(d, e)
        print(f"{e:>6.2f} {n:>7} {roi:>9.3%} {bank:>10.2f} {mdd:>9.1%} {dead}")
