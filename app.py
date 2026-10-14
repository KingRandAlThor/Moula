"""Aide à la décision pour paris sportifs : cotes, variations, value bets, bankroll.
Aucune connexion à Unibet : les mises réelles sont placées manuellement par l'utilisateur.
"""
import json
import random
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

STATE_FILE = Path(__file__).with_name("state.json")
MAX_STAKE_PCT = 0.06
START_BANKROLL = 10.0
TARGET_BOOK = "Unibet (FR)"
DEMO_BOOKS = [TARGET_BOOK, "Bet365", "Pinnacle", "Betclic", "Winamax"]


# ---------- état persistant ----------
def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"bankroll": START_BANKROLL, "bets": [], "dead": False, "peak": START_BANKROLL}


def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=2), encoding="utf-8")


# ---------- maths ----------
def devig(odds: dict) -> dict:
    """Probabilités sans marge à partir d'un dict {issue: cote}."""
    inv = {k: 1 / v for k, v in odds.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def kelly_stake(p, odd, bankroll, kelly_frac):
    b = odd - 1
    f = (p * odd - 1) / b if b > 0 else 0
    if f <= 0:
        return 0.0
    return round(min(bankroll * f * kelly_frac, bankroll * MAX_STAKE_PCT), 2)


# ---------- sources de cotes ----------
def demo_matches(n=6):
    teams = ["PSG", "OM", "Lyon", "Lille", "Nice", "Lens", "Rennes", "Monaco", "Nantes", "Brest", "Reims", "Metz"]
    random.shuffle(teams)
    out = []
    for i in range(n):
        raw = [random.uniform(0.2, 0.5), random.uniform(0.2, 0.3), random.uniform(0.2, 0.5)]
        s = sum(raw)
        true_p = [x / s for x in raw]
        out.append({
            "id": f"m{i}",
            "name": f"{teams[2*i]} - {teams[2*i+1]}",
            "outcomes": [teams[2 * i], "Nul", teams[2 * i + 1]],
            "true_p": true_p,
            "odds": {b: [round(1 / (p * random.uniform(0.9, 1.1)) * 0.95, 2) for p in true_p] for b in DEMO_BOOKS},
        })
    return out


def demo_tick(matches):
    for m in matches:
        for b, odds in m["odds"].items():
            m["odds"][b] = [max(1.05, round(o * random.uniform(0.96, 1.04), 2)) for o in odds]


def live_matches(api_key, sport):
    r = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
        params={"apiKey": api_key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"},
        timeout=15,
    )
    r.raise_for_status()
    out = []
    for ev in r.json():
        books = {}
        names = None
        for bk in ev.get("bookmakers", []):
            mk = next((m for m in bk["markets"] if m["key"] == "h2h"), None)
            if not mk:
                continue
            o = {x["name"]: x["price"] for x in mk["outcomes"]}
            names = names or sorted(o)
            if sorted(o) == names:
                books[bk["title"]] = [o[n] for n in names]
        if names and len(books) >= 3:
            out.append({"id": ev["id"], "name": f'{ev["home_team"]} - {ev["away_team"]}',
                        "outcomes": names, "odds": books})
    return out


# ---------- analyse ----------
def analyse(m, target, min_edge, kelly_frac, bankroll):
    others = [b for b in m["odds"] if b != target]
    if target not in m["odds"] or not others:
        return []
    probs = [devig(dict(zip(m["outcomes"], m["odds"][b]))) for b in others]
    rows = []
    for i, name in enumerate(m["outcomes"]):
        p = sum(x[name] for x in probs) / len(probs)
        odd = m["odds"][target][i]
        edge = p * odd - 1
        rows.append({
            "Match": m["name"], "Issue": name, f"Cote {target}": odd,
            "Proba consensus": round(p, 3), "Edge": round(edge, 3),
            "Mise suggérée": kelly_stake(p, odd, bankroll, kelly_frac) if edge >= min_edge else 0.0,
            "mid": m["id"], "idx": i, "p": p,
        })
    return rows


# ---------- UI ----------
st.set_page_config(page_title="Bankroll Survivor", layout="wide")
S = load_state()
ss = st.session_state
ss.setdefault("history", [])

st.title("🎯 Bankroll Survivor — aide à la décision")
st.caption("Cotes réelles. Aucune connexion à Unibet : tu places la mise toi-même, puis tu l'enregistres ici.")

with st.sidebar:
    st.header("Paramètres")
    api_key = st.text_input("Clé The Odds API (the-odds-api.com, gratuite)", type="password")
    sport = st.text_input("Sport (clé API)", "soccer_france_ligue_one")
    target = st.text_input("Bookmaker cible", TARGET_BOOK)
    min_edge = st.slider("Edge minimum", 0.0, 0.2, 0.03, 0.005)
    kelly_frac = st.slider("Fraction de Kelly", 0.05, 1.0, 0.25, 0.05)
    st.divider()
    if st.button("🔄 Réinitialiser la partie"):
        S = {"bankroll": START_BANKROLL, "bets": [], "dead": False, "peak": START_BANKROLL}
        save_state(S)
        ss.history = []
        st.rerun()

c1, c2, c3 = st.columns(3)
c1.metric("Bankroll", f'{S["bankroll"]:.2f} €')
c2.metric("Mise max (6 %)", f'{S["bankroll"] * MAX_STAKE_PCT:.2f} €')
c3.metric("Plus haut", f'{S["peak"]:.2f} €')

if S["dead"] or S["bankroll"] <= 0:
    S["dead"] = True
    save_state(S)
    st.error("💀 Bankroll à 0 : tu es mort. Réinitialise la partie pour recommencer.")
    st.stop()

# chargement des matchs
if not api_key:
    st.info("Entre ta clé API (gratuite sur the-odds-api.com) pour charger les cotes réelles.")
    st.stop()
if st.button("🔄 Rafraîchir les cotes (à refaire régulièrement pour voir les variations)"):
    try:
        ss.matches = live_matches(api_key, sport)
    except Exception as e:
        st.error(f"Erreur API : {e}")
if not ss.get("matches"):
    st.info("Clique sur « Rafraîchir les cotes ».")
    st.stop()

# historique des cotes cible
tick = len(ss.history) and max(h["t"] for h in ss.history) + 1 or 0
if not ss.history or ss.get("last_snapshot") != json.dumps([m["odds"].get(target) for m in ss.matches]):
    for m in ss.matches:
        for i, name in enumerate(m["outcomes"]):
            if target in m["odds"]:
                ss.history.append({"t": tick, "key": f'{m["name"]} | {name}', "cote": m["odds"][target][i]})
    ss.last_snapshot = json.dumps([m["odds"].get(target) for m in ss.matches])

rows = [r for m in ss.matches for r in analyse(m, target, min_edge, kelly_frac, S["bankroll"])]
df = pd.DataFrame(rows)
tab1, tab2, tab3 = st.tabs(["Value bets", "Variation des cotes", "Mes paris"])

with tab1:
    if df.empty:
        st.warning(f"Pas de cotes pour « {target} ».")
    else:
        st.dataframe(df.drop(columns=["mid", "idx", "p"]).sort_values("Edge", ascending=False), use_container_width=True)
        best = df[df["Mise suggérée"] > 0].sort_values("Edge", ascending=False)
        for _, r in best.iterrows():
            col = st.columns([4, 1])
            col[0].write(f'**{r["Match"]}** — {r["Issue"]} @ {r[f"Cote {target}"]} (edge {r["Edge"]:.1%})')
            if col[1].button(f'J\'ai parié {r["Mise suggérée"]} € sur Unibet', key=f'b{r["mid"]}{r["idx"]}'):
                stake = min(r["Mise suggérée"], S["bankroll"] * MAX_STAKE_PCT)
                S["bankroll"] = round(S["bankroll"] - stake, 2)
                S["bets"].append({"match": r["Match"], "issue": r["Issue"], "odd": r[f"Cote {target}"],
                                  "stake": stake, "p": r["p"], "status": "open", "mid": r["mid"], "idx": int(r["idx"])})
                save_state(S)
                st.rerun()

with tab2:
    if ss.history:
        h = pd.DataFrame(ss.history)
        sel = st.multiselect("Issues", sorted(h["key"].unique()), default=sorted(h["key"].unique())[:3])
        if sel:
            st.line_chart(h[h["key"].isin(sel)].pivot_table(index="t", columns="key", values="cote"))

with tab3:
    for i, b in enumerate(S["bets"]):
        if b["status"] != "open":
            continue
        col = st.columns([4, 1, 1, 1])
        col[0].write(f'{b["match"]} — **{b["issue"]}** @ {b["odd"]} — mise {b["stake"]} €')
        if col[1].button("✅ Gagné", key=f"w{i}"):
            b["status"] = "won"
            S["bankroll"] = round(S["bankroll"] + b["stake"] * b["odd"], 2)
        elif col[2].button("❌ Perdu", key=f"l{i}"):
            b["status"] = "lost"
        else:
            continue
        S["peak"] = max(S["peak"], S["bankroll"])
        if S["bankroll"] <= 0:
            S["dead"] = True
        save_state(S)
        st.rerun()
    closed = [b for b in S["bets"] if b["status"] != "open"]
    if closed:
        st.dataframe(pd.DataFrame(closed).drop(columns=["mid", "idx"]), use_container_width=True)
