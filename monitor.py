"""Moniteur de cotes : compare Unibet au consensus du marché, apprend des résultats, alerte par mail/SMS/ntfy.
Ne place AUCUN pari : l'utilisateur joue manuellement sur Unibet.
Usage : python monitor.py          (boucle continue)
        python monitor.py once     (un seul cycle)
        python monitor.py once --dry   (n'envoie rien, affiche les alertes)
"""
import datetime as dt
import json
import math
import os
import smtplib
import sqlite3
import sys
import time
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

BASE = Path(__file__).parent
DB = BASE / "odds.db"
STATE = BASE / "state.json"
MODEL = BASE / "model.pkl"
API = "https://api.the-odds-api.com/v4"
MAX_STAKE_PCT = 0.06
MIN_STAKE = 0.10
DRY = "--dry" in sys.argv
REMAINING = None


def load_cfg():
    cfg = {
        "ODDS_API_KEY": "", "TARGET_BOOK": "Unibet (FR)",
        "SPORTS": "soccer_france_ligue_one",         "INTERVAL_MIN": "96",
                "ACTIVE_FROM": "0", "ACTIVE_TO": "24", "MIN_EDGE": "0.04", "MAX_EDGE": "0.25",
        "KELLY_FRACTION": "0.25", "RESERVE_REQUESTS": "6", "HORIZON_HOURS": "48",
        "SMTP_HOST": "smtp.gmail.com", "SMTP_USER": "", "SMTP_PASSWORD": "", "MAIL_TO": "",
        "FREE_USER": "", "FREE_PASS": "", "NTFY_TOPIC": "",
        "TELEGRAM_TOKEN": "", "TELEGRAM_CHAT": "",
    }
    f = BASE / "config.env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    for k in list(cfg) + ["BANKROLL", "RESPECT_HOURS"]:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


CFG = load_cfg()


# ---------- base de données ----------
def db():
    con = sqlite3.connect(DB)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS obs(ts REAL, event_id TEXT, sport TEXT, commence REAL, home TEXT, away TEXT,
        outcome TEXT, odd REAL, cons_p REAL, pin_p REAL, n INTEGER);
    CREATE TABLE IF NOT EXISTS results(event_id TEXT PRIMARY KEY, winner TEXT);
    CREATE TABLE IF NOT EXISTS alerts(event_id TEXT, outcome TEXT, ts REAL, PRIMARY KEY(event_id, outcome));
    CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
    CREATE TABLE IF NOT EXISTS bets(id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, outcome TEXT, label TEXT,
        match TEXT, odd REAL, stake REAL, commence REAL, status TEXT, ts REAL);
    CREATE INDEX IF NOT EXISTS ix_obs ON obs(event_id, outcome, ts);
    """)
    return con


def meta_get(con, k, default=None):
    r = con.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return r[0] if r else default


def meta_set(con, k, v):
    con.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(v)))


# ---------- API ----------
def api(path, **params):
    global REMAINING
    params["apiKey"] = CFG["ODDS_API_KEY"]
    r = requests.get(API + path, params=params, timeout=30)
    r.raise_for_status()
    rem = r.headers.get("x-requests-remaining")
    if rem is not None:
        REMAINING = int(float(rem))
    return r.json()


def sports_list():
    if CFG["SPORTS"].strip().lower() == "all":
        return [s["key"] for s in api("/sports") if s["active"] and not s["has_outrights"]]
    return [s.strip() for s in CFG["SPORTS"].split(",") if s.strip()]


# ---------- analyse ----------
def devig(prices):
    inv = [1 / p for p in prices]
    t = sum(inv)
    return [x / t for x in inv]


def analyse(ev):
    books = {}
    for bk in ev.get("bookmakers", []):
        m = next((x for x in bk["markets"] if x["key"] == "h2h"), None)
        if m:
            books[bk["title"]] = {o["name"]: o["price"] for o in m["outcomes"]}
    tgt = books.get(CFG["TARGET_BOOK"])
    if not tgt:
        return []
    names = sorted(tgt)
    others = {b: o for b, o in books.items() if not b.startswith("Unibet") and sorted(o) == names}
    if len(others) < 4:
        return []
    acc, wsum, pin = dict.fromkeys(names, 0.0), 0.0, None
    for b, o in others.items():
        p = dict(zip(names, devig([o[n] for n in names])))
        w = 3.0 if b == "Pinnacle" else 1.0
        for n in names:
            acc[n] += p[n] * w
        wsum += w
        if b == "Pinnacle":
            pin = p
    return [{"outcome": n, "odd": tgt[n], "cons_p": acc[n] / wsum,
             "pin_p": pin[n] if pin else acc[n] / wsum, "n": len(others)} for n in names]


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def features(cons_p, pin_p, odd, drift, n):
    return [logit(cons_p), logit(pin_p), math.log(odd), drift, n]


_model = None


def model_prob(f):
    global _model
    if _model is None and MODEL.exists():
        import pickle
        _model = pickle.loads(MODEL.read_bytes())
    if _model is None:
        return None
    return float(_model.predict_proba([f])[0][1])


# ---------- apprentissage ----------
def collect_results(con):
    if time.time() - float(meta_get(con, "scores_ts", 0)) < 48 * 3600:
        return
    now = time.time()
    rows = con.execute("""SELECT DISTINCT sport FROM obs WHERE commence < ? AND event_id NOT IN
        (SELECT event_id FROM results)""", (now - 3 * 3600,)).fetchall()
    for (sport,) in rows:
        if REMAINING is not None and REMAINING < int(CFG["RESERVE_REQUESTS"]) + 2:
            break
        try:
            for ev in api(f"/sports/{sport}/scores", daysFrom=3):
                if ev.get("completed") and ev.get("scores"):
                    sc = {s["name"]: float(s["score"]) for s in ev["scores"]}
                    top = max(sc.values())
                    winners = [n for n, v in sc.items() if v == top]
                    con.execute("INSERT OR REPLACE INTO results VALUES(?,?)",
                                (ev["id"], winners[0] if len(winners) == 1 else "Draw"))
        except Exception as e:
            print("scores:", e)
    meta_set(con, "scores_ts", now)
    con.commit()


def train(con):
    if time.time() - float(meta_get(con, "train_ts", 0)) < 24 * 3600:
        return
    try:
        import numpy as np
        from sklearn.metrics import log_loss
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return
    rows = con.execute("""
        SELECT o.event_id, o.outcome, o.odd, o.cons_p, o.pin_p, o.n, o.commence, r.winner,
               (SELECT odd FROM obs f WHERE f.event_id=o.event_id AND f.outcome=o.outcome ORDER BY ts LIMIT 1)
        FROM obs o JOIN results r ON r.event_id=o.event_id
        WHERE o.ts = (SELECT MAX(ts) FROM obs x WHERE x.event_id=o.event_id AND x.outcome=o.outcome AND x.ts < o.commence)
        ORDER BY o.commence""").fetchall()
    meta_set(con, "train_ts", time.time())
    con.commit()
    if len(rows) < 600:
        print(f"Apprentissage : {len(rows)} échantillons, 600 requis.")
        return
    X = np.array([features(r[3], r[4], r[2], math.log(r[2] / r[8]), r[5]) for r in rows])
    y = np.array([1 if r[1] == r[7] else 0 for r in rows])
    cut = int(len(rows) * 0.75)
    net = make_pipeline(StandardScaler(), MLPClassifier((32, 16), alpha=1e-2, early_stopping=True,
                                                         max_iter=500, random_state=0))
    net.fit(X[:cut], y[:cut])
    ll_net = log_loss(y[cut:], np.clip(net.predict_proba(X[cut:])[:, 1], 1e-4, 1 - 1e-4))
    ll_mkt = log_loss(y[cut:], np.clip([r[3] for r in rows[cut:]], 1e-4, 1 - 1e-4))
    print(f"Apprentissage : réseau {ll_net:.4f} vs marché {ll_mkt:.4f}")
    if ll_net < ll_mkt:
        import pickle
        net.fit(X, y)
        MODEL.write_bytes(pickle.dumps(net))
        global _model
        _model = None
        print("Modèle activé.")
    elif MODEL.exists():
        MODEL.unlink()
        _model = None


# ---------- notifications ----------
def tg(method, **data):
    r = requests.post(f"https://api.telegram.org/bot{CFG['TELEGRAM_TOKEN']}/{method}", json=data, timeout=30)
    r.raise_for_status()
    return r.json()["result"]


def tg_send(text, buttons=None):
    if DRY or not (CFG["TELEGRAM_TOKEN"] and CFG["TELEGRAM_CHAT"]):
        return False
    data = {"chat_id": CFG["TELEGRAM_CHAT"], "text": text}
    if buttons:
        data["reply_markup"] = {"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in buttons]]}
    tg("sendMessage", **data)
    return True


def notify(subject, body, buttons=None):
    if DRY:
        print(f"--- ALERTE (dry) ---\n{subject}\n{body}\n")
        return
    sent = tg_send(subject + "\n\n" + body, buttons)
    if CFG["SMTP_USER"] and CFG["MAIL_TO"]:
        m = EmailMessage()
        m["From"], m["To"], m["Subject"] = CFG["SMTP_USER"], CFG["MAIL_TO"], subject
        m.set_content(body)
        with smtplib.SMTP_SSL(CFG["SMTP_HOST"], 465) as s:
            s.login(CFG["SMTP_USER"], CFG["SMTP_PASSWORD"])
            s.send_message(m)
        sent = True
    if CFG["FREE_USER"]:
        requests.get("https://smsapi.free-mobile.fr/sendmsg", params={
            "user": CFG["FREE_USER"], "pass": CFG["FREE_PASS"], "msg": subject + "\n" + body}, timeout=15)
        sent = True
    if CFG["NTFY_TOPIC"]:
        requests.post(f"https://ntfy.sh/{CFG['NTFY_TOPIC']}", data=body.encode("utf-8"),
                      headers={"Title": subject.encode("utf-8")}, timeout=15)
        sent = True
    if not sent:
        print("Aucun canal de notification configuré.\n", subject, body)


def bankroll(con):
    b = float(meta_get(con, "bankroll", CFG.get("BANKROLL") or 10.0))
    return b, b <= 0


def set_bankroll(con, b):
    meta_set(con, "bankroll", round(b, 2))
    if b <= 0 and meta_get(con, "dead_notified") != "1":
        meta_set(con, "dead_notified", 1)
        tg_send("💀 Ton solde est à 0 : tu es mort. Fin de la partie. Envoie « solde 10 » pour recommencer.")
    elif b > 0:
        meta_set(con, "dead_notified", 0)


def poll_telegram(con):
    """Lit tes réponses : boutons « placé / ignoré » et commandes texte."""
    if not (CFG["TELEGRAM_TOKEN"] and CFG["TELEGRAM_CHAT"]) or DRY:
        return
    offset = int(meta_get(con, "tg_offset", 0))
    try:
        updates = tg("getUpdates", offset=offset, timeout=0, allowed_updates=["message", "callback_query"])
    except Exception as e:
        print("telegram:", e)
        return
    for u in updates:
        meta_set(con, "tg_offset", u["update_id"] + 1)
        cb, msg = u.get("callback_query"), u.get("message")
        chat = str((cb["message"]["chat"]["id"]) if cb else (msg or {}).get("chat", {}).get("id", ""))
        if chat != str(CFG["TELEGRAM_CHAT"]):
            continue
        if cb:
            act, bid = cb["data"].split(":")
            b = con.execute("SELECT id, label, odd, stake, status FROM bets WHERE id=?", (int(bid),)).fetchone()
            if b and b[4] == "proposed":
                if act == "p":
                    con.execute("UPDATE bets SET status='placed' WHERE id=?", (b[0],))
                    set_bankroll(con, bankroll(con)[0] - b[3])
                    reply = f"✅ Pari enregistré : {b[1]} @ {b[2]} — {b[3]} €. Solde : {bankroll(con)[0]:.2f} €"
                else:
                    con.execute("UPDATE bets SET status='ignored' WHERE id=?", (b[0],))
                    reply = "❌ Pari ignoré."
            else:
                reply = "Ce pari est déjà traité."
            try:
                tg("answerCallbackQuery", callback_query_id=cb["id"])
            except Exception:
                pass
            tg_send(reply)
            continue
        text = (msg or {}).get("text", "").strip().lower()
        if text.startswith("solde"):
            try:
                set_bankroll(con, float(text.split()[1].replace(",", ".")))
            except (IndexError, ValueError):
                pass
            tg_send(f"Solde : {bankroll(con)[0]:.2f} €")
        elif text.startswith("ok"):
            b = con.execute("SELECT id, label, odd, stake FROM bets WHERE status='proposed' ORDER BY id DESC LIMIT 1").fetchone()
            if not b:
                tg_send("Aucun pari en attente.")
                continue
            odd, stake = b[2], b[3]
            for tok in text.split()[1:]:
                try:
                    if tok.startswith("@"):
                        odd = float(tok[1:].replace(",", "."))
                    else:
                        stake = float(tok.replace(",", "."))
                except ValueError:
                    pass
            stake = min(stake, bankroll(con)[0] * MAX_STAKE_PCT + 1e-9)
            con.execute("UPDATE bets SET status='placed', odd=?, stake=? WHERE id=?", (odd, round(stake, 2), b[0]))
            set_bankroll(con, bankroll(con)[0] - round(stake, 2))
            tg_send(f"✅ Enregistré : {b[1]} @ {odd} — {round(stake, 2)} € (plafonné à 6 %). Solde : {bankroll(con)[0]:.2f} €")
        elif text in ("/start", "aide", "help"):
            tg_send("Boutons sous chaque alerte, ou : « ok 0.5 @2.1 » (mise/cote réelles), « solde 12.5 », « paris ».")
        elif text == "paris":
            rows = con.execute("SELECT match, label, odd, stake, status FROM bets WHERE status IN ('placed','won','lost') "
                               "ORDER BY id DESC LIMIT 10").fetchall()
            tg_send("\n".join(f"{r[0]} | {r[1]} @ {r[2]} | {r[3]} € | {r[4]}" for r in rows) or "Aucun pari.")
    con.commit()


def settle_bets(con):
    """Règle automatiquement les paris placés dont le résultat est connu."""
    rows = con.execute("""SELECT b.id, b.label, b.outcome, b.odd, b.stake, b.match, r.winner FROM bets b
        JOIN results r ON r.event_id=b.event_id WHERE b.status='placed'""").fetchall()
    for bid, label, outcome, odd, stake, match, winner in rows:
        won = outcome == winner
        con.execute("UPDATE bets SET status=? WHERE id=?", ("won" if won else "lost", bid))
        if won:
            set_bankroll(con, bankroll(con)[0] + stake * odd)
        else:
            set_bankroll(con, bankroll(con)[0])
        tg_send(f"{'🏆 GAGNÉ' if won else '😞 PERDU'} : {match} — {label} @ {odd}\n"
                f"{'+' + str(round(stake * (odd - 1), 2)) if won else '-' + str(stake)} € | Solde : {bankroll(con)[0]:.2f} €")
    con.commit()


# ---------- cycle ----------
def cycle(con):
    now = time.time()
    poll_telegram(con)
    settle_bets(con)
    if bankroll(con)[1]:
        print("💀 Bankroll à 0 : plus d'alertes.")
        return
    horizon = now + float(CFG["HORIZON_HOURS"]) * 3600
    min_edge, max_edge = float(CFG["MIN_EDGE"]), float(CFG["MAX_EDGE"])
    for sport in sports_list():
        if REMAINING is not None and REMAINING < int(CFG["RESERVE_REQUESTS"]):
            print(f"Quota presque épuisé ({REMAINING}), pause.")
            return
        try:
            events = api(f"/sports/{sport}/odds", regions="eu", markets="h2h", oddsFormat="decimal")
        except Exception as e:
            print(sport, "erreur:", e)
            continue
        for ev in events:
            commence = dt.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).timestamp()
            if not (now < commence < horizon):
                continue
            for a in analyse(ev):
                first = con.execute("SELECT odd FROM obs WHERE event_id=? AND outcome=? ORDER BY ts LIMIT 1",
                                    (ev["id"], a["outcome"])).fetchone()
                drift = math.log(a["odd"] / first[0]) if first else 0.0
                con.execute("INSERT INTO obs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (now, ev["id"], sport, commence, ev["home_team"], ev["away_team"],
                             a["outcome"], a["odd"], a["cons_p"], a["pin_p"], a["n"]))
                p = a["cons_p"]
                mp = model_prob(features(p, a["pin_p"], a["odd"], drift, a["n"]))
                if mp is not None:
                    if mp * a["odd"] - 1 < 0:
                        continue
                    p = (p + mp) / 2
                edge = p * a["odd"] - 1
                if not (min_edge <= edge <= max_edge) or not (1.25 <= a["odd"] <= 8):
                    continue
                if con.execute("SELECT 1 FROM alerts WHERE event_id=? AND outcome=?",
                               (ev["id"], a["outcome"])).fetchone():
                    continue
                bank, dead = bankroll(con)
                if dead:
                    return
                kelly = edge / (a["odd"] - 1) * float(CFG["KELLY_FRACTION"])
                stake = round(min(bank * kelly, bank * MAX_STAKE_PCT), 2)
                if stake < MIN_STAKE:
                    continue
                when = dt.datetime.fromtimestamp(commence, ZoneInfo("Europe/Paris")).strftime("%d/%m %H:%M")
                label = a["outcome"] if a["outcome"] != "Draw" else "Match nul"
                bid = con.execute("INSERT INTO bets(event_id,outcome,label,match,odd,stake,commence,status,ts) "
                                  "VALUES(?,?,?,?,?,?,?,'proposed',?)",
                                  (ev["id"], a["outcome"], label, f"{ev['home_team']} - {ev['away_team']}",
                                   a["odd"], stake, commence, now)).lastrowid
                notify(f"Pari value : {label} @ {a['odd']}",
                       f"{ev['home_team']} - {ev['away_team']} ({when})\n"
                       f"Pari : {label} à la cote {a['odd']} (Unibet)\n"
                       f"Mise conseillée : {stake} € (solde {bank:.2f} €, max 6 % = {bank*0.06:.2f} €)\n"
                       f"Proba estimée {p:.1%} | edge {edge:+.1%} | dérive de cote {drift:+.1%}\n\n"
                       f"Pour jouer : ouvre l'appli Unibet > Sport > cherche « {ev['home_team']} » > "
                       f"résultat du match > « {label} » > mise {stake} € > valider.\n"
                       f"Ensuite confirme ci-dessous (ou réponds « ok 0.5 @2.1 » avec ta mise/cote réelles).",
                       [("✅ Pari placé", f"p:{bid}"), ("❌ Ignoré", f"i:{bid}")])
                con.execute("INSERT INTO alerts VALUES(?,?,?)", (ev["id"], a["outcome"], now))
        con.commit()
    collect_results(con)
    settle_bets(con)
    train(con)
    con.commit()


def main():
    if "chatid" in sys.argv:
        for u in tg("getUpdates"):
            m = u.get("message") or {}
            if m:
                print("chat_id =", m["chat"]["id"], "| de", m["chat"].get("first_name"))
        return
    if not CFG["ODDS_API_KEY"]:
        sys.exit("Renseigne ODDS_API_KEY dans config.env (voir config.example.env).")
    con = db()
    once = "once" in sys.argv
    while True:
        h = dt.datetime.now(ZoneInfo("Europe/Paris")).hour
        in_hours = int(CFG["ACTIVE_FROM"]) <= h < int(CFG["ACTIVE_TO"])
        if (once and not CFG.get("RESPECT_HOURS")) or in_hours:
            cycle(con)
            print(dt.datetime.now().strftime("%H:%M"), "cycle OK, requêtes restantes:", REMAINING)
        if once:
            break
        con.commit()
        time.sleep(float(CFG["INTERVAL_MIN"]) * 60)


if __name__ == "__main__":
    main()
