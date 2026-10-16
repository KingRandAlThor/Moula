# Moula

Bot Python qui surveille les cotes de football, repère les écarts entre Unibet et le consensus du marché, et envoie des alertes sur Telegram. Il tourne gratuitement sur GitHub Actions.

**Le bot ne place aucun pari.** Il n'a aucun accès à Unibet : les mises sont jouées à la main, et la bankroll suivie par le bot est simulée (10 € au départ).

## Fonctionnement

- **Cotes** : [The Odds API](https://the-odds-api.com) (quota gratuit de 500 requêtes par mois, réparti entre 10 championnats).
- **Contexte** : météo, avantage à domicile, arbitre et blessures via API-Football (plan gratuit, 100 requêtes par jour).
- **Modèle** : un classifieur scikit-learn réentraîné sur les résultats observés (`model.pkl`).
- **Alertes** : message Telegram avec boutons de confirmation ; la bankroll est mise à jour selon ta réponse.
- **Règles de jeu** :
  - mise maximale de 6 % de la bankroll ;
  - au moins un pari confirmé par jour entre 12h et 14h (heure de Paris), sinon le bot « meurt » ;
  - il meurt aussi si la bankroll tombe à 0 €.

## Commandes Telegram

| Message | Effet |
|---|---|
| `solde` | Affiche la bankroll ; `solde 12.5` la corrige |
| `mission` | État de la mission du jour |
| `mes paris` | Paris en cours |
| `statut`, `radar` | État du bot, opportunités repérées |
| `sur quoi parier` | Meilleures opportunités actuelles |
| `placé`, `ok 0.5 @2.1` | Confirme un pari (mise et cote facultatives) |
| `aide` | Liste des commandes |

## Installation

1. Crée un bot avec @BotFather, puis récupère ton identifiant de chat avec `python monitor.py chatid`.
2. Obtiens une clé The Odds API et, si tu veux le contexte, une clé API-Football.
3. Copie `config.example.env` en `config.env` et remplis-le (fichier ignoré par git). En local :

```bash
pip install -r requirements.txt
python monitor.py once --dry   # un cycle sans rien envoyer
python monitor.py daemon       # écoute continue
```

## Hébergement sur GitHub Actions

Le workflow `.github/workflows/monitor.yml` lance `python monitor.py daemon` toutes les 6 heures. Chaque exécution dure environ 5 h 45 et interroge Telegram toutes les 30 minutes.

Ajoute dans **Settings → Secrets and variables → Actions** les secrets `ODDS_API_KEY`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT`, `AF_KEY` et les variables décrites dans `config.example.env`. L'état (`odds.db`, `model.pkl`) est conservé via le cache Actions, enregistré en fin d'exécution : un plantage peut faire perdre quelques heures de données.

Pour relancer un bot mort : lancer le workflow à la main avec l'option `revive`.

## Ce que montrent les backtests

Les scripts d'analyse tournent sur 20 501 matchs (8 championnats, 8 saisons, données [football-data.co.uk](https://www.football-data.co.uk)), avec les deux dernières saisons gardées comme test.

- Le marché est efficace : le modèle fait une log-loss de 0,9700 contre 0,9645 pour le marché.
- Aucun facteur de calendrier, favori à l'extérieur, nul à cotes 3-5 ou stratégie issue d'une recherche en grille ne tient hors échantillon.
- Les gros ROI apparus au début venaient d'un biais : l'utilisation de cotes de clôture Pinnacle connues après coup.
- Seul le Over/Under 2,5 reste à surveiller (+3,9 %), mais sans significativité statistique (p = 0,28).

Conclusion : aucun avantage durable n'a été trouvé. Le projet sert d'outil de suivi et d'expérimentation, pas de source de revenus.

## Scripts

`backtest.py`, `pretrain.py`, `pretrain2.py`, `analyze.py`, `own_model.py`, `grid_search.py` : analyses sur l'historique placé dans `hist/` (non versionné, résultats dans `out/`).

## Jeu responsable

Les paris comportent un risque de perte. Joue uniquement ce que tu peux perdre ; aide : joueurs-info-service.fr (09 74 75 13 13).
