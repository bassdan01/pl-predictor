#!/usr/bin/env python3
"""Premier League score predictor (all data free, no keys).

Sources: the FPL API (fixtures, results, xG, injuries, penalties) and the free
football-data.co.uk fixtures file (bookmaker odds, used quietly as supporting
data and never shown). Writes docs/data.json for the page and keeps
docs/history.json so past predictions can be checked against real results.
"""
import csv
import io
import json
import math
import os
import re
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

BASE = "https://fantasy.premierleague.com/api"
ODDS_URL = "https://www.football-data.co.uk/fixtures.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "data.json")
HIST = os.path.join(HERE, "docs", "history.json")

# ---- Tunable settings -------------------------------------------------------
FORM_GAMES = 6        # recent games that count as "form" (fewer early in the season)
DECAY = 0.8           # each older game counts 80% as much as the one after it
FORM_WEIGHT = 0.5     # 0 = season only, 1 = recent form only
FORM_PRIOR = 1.5      # pulls form toward the season rating when the sample is small
XG_WEIGHT = 0.6       # within form: share given to xG vs actual goals
PRIOR_GAMES = 4       # pulls season numbers toward FPL's own team ratings early on
ODDS_WEIGHT = 0.5     # how much the (hidden) market view counts against the model
INJURY_LIMIT = 4      # injuries only matter once a team has MORE than this many
INJURY_STEP = 0.03    # rating drop per extra injured regular
INJURY_CAP = 0.15
DC_RHO = -0.10        # Dixon-Coles low-score correction (more realistic 0-0 / 1-1)
MAX_GOALS = 8
TOP_SCORERS = 3
TOP_ASSISTS = 3
# -----------------------------------------------------------------------------

ALIASES = {
    "manunited": "manutd", "manchesterunited": "manutd", "manchestercity": "mancity",
    "tottenham": "spurs", "tottenhamhotspur": "spurs", "nottinghamforest": "nottmforest",
    "wolverhamptonwanderers": "wolves", "leedsunited": "leeds", "newcastleunited": "newcastle",
    "westhamunited": "westham", "brightonandhovealbion": "brighton", "afcbournemouth": "bournemouth",
}


def canon(name):
    key = re.sub(r"[^a-z]", "", name.lower())
    return ALIASES.get(key, key)


def get_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (pl-predictor)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8-sig", errors="replace")


def get(path):
    return json.loads(get_text(f"{BASE}/{path}"))


def load_odds(fetch_text=get_text):
    """Return {(home, away): [p_home, p_draw, p_away]} with the bookmaker margin removed."""
    rows = csv.DictReader(io.StringIO(fetch_text(ODDS_URL)))
    out = {}
    for r in rows:
        if (r.get("Div") or "").strip() != "E0":
            continue
        for cols in (("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA")):
            try:
                o = [float(r[c]) for c in cols]
            except (KeyError, TypeError, ValueError):
                continue
            if min(o) > 1.0:
                inv = [1 / x for x in o]
                s = sum(inv)
                out[(canon(r["HomeTeam"]), canon(r["AwayTeam"]))] = [x / s for x in inv]
                break
    return out


def pois(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)


def tau(i, j, lh, la):
    if i == 0 and j == 0:
        return 1 - lh * la * DC_RHO
    if i == 0 and j == 1:
        return 1 + lh * DC_RHO
    if i == 1 and j == 0:
        return 1 + la * DC_RHO
    if i == 1 and j == 1:
        return 1 - DC_RHO
    return 1.0


def score_grid(lh, la):
    ph = pd = pa = 0.0
    best, best_p = (0, 0), -1.0
    for i in range(MAX_GOALS + 1):
        pi = pois(i, lh)
        for j in range(MAX_GOALS + 1):
            p = pi * pois(j, la) * tau(i, j, lh, la)
            if i > j:
                ph += p
            elif i == j:
                pd += p
            else:
                pa += p
            if p > best_p:
                best, best_p = (i, j), p
    total = ph + pd + pa
    return ph / total, pd / total, pa / total, best


def fit_lambdas(lh, la, target):
    """Keep total goals, shift the split until P(home win) - P(away win) hits target."""
    total = lh + la
    lo, hi = 0.05, total - 0.05
    for _ in range(40):
        mid = (lo + hi) / 2
        ph, _, pa, _ = score_grid(mid, total - mid)
        if ph - pa < target:
            lo = mid
        else:
            hi = mid
    m = (lo + hi) / 2
    return m, total - m


def outcome(h, a):
    return "H" if h > a else "D" if h == a else "A"


def build(fetch=get, history=None, now=None, odds=None):
    boot = fetch("bootstrap-static/")
    fixtures = fetch("fixtures/")

    teams = {t["id"]: t["name"] for t in boot["teams"]}
    events = boot["events"]
    gw = next((e for e in events if e.get("is_next")), None) or \
        next((e for e in events if not e["finished"]), events[-1])

    def is_played(f):
        return (f.get("finished") or f.get("finished_provisional")) and \
            f.get("team_h_score") is not None and f.get("team_a_score") is not None

    played = sorted((f for f in fixtures if is_played(f)),
                    key=lambda f: f.get("kickoff_time") or "")

    n_fix = defaultdict(int)  # fixtures per team per gameweek (double gameweeks)
    for f in fixtures:
        if f.get("event"):
            n_fix[(f["event"], f["team_h"])] += 1
            n_fix[(f["event"], f["team_a"])] += 1

    results = defaultdict(list)
    home_goals = away_goals = 0
    for f in played:
        h, a, hs, as_ = f["team_h"], f["team_a"], f["team_h_score"], f["team_a_score"]
        home_goals += hs
        away_goals += as_
        results[h].append({"gf": hs, "ga": as_, "gw": f["event"], "opp": a})
        results[a].append({"gf": as_, "ga": hs, "gw": f["event"], "opp": h})

    n_games = len(played)
    mu_h = home_goals / n_games if n_games else 1.5
    mu_a = away_goals / n_games if n_games else 1.2
    mu = (mu_h + mu_a) / 2

    # FPL's own team ratings: the starting point before there is much data
    raw_a = {t["id"]: (t.get("strength_attack_home", 1000) + t.get("strength_attack_away", 1000)) / 2
             for t in boot["teams"]}
    raw_d = {t["id"]: (t.get("strength_defence_home", 1000) + t.get("strength_defence_away", 1000)) / 2
             for t in boot["teams"]}
    if raw_a and all(v > 0 for v in raw_a.values()) and all(v > 0 for v in raw_d.values()):
        avg_a = sum(raw_a.values()) / len(raw_a)
        avg_d = sum(raw_d.values()) / len(raw_d)
        prior_att = {t: (raw_a[t] / avg_a) ** 1.5 for t in teams}
        prior_dfn = {t: (avg_d / raw_d[t]) ** 1.5 for t in teams}
    else:  # FPL ratings missing or zero: start every team at league average
        prior_att = {t: 1.0 for t in teams}
        prior_dfn = {t: 1.0 for t in teams}

    # ---- xG from recent gameweeks (FPL live data) ----------------------------
    form_games = {t: results[t][-FORM_GAMES:] for t in teams}
    form_gws = {t: {g["gw"] for g in form_games[t]} for t in teams}
    needed = sorted(set().union(*form_gws.values())) if form_gws else []

    el_team = {e["id"]: e["team"] for e in boot["elements"]}
    team_xg = defaultdict(float)
    p_xg, p_xa, p_min = defaultdict(float), defaultdict(float), defaultdict(float)
    for g in needed:
        try:
            live = fetch(f"event/{g}/live/")
        except Exception as exc:
            print(f"warning: no live data for GW{g}: {exc}")
            continue
        for el in live.get("elements", []):
            t = el_team.get(el["id"])
            if t is None or g not in form_gws[t]:
                continue
            st = el.get("stats", {})
            xg = float(st.get("expected_goals") or 0)
            team_xg[(g, t)] += xg
            p_xg[el["id"]] += xg
            p_xa[el["id"]] += float(st.get("expected_assists") or 0)
            p_min[el["id"]] += float(st.get("minutes") or 0)
    use_xg = sum(team_xg.values()) > 0

    # ---- injuries: only matter beyond INJURY_LIMIT key players ---------------
    injured = defaultdict(int)
    for e in boot["elements"]:
        if e["status"] == "i" and float(e.get("minutes") or 0) >= 270:
            injured[e["team"]] += 1

    # ---- attack / defence ratings --------------------------------------------
    att0, dfn0 = {}, {}
    for t in teams:
        res = results[t]
        n = len(res)
        att0[t] = (sum(r["gf"] for r in res) / mu + PRIOR_GAMES * prior_att[t]) / (n + PRIOR_GAMES)
        dfn0[t] = (sum(r["ga"] for r in res) / mu + PRIOR_GAMES * prior_dfn[t]) / (n + PRIOR_GAMES)

    att, dfn, form_info = {}, {}, {}
    for t in teams:
        fg = form_games[t]
        num_a = num_d = wsum = 0.0
        xf, xa = [], []
        for idx, r in enumerate(reversed(fg)):          # idx 0 = most recent game
            w = DECAY ** idx
            k = max(n_fix[(r["gw"], t)], 1)
            ko = max(n_fix[(r["gw"], r["opp"])], 1)
            gxf = team_xg[(r["gw"], t)] / k
            gxa = team_xg[(r["gw"], r["opp"])] / ko
            if use_xg:
                bf = XG_WEIGHT * gxf + (1 - XG_WEIGHT) * r["gf"]
                ba = XG_WEIGHT * gxa + (1 - XG_WEIGHT) * r["ga"]
                xf.append(gxf)
                xa.append(gxa)
            else:
                bf, ba = r["gf"], r["ga"]
            # judge each game against how strong that opponent is
            num_a += w * bf / (mu * dfn0[r["opp"]])
            num_d += w * ba / (mu * att0[r["opp"]])
            wsum += w
        form_att = (num_a + FORM_PRIOR * att0[t]) / (wsum + FORM_PRIOR)
        form_dfn = (num_d + FORM_PRIOR * dfn0[t]) / (wsum + FORM_PRIOR)
        att[t] = FORM_WEIGHT * form_att + (1 - FORM_WEIGHT) * att0[t]
        dfn[t] = FORM_WEIGHT * form_dfn + (1 - FORM_WEIGHT) * dfn0[t]

        extra = injured[t] - INJURY_LIMIT
        if extra > 0:
            adj = min(INJURY_CAP, INJURY_STEP * extra)
            att[t] *= 1 - adj
            dfn[t] *= 1 + adj

        form_info[t] = {
            "form": "".join("W" if r["gf"] > r["ga"] else "D" if r["gf"] == r["ga"] else "L"
                            for r in fg),
            "xg_for": round(sum(xf) / len(xf), 2) if xf else None,
            "xg_against": round(sum(xa) / len(xa), 2) if xa else None,
            "goals_for": sum(r["gf"] for r in fg),
            "goals_against": sum(r["ga"] for r in fg),
        }

    # ---- likely scorers and assisters ------------------------------------------
    by_team = defaultdict(list)
    for e in boot["elements"]:
        if e["element_type"] == 1 or e["status"] not in ("a", "d"):
            continue
        if float(e.get("minutes") or 0) < 90:
            continue
        by_team[e["team"]].append(e)

    def rate(e, season_key, recent_total):
        season = float(e.get(season_key) or 0)
        mins = p_min.get(e["id"], 0)
        if mins >= 60:
            return 0.5 * (recent_total[e["id"]] / mins * 90) + 0.5 * season
        return season * 0.8

    def likely(team, opp):
        sc, asst = [], []
        for e in by_team[team]:
            ps = 1 - math.exp(-rate(e, "expected_goals_per_90", p_xg) * 0.85 * dfn[opp])
            pa_ = 1 - math.exp(-rate(e, "expected_assists_per_90", p_xa) * 0.85 * dfn[opp])
            sc.append({"name": e["web_name"], "prob": round(ps, 3), "doubt": e["status"] == "d",
                       "pen": e.get("penalties_order") == 1})
            asst.append({"name": e["web_name"], "prob": round(pa_, 3), "doubt": e["status"] == "d"})
        sc.sort(key=lambda x: x["prob"], reverse=True)
        asst.sort(key=lambda x: x["prob"], reverse=True)
        sc = sc[:TOP_SCORERS]
        for i, s in enumerate(sc):
            s["fav"] = i == 0
        return sc, asst[:TOP_ASSISTS]

    # ---- predict this gameweek -------------------------------------------------
    gw_fix = sorted((f for f in fixtures if f.get("event") == gw["id"]),
                    key=lambda f: f.get("kickoff_time") or "")
    out_fix, ranking, matched = [], [], 0
    for f in gw_fix:
        h, a = f["team_h"], f["team_a"]
        lh = mu_h * att[h] * dfn[a]
        la = mu_a * att[a] * dfn[h]
        ph, pd_, pa, (bh, ba) = score_grid(lh, la)

        mkt = odds.get((canon(teams[h]), canon(teams[a]))) if odds else None
        if mkt:  # quietly blend in the market view; odds themselves are never output
            target = (1 - ODDS_WEIGHT) * (ph - pa) + ODDS_WEIGHT * (mkt[0] - mkt[2])
            lh, la = fit_lambdas(lh, la, target)
            ph, pd_, pa, (bh, ba) = score_grid(lh, la)
            matched += 1

        started = f.get("team_h_score") is not None
        sh, ah = likely(h, a)
        sa, aa = likely(a, h)
        out_fix.append({
            "id": f["id"], "gw": gw["id"],
            "kickoff": f.get("kickoff_time"),
            "home": teams[h], "away": teams[a],
            "xg_home": round(lh, 2), "xg_away": round(la, 2),
            "p_home": round(ph, 3), "p_draw": round(pd_, 3), "p_away": round(pa, 3),
            "pred_home": bh, "pred_away": ba,
            "actual": [f["team_h_score"], f["team_a_score"]] if started else None,
            "finished": bool(f.get("finished") or f.get("finished_provisional")),
            "form_home": form_info[h], "form_away": form_info[a],
            "injured_home": injured[h] if injured[h] > INJURY_LIMIT else 0,
            "injured_away": injured[a] if injured[a] > INJURY_LIMIT else 0,
            "scorers_home": sh, "scorers_away": sa,
            "assists_home": ah, "assists_away": aa,
        })
        ranking.append({"team": teams[h], "opp": teams[a], "venue": "H", "win": round(ph, 3)})
        ranking.append({"team": teams[a], "opp": teams[h], "venue": "A", "win": round(pa, 3)})
    ranking.sort(key=lambda r: r["win"], reverse=True)
    print(f"market data matched {matched} of {len(gw_fix)} fixtures")

    # ---- track how well past predictions did -----------------------------------
    hist = history if history is not None else {}
    now = now or datetime.now(timezone.utc)
    names = {e["id"]: e["web_name"] for e in boot["elements"]}

    def kicked_off(iso):
        return bool(iso) and datetime.fromisoformat(iso.replace("Z", "+00:00")) <= now

    for o in out_fix:                      # saved only before kick-off, then frozen
        if kicked_off(o["kickoff"]):
            continue
        hist[str(o["id"])] = {
            "gw": o["gw"], "kickoff": o["kickoff"],
            "home": o["home"], "away": o["away"],
            "pred": [o["pred_home"], o["pred_away"]],
            "p": [o["p_home"], o["p_draw"], o["p_away"]],
            "scorers_home": [s["name"] for s in o["scorers_home"]],
            "scorers_away": [s["name"] for s in o["scorers_away"]],
            "actual": None,
        }

    by_id = {str(f["id"]): f for f in fixtures}
    for fid, rec in hist.items():
        f = by_id.get(fid)
        if rec.get("actual") or not f or not is_played(f):
            continue
        goals = {"h": [], "a": []}
        for s in f.get("stats", []):
            if s.get("identifier") == "goals_scored":
                for side in ("h", "a"):
                    for x in s.get(side, []):
                        goals[side].append({"name": names.get(x["element"], "?"), "n": x["value"]})
        rec["actual"] = {"score": [f["team_h_score"], f["team_a_score"]],
                         "scorers_home": goals["h"], "scorers_away": goals["a"]}

    done = sorted((r for r in hist.values() if r.get("actual")),
                  key=lambda r: r["kickoff"] or "", reverse=True)
    right = exact = named = hit_total = 0
    res_matches = []
    for r in done:
        a = r["actual"]
        pick = ["H", "D", "A"][r["p"].index(max(r["p"]))]
        ok = pick == outcome(*a["score"])
        ex = r["pred"] == a["score"]
        ph_ = [{"name": n, "hit": n in {x["name"] for x in a["scorers_home"]}} for n in r["scorers_home"]]
        pa_ = [{"name": n, "hit": n in {x["name"] for x in a["scorers_away"]}} for n in r["scorers_away"]]
        right += ok
        exact += ex
        named += len(ph_) + len(pa_)
        hit_total += sum(x["hit"] for x in ph_ + pa_)
        res_matches.append({
            "gw": r["gw"], "kickoff": r["kickoff"], "home": r["home"], "away": r["away"],
            "pred": r["pred"], "actual": a["score"], "result_ok": ok, "exact": ex,
            "scored_home": a["scorers_home"], "scored_away": a["scorers_away"],
            "pred_home": ph_, "pred_away": pa_,
        })
    results_out = {
        "summary": {"matches": len(done), "result_right": right, "exact": exact,
                    "scorers_named": named, "scorers_hit": hit_total},
        "matches": res_matches,
    }

    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gameweek": gw["id"],
        "gameweek_name": gw["name"],
        "deadline": gw.get("deadline_time"),
        "uses_xg": use_xg,
        "games_played": n_games,
        "fixtures": out_fix,
        "ranking": ranking,
        "results": results_out,
    }, hist


def main():
    history = {}
    if os.path.exists(HIST):
        with open(HIST) as fh:
            history = json.load(fh)
    try:
        odds = load_odds()
        print(f"market data loaded for {len(odds)} upcoming matches")
    except Exception as exc:  # odds are optional; carry on without them
        print(f"note: market data unavailable ({exc}); using model only")
        odds = None
    data, history = build(history=history, odds=odds)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=1)
    with open(HIST, "w") as fh:
        json.dump(history, fh, indent=1)
    print(f"Wrote {OUT}: {data['gameweek_name']}, {len(data['fixtures'])} fixtures, "
          f"xG {'on' if data['uses_xg'] else 'off'}")


if __name__ == "__main__":
    main()
                     
