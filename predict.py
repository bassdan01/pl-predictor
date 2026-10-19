#!/usr/bin/env python3
"""Premier League score predictor.

Uses only the free FPL API (no key needed). Run it and it writes docs/data.json
for the current FPL gameweek: predicted scorelines, win/draw/loss chances,
a ranking of teams by win chance, and likely scorers from recent xG.
"""
import json
import math
import os
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

BASE = "https://fantasy.premierleague.com/api"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "data.json")

# ---- Tunable settings -------------------------------------------------------
FORM_GAMES = 3        # how many recent games count as "form"
FORM_WEIGHT = 0.5     # 0 = season only, 1 = last-3 form only
XG_WEIGHT = 0.6       # within form: share given to xG vs actual goals
PRIOR_GAMES = 4       # pulls season numbers toward league average early on
FORM_PRIOR = 1        # same idea for the (small) form sample
MAX_GOALS = 8
TOP_SCORERS = 3       # likely scorers shown per team
# -----------------------------------------------------------------------------


def get(path):
    req = urllib.request.Request(
        f"{BASE}/{path}", headers={"User-Agent": "Mozilla/5.0 (pl-predictor)"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def pois(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)


def score_grid(lh, la):
    ph = pd = pa = 0.0
    best, best_p = (0, 0), -1.0
    for i in range(MAX_GOALS + 1):
        pi = pois(i, lh)
        for j in range(MAX_GOALS + 1):
            p = pi * pois(j, la)
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


def build(fetch=get):
    boot = fetch("bootstrap-static/")
    fixtures = fetch("fixtures/")

    teams = {t["id"]: t["name"] for t in boot["teams"]}
    events = boot["events"]
    gw = next((e for e in events if not e["finished"]), events[-1])

    def is_played(f):
        return (f.get("finished") or f.get("finished_provisional")) and \
            f.get("team_h_score") is not None and f.get("team_a_score") is not None

    played = sorted((f for f in fixtures if is_played(f)),
                    key=lambda f: f.get("kickoff_time") or "")

    # fixtures per team per gameweek (handles double gameweeks)
    n_fix = defaultdict(int)
    for f in fixtures:
        if f.get("event"):
            n_fix[(f["event"], f["team_h"])] += 1
            n_fix[(f["event"], f["team_a"])] += 1

    # results per team, oldest first
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

    # ---- xG from the last few gameweeks (FPL live data) ----------------------
    form_games = {t: results[t][-FORM_GAMES:] for t in teams}
    form_gws = {t: {g["gw"] for g in form_games[t]} for t in teams}
    needed = sorted(set().union(*form_gws.values())) if form_gws else []

    el_team = {e["id"]: e["team"] for e in boot["elements"]}
    team_xg = defaultdict(float)             # (gw, team) -> xG
    p_xg = defaultdict(float)                # player -> recent xG
    p_min = defaultdict(float)               # player -> recent minutes
    for g in needed:
        try:
            live = fetch(f"event/{g}/live/")
        except Exception as exc:  # keep going without xG for this week
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
            p_min[el["id"]] += float(st.get("minutes") or 0)
    use_xg = sum(team_xg.values()) > 0

    # ---- attack / defence strength ratings -----------------------------------
    att, dfn, form_info = {}, {}, {}
    for t in teams:
        res = results[t]
        n = len(res)
        s_for = (sum(r["gf"] for r in res) + PRIOR_GAMES * mu) / (n + PRIOR_GAMES)
        s_ag = (sum(r["ga"] for r in res) + PRIOR_GAMES * mu) / (n + PRIOR_GAMES)

        fg = form_games[t]
        f_for, f_ag, xf, xa = [], [], [], []
        for r in fg:
            k = max(n_fix[(r["gw"], t)], 1)
            ko = max(n_fix[(r["gw"], r["opp"])], 1)
            gxf = team_xg[(r["gw"], t)] / k
            gxa = team_xg[(r["gw"], r["opp"])] / ko
            if use_xg:
                f_for.append(XG_WEIGHT * gxf + (1 - XG_WEIGHT) * r["gf"])
                f_ag.append(XG_WEIGHT * gxa + (1 - XG_WEIGHT) * r["ga"])
                xf.append(gxf)
                xa.append(gxa)
            else:
                f_for.append(r["gf"])
                f_ag.append(r["ga"])
        m = len(fg)
        r_for = (sum(f_for) + FORM_PRIOR * mu) / (m + FORM_PRIOR)
        r_ag = (sum(f_ag) + FORM_PRIOR * mu) / (m + FORM_PRIOR)

        att[t] = (FORM_WEIGHT * r_for + (1 - FORM_WEIGHT) * s_for) / mu
        dfn[t] = (FORM_WEIGHT * r_ag + (1 - FORM_WEIGHT) * s_ag) / mu

        letters = "".join("W" if r["gf"] > r["ga"] else "D" if r["gf"] == r["ga"] else "L"
                          for r in fg)
        form_info[t] = {
            "form": letters,
            "xg_for": round(sum(xf) / len(xf), 2) if xf else None,
            "xg_against": round(sum(xa) / len(xa), 2) if xa else None,
            "goals_for": sum(r["gf"] for r in fg),
            "goals_against": sum(r["ga"] for r in fg),
        }

    # ---- likely scorers -------------------------------------------------------
    by_team = defaultdict(list)
    for e in boot["elements"]:
        if e["element_type"] == 1 or e["status"] not in ("a", "d"):
            continue
        if float(e.get("minutes") or 0) < 90:
            continue
        by_team[e["team"]].append(e)

    def likely_scorers(team, opp):
        out = []
        for e in by_team[team]:
            season = float(e.get("expected_goals_per_90") or 0)
            mins = p_min.get(e["id"], 0)
            if mins >= 60:
                recent = p_xg[e["id"]] / mins * 90
                rate = 0.5 * recent + 0.5 * season
            else:
                rate = season * 0.8  # barely played recently
            p = 1 - math.exp(-rate * 0.85 * dfn[opp])
            out.append({"name": e["web_name"], "prob": round(p, 3),
                        "doubt": e["status"] == "d"})
        out.sort(key=lambda x: x["prob"], reverse=True)
        return out[:TOP_SCORERS]

    # ---- predict this gameweek -----------------------------------------------
    gw_fix = sorted((f for f in fixtures if f.get("event") == gw["id"]),
                    key=lambda f: f.get("kickoff_time") or "")
    out_fix, ranking = [], []
    for f in gw_fix:
        h, a = f["team_h"], f["team_a"]
        lh = mu_h * att[h] * dfn[a]
        la = mu_a * att[a] * dfn[h]
        ph, pd_, pa, (bh, ba) = score_grid(lh, la)
        started = f.get("team_h_score") is not None
        out_fix.append({
            "kickoff": f.get("kickoff_time"),
            "home": teams[h], "away": teams[a],
            "xg_home": round(lh, 2), "xg_away": round(la, 2),
            "p_home": round(ph, 3), "p_draw": round(pd_, 3), "p_away": round(pa, 3),
            "pred_home": bh, "pred_away": ba,
            "actual": [f["team_h_score"], f["team_a_score"]] if started else None,
            "finished": bool(f.get("finished") or f.get("finished_provisional")),
            "form_home": form_info[h], "form_away": form_info[a],
            "scorers_home": likely_scorers(h, a),
            "scorers_away": likely_scorers(a, h),
        })
        ranking.append({"team": teams[h], "opp": teams[a], "venue": "H", "win": round(ph, 3)})
        ranking.append({"team": teams[a], "opp": teams[h], "venue": "A", "win": round(pa, 3)})
    ranking.sort(key=lambda r: r["win"], reverse=True)

    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gameweek": gw["id"],
        "gameweek_name": gw["name"],
        "deadline": gw.get("deadline_time"),
        "uses_xg": use_xg,
        "games_played": n_games,
        "fixtures": out_fix,
        "ranking": ranking,
    }


def main():
    data = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=1)
    print(f"Wrote {OUT}: {data['gameweek_name']}, {len(data['fixtures'])} fixtures, "
          f"xG {'on' if data['uses_xg'] else 'off'}")


if __name__ == "__main__":
    main()
