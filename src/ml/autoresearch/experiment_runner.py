"""
experiment_runner.py — Applies a candidate parameter set to the bot config,
runs a back-test against the last N days of collected token data, and returns
a dict of metrics that the Claude AI agent uses to decide its next proposal.

Design mirrors karpathy/autoresearch: the agent sees the full history of
(params, metrics) pairs and proposes the next experiment.  This file is the
"train.py equivalent" — it is edited by the agent via the config only,
never structurally rewritten.

Metrics returned
────────────────
  win_rate          fraction of labeled trades that hit WIN_MULTIPLIER
  filter_pass_rate  fraction of signals that passed the filter
  kelly_return      half-Kelly geometric mean return (our primary objective)
  n_trades          number of simulated trades
  n_signals         number of signals evaluated
"""

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
import yaml

log = logging.getLogger("experiment_runner")

# ── Param schema with min/max bounds (agent may not exceed these) ─────────────
PARAM_SCHEMA: dict[str, dict] = {
    # Filter thresholds
    "min_liquidity_usd":         {"type": "float",  "min": 10_000,  "max": 500_000,  "default": 50_000},
    "max_liquidity_usd":         {"type": "float",  "min": 100_000, "max": 5_000_000,"default": 2_000_000},
    "min_market_cap_usd":        {"type": "float",  "min": 10_000,  "max": 500_000,  "default": 50_000},
    "max_market_cap_usd":        {"type": "float",  "min": 100_000, "max": 20_000_000,"default":10_000_000},
    "min_lp_lock_days":          {"type": "int",    "min": 0,       "max": 365,      "default": 30},
    "max_sniper_concentration":  {"type": "float",  "min": 0.01,    "max": 0.20,     "default": 0.05},
    "max_dev_holding_pct":       {"type": "float",  "min": 0.01,    "max": 0.20,     "default": 0.05},
    "max_deployer_rug_rate":     {"type": "float",  "min": 0.0,     "max": 0.50,     "default": 0.20},
    "min_ensemble_score":        {"type": "float",  "min": 0.40,    "max": 0.90,     "default": 0.60},
    "min_buy_sell_ratio":        {"type": "float",  "min": 0.40,    "max": 0.80,     "default": 0.55},
    # Strategy params
    "tp_1_multiplier":           {"type": "float",  "min": 1.10,    "max": 3.00,     "default": 1.40},
    "tp_2_multiplier":           {"type": "float",  "min": 1.30,    "max": 5.00,     "default": 2.00},
    "tp_1_sell_pct":             {"type": "float",  "min": 0.20,    "max": 0.80,     "default": 0.50},
    "tp_2_sell_pct":             {"type": "float",  "min": 0.20,    "max": 0.80,     "default": 0.30},
    "hard_stop_loss_pct":        {"type": "float",  "min": 0.05,    "max": 0.40,     "default": 0.20},
    "time_stop_minutes":         {"type": "int",    "min": 10,      "max": 240,      "default": 60},
    "kelly_fraction":            {"type": "float",  "min": 0.10,    "max": 0.75,     "default": 0.50},
    "tp_3_trailing_stop_pct":    {"type": "float",  "min": 0.05,    "max": 0.30,     "default": 0.15},
}


def clamp_params(params: dict[str, Any]) -> dict[str, Any]:
    """Clamp every param to its schema bounds and cast to correct type."""
    result = {}
    for k, spec in PARAM_SCHEMA.items():
        raw = params.get(k, spec["default"])
        val = float(raw)
        val = max(spec["min"], min(spec["max"], val))
        result[k] = int(val) if spec["type"] == "int" else round(val, 6)
    return result


# ── Simulation logic ───────────────────────────────────────────────────────────

def evaluate_params(conn, params: dict, lookback_days: int = 7) -> dict:
    """
    Back-test params against historical data.  Applies filter thresholds
    deterministically and re-labels outcomes using the strategy TP/SL params.
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Pull labeled tokens from the lookback window
    cur.execute("""
        SELECT
            t.mint,
            t.liquidity_usd_at_detection    AS liq,
            t.market_cap_usd_at_detection   AS mcap,
            t.sniper_concentration_pct,
            t.dev_holding_pct,
            t.deployer_rug_rate,
            t.lp_lock_days,
            t.buy_sell_ratio_at_detection   AS buy_sell_ratio,
            t.ml_ensemble_score             AS ensemble_score,
            t.price_usd_at_detection        AS entry_price,
            t.ml_label                      AS base_label,
            (
                SELECT COALESCE(
                    json_agg(json_build_object('high', c.high, 'low', c.low, 'ts', c.ts)
                             ORDER BY c.ts ASC) FILTER (WHERE c.ts < t.first_seen_at + INTERVAL '4 hours'),
                    '[]'::json
                )
                FROM price_candles_2s c
                WHERE c.mint = t.mint AND c.ts >= t.first_seen_at
            ) AS candles
        FROM tokens t
        WHERE t.first_seen_at >= NOW() - (%s || ' days')::INTERVAL
          AND t.ml_label IS NOT NULL
          AND t.price_usd_at_detection > 0
        ORDER BY t.first_seen_at ASC
    """, (str(lookback_days),))

    rows = cur.fetchall()

    n_signals        = len(rows)
    n_passed         = 0
    n_wins           = 0
    n_losses         = 0
    simulated_trades: list[dict] = []

    for row in rows:
        liq          = float(row["liq"] or 0)
        mcap         = float(row["mcap"] or 0)
        sniper       = float(row["sniper_concentration_pct"] or 0)
        dev_hold     = float(row["dev_holding_pct"] or 0)
        rug_rate     = float(row["deployer_rug_rate"] or 0)
        lp_days      = int(row["lp_lock_days"] or 0)
        bsr          = float(row["buy_sell_ratio"] or 0.5)
        ml_score     = float(row["ensemble_score"] or 0)
        entry_price  = float(row["entry_price"] or 0)

        # Apply filter with candidate thresholds
        if liq < params["min_liquidity_usd"]:          continue
        if liq > params["max_liquidity_usd"]:          continue
        if mcap < params["min_market_cap_usd"]:        continue
        if mcap > params["max_market_cap_usd"]:        continue
        if lp_days < params["min_lp_lock_days"]:       continue
        if sniper > params["max_sniper_concentration"]: continue
        if dev_hold > params["max_dev_holding_pct"]:   continue
        if rug_rate > params["max_deployer_rug_rate"]: continue
        if ml_score < params["min_ensemble_score"]:    continue
        if bsr < params["min_buy_sell_ratio"]:         continue

        n_passed += 1

        # Re-evaluate outcome with the candidate TP/SL thresholds
        candles = row["candles"] or []
        outcome = simulate_trade_outcome(
            candles=candles,
            entry_price=entry_price,
            tp1=params["tp_1_multiplier"],
            tp2=params["tp_2_multiplier"],
            tp1_sell_pct=params["tp_1_sell_pct"],
            tp2_sell_pct=params["tp_2_sell_pct"],
            hard_sl=params["hard_stop_loss_pct"],
            time_stop_min=params["time_stop_minutes"],
            trailing_stop_pct=params["tp_3_trailing_stop_pct"],
        )

        simulated_trades.append(outcome)
        if outcome["return"] > 0:
            n_wins += 1
        else:
            n_losses += 1

    # Kelly-adjusted geometric return (primary objective)
    kelly_return = compute_kelly_return(simulated_trades, params["kelly_fraction"])

    return {
        "n_signals":        n_signals,
        "n_passed":         n_passed,
        "filter_pass_rate": round(n_passed / max(n_signals, 1), 4),
        "n_trades":         len(simulated_trades),
        "n_wins":           n_wins,
        "n_losses":         n_losses,
        "win_rate":         round(n_wins / max(len(simulated_trades), 1), 4),
        "kelly_return":     round(kelly_return, 6),
    }


def simulate_trade_outcome(
    candles: list, entry_price: float,
    tp1: float, tp2: float,
    tp1_sell_pct: float, tp2_sell_pct: float,
    hard_sl: float, time_stop_min: int,
    trailing_stop_pct: float,
) -> dict:
    """
    Walk price candles and apply the TP/SL ladder, returning the blended
    return for a $1 position.
    """
    if entry_price <= 0 or not candles:
        return {"return": 0.0, "exit_reason": "no_data"}

    size       = 1.0            # normalised position size
    remaining  = size
    realised   = 0.0
    tp1_hit    = False
    tp2_hit    = False
    trail_stop = None
    sl_price   = entry_price * (1.0 - hard_sl)

    candles_sorted = sorted(candles, key=lambda c: c.get("ts", 0))
    for i, c in enumerate(candles_sorted):
        high = float(c.get("high") or entry_price)
        low  = float(c.get("low")  or entry_price)
        minutes_elapsed = i * (2 / 60)    # 2-second candles → minutes

        # Hard SL
        if low <= sl_price and remaining > 0:
            realised += remaining * (sl_price / entry_price)
            remaining = 0
            return {"return": realised - size, "exit_reason": "hard_sl"}

        # TP1
        if not tp1_hit and high >= entry_price * tp1 and remaining > 0:
            tp1_hit    = True
            sell_qty   = remaining * tp1_sell_pct
            realised  += sell_qty * tp1
            remaining -= sell_qty
            trail_stop = entry_price * tp2 * (1.0 - trailing_stop_pct)

        # TP2
        if tp1_hit and not tp2_hit and high >= entry_price * tp2 and remaining > 0:
            tp2_hit    = True
            sell_qty   = remaining * tp2_sell_pct
            realised  += sell_qty * tp2
            remaining -= sell_qty
            trail_stop = high * (1.0 - trailing_stop_pct)

        # Trailing stop update & trigger
        if trail_stop is not None:
            new_trail = high * (1.0 - trailing_stop_pct)
            if new_trail > trail_stop:
                trail_stop = new_trail
            if low <= trail_stop and remaining > 0:
                realised  += remaining * (trail_stop / entry_price)
                remaining  = 0
                return {"return": realised - size, "exit_reason": "trail_stop"}

        # Time stop
        if minutes_elapsed >= time_stop_min and remaining > 0:
            close = float(c.get("close") or entry_price)
            realised  += remaining * (close / entry_price)
            remaining  = 0
            return {"return": realised - size, "exit_reason": "time_stop"}

    # End of observation window — exit at last close
    if remaining > 0 and candles_sorted:
        close = float(candles_sorted[-1].get("close") or entry_price)
        realised += remaining * (close / entry_price)

    return {"return": realised - size, "exit_reason": "window_end"}


def compute_kelly_return(trades: list[dict], kelly_fraction: float) -> float:
    """Half-Kelly geometric mean return over the simulated trade sequence."""
    if not trades:
        return 0.0
    b = 1.0   # normalised bet
    returns = [t["return"] for t in trades]
    wins    = [r for r in returns if r > 0]
    losses  = [r for r in returns if r <= 0]
    p       = len(wins) / max(len(returns), 1)
    q       = 1.0 - p
    avg_win  = sum(wins)  / max(len(wins),   1)
    avg_loss = abs(sum(losses) / max(len(losses), 1))
    b_ratio  = avg_win / max(avg_loss, 1e-9)
    kelly    = max((p * b_ratio - q) / b_ratio, 0.0)
    bet_size = kelly * kelly_fraction

    log_g = sum(math.log(1.0 + bet_size * r) if (1.0 + bet_size * r) > 0 else -10.0 for r in returns)
    geometric_mean = math.exp(log_g / max(len(returns), 1))
    return geometric_mean - 1.0      # excess return per trade


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url",      required=True)
    parser.add_argument("--params",      required=True, help="JSON string of params to test")
    parser.add_argument("--lookback",    type=int, default=7, help="Days of history to back-test")
    parser.add_argument("--output",      help="Path to write metrics JSON")
    args = parser.parse_args()

    raw_params   = json.loads(args.params)
    clamped      = clamp_params(raw_params)
    conn         = psycopg2.connect(args.db_url)
    metrics      = evaluate_params(conn, clamped, lookback_days=args.lookback)
    conn.close()

    result = {"params": clamped, "metrics": metrics, "timestamp": datetime.now(timezone.utc).isoformat()}
    print(json.dumps(result, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        log.info("Metrics written to %s", args.output)


if __name__ == "__main__":
    main()
