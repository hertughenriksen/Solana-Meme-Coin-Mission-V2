# Autoresearch Program Instructions

This file is read by the Claude AI agent at the start of every experiment.
Edit it to steer the search — the agent will follow your guidance while still
using the experiment history to avoid repeating itself.

---

## Goal

Find the filter-threshold and strategy-parameter combination that maximises
**kelly_return** (Kelly-adjusted geometric mean return per trade) while
satisfying these constraints:

| Metric             | Target        | Hard limit |
|--------------------|---------------|------------|
| win_rate           | ≥ 0.52        | ≥ 0.48     |
| filter_pass_rate   | 0.05 – 0.40   | 0.03 – 0.50 |
| n_trades           | ≥ 20          | ≥ 5        |

If n_trades < 5 the filter is too aggressive — loosen it.
If filter_pass_rate > 0.40 the bot is likely over-trading — tighten it.

---

## Parameter groups and intuition

### Filter thresholds
- **min_liquidity_usd / max_liquidity_usd** — low liquidity means high slippage;
  very high liquidity means the pump already happened. Sweet spot: 50k – 500k.
- **min_lp_lock_days** — LP lock removes immediate rug risk. 14–30 days is good.
- **max_sniper_concentration** — high sniper % means early wallets will dump on you.
  Keep ≤ 0.05.
- **max_dev_holding_pct** — dev holds > 5% is a red flag.
- **max_deployer_rug_rate** — fraction of the deployer's previous tokens that rugged.
  > 0.20 is a strong negative signal.
- **min_ensemble_score** — the ML model's confidence. 0.60 is baseline; raising it
  reduces volume but improves quality.
- **min_buy_sell_ratio** — buy pressure dominance. 0.55+ means more buyers than sellers.

### Strategy params
- **tp_1_multiplier / tp_2_multiplier** — take-profit levels. TP1 at 1.4× is
  conservative but reliable; TP2 at 2–3× captures bigger moves.
- **tp_1_sell_pct / tp_2_sell_pct** — how much to sell at each TP. Must sum ≤ 1.0.
- **hard_stop_loss_pct** — percentage drop that triggers a full exit. 0.15–0.25
  balances giving the trade room vs cutting losses.
- **time_stop_minutes** — exit if the trade hasn't hit TP1 by this time. Shorter
  keeps capital moving; longer gives more room.
- **kelly_fraction** — fraction of Kelly-optimal bet size. 0.5 (half-Kelly) is
  standard; go lower for more conservative sizing.
- **tp_3_trailing_stop_pct** — trail distance after TP2. Tighter = locks in more
  profit but risks premature exit.

---

## Search strategy hints

1. Start by exploring a wide range of `min_ensemble_score` (0.50, 0.65, 0.75, 0.85)
   to understand how the ML filter quality affects downstream metrics.

2. Once you have a baseline, narrow `min_liquidity_usd` — it has the highest
   impact on filter_pass_rate.

3. After fixing good filter settings, optimise the TP ladder:
   the `tp_1_multiplier` × `tp_1_sell_pct` product determines the "guaranteed"
   portion of return; maximise this while keeping win_rate stable.

4. Finish with `kelly_fraction` and `hard_stop_loss_pct` fine-tuning.

5. If win_rate is consistently < 0.50 despite a high ML threshold, the issue
   is likely the filter thresholds not the strategy params — focus there.

---

## What NOT to do

- Do not set `tp_1_sell_pct + tp_2_sell_pct > 0.90` — keep some remainder for
  the trailing stop.
- Do not set `min_lp_lock_days > 60` — very few legitimate tokens lock for that long.
- Do not reduce `max_sniper_concentration` below 0.01 — you'll have almost no trades.
- Do not increase `min_ensemble_score` above 0.80 unless n_trades is still ≥ 20.
