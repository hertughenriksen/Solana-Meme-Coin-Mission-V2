"""
agent.py — AI-driven autoresearch agent for the memecoin bot.

Architecture (mirrors karpathy/autoresearch):
  1. Load full experiment history from history.json
  2. Pass history + program.md instructions to Claude claude-sonnet-4-20250514
  3. Claude proposes the next params as JSON
  4. experiment_runner.py evaluates those params on real historical data
  5. Result is appended to history — Claude will see it next iteration
  6. Repeat until --max-experiments or --hours limit

The agent NEVER repeats a param set it has already tried: the full history
is in the context window for every call, so Claude can see "I tried X → 0.62
win_rate, I tried Y → 0.71" and search intelligently.

Usage
─────
  python -m ml.autoresearch.agent \
      --db-url "$DATABASE_URL" \
      --history-path ml/autoresearch/history.json \
      --program-path ml/autoresearch/program.md \
      --max-experiments 50 \
      --hours 8
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

log = logging.getLogger("autoresearch.agent")

SYSTEM_PROMPT = """\
You are an expert quantitative researcher optimising a Solana memecoin trading bot.
Your job is to propose filter thresholds and strategy parameters that maximise
the Kelly-adjusted geometric mean return (kelly_return) while keeping win_rate
above 0.50 and filter_pass_rate between 0.05 and 0.40.

On each turn you will receive:
  • program.md — human-written description of the codebase and goals
  • The full experiment history (params tried + metrics achieved)

You must respond with ONLY a valid JSON object containing the next param set.
Do not repeat parameters you have already tried. Search intelligently — use
what you know about what worked and what did not.

Param schema (you MUST stay within these bounds):
{schema}

Example response format:
{{
  "min_liquidity_usd": 75000,
  "max_liquidity_usd": 1500000,
  "min_market_cap_usd": 60000,
  "max_market_cap_usd": 8000000,
  "min_lp_lock_days": 14,
  "max_sniper_concentration": 0.04,
  "max_dev_holding_pct": 0.04,
  "max_deployer_rug_rate": 0.15,
  "min_ensemble_score": 0.65,
  "min_buy_sell_ratio": 0.58,
  "tp_1_multiplier": 1.45,
  "tp_2_multiplier": 2.20,
  "tp_1_sell_pct": 0.50,
  "tp_2_sell_pct": 0.30,
  "hard_stop_loss_pct": 0.18,
  "time_stop_minutes": 50,
  "kelly_fraction": 0.45,
  "tp_3_trailing_stop_pct": 0.12
}}
"""

from ml.autoresearch.experiment_runner import PARAM_SCHEMA, clamp_params, evaluate_params


def build_schema_description() -> str:
    lines = []
    for k, spec in PARAM_SCHEMA.items():
        lines.append(f"  {k}: {spec['type']} in [{spec['min']}, {spec['max']}] (default {spec['default']})")
    return "\n".join(lines)


def load_history(history_path: Path) -> list[dict]:
    if history_path.exists():
        try:
            return json.loads(history_path.read_text())
        except json.JSONDecodeError:
            log.warning("history.json is corrupt — starting fresh")
    return []


def save_history(history_path: Path, history: list[dict]):
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))


def load_program_md(program_path: Path) -> str:
    if program_path.exists():
        return program_path.read_text()
    return "(program.md not found — proceeding without extra instructions)"


def format_history_for_prompt(history: list[dict]) -> str:
    if not history:
        return "No experiments have been run yet. This is experiment #1."
    lines = ["Experiment history (most recent last):"]
    for i, entry in enumerate(history, 1):
        m = entry.get("metrics", {})
        p = entry.get("params",  {})
        ts = entry.get("timestamp", "?")
        lines.append(
            f"\n[Experiment {i}] {ts}"
            f"\n  kelly_return={m.get('kelly_return', '?'):.4f}"
            f"  win_rate={m.get('win_rate', '?'):.3f}"
            f"  filter_pass_rate={m.get('filter_pass_rate', '?'):.3f}"
            f"  n_trades={m.get('n_trades', '?')}"
        )
        lines.append(f"  params: {json.dumps(p)}")
    return "\n".join(lines)


def call_claude(client: anthropic.Anthropic, program_md: str, history: list[dict]) -> dict:
    """Call Claude claude-sonnet-4-20250514 and return parsed param dict."""
    user_content = (
        f"## program.md\n\n{program_md}\n\n"
        f"## Experiment history\n\n{format_history_for_prompt(history)}\n\n"
        "Propose the next experiment. Respond with ONLY a JSON object."
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT.format(schema=build_schema_description()),
        messages=[{"role": "user", "content": user_content}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if Claude wrapped the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        params = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude returned invalid JSON: {exc}\n\nRaw:\n{raw}") from exc

    return params


def run_experiment(db_url: str, params: dict, lookback_days: int) -> dict:
    """Run the experiment by calling experiment_runner as a subprocess (safe isolation)."""
    result = subprocess.run(
        [
            sys.executable, "-m", "ml.autoresearch.experiment_runner",
            "--db-url",   db_url,
            "--params",   json.dumps(params),
            "--lookback", str(lookback_days),
        ],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"experiment_runner failed:\n{result.stderr}")
    return json.loads(result.stdout)


def find_best(history: list[dict]) -> dict | None:
    if not history:
        return None
    return max(history, key=lambda e: e.get("metrics", {}).get("kelly_return", -999))


# ── Main agent loop ───────────────────────────────────────────────────────────

def run_agent(
    db_url: str,
    history_path: Path,
    program_path: Path,
    max_experiments: int,
    hours_limit: float,
    lookback_days: int,
    dry_run: bool,
):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set in environment")
        sys.exit(1)

    client     = anthropic.Anthropic(api_key=api_key)
    history    = load_history(history_path)
    program_md = load_program_md(program_path)
    deadline   = time.time() + hours_limit * 3600

    log.info(
        "🚀 Autoresearch agent starting | history=%d experiments | limit=%d or %.1fh",
        len(history), max_experiments, hours_limit,
    )

    experiment_count = 0
    while experiment_count < max_experiments and time.time() < deadline:
        experiment_count += 1
        log.info("── Experiment %d / %d ──────────────────────", experiment_count, max_experiments)

        # 1. Ask Claude for the next param set
        log.info("Calling Claude for next proposal…")
        try:
            raw_params = call_claude(client, program_md, history)
        except Exception as exc:
            log.error("Claude API call failed: %s — retrying in 30s", exc)
            time.sleep(30)
            continue

        clamped = clamp_params(raw_params)
        log.info("Proposed params: %s", json.dumps(clamped))

        if dry_run:
            log.info("[DRY RUN] Skipping actual experiment evaluation")
            history.append({
                "params":    clamped,
                "metrics":   {"kelly_return": 0.0, "win_rate": 0.5, "filter_pass_rate": 0.1, "n_trades": 0},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dry_run":   True,
            })
            save_history(history_path, history)
            continue

        # 2. Run the experiment
        log.info("Running experiment…")
        try:
            result = run_experiment(db_url, clamped, lookback_days)
        except Exception as exc:
            log.error("Experiment runner failed: %s — skipping", exc)
            continue

        metrics = result.get("metrics", {})
        entry   = {
            "params":    result.get("params", clamped),
            "metrics":   metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        history.append(entry)
        save_history(history_path, history)

        kelly  = metrics.get("kelly_return",    "?")
        wr     = metrics.get("win_rate",        "?")
        fpr    = metrics.get("filter_pass_rate","?")
        ntrad  = metrics.get("n_trades",        "?")
        log.info(
            "Result | kelly=%.4f  win_rate=%.3f  pass_rate=%.3f  n_trades=%s",
            kelly if isinstance(kelly, float) else 0,
            wr    if isinstance(wr,    float) else 0,
            fpr   if isinstance(fpr,   float) else 0,
            ntrad,
        )

        # Brief pause so we don't hammer the API
        time.sleep(5)

    # Final summary
    best = find_best(history)
    log.info("═══ Autoresearch complete — %d experiments run ═══", experiment_count)
    if best:
        bm = best["metrics"]
        log.info(
            "Best: kelly=%.4f  win_rate=%.3f  pass_rate=%.3f",
            bm.get("kelly_return", 0), bm.get("win_rate", 0), bm.get("filter_pass_rate", 0),
        )
        log.info("Best params: %s", json.dumps(best["params"], indent=2))
    return history


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url",          required=True)
    parser.add_argument("--history-path",    default="ml/autoresearch/history.json")
    parser.add_argument("--program-path",    default="ml/autoresearch/program.md")
    parser.add_argument("--max-experiments", type=int,   default=50)
    parser.add_argument("--hours",           type=float, default=8.0)
    parser.add_argument("--lookback",        type=int,   default=7)
    parser.add_argument("--dry-run",         action="store_true")
    args = parser.parse_args()

    run_agent(
        db_url          = args.db_url,
        history_path    = Path(args.history_path),
        program_path    = Path(args.program_path),
        max_experiments = args.max_experiments,
        hours_limit     = args.hours,
        lookback_days   = args.lookback,
        dry_run         = args.dry_run,
    )


if __name__ == "__main__":
    main()
