# 💰 Lifetime Financial Simulator

A Streamlit app that runs a **Monte Carlo simulation** of your net worth over
your entire lifetime — accounting for income changes, recurring expenses,
big one-time purchases, and multiple savings/investment vehicles (IRA, CDs,
real estate, brokerage investments, cash, or any custom asset you define).

## Why Monte Carlo?

A single projection (e.g. "assume 7% returns every year") hides the real
risk in long-term financial planning. This app instead simulates hundreds or
thousands of possible futures, drawing a random annual return for each
asset class (and optionally random income shocks) from a normal distribution
around your expected return and volatility. The result is a **range** of
outcomes — a 5th–95th percentile band and a median line — so you can see not
just "what's likely" but "how bad could it get" and "how good could it get."

## Features

- **Time range** — set your current age and how far out to simulate (e.g. to age 90).
- **Income** — base salary, an organic annual raise rate, income volatility
  (job loss / bonus risk), and a **career path table** where you can specify
  income changes at any age (promotions, career switches, sabbaticals, retirement).
- **Recurring expenses** — your baseline "non-redeemable" living costs, growing with inflation.
- **Large one-time expenses** — a table of big purchases/events at specific ages
  (down payment, wedding, tuition, medical bills, etc.).
- **Savings & investment allocation** — a table of asset classes, each with:
  - Initial balance
  - % of each year's leftover cash flow allocated to it
  - Expected annual return and volatility (used to randomly draw returns each year)
  - Liquidity (liquid assets are drawn down first if you have a shortfall year)
- **Scenario comparison** — run a baseline, tweak one variable, run again, and
  overlay the median trajectories to see how small changes compound over decades.
- **Outputs**:
  - Fan chart of net worth (5th/25th/50th/75th/95th percentiles) over time
  - Histogram of final net worth
  - Probability of ever going net-negative
  - Average asset balance breakdown over time
  - CSV export

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Files

- `simulation.py` — UI-agnostic Monte Carlo engine (dataclasses + `run_monte_carlo`)
- `app.py` — Streamlit UI: input forms, scenario management, charts
- `requirements.txt` — Python dependencies

## Notes on the model

- Each asset's annual return is drawn independently from
  `Normal(expected_return, volatility)`, floored at -95% to avoid unrealistic
  runaway negative compounding.
- Allocation percentages are normalized to sum to 100% automatically if you
  enter values that don't add up exactly.
- In a year where expenses exceed income, the shortfall is withdrawn from
  liquid assets first; if those are exhausted, illiquid assets (marked
  "Liquid = False", e.g. IRA/real estate) are tapped, and if everything is
  exhausted the most liquid asset goes negative (representing debt).
- This is an educational planning tool, not financial advice — expected
  returns and volatilities are user-supplied assumptions.
