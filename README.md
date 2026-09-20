# 💰 Lifetime Financial Simulator

A Streamlit app that runs a **Monte Carlo simulation** of your net worth over
your entire lifetime — accounting for income changes, recurring expenses,
big one-time purchases, and multiple savings/investment vehicles (IRA, CDs,
real estate, brokerage investments, cash, or any custom asset you define).

## Why Monte Carlo?

A single projection (e.g. "assume 7% returns every year") hides the real
risk in long-term financial planning. This app instead simulates hundreds or
thousands of possible futures, drawing lognormal annual returns with a shared market shock (and optionally random income shocks). The shared factor correlates assets in the same bad or good year and can persist into adjacent years. The result is a **range** of
outcomes — a 5th–95th percentile band and a median line — so you can see not
just "what's likely" but "how bad could it get" and "how good could it get."

## Features

- **Time range** — set your current age and how far out to simulate (e.g. to age 90).
- **Income** — base salary, an organic annual raise rate, income volatility
  (job loss / bonus risk), an effective tax rate, and a **career path table**
  where you can specify income changes at any age (promotions, career
  switches, sabbaticals, retirement).
- **Recurring expenses** — your baseline "non-redeemable" living costs, growing with inflation, **plus a
  cost-of-living change table** so you can model a jump (or drop) in expenses starting at a given age —
  relocating to a pricier city, paying off a mortgage, downsizing, kids leaving home, etc. Inflation keeps
  compounding on top of the new baseline afterward.
- **Large one-time expenses** — a table of big purchases/events at specific ages
  (down payment, wedding, tuition, medical bills, etc.).
- **Savings & investment allocation** — a table of financial asset classes with explicit account tax type and market-factor exposure, each with:
  - Initial balance
  - % of each year's leftover cash flow allocated to it
  - Expected annual return and volatility (used to randomly draw returns each year)
  - Liquidity (liquid assets are drawn down first if you have a shortfall year)
  - **Annual contribution cap** (optional) — mirrors real-world limits like an IRA's
    ~$7,000/year cap; amounts above the cap automatically overflow into uncapped assets
- **Scenario comparison** — run a baseline, tweak one variable, run again, and
  overlay the median trajectories to see how small changes compound over decades.
- **Today's-dollars toggle** — deflate every chart/metric by the scenario's inflation
  rate so a projection 40+ years out is comparable to what a dollar is worth *right now*,
  avoiding the illusion of huge nominal growth that's mostly just inflation.
- **Outputs**:
  - Fan chart of net worth (5th/25th/50th/75th/95th percentiles) over time
  - Histogram of final net worth
  - Probability of ever going net-negative
  - Average asset balance breakdown over time
  - Expected cash-flow detail table (gross income, after-tax income, expenses, leftover) for sanity-checking assumptions
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

- Financial assets use lognormal gross returns, naturally bounded at -100%. A shared market factor plus asset-specific shocks creates cross-asset correlation, while configurable market-shock persistence introduces correlation across adjacent years.
- Allocation percentages are normalized to sum to 100% automatically if you
  enter values that don't add up exactly.
- **Taxes** distinguish ordinary income, tax-deferred withdrawals, and taxable brokerage gains. Tax-deferred withdrawals are taxed on the full distribution and receive the configured early-withdrawal penalty before age 59.5; taxable withdrawals apply capital-gains tax only to the unrealized-gain fraction tracked through cost basis.
- **Contribution caps**: if an asset's cap would be exceeded in a given year,
  the excess automatically flows to uncapped assets proportional to their
  allocation share. If every asset happens to be capped and there's still
  leftover, that excess is simply not invested that year (a documented
  simplification — in practice you'd put it in a taxable brokerage account).
- In a year where expenses exceed income, the shortfall is withdrawn from
  liquid assets first; if those are exhausted, illiquid assets (marked
  "Liquid = False", e.g. IRA/real estate) are tapped, and if everything is
  exhausted the most liquid asset goes negative (representing debt).
- **Today's dollars**: the inflation-adjustment toggle deflates nominal
  results using each scenario's own expense-inflation rate as a proxy for
  general inflation. This is the more realistic default view for long
  horizons — nominal dollars decades out can look dramatically larger than
  they're actually worth.
- **Real estate** is a dedicated leveraged asset model: appreciation is applied to full property value, mortgages amortize into interest/principal, rental income is vacancy-adjusted, carrying costs include property tax/insurance/maintenance, and only equity contributes to net worth.\n- Remaining simplifications include progressive tax brackets, Social Security, required minimum distributions, transaction costs, depreciation/deductions, and property-sale taxes.
- This is an educational planning tool, not financial advice — expected
  returns, volatilities, and tax rates are user-supplied assumptions.
