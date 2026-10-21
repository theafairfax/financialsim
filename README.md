# Financial Planner — 5 years, 10 years, or longer

A Streamlit Monte Carlo planner for comparing financial decisions. Start with
current balances, pay and spending; model dated changes; inspect cash, investments,
debt and home equity separately. The default horizon is **10 years**, with a
one-click **5-year** option and a custom horizon.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## A useful first plan

1. Choose your horizon and enter your age, income and annual living expenses.
   Take-home income mode avoids relying on the simplified wage tax estimator.
2. Enter existing investment balances and cost bases. Allocation percentages apply
   to **new contributions**, not portfolio weights or automatic rebalancing.
3. Enter non-mortgage debt, its blended APR and required annual payments. Exclude
   those payments from living expenses. Surplus first repays debt, then builds the
   emergency reserve, then follows investment allocations.
4. Set an emergency cash target and, if useful, probabilities for unemployment
   spells and unexpected bills. Optional spending and inflation uncertainty start
   at zero so risk assumptions are an explicit choice.
5. Add dated salary changes, expense changes and one-time bills. These tables
   start empty; the app does not silently assume a promotion or a home purchase.
6. For an existing home, enter value and the full mortgage balance, including an
   underwater mortgage. For a future home, set its purchase age, expected nominal
   purchase price, mortgage and closing costs. Do **not** enter its down payment
   again in the one-time expense table. Remove rent via an expense change if needed.
7. Choose a goal in today's dollars: liquid assets, net worth or cash. Compare
   baseline and first-year market-decline results and inspect 5/10-year checkpoints.
8. Download the scenario assumptions (JSON), annual nominal cash flow (CSV), and
   average balance breakdown (CSV). Named scenarios can be compared within the
   current app session. JSON is an audit/export format, not an import feature.

## Timing and accounting

- Index zero is **today's untouched balance sheet**. A five-year plan has today
  plus five year-end points, with exactly five annual cash flows and returns.
- An event at age A belongs to the year ending at A. First-year income and living
  expenses use today's baseline. Organic growth starts in the second year.
  Dated changes reset the baseline in their event year.
- Investments earn returns on opening balances; savings are contributed at year
  end. This is an **annual model**, not a monthly affordability or payment-timing
  model. It can miss cash shortages within a year.
- Every dollar of surplus stays on the balance sheet unless explicitly spent.
  Contribution-cap overflow and zero allocations fall back to uncapped cash.
  A contribution cap of zero prohibits contributions; the large sentinel displayed
  in the editor means unlimited. Emergency cash is replenished from surplus;
  existing investments are not automatically sold to fill the reserve.
- Shortfalls use cash first, then eligible investments. They never turn an
  investment balance negative. Remaining shortfalls create separate debt, accruing
  the configured APR from the following year. Existing debt accrues a full year
  of interest before annual payments. Debt uses one blended rate; borrowing is
  assumed unlimited. **New borrowing is a distress indicator**, not evidence that
  a lender would approve a loan. Mortgage debt is modeled separately.
- Retirement/illiquid accounts are protected before the selected retirement age
  unless early access is enabled. They become available at retirement. Withdrawal
  taxes and the configured pre-59.5 penalty apply to tax-deferred accounts.
- Employment income stops at the chosen retirement year-end age. Net retirement
  income is entered in today's dollars and grows with simulated inflation; it
  represents pension/Social Security take-home income, not portfolio withdrawals.
- Future property purchases occur at year end. Purchase price minus mortgage plus
  closing costs is paid then; ownership cash flows and amortization begin the next
  year. Property value and debt appear together. Existing homes incur ownership
  costs in the first year. Mortgage principal increases equity while reducing cash.
- Avoid double-counting modeled mortgage payments, taxes, insurance and maintenance
  in living expenses. Property figures are nominal values/rents at entry or
  purchase; entered rents and insurance then grow with the overall price index.
  No property sale, refinancing or equity borrowing is automated.

## Returns, inflation, costs and shocks

- Expected Return is the **arithmetic mean nominal annual simple return**, not
  CAGR. The lognormal gross-return parameters are calibrated to the entered mean
  and standard deviation; positive asset balances cannot lose more than 100%.
- Each asset combines a shared standard normal market shock and its own shock.
  “Market Beta” is a **factor loading**, not conventional equity beta. Its product
  with another loading gives their contemporaneous correlation in latent shocks,
  not an exact simple-return correlation. The market starts in its stationary
  distribution. Persistence defaults to zero and is a sensitivity assumption.
- Fees and approximate tax drag reduce asset balances annually. Returns should be
  entered before those reductions, or both reductions should be zero if returns
  are already net. Tax drag approximates ongoing distribution taxes; it does not
  calculate interest/dividend taxes or update tax lots. Brokerage sale taxes are
  separately estimated from the unrealized-gain fraction and proportional basis.
- Inflation is an optional random annual rate, bounded below at -95%. Living costs
  follow that path, except explicit resets, plus optional temporary spending noise.
  Today's-dollar charts deflate **each simulation by its own cumulative price
  index before taking percentiles**. Expense growth also represents general
  inflation here; category-specific inflation is not modeled.
- Unemployment is an independent annual Bernoulli event reducing that year's wages
  by the selected months/12. It is not a persistent employment-state model.
  Unexpected bills are independent annual Bernoulli events with a fixed
  inflation-adjusted cost. Neither is empirically calibrated or linked to markets.
- The optional stress test replaces first-year financial asset returns with a
  chosen market decline multiplied by the absolute loading. Property appreciation
  is unchanged. Later years resume the original model. The same seed shares other
  shocks across runs. This is a hypothetical sensitivity, not a historical replay.

## Results and their limits

The app shows net-worth percentiles, final distributions, goal attainment,
5/10-year checkpoints, cash reserves, accessible investment balances, non-mortgage
debt, borrowing incidence, and cash-target gaps. Liquid balances exclude home
equity and retirement accounts and are **before sale taxes**. Cash targets use
living expenses only; include extra safety margin for separate ownership and debt
costs. Average cash-flow exports include actual simulated property costs, purchase
cash, unexpected bills and debt movements and remain nominal regardless of chart
settings. Cash flow is reported before extra debt repayment and new saving.

A positive net worth does not imply adequate spendable money. Check borrowing,
cash gaps and the downside checkpoint values alongside the median.

Probabilities are **conditional model outputs**, not validated real-world odds.
More simulations reduce sampling noise but do not fix wrong assumptions. Defaults
are illustrative, not personalized recommendations or calibrated market forecasts.
No historical data are downloaded. Compare several return, inflation and career
assumptions rather than treating one output as an accurate prediction.

### Tax scope

The wage estimator uses 2026 federal brackets/standard deductions, employee Social
Security/Medicare and an effective state rate. Optional threshold indexing scales
federal brackets, the standard deduction and the Social Security wage cap by
simulated inflation; additional Medicare thresholds stay fixed. This is a future
policy assumption, not a tax-law forecast. Payroll tax treats entered household
income as one wage earner. Stipends, self-employment, credits, contribution
deductions/eligibility, shared IRA limits, catch-up contributions, employer
matches, capital-gain bracket stacking, RMDs, Roth conversion/withdrawal ordering,
rental tax deductions/depreciation, property-sale taxes and detailed state/local
taxes are not modeled. Withdrawal tax rates and contribution caps are user-set.
Tax-free withdrawals are simplified. Use take-home inputs and conservative net
return assumptions when these omissions matter.

## Validation

```bash
python -m unittest discover -s tests -v
```

The suite checks conservation of cash, capped/zero allocations, debt and interest,
withdrawal taxes, starting snapshots, dated events, future purchases, underwater
mortgages, retirement, shocks, pathwise inflation, stationary market variance,
repeatable stress comparisons, and actual Streamlit interactions. These checks
validate implementation behavior, not forecast accuracy.
