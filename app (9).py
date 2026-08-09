"""
Lifetime Financial Simulator
-----------------------------
A Streamlit app that Monte-Carlo simulates a person's net worth over their
lifetime, accounting for:
  - a configurable time range (current age -> horizon age)
  - recurring "non-redeemable" living expenses (with inflation)
  - large one-time expenses at specific ages
  - a base income with an organic growth rate, optional income volatility,
    and explicit career-path changes (raises, job switches, layoffs...)
  - multiple savings/investment vehicles (IRA, CDs, real estate,
    brokerage/investment, cash, or any custom asset class) each with its
    own contribution allocation, expected return, and volatility

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation import (
    AssetClass,
    ExpenseChange,
    IncomeChange,
    LargeExpense,
    SimulationConfig,
    deflate_to_real,
    run_monte_carlo,
)

st.set_page_config(page_title="Lifetime Financial Simulator", page_icon="💰", layout="wide")

st.title("💰 Lifetime Financial Simulator")
st.caption(
    "Model your net worth across your lifetime with Monte Carlo simulation. "
    "Adjust income, expenses, big life purchases, and how you split savings across "
    "different asset classes to see a *range* of possible financial futures — not just one guess."
)

if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}  # name -> result dict
if "scenario_configs" not in st.session_state:
    st.session_state.scenario_configs = {}  # name -> human-readable config summary

UNLIMITED_CAP_SENTINEL = 999_999_999  # shown in the editor to mean "no contribution cap"

DEFAULT_ASSETS = pd.DataFrame(
    [
        {"Asset Class": "Cash / Savings", "Initial Balance": 5000, "Allocation %": 10.0,
         "Expected Return %": 2.0, "Volatility %": 0.5, "Liquid": True,
         "Annual Contribution Cap": UNLIMITED_CAP_SENTINEL},
        {"Asset Class": "CDs", "Initial Balance": 5000, "Allocation %": 15.0,
         "Expected Return %": 4.5, "Volatility %": 1.0, "Liquid": True,
         "Annual Contribution Cap": UNLIMITED_CAP_SENTINEL},
        {"Asset Class": "IRA", "Initial Balance": 10000, "Allocation %": 20.0,
         "Expected Return %": 7.0, "Volatility %": 12.0, "Liquid": False,
         "Annual Contribution Cap": 7000},
        {"Asset Class": "Real Estate", "Initial Balance": 0, "Allocation %": 15.0,
         "Expected Return %": 4.0, "Volatility %": 8.0, "Liquid": False,
         "Annual Contribution Cap": UNLIMITED_CAP_SENTINEL},
        {"Asset Class": "Investment Portfolio", "Initial Balance": 10000, "Allocation %": 40.0,
         "Expected Return %": 8.0, "Volatility %": 15.0, "Liquid": True,
         "Annual Contribution Cap": UNLIMITED_CAP_SENTINEL},
    ]
)

# --------------------------------------------------------------------------
# Sidebar: time range, income, expenses, Monte Carlo settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⏳ Time Range")
    start_age = st.number_input("Current Age", min_value=16, max_value=100, value=30, step=1)
    end_age = st.number_input("Simulate Until Age", min_value=int(start_age) + 1, max_value=120, value=90, step=1)

    st.header("💵 Income")
    base_income = st.number_input("Current Annual Income ($)", min_value=0, max_value=10_000_000,
                                   value=75000, step=1000)
    income_growth = st.slider("Organic Annual Raise (%)", 0.0, 15.0, 2.0, 0.1,
                               help="Applied every year that doesn't have an explicit career change below.") / 100
    income_volatility = st.slider("Income Volatility (%)", 0.0, 40.0, 5.0, 0.5,
                                   help="Year-to-year randomness representing bonus/job-loss risk.") / 100
    tax_rate = st.slider("Effective Tax Rate on Income (%)", 0.0, 50.0, 22.0, 1.0,
                          help="A flat effective rate (combined federal + state + payroll) applied to gross "
                               "income before it's available to spend or save. Using effective (not marginal) "
                               "rate gives a more realistic cash-flow picture.") / 100

    st.header("🧾 Recurring Expenses")
    base_expenses = st.number_input("Current Annual Living Expenses ($)", min_value=0, max_value=10_000_000,
                                     value=45000, step=1000,
                                     help="Non-redeemable expenses: rent/mortgage, food, insurance, etc.")
    expense_inflation = st.slider("Expense Inflation Rate (%)", 0.0, 15.0, 3.0, 0.1) / 100

    st.header("🎲 Monte Carlo Settings")
    num_sims = st.select_slider("Number of Simulations", options=[100, 500, 1000, 2000, 5000], value=1000)
    use_seed = st.checkbox("Use fixed random seed (reproducible results)", value=True)
    seed = 42 if use_seed else None

# --------------------------------------------------------------------------
# Career path / income changes
# --------------------------------------------------------------------------
st.subheader("📈 Career Path: Income Changes")
st.caption(
    "Model raises, job changes, or income loss at specific ages. Each row overrides your income "
    "starting that age; organic growth resumes from the new value until the next change."
)
income_changes_df = st.data_editor(
    pd.DataFrame([{"Age": int(start_age) + 5, "New Annual Income": float(base_income) * 1.3, "Note": "Promotion"}]),
    num_rows="dynamic",
    use_container_width=True,
    key="income_changes_editor",
    column_config={
        "Age": st.column_config.NumberColumn(min_value=int(start_age), max_value=int(end_age), step=1),
        "New Annual Income": st.column_config.NumberColumn(min_value=0, step=1000, format="$%d"),
    },
)

# --------------------------------------------------------------------------
# Cost-of-living / expense changes
# --------------------------------------------------------------------------
st.subheader("🏙️ Cost-of-Living Changes")
st.caption(
    "Model a shift in your recurring expenses starting at a given age — relocating to a more/less expensive "
    "city, paying off a mortgage, kids leaving home, downsizing, etc. Each row **resets** your recurring "
    "expense baseline starting that age; inflation keeps compounding from the new baseline afterward."
)
expense_changes_df = st.data_editor(
    pd.DataFrame([{"Age": int(start_age) + 10, "New Annual Expenses": float(base_expenses) * 1.6,
                   "Note": "Relocated to a higher cost-of-living city"}]),
    num_rows="dynamic",
    use_container_width=True,
    key="expense_changes_editor",
    column_config={
        "Age": st.column_config.NumberColumn(min_value=int(start_age), max_value=int(end_age), step=1),
        "New Annual Expenses": st.column_config.NumberColumn(min_value=0, step=1000, format="$%d"),
    },
)

# --------------------------------------------------------------------------
# Large one-time expenses
# --------------------------------------------------------------------------
st.subheader("🏠 Large One-Time Expenses")
st.caption("Model big purchases or events: home down payment, wedding, education, medical costs, a new car, etc.")
large_expenses_df = st.data_editor(
    pd.DataFrame(
        [
            {"Age": int(start_age) + 7, "Amount": 60000, "Description": "House down payment"},
            {"Age": int(start_age) + 20, "Amount": 30000, "Description": "Kid's education"},
        ]
    ),
    num_rows="dynamic",
    use_container_width=True,
    key="large_expenses_editor",
    column_config={
        "Age": st.column_config.NumberColumn(min_value=int(start_age), max_value=int(end_age), step=1),
        "Amount": st.column_config.NumberColumn(min_value=0, step=1000, format="$%d"),
    },
)

# --------------------------------------------------------------------------
# Savings & investment allocation
# --------------------------------------------------------------------------
st.subheader("📊 Savings & Investment Allocation")
st.caption(
    "Define where each year's leftover cash flow goes, and the expected growth characteristics of each asset. "
    "Allocation % should total 100% (auto-normalized if not). 'Liquid' assets are drawn down first if expenses "
    "exceed income in a given year; illiquid assets (IRA, real estate) are only tapped as a last resort. "
    f"'Annual Contribution Cap' mirrors real contribution limits (e.g. an IRA) — leave at "
    f"{UNLIMITED_CAP_SENTINEL:,} for unlimited; amounts above a cap automatically flow to uncapped assets instead."
)
assets_df = st.data_editor(
    DEFAULT_ASSETS,
    num_rows="dynamic",
    use_container_width=True,
    key="assets_editor",
    column_config={
        "Initial Balance": st.column_config.NumberColumn(min_value=0, step=500, format="$%d"),
        "Allocation %": st.column_config.NumberColumn(min_value=0, max_value=100, step=1),
        "Expected Return %": st.column_config.NumberColumn(step=0.5),
        "Volatility %": st.column_config.NumberColumn(min_value=0, step=0.5),
        "Liquid": st.column_config.CheckboxColumn(),
        "Annual Contribution Cap": st.column_config.NumberColumn(min_value=0, step=500, format="$%d"),
    },
)

alloc_sum = pd.to_numeric(assets_df["Allocation %"], errors="coerce").sum()
if alloc_sum > 0 and abs(alloc_sum - 100) > 0.5:
    st.warning(f"Allocation percentages currently sum to {alloc_sum:.1f}%, not 100%. "
               "They'll be automatically normalized when the simulation runs.")

# --------------------------------------------------------------------------
# Run simulation
# --------------------------------------------------------------------------
st.divider()
scenario_name = st.text_input("Scenario Name", value=f"Scenario {len(st.session_state.scenarios) + 1}")
run_col1, run_col2 = st.columns([1, 1])
run_clicked = run_col1.button("🚀 Run Simulation", type="primary", use_container_width=True)
clear_clicked = run_col2.button("🗑️ Clear All Scenarios", use_container_width=True)

if clear_clicked:
    st.session_state.scenarios = {}
    st.session_state.scenario_configs = {}
    st.rerun()

if run_clicked:
    income_changes = [
        IncomeChange(age=int(r["Age"]), new_annual_income=float(r["New Annual Income"]),
                     description=str(r.get("Note", "") or ""))
        for _, r in income_changes_df.dropna(subset=["Age", "New Annual Income"]).iterrows()
    ]
    expense_changes = [
        ExpenseChange(age=int(r["Age"]), new_annual_expenses=float(r["New Annual Expenses"]),
                      description=str(r.get("Note", "") or ""))
        for _, r in expense_changes_df.dropna(subset=["Age", "New Annual Expenses"]).iterrows()
    ]
    large_expenses = [
        LargeExpense(age=int(r["Age"]), amount=float(r["Amount"]),
                     description=str(r.get("Description", "") or ""))
        for _, r in large_expenses_df.dropna(subset=["Age", "Amount"]).iterrows()
    ]

    valid_assets_df = assets_df.dropna(subset=["Asset Class", "Initial Balance", "Allocation %",
                                                 "Expected Return %", "Volatility %"])
    total_alloc = pd.to_numeric(valid_assets_df["Allocation %"], errors="coerce").sum()
    assets = []
    for _, r in valid_assets_df.iterrows():
        raw_alloc = float(r["Allocation %"])
        alloc_pct = raw_alloc / total_alloc if total_alloc > 0 else 0.0
        cap_raw = r.get("Annual Contribution Cap", UNLIMITED_CAP_SENTINEL)
        cap_raw = UNLIMITED_CAP_SENTINEL if pd.isna(cap_raw) else float(cap_raw)
        contribution_cap = None if cap_raw <= 0 or cap_raw >= UNLIMITED_CAP_SENTINEL else cap_raw
        assets.append(
            AssetClass(
                name=str(r["Asset Class"]),
                initial_balance=float(r["Initial Balance"]),
                allocation_pct=alloc_pct,
                expected_return=float(r["Expected Return %"]) / 100,
                volatility=float(r["Volatility %"]) / 100,
                liquidity="liquid" if bool(r.get("Liquid", True)) else "illiquid",
                annual_contribution_cap=contribution_cap,
            )
        )

    config = SimulationConfig(
        start_age=int(start_age),
        end_age=int(end_age),
        base_income=float(base_income),
        income_growth_rate=income_growth,
        income_changes=income_changes,
        income_volatility=income_volatility,
        tax_rate=tax_rate,
        base_expenses=float(base_expenses),
        expense_inflation=expense_inflation,
        expense_changes=expense_changes,
        large_expenses=large_expenses,
        assets=assets,
        num_simulations=int(num_sims),
        seed=seed,
    )

    with st.spinner(f"Running {num_sims:,} simulated lifetimes..."):
        result = run_monte_carlo(config)

    name = scenario_name or f"Scenario {len(st.session_state.scenarios) + 1}"
    st.session_state.scenarios[name] = result
    st.session_state.scenario_configs[name] = {
        "Start Age": start_age, "End Age": end_age, "Base Income": base_income,
        "Base Expenses": base_expenses, "# Simulations": num_sims,
    }
    st.success(f"Scenario **{name}** simulated across {num_sims:,} possible futures!")

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
if st.session_state.scenarios:
    st.divider()
    st.header("📉 Results")

    scenario_names = list(st.session_state.scenarios.keys())
    selected = st.multiselect("Scenarios to display", scenario_names, default=scenario_names[-1:])
    show_real = st.checkbox(
        "💵 Show values in today's dollars (inflation-adjusted)", value=True,
        help="Deflates every chart and metric below using each scenario's own expense-inflation rate, so "
             "distant-future numbers are comparable to what a dollar buys today. Turn off to see raw nominal "
             "(future, un-inflation-adjusted) dollar amounts instead."
    )

    def _adj(values, r):
        """Deflate to real dollars if the toggle is on, else pass through nominal."""
        return deflate_to_real(values, r["ages"], r["inflation_rate"]) if show_real else values

    if selected:
        primary = selected[0]
        res = st.session_state.scenarios[primary]
        ages = res["ages"]
        nw = _adj(res["net_worth"], res)
        pct = {p: np.percentile(nw, p, axis=0) for p in [5, 25, 50, 75, 95]}
        dollar_label = "Net Worth in Today's Dollars ($)" if show_real else "Net Worth, Nominal ($)"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ages + ages[::-1], y=list(pct[95]) + list(pct[5])[::-1],
            fill="toself", fillcolor="rgba(99,110,250,0.15)", line=dict(width=0),
            name="5th–95th percentile", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=ages + ages[::-1], y=list(pct[75]) + list(pct[25])[::-1],
            fill="toself", fillcolor="rgba(99,110,250,0.32)", line=dict(width=0),
            name="25th–75th percentile", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=ages, y=pct[50], line=dict(color="rgb(80,90,230)", width=3), name="Median",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="red", opacity=0.5)
        fig.update_layout(
            title=f"Net Worth Projection — {primary}",
            xaxis_title="Age", yaxis_title=dollar_label,
            hovermode="x unified", height=500, legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(selected) > 1:
            fig2 = go.Figure()
            for name in selected:
                r = st.session_state.scenarios[name]
                med = np.percentile(_adj(r["net_worth"], r), 50, axis=0)
                fig2.add_trace(go.Scatter(x=r["ages"], y=med, name=name, mode="lines", line=dict(width=3)))
            fig2.add_hline(y=0, line_dash="dot", line_color="red", opacity=0.5)
            fig2.update_layout(
                title="Scenario Comparison (Median Net Worth)",
                xaxis_title="Age", yaxis_title=dollar_label, height=450,
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                "Compare how small tweaks (a higher savings rate, a career change, a bigger emergency fund) "
                "shift your median trajectory. Run additional scenarios above without clearing to keep building this comparison."
            )

        final_nw = nw[:, -1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median Final Net Worth", f"${np.median(final_nw):,.0f}")
        c2.metric("10th Percentile (Bad Luck)", f"${np.percentile(final_nw, 10):,.0f}")
        c3.metric("90th Percentile (Good Luck)", f"${np.percentile(final_nw, 90):,.0f}")
        prob_negative = (nw.min(axis=1) < 0).mean() * 100
        c4.metric("Chance Net Worth Ever Goes Negative", f"{prob_negative:.1f}%")

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=final_nw, nbinsx=50, marker_color="rgb(99,110,250)"))
        fig3.update_layout(
            title=f"Distribution of Net Worth at Age {ages[-1]} — {primary}",
            xaxis_title=dollar_label, yaxis_title="Number of Simulations", height=400,
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Asset Breakdown Over Time (Average Across Simulations)")
        balances = _adj(res["balances"], res)  # deflate_to_real broadcasts over the last (age) axis
        mean_balances = balances.mean(axis=0)
        df_breakdown = pd.DataFrame(mean_balances, columns=res["asset_names"])
        df_breakdown.insert(0, "Age", ages)

        fig4 = go.Figure()
        for col in res["asset_names"]:
            fig4.add_trace(go.Scatter(x=df_breakdown["Age"], y=df_breakdown[col], stackgroup="one", name=col))
        fig4.update_layout(
            title="Average Asset Balances Over Time", xaxis_title="Age", yaxis_title=dollar_label, height=450,
        )
        st.plotly_chart(fig4, use_container_width=True)

        csv = df_breakdown.to_csv(index=False)
        st.download_button("📥 Download Asset Breakdown (CSV)", csv,
                            file_name=f"{primary.replace(' ', '_')}_breakdown.csv", mime="text/csv")

        with st.expander("🧮 Cash Flow Detail (expected path, before investment-return randomness)"):
            st.caption(
                "Sanity-check your assumptions here: this shows the *expected* year-by-year cash flow — gross "
                "income, after-tax income, recurring expenses (including any cost-of-living changes), and large "
                "one-time expenses — independent of investment-return randomness. Income volatility is not "
                "reflected here since it varies per simulation; this is the average path around which each "
                "simulation's income randomly fluctuates."
            )
            cash_flow_df = pd.DataFrame({
                "Age": res["ages"],
                "Gross Income": res["income_path"],
                "After-Tax Income": res["after_tax_income_path"],
                "Recurring Expenses": res["expense_path"],
                "Large Expenses": res["large_expense_path"],
            })
            cash_flow_df["Net Leftover"] = (
                cash_flow_df["After-Tax Income"] - cash_flow_df["Recurring Expenses"] - cash_flow_df["Large Expenses"]
            )
            st.dataframe(
                cash_flow_df.style.format({c: "${:,.0f}" for c in cash_flow_df.columns if c != "Age"}),
                use_container_width=True, height=300,
            )

        with st.expander("ℹ️ How to read this"):
            st.markdown(
                "- **Bands** show the range of outcomes across every simulated lifetime — wider bands mean more "
                "uncertainty (usually from higher-volatility assets or higher income volatility).\n"
                "- **Median** is the middle outcome: half of simulated futures end up above it, half below.\n"
                "- **Today's dollars** (the default) strips out inflation so a projection 40 years out is "
                "comparable to what money is worth right now — this avoids the illusion of huge nominal growth "
                "that's mostly just inflation. Toggle it off to see raw future dollar amounts instead.\n"
                "- Try duplicating a scenario with one change — e.g. a 5% higher savings allocation to "
                "Investments, a cost-of-living jump from relocating, or delaying a large expense by a few "
                "years — and compare the two median lines above."
            )
else:
    st.info("Configure your parameters above and click **Run Simulation** to see your lifetime financial projection.")
