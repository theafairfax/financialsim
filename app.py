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

from dataclasses import asdict
import json

from simulation import (
    AssetClass,
    RealEstateAsset,
    ExpenseChange,
    IncomeChange,
    LargeExpense,
    SimulationConfig,
    deflate_to_real,
    estimate_income_taxes,
    run_monte_carlo,
)

st.set_page_config(page_title="Financial Planner", page_icon="💰", layout="wide")

st.title("💰 Financial Planner")
st.caption(
    "Plan the next 5 or 10 years, or explore a longer horizon with Monte Carlo simulation. "
    "Adjust income, expenses, big life purchases, and how you split savings across "
    "different asset classes to see a *range* of possible financial futures — not just one guess."
)

if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}  # name -> result dict
if "scenario_configs" not in st.session_state:
    st.session_state.scenario_configs = {}  # name -> human-readable config summary

UNLIMITED_CAP_SENTINEL = 999_999_999  # shown in the editor to mean "no contribution cap"

US_STATES = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida",
    "Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine",
    "Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska",
    "Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota",
    "Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee",
    "Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"
]
NO_INDIVIDUAL_INCOME_TAX_STATES = {"Alaska","Florida","Nevada","New Hampshire","South Dakota","Tennessee","Texas","Washington","Wyoming"}

DEFAULT_ASSETS = pd.DataFrame(
    [
        {"Asset Class": "Cash / Savings", "Initial Balance": 5000, "Allocation %": 10.0,
         "Expected Return %": 2.0, "Volatility %": 0.5, "Liquid": True, "Account Type": "cash",
         "Market Beta": 0.05, "Cost Basis": 5000, "Annual Contribution Cap": UNLIMITED_CAP_SENTINEL},
        {"Asset Class": "CDs", "Initial Balance": 5000, "Allocation %": 15.0,
         "Expected Return %": 4.5, "Volatility %": 1.0, "Liquid": True, "Account Type": "taxable",
         "Market Beta": 0.10, "Cost Basis": 5000, "Annual Contribution Cap": UNLIMITED_CAP_SENTINEL},
        {"Asset Class": "IRA", "Initial Balance": 10000, "Allocation %": 25.0,
         "Expected Return %": 7.0, "Volatility %": 12.0, "Liquid": False, "Account Type": "tax_deferred",
         "Market Beta": 0.80, "Cost Basis": 0, "Annual Contribution Cap": 7000},
        {"Asset Class": "Investment Portfolio", "Initial Balance": 10000, "Allocation %": 50.0,
         "Expected Return %": 8.0, "Volatility %": 15.0, "Liquid": True, "Account Type": "taxable",
         "Market Beta": 0.85, "Cost Basis": 10000, "Annual Contribution Cap": UNLIMITED_CAP_SENTINEL},
    ]
)

# --------------------------------------------------------------------------
# Sidebar: time range, income, expenses, Monte Carlo settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⏳ Time Range")
    start_age = st.number_input("Current Age", min_value=16, max_value=100, value=30, step=1)
    horizon = st.radio("Planning horizon", ["5 years", "10 years", "Custom"], index=1, horizontal=True)
    horizon_years = 5 if horizon == "5 years" else 10
    end_age = st.number_input("Simulate Until Age", min_value=int(start_age) + 1, max_value=120, value=min(int(start_age) + horizon_years, 120), step=1, disabled=horizon != "Custom")
    if horizon != "Custom":
        end_age = min(int(start_age) + horizon_years, 120)
    st.caption("Today is the starting balance sheet. Each later age is a year-end result; future events use the age shown at that year end.")

    st.header("💵 Income")
    income_entry_mode = st.radio(
        "Income Entry Mode",
        ["Gross income (estimate taxes)", "Post-tax income"],
        help="Use gross income to estimate federal, payroll, and state taxes, or enter take-home income directly."
    )
    income_is_post_tax = income_entry_mode == "Post-tax income"
    income_label = "Current Annual Post-Tax Income ($)" if income_is_post_tax else "Current Annual Gross Income ($)"
    base_income = st.number_input(income_label, min_value=0, max_value=10_000_000, value=75000, step=1000)

    filing_status = "Single"
    state_name = "Alabama"
    state_effective_tax_rate = 0.0
    additional_deductions = 0.0
    tax_rate = 0.0
    if not income_is_post_tax:
        filing_status = st.selectbox(
            "Federal Filing Status",
            ["Single", "Married Filing Jointly", "Married Filing Separately", "Head of Household"],
        )
        state_name = st.selectbox("State", US_STATES, index=US_STATES.index("Alabama"))
        no_state_income_tax = state_name in NO_INDIVIDUAL_INCOME_TAX_STATES
        state_effective_tax_rate = st.number_input(
            "Estimated Effective State Income Tax Rate (%)",
            min_value=0.0, max_value=20.0, value=0.0 if no_state_income_tax else 4.0, step=0.1,
            disabled=no_state_income_tax,
            help="State taxes vary by deductions, credits, local taxes, and progressive schedules. Use an effective rate for your situation."
        ) / 100
        additional_deductions = st.number_input(
            "Additional Federal Deductions Beyond Standard Deduction ($)",
            min_value=0, max_value=5_000_000, value=0, step=500,
        )
        tax_preview = estimate_income_taxes(
            float(base_income), filing_status, state_effective_tax_rate, float(additional_deductions)
        )
        tax_rate = float(tax_preview["total_tax"] / base_income) if base_income else 0.0
        st.caption(
            f"Estimated current taxes: federal ${float(tax_preview['federal_income_tax']):,.0f} + "
            f"payroll ${float(tax_preview['payroll_tax']):,.0f} + state ${float(tax_preview['state_income_tax']):,.0f} "
            f"= ${float(tax_preview['total_tax']):,.0f}; estimated take-home ${float(tax_preview['after_tax_income']):,.0f}."
        )

    income_growth = st.slider("Organic Annual Raise (%)", 0.0, 15.0, 2.0, 0.1,
                               help="Applied every year that doesn't have an explicit career change below.") / 100
    income_volatility = st.slider("Income Volatility (%)", 0.0, 40.0, 5.0, 0.5,
                                   help="Temporary pay/bonus variation. Model unemployment separately below.") / 100
    ordinary_withdrawal_tax_rate = st.slider("Tax Rate on Tax-Deferred Withdrawals (%)", 0.0, 50.0, 22.0, 1.0) / 100
    capital_gains_tax_rate = st.slider("Capital Gains Tax Rate (%)", 0.0, 40.0, 15.0, 1.0) / 100
    early_withdrawal_penalty_rate = st.slider("Early Retirement Withdrawal Penalty (%)", 0.0, 20.0, 10.0, 1.0) / 100

    st.header("🧾 Recurring Expenses")
    base_expenses = st.number_input("Current Annual Living Expenses ($)", min_value=0, max_value=10_000_000,
                                     value=45000, step=1000,
                                     help="Non-redeemable expenses: rent/mortgage, food, insurance, etc.")
    expense_inflation = st.slider("Expense Inflation Rate (%)", 0.0, 15.0, 3.0, 0.1) / 100

    st.header("🎲 Monte Carlo Settings")
    num_sims = st.select_slider("Number of Simulations", options=[100, 500, 1000, 2000, 5000], value=1000)
    market_autocorrelation = st.slider("Market Shock Persistence", 0.0, 0.8, 0.0, 0.05,
        help="Correlates broad market shocks across adjacent years; 0 is the default. Positive persistence is a sensitivity assumption, not a calibrated forecast.")
    use_seed = st.checkbox("Use fixed random seed (reproducible results)", value=True)
    seed = 42 if use_seed else None

# --------------------------------------------------------------------------
# Career path / income changes
# --------------------------------------------------------------------------
st.info("Start with your current balances, take-home pay and spending. Choose a 5- or 10-year horizon, then add dated changes. The ranges depend on your assumptions; they are not validated forecasts.")

with st.sidebar.expander("Cash reserve, debt and shocks", expanded=True):
    emergency_fund_months = st.number_input("Target emergency cash (months of living expenses)", 0.0, 24.0, 3.0, 0.5)
    initial_debt = st.number_input("Current non-mortgage debt ($)", 0.0, 100_000_000.0, 0.0, 1000.0)
    debt_interest_rate = st.number_input("Non-mortgage debt APR (%)", 0.0, 100.0, 10.0, 0.5) / 100
    annual_debt_payment = st.number_input("Required annual non-mortgage debt payments ($)", 0.0, 10_000_000.0, 0.0, 500.0)
    st.caption("Exclude these payments from living expenses. Surplus pays down this debt first, then refills emergency cash. Unfunded costs become new borrowing at this APR; borrowing capacity is assumed unlimited and flagged in results.")
    unemployment_probability = st.slider("Annual chance of an unemployment spell (%)", 0, 100, 0) / 100
    unemployment_months = st.slider("Months of pay lost during a spell", 0, 12, 3)
    emergency_probability = st.slider("Annual chance of a large unexpected bill (%)", 0, 100, 0) / 100
    emergency_cost = st.number_input("Unexpected bill in today's dollars ($)", 0.0, 10_000_000.0, 5000.0, 500.0)
    inflation_volatility = st.slider("Inflation uncertainty (annual percentage points)", 0.0, 10.0, 0.0, 0.5) / 100
    expense_volatility = st.slider("Temporary annual spending variation (%)", 0.0, 50.0, 0.0, 1.0) / 100

with st.sidebar.expander("Retirement and tax assumptions"):
    enable_retirement = st.checkbox("Stop employment income at retirement")
    retirement_age = st.number_input("Retirement year-end age", int(start_age), 120, max(int(start_age), 65)) if enable_retirement else None
    retirement_net_income = st.number_input("Annual retirement take-home income (today's dollars)", 0.0, 10_000_000.0, 0.0, 1000.0)
    st.caption("Enter net pension / Social Security income here; portfolio withdrawals are calculated separately.")
    allow_retirement_withdrawals = st.checkbox("Allow early withdrawals from retirement / illiquid accounts", value=False)
    st.caption("Retirement accounts become available at your selected retirement age. Before then, this switch permits early access; otherwise cash shortages create borrowing. Tax-free account access is simplified; Roth qualification and ordering rules are not modeled.")
    index_tax_thresholds = st.checkbox("Adjust federal brackets, standard deduction and Social Security wage cap with inflation", value=True)
    st.caption("A planning assumption, not a prediction of future tax law. State taxes use your effective rate; deductions, retirement contributions and withdrawal tax rates remain simplified.")

st.subheader("📈 Career Path: Income Changes")
st.caption(
    "Model raises, job changes, or income loss at specific ages. Each row overrides your income "
    "starting that age; organic growth resumes from the new value until the next change."
)
income_changes_df = st.data_editor(
    pd.DataFrame(columns=["Age", "New Annual Income", "Note"]),
    num_rows="dynamic",
    width="stretch",
    key="income_changes_editor",
    column_config={
        "Age": st.column_config.NumberColumn(min_value=int(start_age)+1, max_value=int(end_age), step=1),
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
    pd.DataFrame(columns=["Age", "New Annual Expenses", "Note"]),
    num_rows="dynamic",
    width="stretch",
    key="expense_changes_editor",
    column_config={
        "Age": st.column_config.NumberColumn(min_value=int(start_age)+1, max_value=int(end_age), step=1),
        "New Annual Expenses": st.column_config.NumberColumn(min_value=0, step=1000, format="$%d"),
    },
)

# --------------------------------------------------------------------------
# Large one-time expenses
# --------------------------------------------------------------------------
st.subheader("🏠 Large One-Time Expenses")
st.caption("Model one-time spending such as education, medical bills, or a car. Enter home purchases in Real Estate so equity and mortgage debt are created together.")
large_expenses_df = st.data_editor(
    pd.DataFrame(columns=["Age", "Amount", "Description"]),
    num_rows="dynamic",
    width="stretch",
    key="large_expenses_editor",
    column_config={
        "Age": st.column_config.NumberColumn(min_value=int(start_age)+1, max_value=int(end_age), step=1),
        "Amount": st.column_config.NumberColumn(min_value=0, step=1000, format="$%d"),
    },
)

# --------------------------------------------------------------------------
# Savings & investment allocation
# --------------------------------------------------------------------------
st.subheader("📊 Savings & Investment Allocation")
st.caption(
    "Define where each year's leftover cash flow goes. 'Lost Income' is money you intentionally leave "
    "unallocated to model unplanned spending, emergencies, leakage, or other costs not captured elsewhere. "
    "The remaining investable cash is allocated across the asset table below."
)
lost_income_pct = st.number_input(
    "Lost Income / Unplanned Spending (% of positive leftover income)",
    min_value=0.0, max_value=100.0, value=0.0, step=1.0,
    help="This share of positive leftover cash flow is treated as spent/lost and does not increase net worth."
) / 100
st.caption(
    "Emergency cash is filled before investing. Allocation percentages direct new contributions, not portfolio rebalancing. Asset Allocation % should total 100% of the cash remaining after Lost Income (auto-normalized if not). "
    "'Liquid' assets are drawn down first during shortfalls. Contribution caps overflow into uncapped assets."
)
st.caption("Expected Return is the arithmetic mean annual nominal return before the fees / tax drag entered below. Market Beta is a shared shock loading, not conventional stock beta. Tax drag is an approximate annual reduction, not a tax return calculation.")
asset_defaults = DEFAULT_ASSETS.copy()
asset_defaults["Annual Fee %"] = 0.0
asset_defaults["Annual Tax Drag %"] = 0.0
assets_df = st.data_editor(
    asset_defaults,
    num_rows="dynamic",
    width="stretch",
    key="assets_editor",
    column_config={
        "Initial Balance": st.column_config.NumberColumn(min_value=0, step=500, format="$%d"),
        "Allocation %": st.column_config.NumberColumn(min_value=0, max_value=100, step=1),
        "Expected Return %": st.column_config.NumberColumn(step=0.5),
        "Volatility %": st.column_config.NumberColumn(min_value=0, step=0.5),
        "Liquid": st.column_config.CheckboxColumn(),
        "Account Type": st.column_config.SelectboxColumn(options=["cash", "taxable", "tax_deferred", "tax_free"]),
        "Market Beta": st.column_config.NumberColumn(min_value=0.0, max_value=0.99, step=0.05),
        "Cost Basis": st.column_config.NumberColumn(min_value=0, step=500, format="$%d"),
        "Annual Contribution Cap": st.column_config.NumberColumn(min_value=0, step=500, format="$%d"),
        "Annual Fee %": st.column_config.NumberColumn(min_value=0.0, max_value=20.0, step=0.05),
        "Annual Tax Drag %": st.column_config.NumberColumn(min_value=0.0, max_value=20.0, step=0.1),
    },
)

st.subheader("🏘️ Real Estate")
property_mode = st.radio(
    "Property Setup",
    ["Single Homestead", "Rental Property / Portfolio"],
    horizontal=True,
    help="Homestead mode hides rental-only inputs. Rental mode exposes rent, vacancy, and supports multiple properties."
)
base_property_df = pd.DataFrame([{
    "Property": "Primary Residence" if property_mode == "Single Homestead" else "Rental Property",
    "Property Value": 0, "Mortgage Balance": 0, "Mortgage Rate %": 6.5, "Mortgage Term Years": 30,
    "Purchase Age (0 = owned now)": 0, "Closing Cost %": 3.0,
    "Appreciation %": 3.0, "Appreciation Volatility %": 8.0, "Market Beta": 0.35,
    "Annual Rent": 0, "Vacancy %": 5.0, "Property Tax %": 1.0, "Annual Insurance": 0, "Maintenance %": 1.0
}])
if property_mode == "Single Homestead":
    st.caption("Simple owner-occupied home model: property value, mortgage, appreciation, taxes, insurance, and maintenance. Rental assumptions are omitted.")
    real_estate_df = st.data_editor(
        base_property_df,
        num_rows="fixed",
        width="stretch",
        key="homestead_editor",
        column_order=[
            "Property", "Property Value", "Mortgage Balance", "Mortgage Rate %", "Mortgage Term Years",
            "Purchase Age (0 = owned now)", "Closing Cost %",
            "Appreciation %", "Appreciation Volatility %", "Market Beta", "Property Tax %",
            "Annual Insurance", "Maintenance %"
        ],
    )
else:
    st.caption("Rental mode includes vacancy-adjusted rent and carrying costs, and supports multiple properties.")
    real_estate_df = st.data_editor(
        base_property_df,
        num_rows="dynamic",
        width="stretch",
        key="rental_property_editor",
    )

st.caption("For a future property, enter its purchase year-end age, expected nominal purchase price and initial mortgage. Down payment and closing costs are deducted automatically; do not repeat them as large expenses. Ownership costs start the following year. Exclude modeled mortgage payments, property tax, insurance and maintenance from living expenses; remove rent with a dated expense change when appropriate.")

st.subheader("🎯 Planning goal and stress test")
goal_type = st.radio("Goal measure", ["Liquid assets", "Net worth", "Cash reserve"], horizontal=True)
goal_amount = st.number_input("Goal at the end of the plan (today's dollars)", 0.0, 1_000_000_000.0, 100000.0, 5000.0)
stress_test = st.checkbox("Also run a first-year market decline", value=True)
stress_decline = st.slider("First-year broad market decline (%)", -60, 0, -30, 5) / 100
st.caption("The stress run replaces first-year asset returns with the decline multiplied by each asset's shared shock loading; property appreciation is unchanged. Later years resume the model. This is a hypothetical sensitivity test, not a historical replay or an assigned crash probability.")

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
run_clicked = run_col1.button("🚀 Run Simulation", type="primary", width="stretch")
clear_clicked = run_col2.button("🗑️ Clear All Scenarios", width="stretch")

if clear_clicked:
    st.session_state.scenarios = {}
    st.session_state.scenario_configs = {}
    st.rerun()

if run_clicked:
    try:
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
            contribution_cap = None if cap_raw >= UNLIMITED_CAP_SENTINEL else cap_raw
            assets.append(
                AssetClass(
                    name=str(r["Asset Class"]),
                    initial_balance=float(r["Initial Balance"]),
                    allocation_pct=alloc_pct,
                    expected_return=float(r["Expected Return %"]) / 100,
                    volatility=float(r["Volatility %"]) / 100,
                    liquidity="liquid" if bool(r.get("Liquid", True)) else "illiquid",
                    annual_contribution_cap=contribution_cap,
                    account_type=str(r.get("Account Type", "taxable")),
                    market_beta=float(r.get("Market Beta", 0.7)),
                    capital_gains_tax_rate=capital_gains_tax_rate,
                    annual_fee=float(r.get("Annual Fee %", 0)) / 100,
                    annual_tax_drag=float(r.get("Annual Tax Drag %", 0)) / 100,
                    cost_basis=float(r.get("Cost Basis", r["Initial Balance"])) if not pd.isna(r.get("Cost Basis", r["Initial Balance"])) else float(r["Initial Balance"]),
                )
            )

        real_estate = []
        for _, r in real_estate_df.dropna(subset=["Property", "Property Value"]).iterrows():
            if float(r["Property Value"]) <= 0:
                continue
            real_estate.append(RealEstateAsset(
                name=str(r["Property"]), property_value=float(r["Property Value"]),
                mortgage_balance=float(r.get("Mortgage Balance", 0)), mortgage_rate=float(r.get("Mortgage Rate %", 0))/100,
                mortgage_term_years=int(r.get("Mortgage Term Years", 30)), appreciation_rate=float(r.get("Appreciation %", 3))/100,
                appreciation_volatility=float(r.get("Appreciation Volatility %", 8))/100, market_beta=float(r.get("Market Beta", 0.35)),
                annual_rent=float(r.get("Annual Rent", 0)), vacancy_rate=float(r.get("Vacancy %", 5))/100,
                property_tax_rate=float(r.get("Property Tax %", 1))/100, annual_insurance=float(r.get("Annual Insurance", 0)),
                maintenance_rate=float(r.get("Maintenance %", 1))/100,
                purchase_age=int(r.get("Purchase Age (0 = owned now)", 0)) or None,
                closing_cost_rate=float(r.get("Closing Cost %", 3))/100,
            ))

        config = SimulationConfig(
            start_age=int(start_age),
            end_age=int(end_age),
            base_income=float(base_income),
            income_growth_rate=income_growth,
            income_changes=income_changes,
            income_volatility=income_volatility,
            tax_rate=tax_rate,
            ordinary_withdrawal_tax_rate=ordinary_withdrawal_tax_rate,
            capital_gains_tax_rate=capital_gains_tax_rate,
            early_withdrawal_penalty_rate=early_withdrawal_penalty_rate,
            base_expenses=float(base_expenses),
            expense_inflation=expense_inflation,
            expense_changes=expense_changes,
            large_expenses=large_expenses,
            assets=assets,
            real_estate=real_estate,
            income_is_post_tax=income_is_post_tax,
            filing_status=filing_status,
            state_effective_tax_rate=state_effective_tax_rate,
            additional_deductions=float(additional_deductions),
            lost_income_pct=lost_income_pct,
            market_autocorrelation=market_autocorrelation,
            inflation_volatility=inflation_volatility, expense_volatility=expense_volatility,
            unemployment_probability=unemployment_probability, unemployment_months=unemployment_months,
            emergency_probability=emergency_probability, emergency_cost=emergency_cost,
            initial_debt=initial_debt, debt_interest_rate=debt_interest_rate,
            annual_debt_payment=annual_debt_payment, emergency_fund_months=emergency_fund_months,
            allow_retirement_withdrawals=allow_retirement_withdrawals,
            retirement_age=int(retirement_age) if retirement_age is not None else None,
            retirement_annual_net_income=retirement_net_income, index_tax_thresholds=index_tax_thresholds,
            num_simulations=int(num_sims),
            seed=seed,
        )

        with st.spinner(f"Running {num_sims:,} simulated plans..."):
            try:
                result = run_monte_carlo(config)
                result["configuration"] = asdict(config)
                result["goal_type"], result["goal_amount"] = goal_type, goal_amount
                if stress_test:
                    from dataclasses import replace
                    result["stress_result"] = run_monte_carlo(replace(config, stress_first_year_return=stress_decline))
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

        name = scenario_name or f"Scenario {len(st.session_state.scenarios) + 1}"
        st.session_state.scenarios[name] = result
        st.session_state.scenario_configs[name] = {
            "Start Age": start_age, "End Age": end_age, "Base Income": base_income,
            "Income Mode": income_entry_mode, "Property Mode": property_mode,
            "Lost Income %": lost_income_pct * 100,
            "Base Expenses": base_expenses, "# Simulations": num_sims,
        }
        st.success(f"Scenario **{name}** simulated across {num_sims:,} possible futures!")
    except (ValueError, TypeError, OverflowError) as exc:
        st.error(f"Check your inputs: {exc}")
        st.stop()

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
        if not show_real:
            return values
        if "price_index" in r and values.ndim >= 2:
            divisor = r["price_index"] if values.ndim == 2 else r["price_index"][:, :, None]
            return values / divisor
        return deflate_to_real(values, r["ages"], r["inflation_rate"])

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
        st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig2, width="stretch")
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
        c4.metric("Runs with Negative Net Worth", f"{prob_negative:.1f}%")

        st.caption("Today contains no annual cash flow. Percentiles describe this model and its assumptions, not calibrated real-world probabilities.")
        d1, d2, d3 = st.columns(3)
        d1.metric("Runs needing new borrowing", f"{(res['new_borrowing'][:,1:].max(axis=1) > 0.01).mean()*100:.1f}%")
        d2.metric("Median final non-mortgage debt", f"${np.median(_adj(res['debt'], res)[:,-1]):,.0f}")
        d3.metric("Runs below cash target during plan", f"{(res['reserve_gap'][:,1:].max(axis=1) > 0.01).mean()*100:.1f}%")
        measure_key = {"Liquid assets":"liquid_assets", "Net worth":"net_worth", "Cash reserve":"cash_reserve"}[res['goal_type']]
        goal_values = res[measure_key][:,-1] / res['price_index'][:,-1]
        st.metric(f"Model probability: {res['goal_type'].lower()} ≥ ${res['goal_amount']:,.0f} in today's dollars at age {ages[-1]}",
                  f"{(goal_values >= res['goal_amount']).mean()*100:.1f}%")
        checkpoints = []
        for years in sorted(set([min(5,len(ages)-1), min(10,len(ages)-1), len(ages)-1])):
            checkpoints.append({"Years ahead":years, "Age":ages[years],
                "Median net worth":float(np.median(nw[:,years])),
                "10th percentile net worth":float(np.percentile(nw[:,years],10)),
                "Median liquid assets":float(np.median(_adj(res['liquid_assets'],res)[:,years])),
                "Median cash":float(np.median(_adj(res['cash_reserve'],res)[:,years])),
                "Median debt":float(np.median(_adj(res['debt'],res)[:,years]))})
        st.subheader("5- and 10-year checkpoints")
        st.dataframe(pd.DataFrame(checkpoints), hide_index=True, width="stretch")
        st.caption("Liquid assets exclude home equity and retirement accounts; balances are before any tax due on sale. Cash targets use living expenses only, excluding separate property and debt costs.")
        if "stress_result" in res:
            stress = res['stress_result']
            stress_fig = go.Figure()
            for label, run in [("Baseline", res), ("First-year decline", stress)]:
                stress_fig.add_trace(go.Scatter(x=run['ages'], y=np.median(_adj(run['net_worth'],run),axis=0), name=label))
            stress_fig.update_layout(title="Sensitivity to a first-year market decline", xaxis_title="Age", yaxis_title=dollar_label)
            st.plotly_chart(stress_fig, width="stretch")
        st.download_button("Download scenario assumptions (JSON)", json.dumps(res['configuration'],indent=2),
                           file_name="financial_plan_assumptions.json", mime="application/json")

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=final_nw, nbinsx=50, marker_color="rgb(99,110,250)"))
        fig3.update_layout(
            title=f"Distribution of Net Worth at Age {ages[-1]} — {primary}",
            xaxis_title=dollar_label, yaxis_title="Number of Simulations", height=400,
        )
        st.plotly_chart(fig3, width="stretch")

        st.subheader("Asset Breakdown Over Time (Average Across Simulations)")
        balances = _adj(res["balances"], res)
        mean_balances = balances.mean(axis=0)
        df_breakdown = pd.DataFrame(mean_balances, columns=res["asset_names"])
        if res.get("real_estate_names"):
            re_equity = _adj(res["real_estate_equity"], res).mean(axis=0)
            for i, name in enumerate(res["real_estate_names"]):
                df_breakdown[f"{name} (equity)"] = re_equity[:, i]
        df_breakdown["Non-mortgage debt (liability)"] = -_adj(res["debt"], res).mean(axis=0)
        df_breakdown.insert(0, "Age", ages)

        fig4 = go.Figure()
        for col in [c for c in df_breakdown.columns if c != "Age"]:
            fig4.add_trace(go.Scatter(x=df_breakdown["Age"], y=df_breakdown[col], stackgroup="one", name=col))
        fig4.update_layout(
            title="Average Asset Balances Over Time", xaxis_title="Age", yaxis_title=dollar_label, height=450,
        )
        st.plotly_chart(fig4, width="stretch")

        csv = df_breakdown.to_csv(index=False)
        st.download_button("📥 Download Asset Breakdown (CSV)", csv,
                            file_name=f"{primary.replace(' ', '_')}_breakdown.csv", mime="text/csv")

        with st.expander("🧮 Cash Flow Detail (average simulated nominal dollars)"):
            st.caption("Average actual simulated cash flows, including income shocks, unexpected bills and property costs. Values are nominal dollars even when the charts show today's dollars. The starting row contains no cash flow.")
            income_col = "Post-Tax Income Entered" if res.get("income_is_post_tax", False) else "Gross Income"
            cash_flow_df = pd.DataFrame({
                "Age": res["ages"], income_col: res["income_path"],
                "After-Tax Income": res["after_tax_income_path"],
                "Living Expenses": res["expense_path"], "Large Expenses": res["large_expense_path"],
                "Property Net Cash Flow": res["real_estate_cash_flow"].sum(axis=2).mean(axis=0),
                "Property Purchase Cash": res["property_purchase_cash"].mean(axis=0),
                "Unexpected Bills": res["emergency_expenses"].mean(axis=0),
                "Lost Income / Unplanned Spending": res["expected_lost_income_path"],
                "Debt Interest (added to debt)": res["debt_interest"].mean(axis=0),
                "Debt Payments": res["debt_payments"].mean(axis=0),
                "New Borrowing": res["new_borrowing"].mean(axis=0),
                "Net Cash Flow Before Extra Debt Repayment / Saving": res["cash_flow"].mean(axis=0),
            })
            st.download_button("Download annual cash flow (nominal CSV)", cash_flow_df.to_csv(index=False),
                               file_name="financial_plan_cash_flow.csv", mime="text/csv")
            st.dataframe(
                cash_flow_df.style.format({c: "${:,.0f}" for c in cash_flow_df.columns if c != "Age"}),
                width="stretch", height=300,
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
