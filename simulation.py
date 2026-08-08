"""
simulation.py
--------------
Core Monte Carlo simulation engine for the Lifetime Financial Simulator.

This module is UI-agnostic: it exposes plain dataclasses for configuration
and a single entry point, `run_monte_carlo`, that returns numpy arrays
describing net worth and per-asset balances across many simulated futures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


# --------------------------------------------------------------------------
# Configuration objects
# --------------------------------------------------------------------------

@dataclass
class IncomeChange:
    """A discrete change to annual income starting at a given age
    (e.g. a promotion, a career switch, a layoff, retirement)."""
    age: int
    new_annual_income: float
    description: str = ""


@dataclass
class LargeExpense:
    """A one-time large expense hitting at a specific age
    (e.g. house down payment, wedding, tuition, medical event)."""
    age: int
    amount: float
    description: str = ""


@dataclass
class AssetClass:
    """A savings/investment vehicle the user allocates leftover cash flow to.

    allocation_pct: fraction (0-1) of each year's positive leftover cash flow
        that is contributed to this asset. Across all assets these should
        sum to ~1.0 (the simulation normalizes them if they don't).
    expected_return / volatility: mean and standard deviation of the
        asset's annual return, used to draw a random return each
        simulated year (Monte Carlo).
    liquidity: "liquid" assets are drawn down first to cover shortfalls
        (years where expenses exceed income); "illiquid" assets
        (e.g. real estate, retirement accounts) are only tapped once all
        liquid assets are exhausted, mimicking real-world penalties /
        difficulty accessing that capital early.
    """
    name: str
    initial_balance: float
    allocation_pct: float
    expected_return: float
    volatility: float
    liquidity: str = "liquid"  # "liquid" or "illiquid"


@dataclass
class SimulationConfig:
    start_age: int
    end_age: int

    base_income: float
    income_growth_rate: float = 0.02      # organic annual raise, applied between explicit changes
    income_changes: List[IncomeChange] = field(default_factory=list)
    income_volatility: float = 0.0        # year-to-year income randomness (job loss / bonus risk)

    base_expenses: float = 0.0
    expense_inflation: float = 0.03
    large_expenses: List[LargeExpense] = field(default_factory=list)

    assets: List[AssetClass] = field(default_factory=list)

    num_simulations: int = 1000
    seed: Optional[int] = 42


# --------------------------------------------------------------------------
# Schedule builders (deterministic timelines keyed by age)
# --------------------------------------------------------------------------

def build_income_schedule(config: SimulationConfig, ages: List[int]) -> Dict[int, float]:
    """Returns {age: base_annual_income_that_year} before applying
    year-to-year volatility. Explicit income_changes override the
    organic growth trajectory starting at their given age."""
    changes_by_age = {c.age: c.new_annual_income for c in config.income_changes}
    schedule = {}
    current = config.base_income
    for i, age in enumerate(ages):
        if age in changes_by_age:
            current = changes_by_age[age]
        elif i > 0:
            current = current * (1 + config.income_growth_rate)
        schedule[age] = current
    return schedule


def build_expense_schedule(config: SimulationConfig, ages: List[int]) -> Dict[int, float]:
    """Returns {age: recurring_non_redeemable_expenses_that_year},
    growing with inflation."""
    schedule = {}
    current = config.base_expenses
    for i, age in enumerate(ages):
        if i > 0:
            current = current * (1 + config.expense_inflation)
        schedule[age] = current
    return schedule


def build_large_expense_map(config: SimulationConfig) -> Dict[int, float]:
    """Returns {age: total_lump_sum_expense_that_year}."""
    out: Dict[int, float] = {}
    for exp in config.large_expenses:
        out[exp.age] = out.get(exp.age, 0.0) + exp.amount
    return out


# --------------------------------------------------------------------------
# Allocation / withdrawal logic
# --------------------------------------------------------------------------

def _allocate_and_withdraw(balances: np.ndarray, t: int, leftover: np.ndarray,
                            assets: List[AssetClass]) -> None:
    """Mutates balances[:, t, :] in place: distributes positive leftover
    cash flow across assets by allocation_pct, and funds negative leftover
    (a shortfall year) by withdrawing from liquid assets first, then
    illiquid assets, allowing the last-resort asset to go negative
    (representing debt) if nothing is left."""
    positive = leftover > 0
    negative = leftover < 0

    # --- allocate positive leftover ---
    for a_idx, asset in enumerate(assets):
        if asset.allocation_pct > 0:
            balances[positive, t, a_idx] += leftover[positive] * asset.allocation_pct

    if not negative.any():
        return

    # --- withdraw to cover shortfall, liquid assets first ---
    shortfall = np.where(negative, -leftover, 0.0)
    order = sorted(range(len(assets)), key=lambda i: 0 if assets[i].liquidity == "liquid" else 1)

    for a_idx in order:
        avail = np.maximum(balances[:, t, a_idx], 0.0)
        withdraw = np.minimum(shortfall, avail)
        withdraw = np.where(negative, withdraw, 0.0)
        balances[:, t, a_idx] -= withdraw
        shortfall -= withdraw

    # anything still unmet goes to debt on the first (most liquid) asset
    if order and shortfall.any():
        balances[:, t, order[0]] -= shortfall


# --------------------------------------------------------------------------
# Main Monte Carlo entry point
# --------------------------------------------------------------------------

def run_monte_carlo(config: SimulationConfig) -> dict:
    """Runs a vectorized Monte Carlo simulation of net worth over the
    configured lifetime.

    Returns a dict with:
        ages:        list[int]                     length = n_years
        net_worth:   np.ndarray (n_sims, n_years)
        balances:    np.ndarray (n_sims, n_years, n_assets)
        asset_names: list[str]
        income_path: np.ndarray (n_years,)          deterministic base income (pre-volatility)
        expense_path: np.ndarray (n_years,)          recurring expenses (pre large expenses)
        large_expense_path: np.ndarray (n_years,)
    """
    ages = list(range(config.start_age, config.end_age + 1))
    n_years = len(ages)
    n_sims = max(1, int(config.num_simulations))

    assets = config.assets if config.assets else []
    n_assets = len(assets)

    # normalize allocations so they sum to 1.0 (unallocated leftover would
    # otherwise silently vanish rather than compound anywhere)
    total_alloc = sum(a.allocation_pct for a in assets)
    if total_alloc > 0 and abs(total_alloc - 1.0) > 1e-9:
        assets = [
            AssetClass(a.name, a.initial_balance, a.allocation_pct / total_alloc,
                       a.expected_return, a.volatility, a.liquidity)
            for a in assets
        ]

    rng = np.random.default_rng(config.seed)

    balances = np.zeros((n_sims, n_years, max(n_assets, 1)))
    if n_assets == 0:
        # no asset classes configured -> track net worth as plain cash, 0% return
        assets = [AssetClass("Cash", 0.0, 1.0, 0.0, 0.0, "liquid")]
        n_assets = 1

    for a_idx, asset in enumerate(assets):
        balances[:, 0, a_idx] = asset.initial_balance

    income_schedule = build_income_schedule(config, ages)
    expense_schedule = build_expense_schedule(config, ages)
    large_expense_map = build_large_expense_map(config)

    income_path = np.array([income_schedule[a] for a in ages])
    expense_path = np.array([expense_schedule[a] for a in ages])
    large_expense_path = np.array([large_expense_map.get(a, 0.0) for a in ages])

    net_worth = np.zeros((n_sims, n_years))
    net_worth[:, 0] = balances[:, 0, :].sum(axis=1)

    # Year 0: no growth yet, but still apply that year's cash flow
    leftover0 = np.full(n_sims, income_path[0] - expense_path[0] - large_expense_path[0])
    if config.income_volatility > 0:
        leftover0 = leftover0 + income_path[0] * rng.normal(0, config.income_volatility, n_sims)
    _allocate_and_withdraw(balances, 0, leftover0, assets)
    net_worth[:, 0] = balances[:, 0, :].sum(axis=1)

    for t in range(1, n_years):
        # 1) grow existing balances with a random draw per asset per simulation
        for a_idx, asset in enumerate(assets):
            prev_bal = balances[:, t - 1, a_idx]
            returns = rng.normal(asset.expected_return, asset.volatility, n_sims)
            returns = np.maximum(returns, -0.95)  # floor: can't lose more than 95% in a year
            balances[:, t, a_idx] = prev_bal * (1 + returns)

        # 2) this year's income (with optional volatility) minus expenses
        income = np.full(n_sims, income_path[t])
        if config.income_volatility > 0:
            income = income * (1 + rng.normal(0, config.income_volatility, n_sims))
            income = np.maximum(income, 0.0)

        leftover = income - expense_path[t] - large_expense_path[t]

        # 3) allocate savings or withdraw to cover shortfall
        _allocate_and_withdraw(balances, t, leftover, assets)

        net_worth[:, t] = balances[:, t, :].sum(axis=1)

    return {
        "ages": ages,
        "net_worth": net_worth,
        "balances": balances,
        "asset_names": [a.name for a in assets],
        "income_path": income_path,
        "expense_path": expense_path,
        "large_expense_path": large_expense_path,
    }


def percentile_summary(net_worth: np.ndarray, percentiles=(5, 25, 50, 75, 95)) -> Dict[int, np.ndarray]:
    """Convenience helper: {percentile: array_over_time}."""
    return {p: np.percentile(net_worth, p, axis=0) for p in percentiles}
