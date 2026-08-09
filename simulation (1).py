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
    """A discrete change to annual (gross) income starting at a given age
    (e.g. a promotion, a career switch, a layoff, retirement)."""
    age: int
    new_annual_income: float
    description: str = ""


@dataclass
class ExpenseChange:
    """A discrete change to the recurring annual expense *baseline* starting
    at a given age (e.g. relocating to a more/less expensive city, paying
    off a mortgage, kids leaving home). Inflation continues to compound
    on top of the new baseline from this age forward."""
    age: int
    new_annual_expenses: float
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
    annual_contribution_cap: optional ceiling (in dollars) on how much can
        be contributed to this asset in a single year, mirroring real
        contribution limits on tax-advantaged accounts (e.g. an IRA).
        Amounts above the cap "overflow" and are redistributed to
        uncapped assets proportional to their allocation share. None
        means unlimited.
    """
    name: str
    initial_balance: float
    allocation_pct: float
    expected_return: float
    volatility: float
    liquidity: str = "liquid"  # "liquid" or "illiquid"
    annual_contribution_cap: Optional[float] = None


@dataclass
class SimulationConfig:
    start_age: int
    end_age: int

    base_income: float
    income_growth_rate: float = 0.02      # organic annual raise, applied between explicit changes
    income_changes: List[IncomeChange] = field(default_factory=list)
    income_volatility: float = 0.0        # year-to-year income randomness (job loss / bonus risk)
    tax_rate: float = 0.0                 # flat effective tax rate applied to gross income

    base_expenses: float = 0.0
    expense_inflation: float = 0.03
    expense_changes: List[ExpenseChange] = field(default_factory=list)
    large_expenses: List[LargeExpense] = field(default_factory=list)

    assets: List[AssetClass] = field(default_factory=list)

    num_simulations: int = 1000
    seed: Optional[int] = 42


# --------------------------------------------------------------------------
# Schedule builders (deterministic timelines keyed by age)
# --------------------------------------------------------------------------

def build_income_schedule(config: SimulationConfig, ages: List[int]) -> Dict[int, float]:
    """Returns {age: gross_annual_income_that_year} before applying
    year-to-year volatility or tax. Explicit income_changes override the
    organic growth trajectory starting at their given age; growth resumes
    from the new value afterward."""
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
    """Returns {age: recurring_non_redeemable_expenses_that_year}, growing
    with inflation. Explicit expense_changes rebase the baseline starting
    at their given age (e.g. moving to a higher cost-of-living area);
    inflation continues to compound on top of the new baseline afterward."""
    changes_by_age = {c.age: c.new_annual_expenses for c in config.expense_changes}
    schedule = {}
    current = config.base_expenses
    for i, age in enumerate(ages):
        if age in changes_by_age:
            current = changes_by_age[age]
        elif i > 0:
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

def _allocate_positive(balances: np.ndarray, t: int, leftover_pos: np.ndarray,
                        assets: List[AssetClass]) -> None:
    """Distributes positive leftover cash flow across assets by
    allocation_pct, respecting each asset's optional annual contribution
    cap. Any amount that would exceed a cap "overflows" and is
    redistributed across uncapped assets proportional to their relative
    allocation share. If every asset is capped and contributions still
    exceed total capacity, the remainder is simply not invested that year
    (mirrors real life: money with nowhere tax-advantaged to go sits idle
    or must be spent/invested outside the modeled accounts)."""
    n_sims = balances.shape[0]
    n_assets = len(assets)
    desired = np.zeros((n_sims, n_assets))
    for i, a in enumerate(assets):
        desired[:, i] = leftover_pos * a.allocation_pct

    final_contrib = desired.copy()
    overflow = np.zeros(n_sims)
    uncapped_idx = []
    uncapped_alloc_sum = 0.0

    for i, a in enumerate(assets):
        if a.annual_contribution_cap is not None:
            capped_amt = np.minimum(desired[:, i], a.annual_contribution_cap)
            overflow += desired[:, i] - capped_amt
            final_contrib[:, i] = capped_amt
        else:
            uncapped_idx.append(i)
            uncapped_alloc_sum += a.allocation_pct

    if uncapped_idx and overflow.any():
        for i in uncapped_idx:
            share = (assets[i].allocation_pct / uncapped_alloc_sum) if uncapped_alloc_sum > 0 else (1.0 / len(uncapped_idx))
            final_contrib[:, i] += overflow * share

    for i in range(n_assets):
        balances[:, t, i] += final_contrib[:, i]


def _withdraw_for_shortfall(balances: np.ndarray, t: int, shortfall: np.ndarray,
                             negative_mask: np.ndarray, assets: List[AssetClass]) -> None:
    """Funds a shortfall year (expenses > after-tax income) by withdrawing
    from liquid assets first, then illiquid assets. If everything is
    exhausted, the most liquid asset is allowed to go negative,
    representing debt."""
    order = sorted(range(len(assets)), key=lambda i: 0 if assets[i].liquidity == "liquid" else 1)
    remaining = shortfall.copy()

    for a_idx in order:
        avail = np.maximum(balances[:, t, a_idx], 0.0)
        withdraw = np.minimum(remaining, avail)
        withdraw = np.where(negative_mask, withdraw, 0.0)
        balances[:, t, a_idx] -= withdraw
        remaining -= withdraw

    if order and remaining.any():
        balances[:, t, order[0]] -= remaining


def _allocate_and_withdraw(balances: np.ndarray, t: int, leftover: np.ndarray,
                            assets: List[AssetClass]) -> None:
    positive_amounts = np.where(leftover > 0, leftover, 0.0)
    if positive_amounts.any():
        _allocate_positive(balances, t, positive_amounts, assets)

    negative_mask = leftover < 0
    if negative_mask.any():
        shortfall = np.where(negative_mask, -leftover, 0.0)
        _withdraw_for_shortfall(balances, t, shortfall, negative_mask, assets)


# --------------------------------------------------------------------------
# Main Monte Carlo entry point
# --------------------------------------------------------------------------

def run_monte_carlo(config: SimulationConfig) -> dict:
    """Runs a vectorized Monte Carlo simulation of net worth over the
    configured lifetime.

    Returns a dict with:
        ages:                 list[int]                length = n_years
        net_worth:             np.ndarray (n_sims, n_years)
        balances:               np.ndarray (n_sims, n_years, n_assets)
        asset_names:            list[str]
        income_path:             np.ndarray (n_years,)  gross income, pre-volatility/tax
        after_tax_income_path:    np.ndarray (n_years,)  expected after-tax income, pre-volatility
        expense_path:             np.ndarray (n_years,)  recurring expenses (pre large expenses)
        large_expense_path:        np.ndarray (n_years,)
        inflation_rate:              float               expense_inflation used (for deflating to real $)
        tax_rate:                     float
    """
    ages = list(range(config.start_age, config.end_age + 1))
    n_years = len(ages)
    n_sims = max(1, int(config.num_simulations))

    assets = list(config.assets) if config.assets else [AssetClass("Cash", 0.0, 1.0, 0.0, 0.0, "liquid")]

    # normalize allocations so they sum to 1.0 (unallocated leftover would
    # otherwise silently vanish rather than compound anywhere)
    total_alloc = sum(a.allocation_pct for a in assets)
    if total_alloc > 0 and abs(total_alloc - 1.0) > 1e-9:
        assets = [
            AssetClass(a.name, a.initial_balance, a.allocation_pct / total_alloc,
                       a.expected_return, a.volatility, a.liquidity, a.annual_contribution_cap)
            for a in assets
        ]

    n_assets = len(assets)
    rng = np.random.default_rng(config.seed)

    balances = np.zeros((n_sims, n_years, n_assets))
    for i, asset in enumerate(assets):
        balances[:, 0, i] = asset.initial_balance

    income_schedule = build_income_schedule(config, ages)
    expense_schedule = build_expense_schedule(config, ages)
    large_expense_map = build_large_expense_map(config)

    income_path = np.array([income_schedule[a] for a in ages])
    expense_path = np.array([expense_schedule[a] for a in ages])
    large_expense_path = np.array([large_expense_map.get(a, 0.0) for a in ages])
    after_tax_income_path = income_path * (1 - config.tax_rate)

    net_worth = np.zeros((n_sims, n_years))

    for t in range(n_years):
        if t > 0:
            # grow existing balances with a random draw per asset per simulation
            for i, asset in enumerate(assets):
                prev_bal = balances[:, t - 1, i]
                returns = rng.normal(asset.expected_return, asset.volatility, n_sims)
                returns = np.maximum(returns, -0.95)  # floor: can't lose more than 95% in a year
                balances[:, t, i] = prev_bal * (1 + returns)
            # t == 0: balances already hold each asset's initial_balance

        # this year's income (with optional volatility), taxed, minus expenses
        gross_income = np.full(n_sims, income_path[t])
        if config.income_volatility > 0:
            gross_income = gross_income * (1 + rng.normal(0, config.income_volatility, n_sims))
            gross_income = np.maximum(gross_income, 0.0)
        after_tax_income = gross_income * (1 - config.tax_rate)

        leftover = after_tax_income - expense_path[t] - large_expense_path[t]

        _allocate_and_withdraw(balances, t, leftover, assets)

        net_worth[:, t] = balances[:, t, :].sum(axis=1)

    return {
        "ages": ages,
        "net_worth": net_worth,
        "balances": balances,
        "asset_names": [a.name for a in assets],
        "income_path": income_path,
        "after_tax_income_path": after_tax_income_path,
        "expense_path": expense_path,
        "large_expense_path": large_expense_path,
        "inflation_rate": config.expense_inflation,
        "tax_rate": config.tax_rate,
    }


def percentile_summary(net_worth: np.ndarray, percentiles=(5, 25, 50, 75, 95)) -> Dict[int, np.ndarray]:
    """Convenience helper: {percentile: array_over_time}."""
    return {p: np.percentile(net_worth, p, axis=0) for p in percentiles}


def deflate_to_real(values: np.ndarray, ages: List[int], inflation_rate: float) -> np.ndarray:
    """Converts nominal dollar values into 'today's dollars' (using ages[0]
    as the base year), dividing by compound inflation. Works for 1D
    (n_years,) or 2D (n_sims, n_years) arrays -- the inflation factor
    broadcasts along the last axis."""
    base_age = ages[0]
    factors = np.array([(1 + inflation_rate) ** (a - base_age) for a in ages])
    return values / factors
