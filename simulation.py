"""Core Monte Carlo engine for the Lifetime Financial Simulator."""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


FEDERAL_TAX_BRACKETS_2026 = {
    "Single": [(12400, 0.10), (50400, 0.12), (105700, 0.22), (201775, 0.24), (256225, 0.32), (640600, 0.35), (float("inf"), 0.37)],
    "Married Filing Jointly": [(24800, 0.10), (100800, 0.12), (211400, 0.22), (403550, 0.24), (512450, 0.32), (768700, 0.35), (float("inf"), 0.37)],
    "Married Filing Separately": [(12400, 0.10), (50400, 0.12), (105700, 0.22), (201775, 0.24), (256225, 0.32), (384350, 0.35), (float("inf"), 0.37)],
    "Head of Household": [(17700, 0.10), (67450, 0.12), (105700, 0.22), (201750, 0.24), (256200, 0.32), (640600, 0.35), (float("inf"), 0.37)],
}
STANDARD_DEDUCTION_2026 = {
    "Single": 16100,
    "Married Filing Jointly": 32200,
    "Married Filing Separately": 16100,
    "Head of Household": 24150,
}

def estimate_income_taxes(gross_income, filing_status="Single", state_effective_rate=0.0, deductions=0.0, threshold_scale=1.0):
    """Estimate 2026 U.S. federal income tax, employee FICA, and state income tax."""
    gross = np.maximum(np.asarray(gross_income, dtype=float), 0.0)
    status = filing_status if filing_status in FEDERAL_TAX_BRACKETS_2026 else "Single"
    deduction = STANDARD_DEDUCTION_2026[status] * threshold_scale + max(float(deductions), 0.0)
    taxable = np.maximum(gross - deduction, 0.0)
    federal = np.zeros_like(taxable)
    lower = 0.0
    for upper, rate in FEDERAL_TAX_BRACKETS_2026[status]:
        upper = upper * threshold_scale
        federal += np.maximum(np.minimum(taxable, upper) - lower, 0.0) * rate
        lower = upper
    social_security = np.minimum(gross, 184500.0 * threshold_scale) * 0.062
    medicare = gross * 0.0145
    threshold = 250000.0 if status == "Married Filing Jointly" else 125000.0 if status == "Married Filing Separately" else 200000.0
    additional_medicare = np.maximum(gross - threshold, 0.0) * 0.009
    state = gross * np.clip(float(state_effective_rate), 0.0, 1.0)
    total = federal + social_security + medicare + additional_medicare + state
    return {
        "federal_income_tax": federal,
        "payroll_tax": social_security + medicare + additional_medicare,
        "state_income_tax": state,
        "total_tax": total,
        "after_tax_income": np.maximum(gross - total, 0.0),
    }

@dataclass
class IncomeChange:
    age: int
    new_annual_income: float
    description: str = ""

@dataclass
class ExpenseChange:
    age: int
    new_annual_expenses: float
    description: str = ""

@dataclass
class LargeExpense:
    age: int
    amount: float
    description: str = ""

@dataclass
class AssetClass:
    name: str
    initial_balance: float
    allocation_pct: float
    expected_return: float
    volatility: float
    liquidity: str = "liquid"
    annual_contribution_cap: Optional[float] = None
    account_type: str = "taxable"  # cash, taxable, tax_deferred, tax_free
    market_beta: float = 0.7
    capital_gains_tax_rate: Optional[float] = None
    cost_basis: Optional[float] = None
    annual_fee: float = 0.0
    annual_tax_drag: float = 0.0

@dataclass
class RealEstateAsset:
    name: str
    property_value: float
    mortgage_balance: float
    mortgage_rate: float
    mortgage_term_years: int
    allocation_pct: float = 0.0
    appreciation_rate: float = 0.03
    appreciation_volatility: float = 0.08
    market_beta: float = 0.35
    annual_rent: float = 0.0
    vacancy_rate: float = 0.05
    property_tax_rate: float = 0.01
    annual_insurance: float = 0.0
    maintenance_rate: float = 0.01
    purchase_age: Optional[int] = None
    closing_cost_rate: float = 0.03

@dataclass
class SimulationConfig:
    start_age: int
    end_age: int
    base_income: float
    income_growth_rate: float = 0.02
    income_changes: List[IncomeChange] = field(default_factory=list)
    income_volatility: float = 0.0
    tax_rate: float = 0.0
    ordinary_withdrawal_tax_rate: Optional[float] = None
    capital_gains_tax_rate: float = 0.15
    early_withdrawal_penalty_rate: float = 0.10
    base_expenses: float = 0.0
    expense_inflation: float = 0.03
    expense_changes: List[ExpenseChange] = field(default_factory=list)
    large_expenses: List[LargeExpense] = field(default_factory=list)
    assets: List[AssetClass] = field(default_factory=list)
    real_estate: List[RealEstateAsset] = field(default_factory=list)
    income_is_post_tax: bool = False
    filing_status: str = "Single"
    state_effective_tax_rate: float = 0.0
    additional_deductions: float = 0.0
    lost_income_pct: float = 0.0
    market_autocorrelation: float = 0.0
    inflation_volatility: float = 0.0
    expense_volatility: float = 0.0
    unemployment_probability: float = 0.0
    unemployment_months: float = 3.0
    emergency_probability: float = 0.0
    emergency_cost: float = 5000.0
    initial_debt: float = 0.0
    debt_interest_rate: float = 0.10
    annual_debt_payment: float = 0.0
    emergency_fund_months: float = 3.0
    allow_retirement_withdrawals: bool = False
    retirement_age: Optional[int] = None
    retirement_annual_net_income: float = 0.0
    index_tax_thresholds: bool = True
    stress_first_year_return: Optional[float] = None
    num_simulations: int = 1000
    seed: Optional[int] = 42

def build_income_schedule(config, ages):
    changes={c.age:c.new_annual_income for c in config.income_changes}; out={}; cur=config.base_income
    for i,age in enumerate(ages):
        if age in changes: cur=changes[age]
        elif i>0: cur*=1+config.income_growth_rate
        out[age]=cur
    return out

def build_expense_schedule(config, ages):
    changes={c.age:c.new_annual_expenses for c in config.expense_changes}; out={}; cur=config.base_expenses
    for i,age in enumerate(ages):
        if age in changes: cur=changes[age]
        elif i>0: cur*=1+config.expense_inflation
        out[age]=cur
    return out

def build_large_expense_map(config):
    out={}
    for e in config.large_expenses: out[e.age]=out.get(e.age,0.0)+e.amount
    return out

def _lognormal_simple_return(rng, mean, vol, z):
    if vol <= 0: return np.full_like(z, mean, dtype=float)
    gross_mean=max(1.0+mean,1e-9)
    sigma2=np.log(1.0+(vol*vol)/(gross_mean*gross_mean))
    sigma=np.sqrt(max(sigma2,0.0))
    mu=np.log(gross_mean)-0.5*sigma2
    return np.exp(mu+sigma*z)-1.0

def _mortgage_payment(balance, annual_rate, term_years):
    n=max(int(term_years*12),1)
    if balance<=0: return 0.0
    r=annual_rate/12.0
    if r==0: return balance/n
    return balance*r/(1-(1+r)**(-n))

def _amortize_one_year(balance, original_balance, annual_rate, term_years):
    payment=_mortgage_payment(original_balance,annual_rate,term_years)
    bal=balance.copy(); interest_paid=np.zeros_like(bal); principal_paid=np.zeros_like(bal)
    r=annual_rate/12.0
    for _ in range(12):
        interest=bal*r
        actual=np.minimum(bal+interest,payment)
        principal=np.minimum(bal,np.maximum(actual-interest,0.0))
        interest_paid+=np.minimum(interest,actual)
        principal_paid+=principal
        bal=np.maximum(bal-principal,0.0)
    return bal, interest_paid, principal_paid, interest_paid+principal_paid

def _allocate_positive(balances,bases,t,leftover,assets):
    n_sims=balances.shape[0]; desired=np.zeros((n_sims,len(assets)))
    for i,a in enumerate(assets): desired[:,i]=leftover*a.allocation_pct
    final=desired.copy(); overflow=np.zeros(n_sims); uncapped=[]; share_sum=0.0
    for i,a in enumerate(assets):
        if a.annual_contribution_cap is not None:
            x=np.minimum(desired[:,i],a.annual_contribution_cap); overflow+=desired[:,i]-x; final[:,i]=x
        else: uncapped.append(i); share_sum+=a.allocation_pct
    if uncapped and overflow.any():
        for i in uncapped: final[:,i]+=overflow*((assets[i].allocation_pct/share_sum) if share_sum>0 else 1/len(uncapped))
    # Preserve savings even when every destination is capped or allocations are zero.
    residual = np.maximum(leftover - final.sum(axis=1), 0.0)
    cash_index = next(i for i,a in enumerate(assets) if a.account_type == "cash")
    final[:,cash_index] += residual
    for i in range(len(assets)):
        balances[:,t,i]+=final[:,i]
        if assets[i].account_type=="taxable": bases[:,t,i]+=final[:,i]

def _withdraw_available(balances, bases, t, need, assets, age, config):
    """Return cash raised and remaining shortfall; never turn investments into debt."""
    remaining = need.copy()
    order = sorted(range(len(assets)), key=lambda i: (
        assets[i].account_type != "cash", assets[i].liquidity != "liquid"))
    for i in order:
        asset = assets[i]
        can_access = config.allow_retirement_withdrawals or (config.retirement_age is not None and age >= config.retirement_age)
        if asset.liquidity != "liquid" and not can_access:
            continue
        if asset.account_type in ("tax_deferred", "tax_free") and not can_access:
            continue
        balance = balances[:, t, i]
        if asset.account_type == "tax_deferred":
            rate = config.ordinary_withdrawal_tax_rate
            rate = config.tax_rate if rate is None else rate
            rate += config.early_withdrawal_penalty_rate if age < 59.5 else 0
            factor = np.full_like(balance, max(1 - rate, 1e-9))
        elif asset.account_type == "taxable":
            gain_fraction = np.divide(np.maximum(balance-bases[:, t, i], 0), balance,
                                      out=np.zeros_like(balance), where=balance > 0)
            rate = asset.capital_gains_tax_rate
            rate = config.capital_gains_tax_rate if rate is None else rate
            factor = np.maximum(1 - gain_fraction * rate, 1e-9)
        else:
            factor = np.ones_like(balance)
        gross = np.minimum(balance, remaining / factor)
        fraction = np.divide(gross, balance, out=np.zeros_like(balance), where=balance > 0)
        bases[:, t, i] *= 1 - fraction
        balances[:, t, i] -= gross
        remaining = np.maximum(remaining - gross * factor, 0)
    return need - remaining, remaining


def _validate(config):
    if config.end_age <= config.start_age:
        raise ValueError("The horizon must end after the current age.")
    if not 1 <= config.num_simulations <= 10000:
        raise ValueError("Use 1 to 10,000 simulations.")
    for name in ("base_income", "base_expenses", "initial_debt", "annual_debt_payment",
                 "emergency_fund_months", "emergency_cost", "inflation_volatility",
                 "expense_volatility", "income_volatility", "retirement_annual_net_income"):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    for name in ("unemployment_probability", "emergency_probability", "lost_income_pct", "debt_interest_rate"):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    if not 0 <= config.unemployment_months <= 12 or config.expense_inflation <= -1:
        raise ValueError("Check unemployment months and inflation.")
    for name in ("income_growth_rate", "expense_inflation", "market_autocorrelation",
                 "capital_gains_tax_rate", "early_withdrawal_penalty_rate", "tax_rate",
                 "state_effective_tax_rate", "additional_deductions"):
        if not np.isfinite(getattr(config, name)):
            raise ValueError(f"{name} must be finite.")
    for name in ("capital_gains_tax_rate", "early_withdrawal_penalty_rate", "tax_rate", "state_effective_tax_rate"):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    if config.ordinary_withdrawal_tax_rate is not None and not 0 <= config.ordinary_withdrawal_tax_rate <= 1:
        raise ValueError("Withdrawal tax rate must be between 0 and 1.")
    if config.stress_first_year_return is not None and not -1 < config.stress_first_year_return <= 0:
        raise ValueError("Stress decline must be between -100% and 0%.")
    if config.income_growth_rate <= -1:
        raise ValueError("Income growth must exceed -100%.")
    for event in [*config.income_changes, *config.expense_changes, *config.large_expenses]:
        if not config.start_age < event.age <= config.end_age:
            raise ValueError("Dated changes and bills must occur after today and within the horizon.")
        amount = event.new_annual_income if isinstance(event, IncomeChange) else event.new_annual_expenses if isinstance(event, ExpenseChange) else event.amount
        if not np.isfinite(amount) or amount < 0:
            raise ValueError("Dated changes and bills must have finite, nonnegative amounts.")
    for asset in config.assets:
        if not all(np.isfinite(v) for v in (asset.initial_balance, asset.allocation_pct,
                                            asset.expected_return, asset.volatility, asset.market_beta,
                                            asset.annual_fee, asset.annual_tax_drag)):
            raise ValueError("Asset assumptions must be finite.")
        if min(asset.initial_balance, asset.allocation_pct, asset.volatility) < 0 or asset.expected_return <= -1:
            raise ValueError("Asset balances, allocations and volatility must be nonnegative; return must exceed -100%.")
        if abs(asset.market_beta) > 1 or not 0 <= asset.annual_fee < 1 or not 0 <= asset.annual_tax_drag < 1:
            raise ValueError("Check factor loadings, fees and annual tax drag.")
        if asset.account_type not in ("cash", "taxable", "tax_deferred", "tax_free"):
            raise ValueError("Unknown account type.")
        if asset.annual_contribution_cap is not None and (not np.isfinite(asset.annual_contribution_cap) or asset.annual_contribution_cap < 0):
            raise ValueError("Contribution caps cannot be negative.")
        if asset.capital_gains_tax_rate is not None and not 0 <= asset.capital_gains_tax_rate <= 1:
            raise ValueError("Capital gains tax rates must be between 0 and 1.")
        if asset.cost_basis is not None and (not np.isfinite(asset.cost_basis) or asset.cost_basis < 0):
            raise ValueError("Cost basis must be finite and nonnegative.")
    for prop in config.real_estate:
        if not all(np.isfinite(v) for v in (prop.property_value, prop.mortgage_balance,
                    prop.mortgage_rate, prop.appreciation_rate, prop.appreciation_volatility,
                    prop.market_beta, prop.annual_rent, prop.vacancy_rate, prop.property_tax_rate,
                    prop.annual_insurance, prop.maintenance_rate, prop.closing_cost_rate)):
            raise ValueError("Property assumptions must be finite.")
        if prop.appreciation_rate <= -1 or abs(prop.market_beta) > 1 or min(prop.annual_rent,
                  prop.property_tax_rate, prop.annual_insurance, prop.maintenance_rate, prop.closing_cost_rate) < 0:
            raise ValueError("Check property growth, costs and market loading.")
        if prop.purchase_age is not None and prop.purchase_age > config.end_age:
            raise ValueError("Future property purchases must fall within the horizon.")
        if min(prop.property_value, prop.mortgage_balance, prop.mortgage_rate, prop.appreciation_volatility) < 0:
            raise ValueError("Property values, debt, rates and volatility cannot be negative.")
        if prop.mortgage_term_years < 1 or not 0 <= prop.vacancy_rate <= 1:
            raise ValueError("Check mortgage term and vacancy.")
        if prop.purchase_age is not None and prop.purchase_age > config.start_age and prop.mortgage_balance > prop.property_value:
            raise ValueError("A future purchase needs a mortgage no greater than its purchase price.")


def run_monte_carlo(config):
    """Annual planning model. Index 0 is today's untouched balance sheet.

    Events at age A occur in the year ending at A. First-year wages/expenses
    use today's baseline. Contributions are deposited at year end.
    """
    _validate(config)
    from dataclasses import replace
    ages = list(range(config.start_age, config.end_age + 1))
    ny, ns = len(ages), int(config.num_simulations)
    assets = [replace(a) for a in config.assets]
    # Dedicated uncapped cash guarantees conservation of every dollar.
    cash_index = next((i for i,a in enumerate(assets) if a.account_type == "cash" and
                       a.liquidity == "liquid" and a.annual_contribution_cap is None), None)
    if cash_index is None:
        assets.append(AssetClass("Unallocated cash", 0, 0, 0, 0, account_type="cash", market_beta=0))
        cash_index = len(assets)-1
    total = sum(a.allocation_pct for a in assets)
    if total > 0:
        assets = [replace(a, allocation_pct=a.allocation_pct/total) for a in assets]
    # Put the uncapped reserve first so cap overflow never lands in a capped account.
    assets.insert(0, assets.pop(cash_index))
    cash_index = 0
    rng = np.random.default_rng(config.seed)
    shape = (ns, ny)
    balances = np.zeros((*shape, len(assets)))
    bases = np.zeros_like(balances)
    for i,a in enumerate(assets):
        balances[:,0,i] = a.initial_balance
        bases[:,0,i] = a.initial_balance if a.cost_basis is None else a.cost_basis
    properties = config.real_estate
    prop = np.zeros((*shape, len(properties)))
    mort = np.zeros_like(prop)
    equity = np.zeros_like(prop)
    re_cash = np.zeros_like(prop)
    re_interest = np.zeros_like(prop)
    re_principal = np.zeros_like(prop)
    for j,p in enumerate(properties):
        if p.purchase_age is None or p.purchase_age <= config.start_age:
            prop[:,0,j], mort[:,0,j] = p.property_value, p.mortgage_balance
    equity[:,0,:] = prop[:,0,:] - mort[:,0,:]
    debt = np.zeros(shape); debt[:,0] = config.initial_debt
    price_index = np.ones(shape)
    income = np.zeros(shape); expenses = np.zeros(shape); take_home_path = np.zeros(shape)
    lost = np.zeros(shape); emergency = np.zeros(shape); borrowing = np.zeros(shape)
    interest_path = np.zeros(shape); debt_paid = np.zeros(shape); surplus_path = np.zeros(shape)
    large_path = np.zeros(ny); purchase_cash = np.zeros(shape)
    liquid = np.zeros(shape); cash = np.zeros(shape); reserve_gap = np.zeros(shape)
    nw = np.zeros(shape); market = np.zeros(shape)
    market[:,0] = rng.normal(size=ns)  # stationary initialization
    current_expense = np.full(ns, config.base_expenses, dtype=float)
    annual_income = config.base_income
    income_changes = {c.age:c.new_annual_income for c in config.income_changes}
    expense_changes = {c.age:c.new_annual_expenses for c in config.expense_changes}
    large_map = build_large_expense_map(config)
    liquid_indices = [i for i,a in enumerate(assets) if a.liquidity == "liquid" and
                      a.account_type not in ("tax_deferred", "tax_free")]
    cash_indices = [i for i,a in enumerate(assets) if a.account_type == "cash" and a.liquidity == "liquid"]
    def snapshot(t):
        nw[:,t] = balances[:,t,:].sum(axis=1) + equity[:,t,:].sum(axis=1) - debt[:,t]
        liquid[:,t] = balances[:,t,liquid_indices].sum(axis=1)
        cash[:,t] = balances[:,t,cash_indices].sum(axis=1)
        baseline = current_expense if t else np.full(ns, config.base_expenses)
        reserve_gap[:,t] = np.maximum(baseline * config.emergency_fund_months / 12 - cash[:,t], 0)
    snapshot(0)
    phi = float(np.clip(config.market_autocorrelation, -0.95, 0.95))
    for t in range(1, ny):
        age = ages[t]
        inflation = np.maximum(rng.normal(config.expense_inflation, config.inflation_volatility, ns), -0.95)
        price_index[:,t] = price_index[:,t-1] * (1 + inflation)
        market[:,t] = phi * market[:,t-1] + np.sqrt(1-phi*phi) * rng.normal(size=ns)
        for i,a in enumerate(assets):
            z = a.market_beta*market[:,t] + np.sqrt(1-a.market_beta*a.market_beta)*rng.normal(size=ns)
            returns = _lognormal_simple_return(rng, a.expected_return, a.volatility, z)
            if t == 1 and config.stress_first_year_return is not None:
                returns = np.full(ns, max(config.stress_first_year_return, -0.99) * abs(a.market_beta))
            balances[:,t,i] = balances[:,t-1,i] * (1+returns) * (1-a.annual_fee) * (1-a.annual_tax_drag)
            bases[:,t,i] = bases[:,t-1,i]
        debt[:,t] = debt[:,t-1] * (1+config.debt_interest_rate)
        interest_path[:,t] = debt[:,t] - debt[:,t-1]
        if age in income_changes:
            annual_income = income_changes[age]
        elif t > 1:
            annual_income *= 1+config.income_growth_rate
        if age in expense_changes:
            current_expense = np.full(ns, expense_changes[age], dtype=float)
        elif t > 1:
            current_expense *= 1+inflation
        expenses[:,t] = np.maximum(current_expense * (1+rng.normal(0, config.expense_volatility, ns)), 0)
        gross = np.maximum(annual_income*(1+rng.normal(0, config.income_volatility, ns)), 0)
        unemployed = rng.random(ns) < config.unemployment_probability
        gross *= 1-unemployed * config.unemployment_months/12
        retired = config.retirement_age is not None and age >= config.retirement_age
        if retired:
            gross[:] = 0
            take_home = config.retirement_annual_net_income * price_index[:,t-1]
        else:
            scale = price_index[:,t-1] if config.index_tax_thresholds else 1.0
            take_home = gross if config.income_is_post_tax else estimate_income_taxes(
                gross, config.filing_status, config.state_effective_tax_rate,
                config.additional_deductions, scale)["after_tax_income"]
        income[:,t], take_home_path[:,t] = gross, take_home
        large_path[t] = large_map.get(age, 0)
        emergency[:,t] = (rng.random(ns) < config.emergency_probability) * config.emergency_cost * price_index[:,t-1]
        for j,p in enumerate(properties):
            purchased = p.purchase_age is not None and p.purchase_age > config.start_age
            if purchased and age < p.purchase_age:
                continue
            new_purchase = purchased and age == p.purchase_age
            if new_purchase:
                prop[:,t,j], mort[:,t,j] = p.property_value, p.mortgage_balance
                purchase_cash[:,t] += p.property_value-p.mortgage_balance+p.property_value*p.closing_cost_rate
            else:
                z = p.market_beta*market[:,t]+np.sqrt(max(1-p.market_beta*p.market_beta,0))*rng.normal(size=ns)
                prop[:,t,j] = prop[:,t-1,j]*(1+_lognormal_simple_return(rng,p.appreciation_rate,p.appreciation_volatility,z))
                mort[:,t,j] = mort[:,t-1,j]
            # Purchases occur at year end; ownership costs begin the following year.
            if not new_purchase:
                mort[:,t,j], re_interest[:,t,j], re_principal[:,t,j], service = _amortize_one_year(
                    mort[:,t,j], p.mortgage_balance, p.mortgage_rate, p.mortgage_term_years)
                rent_price_index = price_index[:,t-1]
                if purchased:
                    rent_price_index = rent_price_index / price_index[:,p.purchase_age-config.start_age]
                rent = p.annual_rent*rent_price_index*(1-p.vacancy_rate)
                carrying = prop[:,t,j]*(p.property_tax_rate+p.maintenance_rate)+p.annual_insurance*rent_price_index+service
                re_cash[:,t,j] = rent-carrying
            equity[:,t,j] = prop[:,t,j]-mort[:,t,j]
        scheduled = np.minimum(debt[:,t], config.annual_debt_payment)
        available = take_home-expenses[:,t]-large_path[t]-emergency[:,t]+re_cash[:,t,:].sum(axis=1)-purchase_cash[:,t]-scheduled
        debt[:,t] -= scheduled
        debt_paid[:,t] = scheduled
        lost[:,t] = np.maximum(available,0)*config.lost_income_pct
        available -= lost[:,t]
        surplus_path[:,t] = available
        _, shortfall = _withdraw_available(balances,bases,t,np.maximum(-available,0),assets,age,config)
        debt[:,t] += shortfall
        borrowing[:,t] = shortfall
        positive = np.maximum(available,0)
        extra_payment = np.minimum(positive,debt[:,t])
        debt[:,t] -= extra_payment; positive -= extra_payment
        debt_paid[:,t] += extra_payment
        # Refill the reserve before allocating longer-term contributions.
        cash_total = balances[:,t,cash_indices].sum(axis=1)
        reserve = np.minimum(positive,np.maximum(current_expense*config.emergency_fund_months/12-cash_total,0))
        balances[:,t,cash_index] += reserve; positive -= reserve
        _allocate_positive(balances,bases,t,positive,assets)
        snapshot(t)
    return {
        "ages":ages, "net_worth":nw, "balances":balances, "asset_names":[a.name for a in assets],
        "income_path":income.mean(axis=0), "after_tax_income_path":take_home_path.mean(axis=0),
        "expense_path":expenses.mean(axis=0), "large_expense_path":large_path,
        "lost_income":lost, "expected_lost_income_path":lost.mean(axis=0),
        "lost_income_pct":config.lost_income_pct, "income_is_post_tax":config.income_is_post_tax,
        "inflation_rate":config.expense_inflation, "tax_rate":config.tax_rate, "market_factor":market,
        "price_index":price_index, "liquid_assets":liquid, "cash_reserve":cash,
        "reserve_gap":reserve_gap, "debt":debt, "new_borrowing":borrowing,
        "debt_interest":interest_path, "debt_payments":debt_paid,
        "emergency_expenses":emergency, "property_purchase_cash":purchase_cash,
        "cash_flow":surplus_path, "real_estate_names":[p.name for p in properties],
        "real_estate_property_values":prop, "real_estate_mortgage_balances":mort,
        "real_estate_equity":equity, "real_estate_cash_flow":re_cash,
        "real_estate_interest_paid":re_interest, "real_estate_principal_paid":re_principal,
    }


def percentile_summary(net_worth,percentiles=(5,25,50,75,95)):
    return {p:np.percentile(net_worth,p,axis=0) for p in percentiles}

def deflate_to_real(values,ages,inflation_rate):
    factors=np.array([(1+inflation_rate)**(a-ages[0]) for a in ages])
    if values.ndim==3: factors=factors[np.newaxis,:,np.newaxis]
    elif values.ndim==2:
        factors=factors[:,np.newaxis] if values.shape[0]==len(ages) else factors[np.newaxis,:]
    return values/factors
