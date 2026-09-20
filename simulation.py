"""Core Monte Carlo engine for the Lifetime Financial Simulator."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

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
    market_autocorrelation: float = 0.25
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
    for i in range(len(assets)):
        balances[:,t,i]+=final[:,i]
        if assets[i].account_type=="taxable": bases[:,t,i]+=final[:,i]

def _net_cash_from_gross(asset,gross,balance,basis,age,config):
    if asset.account_type=="tax_deferred":
        rate=config.ordinary_withdrawal_tax_rate if config.ordinary_withdrawal_tax_rate is not None else config.tax_rate
        if age<59.5: rate+=config.early_withdrawal_penalty_rate
        return gross*np.maximum(0.0,1.0-rate)
    if asset.account_type=="taxable" and balance>0:
        gain_fraction=max((balance-basis)/balance,0.0)
        cg=asset.capital_gains_tax_rate if asset.capital_gains_tax_rate is not None else config.capital_gains_tax_rate
        return gross*(1.0-gain_fraction*cg)
    return gross

def _withdraw_for_shortfall(balances,bases,t,shortfall,mask,assets,age,config):
    order=sorted(range(len(assets)),key=lambda i:0 if assets[i].liquidity=="liquid" else 1); rem=shortfall.copy()
    for i in order:
        bal=np.maximum(balances[:,t,i],0.0); bas=np.maximum(bases[:,t,i],0.0)
        if not np.any((rem>0)&mask& (bal>0)): continue
        if assets[i].account_type=="tax_deferred":
            rate=config.ordinary_withdrawal_tax_rate if config.ordinary_withdrawal_tax_rate is not None else config.tax_rate
            if age<59.5: rate+=config.early_withdrawal_penalty_rate
            net_factor=max(1.0-rate,1e-9); gross=np.minimum(bal,rem/net_factor); net=gross*net_factor
        elif assets[i].account_type=="taxable":
            gain_frac=np.where(bal>0,np.maximum((bal-bas)/bal,0.0),0.0)
            cg=assets[i].capital_gains_tax_rate if assets[i].capital_gains_tax_rate is not None else config.capital_gains_tax_rate
            net_factor=np.maximum(1.0-gain_frac*cg,1e-9); gross=np.minimum(bal,rem/net_factor); net=gross*net_factor
            ratio=np.where(bal>0,np.minimum(gross/bal,1.0),0.0); bases[:,t,i]-=bas*ratio
        else:
            gross=np.minimum(bal,rem); net=gross
        gross=np.where(mask,gross,0.0); net=np.where(mask,net,0.0)
        balances[:,t,i]-=gross; rem=np.maximum(rem-net,0.0)
    if order and rem.any(): balances[:,t,order[0]]-=rem

def _allocate_and_withdraw(balances,bases,t,leftover,assets,age,config):
    pos=np.where(leftover>0,leftover,0.0)
    if pos.any(): _allocate_positive(balances,bases,t,pos,assets)
    neg=leftover<0
    if neg.any(): _withdraw_for_shortfall(balances,bases,t,np.where(neg,-leftover,0.0),neg,assets,age,config)

def run_monte_carlo(config):
    ages=list(range(config.start_age,config.end_age+1)); ny=len(ages); ns=max(1,int(config.num_simulations))
    assets=list(config.assets) if config.assets else [AssetClass("Cash",0.0,1.0,0.0,0.0,account_type="cash",market_beta=0.0)]
    real_estate=list(config.real_estate)
    total_alloc=sum(a.allocation_pct for a in assets)
    if total_alloc>0 and abs(total_alloc-1.0)>1e-9:
        assets=[AssetClass(a.name,a.initial_balance,a.allocation_pct/total_alloc,a.expected_return,a.volatility,a.liquidity,a.annual_contribution_cap,a.account_type,a.market_beta,a.capital_gains_tax_rate,a.cost_basis) for a in assets]
    rng=np.random.default_rng(config.seed); na=len(assets); nr=len(real_estate)
    balances=np.zeros((ns,ny,na)); bases=np.zeros_like(balances)
    for i,a in enumerate(assets):
        balances[:,0,i]=a.initial_balance
        bases[:,0,i]=a.initial_balance if a.account_type=="taxable" and a.cost_basis is None else (a.cost_basis or 0.0)
    prop=np.zeros((ns,ny,nr)); mort=np.zeros((ns,ny,nr)); equity=np.zeros((ns,ny,nr)); re_cash=np.zeros((ns,ny,nr)); re_interest=np.zeros((ns,ny,nr)); re_principal=np.zeros((ns,ny,nr))
    for j,r in enumerate(real_estate):
        prop[:,0,j]=r.property_value; mort[:,0,j]=min(r.mortgage_balance,r.property_value); equity[:,0,j]=prop[:,0,j]-mort[:,0,j]
    inc=np.array([build_income_schedule(config,ages)[a] for a in ages]); exp=np.array([build_expense_schedule(config,ages)[a] for a in ages]); large_map=build_large_expense_map(config)
    large=np.array([large_map.get(a,0.0) for a in ages]); after=inc*(1-config.tax_rate); nw=np.zeros((ns,ny)); market=np.zeros((ns,ny))
    phi=float(np.clip(config.market_autocorrelation,-0.95,0.95))
    for t,age in enumerate(ages):
        if t>0:
            eps=rng.normal(size=ns); market[:,t]=phi*market[:,t-1]+np.sqrt(max(1-phi*phi,0))*eps
            for i,a in enumerate(assets):
                z=np.clip(a.market_beta,-0.999,0.999)*market[:,t]+np.sqrt(max(1-a.market_beta*a.market_beta,0))*rng.normal(size=ns)
                balances[:,t,i]=balances[:,t-1,i]*(1+_lognormal_simple_return(rng,a.expected_return,a.volatility,z)); bases[:,t,i]=bases[:,t-1,i]
            for j,r in enumerate(real_estate):
                z=np.clip(r.market_beta,-0.999,0.999)*market[:,t]+np.sqrt(max(1-r.market_beta*r.market_beta,0))*rng.normal(size=ns)
                prop[:,t,j]=prop[:,t-1,j]*(1+_lognormal_simple_return(rng,r.appreciation_rate,r.appreciation_volatility,z))
        for j,r in enumerate(real_estate):
            prev=mort[:,t-1,j] if t>0 else mort[:,0,j]
            new_bal, interest_paid, principal_paid, debt_service=_amortize_one_year(prev,r.mortgage_balance,r.mortgage_rate,r.mortgage_term_years)
            mort[:,t,j]=new_bal
            re_interest[:,t,j]=interest_paid
            re_principal[:,t,j]=principal_paid
            rent=r.annual_rent*((1+config.expense_inflation)**t)*(1-r.vacancy_rate)
            carrying=prop[:,t,j]*(r.property_tax_rate+r.maintenance_rate)+r.annual_insurance*((1+config.expense_inflation)**t)+debt_service
            re_cash[:,t,j]=rent-carrying
            equity[:,t,j]=prop[:,t,j]-mort[:,t,j]
        gross=np.full(ns,inc[t])
        if config.income_volatility>0: gross=np.maximum(gross*(1+rng.normal(0,config.income_volatility,ns)),0.0)
        leftover=gross*(1-config.tax_rate)-exp[t]-large[t]+(re_cash[:,t,:].sum(axis=1) if nr else 0.0)
        _allocate_and_withdraw(balances,bases,t,leftover,assets,age,config)
        nw[:,t]=balances[:,t,:].sum(axis=1)+(equity[:,t,:].sum(axis=1) if nr else 0.0)
    return {"ages":ages,"net_worth":nw,"balances":balances,"asset_names":[a.name for a in assets],"income_path":inc,"after_tax_income_path":after,"expense_path":exp,"large_expense_path":large,"inflation_rate":config.expense_inflation,"tax_rate":config.tax_rate,"market_factor":market,"real_estate_names":[r.name for r in real_estate],"real_estate_property_values":prop,"real_estate_mortgage_balances":mort,"real_estate_equity":equity,"real_estate_cash_flow":re_cash,"real_estate_interest_paid":re_interest,"real_estate_principal_paid":re_principal}

def percentile_summary(net_worth,percentiles=(5,25,50,75,95)):
    return {p:np.percentile(net_worth,p,axis=0) for p in percentiles}

def deflate_to_real(values,ages,inflation_rate):
    factors=np.array([(1+inflation_rate)**(a-ages[0]) for a in ages])
    if values.ndim==3: factors=factors[np.newaxis,:,np.newaxis]
    elif values.ndim==2:
        factors=factors[:,np.newaxis] if values.shape[0]==len(ages) else factors[np.newaxis,:]
    return values/factors
