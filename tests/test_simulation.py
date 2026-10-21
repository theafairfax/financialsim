"""Accounting invariants and deterministic planning examples."""
import unittest
from dataclasses import replace
import numpy as np
from simulation import AssetClass, RealEstateAsset, SimulationConfig, IncomeChange, LargeExpense, run_monte_carlo, estimate_income_taxes


def cash(balance=0, allocation=1):
    return AssetClass('Cash', balance, allocation, 0, 0, account_type='cash', market_beta=0)


def config(**kwargs):
    values = dict(start_age=30, end_age=35, base_income=10000, income_is_post_tax=True,
                  income_growth_rate=0, expense_inflation=0, emergency_fund_months=0,
                  assets=[cash()], num_simulations=3)
    values.update(kwargs)
    return SimulationConfig(**values)


class PlanningTests(unittest.TestCase):
    def test_today_is_untouched_and_five_years_means_five_cash_flows(self):
        r = run_monte_carlo(config(assets=[cash(5000)]))
        np.testing.assert_allclose(r['net_worth'][0], [5000,15000,25000,35000,45000,55000])
        self.assertEqual(r['income_path'][0], 0)

    def test_all_capped_surplus_is_preserved(self):
        a = AssetClass('IRA',0,1,0,0,annual_contribution_cap=1000,account_type='tax_deferred')
        r = run_monte_carlo(config(assets=[a]))
        self.assertEqual(r['net_worth'][0,-1], 50000)
        self.assertEqual(r['balances'][0,-1,r['asset_names'].index('IRA')], 5000)

    def test_zero_allocation_preserves_every_dollar(self):
        r = run_monte_carlo(config(assets=[cash(allocation=0)]))
        self.assertEqual(r['net_worth'][0,-1],50000)

    def test_zero_cap_means_no_contribution(self):
        a = AssetClass('IRA',0,1,0,0,annual_contribution_cap=0,account_type='tax_deferred')
        r = run_monte_carlo(config(assets=[a]))
        self.assertEqual(r['balances'][0,-1,r['asset_names'].index('IRA')],0)
        self.assertEqual(r['net_worth'][0,-1],50000)

    def test_debt_interest_not_stock_returns(self):
        r=run_monte_carlo(config(base_income=0, initial_debt=10000, debt_interest_rate=.1))
        self.assertAlmostEqual(r['debt'][0,-1],10000*1.1**5)
        self.assertTrue((r['balances']>=0).all())

    def test_cash_shortage_creates_explicit_borrowing(self):
        r=run_monte_carlo(config(base_income=0,base_expenses=1000,debt_interest_rate=.1))
        self.assertEqual(r['debt'][0,1],1000)
        self.assertEqual(r['debt'][0,2],2100)
        self.assertEqual(r['new_borrowing'][0,2],1000)

    def test_required_and_extra_debt_payments_conserve_cash(self):
        r=run_monte_carlo(config(initial_debt=15000,debt_interest_rate=0,annual_debt_payment=1000))
        np.testing.assert_allclose(r['net_worth'][0],[-15000,-5000,5000,15000,25000,35000])
        self.assertEqual(r['debt'][0,-1],0)

    def test_emergency_reserve_precedes_stock_contributions(self):
        stock=AssetClass('Stock',0,1,0,0)
        r=run_monte_carlo(config(base_expenses=4000,emergency_fund_months=3,assets=[stock]))
        self.assertEqual(r['cash_reserve'][0,1],1000)
        self.assertEqual(r['balances'][0,1,r['asset_names'].index('Stock')],5000)

    def test_retirement_accounts_protected_until_enabled(self):
        ira=AssetClass('IRA',10000,1,0,0,liquidity='illiquid',account_type='tax_deferred')
        c=config(base_income=0,base_expenses=1000,assets=[ira],ordinary_withdrawal_tax_rate=.2)
        protected=run_monte_carlo(c)
        allowed=run_monte_carlo(replace(c,allow_retirement_withdrawals=True))
        self.assertEqual(protected['new_borrowing'][0,1],1000)
        self.assertEqual(allowed['new_borrowing'][0,1],0)
        self.assertAlmostEqual(allowed['balances'][0,1,1],10000-1000/.7)

    def test_taxable_withdrawal_only_taxes_gain(self):
        a=AssetClass('Brokerage',10000,1,0,0,cost_basis=5000,capital_gains_tax_rate=.2)
        r=run_monte_carlo(config(base_income=0,base_expenses=900,assets=[a]))
        self.assertEqual(r['balances'][0,1,1],9000)
        self.assertEqual(r['new_borrowing'][0,1],0)

    def test_underwater_mortgage_is_not_erased(self):
        p=RealEstateAsset('Home',100000,150000,0,30,appreciation_rate=0,appreciation_volatility=0,
                          property_tax_rate=0,maintenance_rate=0)
        r=run_monte_carlo(config(base_income=0,real_estate=[p]))
        self.assertEqual(r['net_worth'][0,0],-50000)
        self.assertAlmostEqual(r['real_estate_mortgage_balances'][0,1,0],145000)
        self.assertAlmostEqual(r['net_worth'][0,1],-50000)

    def test_future_purchase_creates_equity_and_charges_closing_costs(self):
        p=RealEstateAsset('Home',100000,80000,0,30,purchase_age=32,closing_cost_rate=.03,
                          appreciation_rate=0,appreciation_volatility=0,property_tax_rate=0,maintenance_rate=0)
        r=run_monte_carlo(config(base_income=0,assets=[cash(30000)],real_estate=[p]))
        self.assertEqual(r['real_estate_property_values'][0,1,0],0)
        self.assertEqual(r['property_purchase_cash'][0,2],23000)
        self.assertEqual(r['net_worth'][0,2],27000)
        self.assertEqual(r['real_estate_mortgage_balances'][0,2,0],80000)
        self.assertAlmostEqual(r['net_worth'][0,3],27000)

    def test_retirement_stops_wages_and_uses_net_benefit(self):
        r=run_monte_carlo(config(retirement_age=33,retirement_annual_net_income=2000))
        np.testing.assert_allclose(r['after_tax_income_path'],[0,10000,10000,2000,2000,2000])

    def test_dated_income_and_bill(self):
        r=run_monte_carlo(config(income_changes=[IncomeChange(32,20000)],large_expenses=[LargeExpense(32,5000)]))
        np.testing.assert_allclose(r['net_worth'][0],[0,10000,25000,45000,65000,85000])

    def test_guaranteed_shocks_have_expected_dollar_effect(self):
        r=run_monte_carlo(config(unemployment_probability=1,unemployment_months=6,
                                emergency_probability=1,emergency_cost=1000))
        self.assertEqual(r['net_worth'][0,1],4000)

    def test_fees_and_tax_drag_compound(self):
        a=replace(cash(10000),annual_fee=.01,annual_tax_drag=.02)
        r=run_monte_carlo(config(base_income=0,assets=[a]))
        self.assertAlmostEqual(r['net_worth'][0,-1],10000*(.99*.98)**5)

    def test_market_variance_is_stationary_from_first_year(self):
        a=AssetClass('Stock',10000,1,.07,.2,market_beta=.9)
        r=run_monte_carlo(config(base_income=0,assets=[a],num_simulations=10000,market_autocorrelation=.5))
        self.assertAlmostEqual(r['market_factor'][:,1].var(),1,delta=.05)
        self.assertAlmostEqual(r['market_factor'][:,4].var(),1,delta=.05)

    def test_reproducible_seed_and_stress_comparison(self):
        a=AssetClass('Stock',10000,1,.07,.2,market_beta=1)
        c=config(base_income=0,assets=[a],inflation_volatility=.02)
        a1=run_monte_carlo(c); a2=run_monte_carlo(c)
        np.testing.assert_array_equal(a1['net_worth'],a2['net_worth'])
        stress=run_monte_carlo(replace(c,stress_first_year_return=-.3))
        np.testing.assert_array_equal(a1['price_index'],stress['price_index'])
        np.testing.assert_allclose(stress['net_worth'][:,1],7000)

    def test_taxes_accept_pathwise_indexing(self):
        tax=estimate_income_taxes(np.array([75000.,150000.]),threshold_scale=np.array([1.,2.]))
        self.assertAlmostEqual(tax['federal_income_tax'][1],2*tax['federal_income_tax'][0])

    def test_positive_home_equity_does_not_hide_cash_shortages(self):
        p=RealEstateAsset('Home',100000,0,0,30,appreciation_rate=0,appreciation_volatility=0,
                          property_tax_rate=0,maintenance_rate=0)
        r=run_monte_carlo(config(base_income=0,base_expenses=1000,real_estate=[p]))
        self.assertGreater(r['net_worth'][0,-1],0)
        self.assertGreater(r['new_borrowing'][0,-1],0)

    def test_inflation_paths_are_positive_and_deflate_pathwise(self):
        r=run_monte_carlo(config(inflation_volatility=.1))
        self.assertTrue((r['price_index']>0).all())
        self.assertTrue(np.isfinite(r['net_worth']/r['price_index']).all())

    def test_retirement_unlocks_accounts_without_early_access(self):
        ira=AssetClass('IRA',10000,1,0,0,liquidity='illiquid',account_type='tax_deferred')
        r=run_monte_carlo(config(start_age=64,end_age=66,base_income=0,base_expenses=1000,
                                 assets=[ira],retirement_age=65,ordinary_withdrawal_tax_rate=.2))
        self.assertEqual(r['new_borrowing'][0,1],0)
        self.assertEqual(r['balances'][0,1,1],8750)

    def test_future_rent_uses_purchase_year_dollars(self):
        p=RealEstateAsset('Rental',100000,0,0,30,purchase_age=32,closing_cost_rate=0,
                          appreciation_rate=0,appreciation_volatility=0,property_tax_rate=0,
                          maintenance_rate=0,annual_rent=12000,vacancy_rate=0)
        r=run_monte_carlo(config(base_income=0,assets=[cash(200000)],real_estate=[p],expense_inflation=.1))
        self.assertEqual(r['real_estate_cash_flow'][0,3,0],12000)
        self.assertAlmostEqual(r['real_estate_cash_flow'][0,4,0],13200)

    def test_bad_input_rejected(self):
        for c in [config(end_age=30),config(initial_debt=-1),config(unemployment_months=13),
                  config(assets=[replace(cash(),market_beta=2)])]:
            with self.assertRaises(ValueError):run_monte_carlo(c)

if __name__=='__main__':unittest.main()
