"""Exercise actual Streamlit controls and result rendering."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def app(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'), default_timeout=30).run()
        self.assertEqual(list(app.exception),[])
        return app

    def by_label(self, elements, label):
        return next(x for x in elements if x.label==label)

    def test_default_ten_year_plan_and_stress_render(self):
        app=self.app()
        app.button[0].click().run()
        self.assertEqual(list(app.exception),[])
        self.assertEqual(list(app.error),[])
        result=list(app.session_state['scenarios'].values())[0]
        self.assertEqual(len(result['ages']),11)
        self.assertIn('stress_result',result)
        self.assertEqual(len(app.metric),8)
        self.by_label(app.checkbox,"💵 Show values in today's dollars (inflation-adjusted)").uncheck().run()
        self.assertEqual(list(app.exception),[])

    def test_five_year_post_tax_rental_and_inflation(self):
        app=self.app()
        self.by_label(app.radio,'Planning horizon').set_value('5 years')
        self.by_label(app.radio,'Income Entry Mode').set_value('Post-tax income')
        self.by_label(app.radio,'Property Setup').set_value('Rental Property / Portfolio')
        self.by_label(app.slider,'Inflation uncertainty (annual percentage points)').set_value(2.0)
        app.run()
        app.button[0].click().run()
        self.assertEqual(list(app.exception),[])
        self.assertEqual(list(app.error),[])
        result=list(app.session_state['scenarios'].values())[0]
        self.assertEqual(len(result['ages']),6)
        self.assertTrue(result['income_is_post_tax'])

    def test_retirement_and_multiple_scenarios(self):
        app=self.app()
        self.by_label(app.checkbox,'Stop employment income at retirement').check().run()
        self.by_label(app.number_input,'Retirement year-end age').set_value(33).run()
        app.button[0].click().run()
        self.assertEqual(list(app.exception),[])
        result=list(app.session_state['scenarios'].values())[0]
        self.assertEqual(result['income_path'][3],0)
        self.by_label(app.text_input,'Scenario Name').set_value('Alternative').run()
        app.button[0].click().run()
        self.by_label(app.multiselect,'Scenarios to display').set_value(['Scenario 1','Alternative']).run()
        self.assertEqual(list(app.exception),[])

if __name__=='__main__':unittest.main()
