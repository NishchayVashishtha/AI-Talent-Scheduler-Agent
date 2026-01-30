# inference_engine.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyEvaluator:
    def __init__(self):
        """
        Initializes the Fuzzy Inference System.
        Matches Course Requirement: Dealing with Uncertainty & Fuzzy Logic.
        """
        # 1. Define Fuzzy Variables (Antecedents & Consequents)
        # Universe of Discourse: Range of values
        self.skill_match = ctrl.Antecedent(np.arange(0, 101, 1), 'skill_match')
        self.experience = ctrl.Antecedent(np.arange(0, 11, 1), 'experience')
        self.suitability = ctrl.Consequent(np.arange(0, 11, 1), 'suitability')

        # 2. Define Membership Functions (Fuzzification)
        # Kaise define karenge ki kya "Low", "Medium", ya "High" hai?
        
        # Skill Match Categories
        self.skill_match['poor'] = fuzz.trimf(self.skill_match.universe, [0, 0, 50])
        self.skill_match['average'] = fuzz.trimf(self.skill_match.universe, [30, 50, 70])
        self.skill_match['excellent'] = fuzz.trimf(self.skill_match.universe, [60, 100, 100])

        # Experience Categories
        self.experience['junior'] = fuzz.trimf(self.experience.universe, [0, 0, 3])
        self.experience['mid'] = fuzz.trimf(self.experience.universe, [2, 5, 8])
        self.experience['senior'] = fuzz.trimf(self.experience.universe, [6, 10, 10])

        # Output Categories
        self.suitability['low'] = fuzz.trimf(self.suitability.universe, [0, 0, 5])
        self.suitability['medium'] = fuzz.trimf(self.suitability.universe, [3, 5, 7])
        self.suitability['high'] = fuzz.trimf(self.suitability.universe, [6, 10, 10])

        # 3. Define Fuzzy Rules (Inference Engine)
        # Human-like reasoning logic
        
        # Rule 1: Agar skills bekar hain, to experience matter nahi karta -> Low Suitability
        rule1 = ctrl.Rule(self.skill_match['poor'], self.suitability['low'])
        
        # Rule 2: Agar skills average hain aur experience kam hai -> Low Suitability
        rule2 = ctrl.Rule(self.skill_match['average'] & self.experience['junior'], self.suitability['low'])
        
        # Rule 3: Skills average hain par experience acha hai -> Medium Suitability
        rule3 = ctrl.Rule(self.skill_match['average'] & self.experience['mid'], self.suitability['medium'])
        
        # Rule 4: Skills excellent hain, chahe junior ho -> Medium/High (Strong Potential)
        rule4 = ctrl.Rule(self.skill_match['excellent'] & self.experience['junior'], self.suitability['medium'])
        
        # Rule 5: Skills excellent aur experience bhi senior -> High Suitability
        rule5 = ctrl.Rule(self.skill_match['excellent'] & self.experience['senior'], self.suitability['high'])

        # 4. Build Control System
        self.hiring_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        self.hiring_sim = ctrl.ControlSystemSimulation(self.hiring_ctrl)

    def evaluate_candidate(self, skill_percentage, years_exp):
        """
        Performs Fuzzification -> Inference -> Defuzzification
        """
        try:
            # Pass clean inputs
            self.hiring_sim.input['skill_match'] = float(skill_percentage)
            
            # Cap experience at 10 for logic purposes
            exp_input = 10 if years_exp > 10 else float(years_exp)
            self.hiring_sim.input['experience'] = exp_input

            # Crunch the numbers
            self.hiring_sim.compute()

            # Return Defuzzified output (Crisp Score)
            return round(self.hiring_sim.output['suitability'], 2)
            
        except Exception as e:
            print(f"Fuzzy Logic Error: {e}")
            return 0

# Test run
if __name__ == "__main__":
    evaluator = FuzzyEvaluator()
    
    # Scenario 1: Good skills (80%), Low Experience (1 year) -> Should be decent score
    score1 = evaluator.evaluate_candidate(80, 1)
    print(f"Scenario 1 (High Skill, Low Exp): {score1}/10")
    
    # Scenario 2: Low Skills (20%), High Exp (8 years) -> Should be low score
    score2 = evaluator.evaluate_candidate(20, 8)
    print(f"Scenario 2 (Low Skill, High Exp): {score2}/10")