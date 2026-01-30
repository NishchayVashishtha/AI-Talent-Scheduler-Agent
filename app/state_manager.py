# state_manager.py
import numpy as np

class CareerState:
    """
    Matches Course Requirement: Factored State Representation
    State is defined not as a single string, but as a collection of variables (Vector).
    """
    def __init__(self, current_skills, experience_years, budget_hours):
        # This is our "State Vector" components
        self.skills = set(current_skills) # Boolean vector conceptually (Present/Absent)
        self.experience = experience_years # Continuous variable
        self.study_budget = budget_hours   # Constraint variable (CSP)
    
    def to_vector(self, all_possible_skills):
        """
        Converts internal state to a mathematical vector for AI processing.
        Example: [1, 0, 1, 0...] for [Python, Java, SQL, C++]
        """
        vector = []
        for skill in all_possible_skills:
            if skill in self.skills:
                vector.append(1)
            else:
                vector.append(0)
        return np.array(vector)

    def is_goal_reached(self, target_job_skills):
        """
        Goal Test: Checks if current state satisfies the Goal State requirements.
        """
        # Check if all target skills are present in current skills
        missing = [s for s in target_job_skills if s not in self.skills]
        return len(missing) == 0, missing

    def __repr__(self):
        return f"State(Skills={len(self.skills)}, Exp={self.experience})"