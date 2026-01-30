# search_agent.py
import heapq
from knowledge_base import SkillOntology

class CareerPathPlanner:
    def __init__(self):
        self.kb = SkillOntology()
        # Cost table: Estimated weeks to learn a skill
        self.learning_costs = {
            "HTML": 1, "CSS": 2, "JavaScript": 4, "React": 4, "Vue": 3,
            "Python": 4, "Django": 6, "Flask": 3, "SQL": 3, "MongoDB": 2,
            "Machine Learning": 8, "TensorFlow": 6, "Pandas": 2, "Git": 1
        }

    def heuristic(self, current_skills, goal_skills):
        """
        Heuristic (h): Number of missing skills.
        Logic: The more skills missing, the farther we are from the goal.
        """
        missing = [s for s in goal_skills if s not in current_skills]
        return len(missing)

    def get_valid_next_skills(self, current_skills):
        """
        CSP Logic: Returns skills whose prerequisites are met.
        """
        possible_moves = []
        all_known_skills = self.learning_costs.keys()
        
        for skill in all_known_skills:
            if skill in current_skills:
                continue 
            
            prereqs = self.kb.get_prerequisites(skill)
            if all(p in current_skills for p in prereqs):
                possible_moves.append(skill)
                
        return possible_moves

    def plan_career_path(self, start_skills, goal_skills):
        """
        A* Search with Explainability Trace.
        Returns: Path, Total Cost, and Reasoning Log.
        """
        start_tuple = tuple(sorted(start_skills))
        
        open_set = []
        initial_h = self.heuristic(start_skills, goal_skills)
        
        # Priority Queue stores: (f_score, g_score, current_skills_tuple, path)
        heapq.heappush(open_set, (initial_h, 0, start_tuple, []))
        
        visited = set()
        visited.add(start_tuple)

        # Explainability: Log every step the AI takes
        search_trace = []

        while open_set:
            f, g, current_skills, path = heapq.heappop(open_set)
            current_skills_set = set(current_skills)

            # Log the decision
            search_trace.append({
                "step_type": "Expanded Node",
                "skills": list(current_skills),
                "g_score": g,
                "h_score": f - g, # derived from f = g + h
                "f_score": f,
                "message": f"Explored state with {len(current_skills)} skills. Cost so far: {g}"
            })

            # 1. Goal Test
            missing_skills = [s for s in goal_skills if s not in current_skills_set]
            if not missing_skills:
                return path, g, search_trace # Return trace as well

            # 2. Generate Successors
            valid_next_skills = self.get_valid_next_skills(current_skills_set)
            
            for skill in valid_next_skills:
                new_skills = list(current_skills) + [skill]
                new_skills_set = set(new_skills)
                new_skills_tuple = tuple(sorted(new_skills))
                
                if new_skills_tuple in visited:
                    continue
                
                step_cost = self.learning_costs.get(skill, 2)
                new_g = g + step_cost
                new_h = self.heuristic(new_skills_set, goal_skills)
                new_f = new_g + new_h
                
                new_path = path + [skill]
                
                heapq.heappush(open_set, (new_f, new_g, new_skills_tuple, new_path))
                visited.add(new_skills_tuple)

        return None, 0, search_trace