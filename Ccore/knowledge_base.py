# knowledge_base.py
import networkx as nx

class SkillOntology:
    def __init__(self):
        # Graph-based Knowledge Representation
        # Concept: Hierarchical Taxonomy (Parent -> Child relationships)
        self.graph = nx.DiGraph()
        self.build_knowledge_base()

    def build_knowledge_base(self):
        """
        Constructs the domain knowledge using a Directed Graph.
        Matches Course Requirement: Knowledge Representation (Ontology)
        """
        # 1. Root Domains
        self.graph.add_node("CS_Student", type="Role")
        
        # 2. Web Development Branch
        self.graph.add_edge("CS_Student", "Web Development")
        self.graph.add_edge("Web Development", "Frontend")
        self.graph.add_edge("Web Development", "Backend")
        
        # Frontend Skills
        self.graph.add_edge("Frontend", "HTML")
        self.graph.add_edge("Frontend", "CSS")
        self.graph.add_edge("Frontend", "JavaScript")
        self.graph.add_edge("JavaScript", "React")  # React requires JS
        self.graph.add_edge("JavaScript", "Vue")
        
        # Backend Skills
        self.graph.add_edge("Backend", "Python")
        self.graph.add_edge("Backend", "Node.js")
        self.graph.add_edge("Backend", "Databases")
        self.graph.add_edge("Python", "Django")     # Django requires Python
        self.graph.add_edge("Python", "Flask")
        self.graph.add_edge("Databases", "SQL")
        self.graph.add_edge("Databases", "MongoDB")

        # 3. Data Science Branch
        self.graph.add_edge("CS_Student", "Data Science")
        self.graph.add_edge("Data Science", "Machine Learning")
        self.graph.add_edge("Data Science", "Data Analysis")
        self.graph.add_edge("Machine Learning", "Python") # Shared dependency
        self.graph.add_edge("Machine Learning", "TensorFlow")
        self.graph.add_edge("Data Analysis", "Pandas")

        

    def get_prerequisites(self, skill):
        """
        Inference Rule: To learn a child skill, you ideally need the parent skill.
        """
        return list(self.graph.predecessors(skill))

    def get_related_skills(self, skill):
        """
        Reasoning: Finds siblings (e.g., if you know React, Vue is related).
        """
        try:
            parent = list(self.graph.predecessors(skill))[0]
            return list(self.graph.successors(parent))
        except IndexError:
            return []

# Test run (sirf check karne ke liye)
if __name__ == "__main__":
    kb = SkillOntology()
    print("Prerequisites for Django:", kb.get_prerequisites("Django"))
    print("Skills related to React:", kb.get_related_skills("React"))