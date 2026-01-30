# resume_parser.py
import pdfplumber
import spacy
import re

# Load English tokenizer, tagger, parser and NER
# Ensure you ran: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Spacy model not found. Please run: python -m spacy download en_core_web_sm")

class ResumeParser:
    def __init__(self):
        # Pre-defined list of skills to look for (In a real AI, this would be a massive database)
        # This acts as our "Pattern Matching" logic for Perception
        self.known_skills = [
            "Python", "Java", "C++", "HTML", "CSS", "JavaScript", "React", "Vue",
            "Node.js", "Django", "Flask", "SQL", "MongoDB", "PostgreSQL",
            "Machine Learning", "TensorFlow", "Keras", "Pandas", "NumPy",
            "AWS", "Docker", "Git", "Communication", "Leadership"
        ]

    def extract_text_from_pdf(self, pdf_file):
        """
        Raw Perception: Converts physical file bytes into string data.
        """
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    def extract_skills(self, text):
        """
        Feature Extraction: Processing raw text to find specific 'State' variables (Skills).
        Uses NLP tokenization to match words against known skills.
        """
        doc = nlp(text)
        found_skills = set()

        # 1. Direct Phrase Matching (Simple & Fast)
        # We normalize text to lowercase for comparison
        text_lower = text.lower()
        
        for skill in self.known_skills:
            # Regex ensures we match "Java" but not "JavaScript" when looking for "Java"
            # \b denotes word boundary
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if re.search(pattern, text_lower):
                found_skills.add(skill)

        return list(found_skills)

    def get_experience_level(self, text):
        """
        Heuristic Extraction: Estimating experience years from text.
        Looks for patterns like '5 years experience' or dates.
        """
        # Simple heuristic: looking for "X years" pattern
        # This helps build the 'experience' factor of our State Vector
        exp_pattern = r"(\d+)\+?\s+years?"
        matches = re.findall(exp_pattern, text.lower())
        
        if matches:
            # Take the maximum number found as years of experience
            return max([int(m) for m in matches])
        else:
            return 0 # Default to fresher

# Testing
if __name__ == "__main__":
    # Dummy text simulation (If you don't have a PDF right now)
    dummy_resume = """
    I am a software engineer with 3 years of experience.
    I know Python, Django, and have worked with SQL databases.
    I am learning React.
    """
    
    parser = ResumeParser()
    skills = parser.extract_skills(dummy_resume)
    exp = parser.get_experience_level(dummy_resume)
    
    print(f"Perceived Skills: {skills}")
    print(f"Perceived Experience: {exp} Years")