# genetic_scheduler.py
import random
import pandas as pd

class GeneticScheduler:
    def __init__(self, skills_to_learn, hours_per_day=2, days=7):
        """
        Genetic Algorithm for Study Schedule Optimization.
        Population: Set of different Weekly Schedules.
        Gene: A specific time slot assigned to a subject.
        """
        self.skills = skills_to_learn
        if not self.skills:
            self.skills = ["Revision", "Practice"] # Fallback
        self.hours = hours_per_day
        self.days = days
        self.population_size = 20
        self.generations = 50
        self.mutation_rate = 0.1
        
        # Time slots available (e.g., Day 1 Slot 1, Day 1 Slot 2...)
        self.total_slots = self.days * self.hours

    def create_genome(self):
        """Create a random schedule (Chromosome)."""
        # Randomly fill slots with skills or 'Rest'
        genome = [random.choice(self.skills + ['Rest']) for _ in range(self.total_slots)]
        return genome

    def fitness(self, genome):
        """
        Fitness Function: Evaluates how 'good' a schedule is.
        Rules:
        1. Reward (+): If all target skills are present at least once.
        2. Penalty (-): If 'Rest' is too frequent or too rare.
        3. Penalty (-): If the same subject is repeated 3+ times in a row (Burnout).
        """
        score = 100
        
        # Rule 1: Coverage
        unique_subjects = set([g for g in genome if g != 'Rest'])
        coverage = len(unique_subjects) / len(self.skills) if self.skills else 0
        score += (coverage * 50)

        # Rule 2: Burnout Check (consecutive same subjects)
        for i in range(len(genome) - 1):
            if genome[i] == genome[i+1] and genome[i] != 'Rest':
                score -= 5 # Penalty for monotony

        return score

    def crossover(self, parent1, parent2):
        """Single Point Crossover: Combine two schedules."""
        if len(parent1) < 2: return parent1
        split = random.randint(1, len(parent1) - 1)
        child = parent1[:split] + parent2[split:]
        return child

    def mutate(self, genome):
        """Randomly change a slot in the schedule."""
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                genome[i] = random.choice(self.skills + ['Rest'])
        return genome

    def run_evolution(self):
        """Main GA Loop: Selection -> Crossover -> Mutation"""
        # 1. Initialize Population
        population = [self.create_genome() for _ in range(self.population_size)]

        for generation in range(self.generations):
            # 2. Selection (Sort by Fitness)
            population = sorted(population, key=lambda x: self.fitness(x), reverse=True)
            
            # Check if we found a perfect schedule
            if self.fitness(population[0]) >= 150:
                break

            # 3. Reproduction (Top 50% survive)
            next_gen = population[:self.population_size // 2]
            
            # Fill rest with children
            while len(next_gen) < self.population_size:
                parent1 = random.choice(next_gen)
                parent2 = random.choice(next_gen)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_gen.append(child)
            
            population = next_gen

        # Return best schedule
        return population[0]

    def format_schedule(self, best_genome):
        """Converts the gene list into a readable Table (DataFrame)."""
        week_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        schedule_map = {}
        
        slot_idx = 0
        for day in week_days:
            daily_slots = []
            for _ in range(self.hours):
                if slot_idx < len(best_genome):
                    daily_slots.append(best_genome[slot_idx])
                    slot_idx += 1
            schedule_map[day] = daily_slots
            
        # Pad if lengths differ (just in case)
        df = pd.DataFrame.from_dict(schedule_map, orient='index').transpose()
        df.columns.name = "Day"
        df.index.name = "Hour"
        return df