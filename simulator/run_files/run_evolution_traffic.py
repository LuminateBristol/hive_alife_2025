import sys
import os
import random
import numpy as np
from itertools import combinations
from collections import defaultdict

# We need to setup parent directories to properly import other modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from simulator_files import *
from lib import Config

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, parent_dir)

from simulator import CFG_FILES

###### Experiment parameters ######
export_data = False
verbose = True
ex_id = 'exp_3_traffic'

###### Config class ######
default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['exp_setup']
map_file = CFG_FILES['map']

###### GA Hyperparameters ######
POPULATION_SIZE = 10
NUM_GENERATIONS = 20
MUTATION_RATE = 0.4             # Set to 0.0 for no mutation
CROSSOVER_RATE = 0.2            # Set to 0.0 for no crossover
ELITISM_RATE = 0.2
TOURNAMENT_SIZE = 5
DEPENDENCY_UPDATE_INTERVAL = 2  # Update dependency tracking every 5 generations
NUM_ITERATIONS = 10             # Number of iterations ran per generation


class GeneticOptimisation:
    def __init__(self):
        self.cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id, map=map_file)
        self.num_iterations = NUM_ITERATIONS

        # Fitness function weights
        self.performance_weight = 1
        self.communication_weight = 0.2

        # Track dependencies dynamically
        self.dependency_groups = {}

        self.result_file_path = self.create_result_file()

    def create_result_file(self):
        """Generate a filename containing key GA parameters."""
        filename = f"GA_results_pop{POPULATION_SIZE}_gen{NUM_GENERATIONS}_cross{CROSSOVER_RATE}_elit{ELITISM_RATE}_tour{TOURNAMENT_SIZE}.txt"
        file_path = os.path.join(current_dir, filename)
        info_types = self.get_hive_mind_info_types()

        # Write headers + info type mapping in the first line
        with open(file_path, 'w') as file:
            file.write("INFO_TYPES: " + "\t".join(info_types) + "\n")  # First line: Info type labels
            file.write("generation\tgenome\tfitness\n")  # Headers

        return file_path

    def save_results(self, generation, population, fitness_scores):
        """Save results to a file including generation, genome, and fitness."""
        with open(self.result_file_path, 'a') as file:
            for i in range(len(population)):
                genome = population[i]
                fitness = fitness_scores[i]
                file.write(f"{generation}\t{genome}\t{fitness}\n")

    def get_hive_mind_info_types(self):
        """Retrieve all nodes with 'weight' attribute and group them by their 'type' attribute."""
        sim = Simulator(self.cfg_obj, verbose=verbose)
        hive_mind = sim.optimisation_hive_mind.graph

        nodes_with_weight = [n for n, d in hive_mind.nodes(data=True) if 'weight' in d]
        groups = {}
        for node in nodes_with_weight:
            node_type = hive_mind.nodes[node]['type']
            if node_type not in groups:
                groups[node_type] = []
            groups[node_type].append(node)

        return list(groups.keys())  # Return list of unique info types (genes)

    def run_simulation(self, selected_info_types):
        """Run the simulation multiple times and return the average completion time."""
        total_time = 0
        run_times = []

        for i in range(self.num_iterations):
            sim = Simulator(self.cfg_obj, verbose=verbose)

            # Update weights for selected info types
            for node in sim.Hive_Mind.graph.nodes(data=True):
                node_id, attributes = node
                if 'type' in attributes and attributes['type'] in selected_info_types:
                    sim.Hive_Mind.graph.nodes[node_id]['weight'] = 1

            run_time = sim.run(iteration=i)
            total_time += run_time
            run_times.append(run_time)

        average_time = total_time / self.num_iterations
        return average_time

    def evaluate_fitness(self, genome):
        """
        Evaluate fitness of a given genome (set of selected info types).
        Genome = population = a list of selected information types to be ran for this run of th simulation.
        """
        avg_time = self.run_simulation(genome)
        w = len(genome)  # Number of info types
        fitness = self.performance_weight * avg_time + self.communication_weight * w
        return fitness

    def initialize_population(self, num_info_types):
        """
        Generate an initial random population of info type selections.
        A population is a list of information types that will be set to one for this run.
        """
        return [[random.choice([0, 1]) for _ in range(num_info_types)] for _ in range(POPULATION_SIZE)]

    def roulette_wheel_selection(self, population, fitness_scores):
        """Select parents using fitness-proportionate selection."""
        total_fitness = sum(fitness_scores)
        selection_probs = [1 - (fitness_scores[i] / total_fitness) for i in range(len(population))]  # Invert since lower is better
        selected = random.choices(population, weights=selection_probs, k=POPULATION_SIZE)
        return selected

    def tournament_selection(self, population, fitness_scores):
        """Select individuals for crossover using tournament selection while ensuring a large enough group."""
        selected = []

        while len(selected) < POPULATION_SIZE:  # Ensure enough parents for the next generation
            tournament = random.sample(population, min(TOURNAMENT_SIZE, len(population)))  # Avoid oversampling
            best_individual = min(tournament, key=lambda x: fitness_scores[tuple(x)])  # Min fitness is better
            selected.append(best_individual)

        return selected

    def crossover(self, parent1, parent2):
        """Perform crossover to create new offspring."""
        if random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()  # No crossover, return parents as-is

        split = random.randint(1, len(parent1) - 1)
        return parent1[:split] + parent2[split:], parent2[:split] + parent1[split:]

    def mutate(self, genome):
        """Bit-flip mutation to encourage exploration."""
        for i in range(len(genome)):
            if random.random() < MUTATION_RATE:
                genome[i] = 1 - genome[i]  # Flip bit

    def update_dependency_tracking(self, population, fitness_scores):
        """Track co-occurrences of info types in high-fitness solutions."""
        dependency_counts = defaultdict(int)

        # Analyze top-performing individuals
        top_individuals = sorted(population, key=lambda x: fitness_scores[tuple(x)])[:POPULATION_SIZE // 2]
        for individual in top_individuals:
            for combo in combinations(individual, 2):  # Track pairwise co-occurrences
                dependency_counts[combo] += 1

        # Filter for strong dependencies
        self.dependency_groups = {k: v for k, v in dependency_counts.items() if v > len(top_individuals) // 2}

    def main(self):
        """Main loop for genetic algorithm."""
        info_types = self.get_hive_mind_info_types()  # Get all available groups (info types)
        num_info_types = len(info_types)

        # Initialize population genomes
        population = self.initialize_population(num_info_types)

        for generation in range(NUM_GENERATIONS):
            print(f"----------------------------Generation {generation+1} / {NUM_GENERATIONS}----------------------------")
            print(population)

            # Convert population of bitwise information genomes into population of information types format
            population_info_type = []
            fitness_scores = []
            for genome in population:
                population_info_type.append([info_types[i] for i in range(len(info_types)) if genome[i] == 1])
                fitness_scores.append(self.evaluate_fitness(genome))

            # Save results for this generation
            self.save_results(generation, population, fitness_scores)

            # # Track dependencies dynamically
            # if generation % DEPENDENCY_UPDATE_INTERVAL == 0:
            #     self.update_dependency_tracking(population, fitness_scores)

            # Selection
            selected_parents = self.roulette_wheel_selection(population, fitness_scores) # Enter the selection method here...

            # Ensure selected parents count is even for pairing used in crossover
            if len(selected_parents) % 2 == 1:
                selected_parents.append(random.choice(selected_parents))  # Duplicate one parent if odd

            # Crossover
            offspring = []
            for i in range(0, len(selected_parents) - 1, 2):
                child1, child2 = self.crossover(selected_parents[i], selected_parents[i+1])
                offspring.extend([child1, child2])

            # Mutation
            for child in offspring:
                self.mutate(child)

            # Apply elitism: Keep best solutions using fitness scores
            num_elites = ELITISM_RATE * POPULATION_SIZE
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:num_elites]
            elite_individuals = [population[i] for i in elite_indices]

            # Fill the rest of the NEW population with offspring (truncated for max population size)
            population = elite_individuals + offspring[:POPULATION_SIZE - num_elites]

        # Final best solution
        best_index = fitness_scores.index(min(fitness_scores))  # Get index of best solution
        best_solution = population[best_index]
        print(f"\nOptimal Information Set: {best_solution} | Fitness: {fitness_scores[best_index]}")


###### Run experiment ######
if __name__ == "__main__":
    genetic_optimisation = GeneticOptimisation()
    genetic_optimisation.main()
