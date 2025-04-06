import sys
import os
import random
import numpy as np
from itertools import combinations
from collections import defaultdict
import multiprocessing

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
gen_cfg = CFG_FILES['default']
exp_cfg = CFG_FILES['exp_setup']
map_cfg = CFG_FILES['map']

# Setup config for this experiment
gen_cfg = Config(cfg_path=gen_cfg)
exp_cfg = Config(cfg_path=exp_cfg, ex_id=ex_id)
map_cfg = Config(cfg_path=map_cfg)

###### GA Parameters ######
POPULATION_SIZE = 10
NUM_GENERATIONS = 10
MUTATION_RATE = 0.1             # Set to 0.0 for no mutation
CROSSOVER_RATE = 0.5            # Set to 0.0 for no crossover
ELITISM_RATE = 0.3
# TOURNAMENT_SIZE = 5
# DEPENDENCY_UPDATE_INTERVAL = 2  # Update dependency tracking every 5 generations
NUM_ITERATIONS = 5             # Number of iterations ran per generation

# Multi processing wrapper:
def evaluate_genome_wrapper(args): # TODO: at the moment this only parallelises the genomes - i.e. max cores = 10. Future work needs to flatten this to we parallise over Num_genomes + Num_iterations - i;e. setup list of jobs which contains all genomres * NUM_ITERATIONS and send this to multiprocessing. Following this, we can then combine them again and take averages for the entire genome 
        self_instance, genome, info_types = args
        selected_info_types = []
        for index, i in enumerate(genome):
            if i:
                selected_info_types.append(info_types[index])
        return self_instance.evaluate_fitness(selected_info_types)

# Genetic algorithm:
class GeneticOptimisation:
    def __init__(self):
        self.num_iterations = NUM_ITERATIONS

        # Fitness function weights
        self.performance_weight = 1
        self.communication_weight = 0.2

        # Track dependencies dynamically
        self.dependency_groups = {}

        # File save setup
        self.result_file_path = self.create_result_file()

        # Multiprocessing setup
        self.num_cores = 10

    def create_result_file(self):
        filename = f"GA_results_pop{POPULATION_SIZE}_gen{NUM_GENERATIONS}_cross{CROSSOVER_RATE}_elit{ELITISM_RATE}_roulette.txt"
        file_path = os.path.join(current_dir, filename)
        info_types = self.get_hive_mind_info_types()

        with open(file_path, 'w') as file:
            file.write("INFO_TYPES: " + "\t".join(info_types) + "\n")
            file.write("generation\tgenome\tave_time\ttot_weight\tfitness\n")

        return file_path

    def save_results(self, generation, population, ave_time, tot_weight, fitness_scores):
        with open(self.result_file_path, 'a') as file:
            for i in range(len(population)):
                genome = population[i]
                _ave_time = ave_time[i]
                _tot_weight = tot_weight[i]
                fitness = fitness_scores[i]
                file.write(f"{generation}\t{genome}\t{_ave_time}\t{_tot_weight}\t{fitness}\n")

    def get_hive_mind_info_types(self):
        sim = Simulator(gen_cfg, exp_cfg, map_cfg, verbose=verbose)
        hive_mind = sim.optimisation_hive_mind.graph
        # Find a groups of all information nodes with the same name
        information_nodes = [n for n, d in hive_mind.nodes(data=True) if 'weight' in d]
        groups = {}
        for node in information_nodes:
            node_type = hive_mind.nodes[node]['type']
            if node_type not in groups:
                groups[node_type] = []
            groups[node_type].append(node)

        # Return list of information types available for weight optimisation
        return list(groups.keys())

    def run_simulation(self, selected_info_types):
        """Run the simulation multiple times and return the average completion time."""
        total_time = 0
        run_times = []

        print(f'eit: "{selected_info_types}')

        for i in range(self.num_iterations):
            sim = Simulator(gen_cfg, exp_cfg, map_cfg, verbose=False)

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
        w = sum(genome)  # Number of info types
        fitness = self.performance_weight * avg_time + self.communication_weight * w # TODO: fix this so that it normalises (both?) for best fitness
        return avg_time, w, fitness

    def initialize_population(self, num_info_types):
        """
        Generate an initial random population of info type selections.
        A population is a list of information types that will be set to one for this run.
        """
        return [[random.choice([0, 1]) for _ in range(num_info_types)] for _ in range(POPULATION_SIZE)]

    def roulette_wheel_selection(self, population, fitness_scores):
        """Select parents using fitness-proportionate selection."""
        total_fitness = sum(fitness_scores)
        selection_probs = [1 - (fitness_scores[i] / total_fitness) for i in range(len(population))]
        # Selects k=POPULATION_SIZE genomes with a bias based on selection_probs ie. probability of selection
        # Note - this method allows for replacement selection - i.e. high selection_prob genomes can be selected more than once
        selected = random.choices(population, weights=selection_probs, k=POPULATION_SIZE)
        return selected

    # # def tournament_selection(self, population, fitness_scores):
    #     """Select individuals for crossover using tournament selection while ensuring a large enough group."""
    #     selected = []

    #     while len(selected) < POPULATION_SIZE:  # Ensure enough parents for the next generation
    #         tournament = random.sample(population, min(TOURNAMENT_SIZE, len(population)))
    #         best_individual = min(tournament, key=lambda x: fitness_scores[tuple(x)])
    #         selected.append(best_individual)

    #     return selected

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

    # # def update_dependency_tracking(self, population, fitness_scores):
    #     """Track co-occurrences of info types in high-fitness solutions."""
    #     dependency_counts = defaultdict(int)

    #     # Analyze top-performing individuals
    #     top_individuals = sorted(population, key=lambda x: fitness_scores[tuple(x)])[:POPULATION_SIZE // 2]
    #     for individual in top_individuals:
    #         for combo in combinations(individual, 2):  # Track pairwise co-occurrences
    #             dependency_counts[combo] += 1

    #     # Filter for strong dependencies
    #     self.dependency_groups = {k: v for k, v in dependency_counts.items() if v > len(top_individuals) // 2}
    
    def main(self):
        """
        The genetic algorithm works as follows:
        1. Init population - random 1s and 0s for every available information type on the Hive
        2. Evaluate fitness - run the simulation on each genome foe self.num_iterations
                            - return the average completion time over all iterations
                            - return the hive compute (num of shared infos) TODO: update this to num*[sizes_of_info_types]
                            - calculate fitness as a function of completion time and hive compute
        3. Use roulette wheel to method to select parents for mating - add random parent if odd number
        4. Use crossover function to create offspring for all parents.
                            - Only crossover based on CROSSOVER_RATE probability, otherwise return original parents
                            - For every parent entered - there are always two children produced
        5. For the offspring - apply mutation - bit-flip each of the info weights basd on a probability = MUTATION_RATE
        6. Select elite genomes based on percentage ELITISM_RATE
        7. Add together all elite genomes + enough offspring to make up a full POPULATION_SIZE

        Note - there is options to track dependencies but this is not currently implemented
        """
        info_types = self.get_hive_mind_info_types()  # Get all available groups (info types)
        num_info_types = len(info_types)
        print(info_types)

        # Initialize population genomes
        population = self.initialize_population(num_info_types)

        for generation in range(NUM_GENERATIONS):
            print(f"----------------------------Generation {generation+1} / {NUM_GENERATIONS}----------------------------")
            # print(population)

            # Multiprocessing setup
            with multiprocessing.Pool(processes=self.num_cores) as pool:    
                # Pass self along with genome for each evaluation
                results = pool.map(evaluate_genome_wrapper, [(self, genome, info_types) for genome in population])

            # Convert population of bitwise information genomes into population of information types format
            population_info_type = []
            ave_times = []
            tot_weights = []
            fitness_scores = []
            for i, genome in enumerate(population):
                population_info_type.append([info_types[i] for i in range(len(info_types)) if genome[i] == 1])
                avg_time, w, fitness = results[i]
                ave_times.append(avg_time)
                tot_weights.append(w)
                fitness_scores.append(fitness)

            # Update fitness score based using normalisation for this population set
            # TODO: this is a hacky way of doing this - need to update code!
            max_ave_time = max(ave_times)
            normalised_fitness_scores = []
            for i in range(len(population)):
                normalised_fitness = (ave_times[i] / max_ave_time) + 0.2 * tot_weights[i]
                normalised_fitness_scores.append(normalised_fitness)

            # Save results for this generation
            self.save_results(generation, population, ave_times, tot_weights, normalised_fitness_scores)

            # # Track dependencies dynamically
            # if generation % DEPENDENCY_UPDATE_INTERVAL == 0:
            #     self.update_dependency_tracking(population, fitness_scores)

            # Selection
            selected_parents = self.roulette_wheel_selection(population, normalised_fitness_scores)

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
            num_elites = int(ELITISM_RATE * POPULATION_SIZE)
            elite_indices = sorted(range(len(normalised_fitness_scores)), key=lambda i: normalised_fitness_scores[i])[:num_elites]
            elite_individuals = [population[i] for i in elite_indices]

            # Fill the rest of the NEW population with offspring (truncated for max population size)
            population = elite_individuals + offspring[:POPULATION_SIZE - num_elites]

        # Final best solution
        best_index = normalised_fitness_scores.index(min(normalised_fitness_scores))  # Get index of best solution
        best_solution = population[best_index]
        print(f"\nOptimal Information Set: {best_solution} | Fitness: {normalised_fitness_scores[best_index]}")


###### Run experiment ######
if __name__ == "__main__":
    genetic_optimisation = GeneticOptimisation()
    genetic_optimisation.main()
