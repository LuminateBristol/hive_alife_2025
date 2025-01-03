import sys
import os
import random
import time
import matplotlib.pyplot as plt

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
ex_id = 'exp_2_area_coverage'

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['aamas_exps']
map_file = CFG_FILES['map']


###### Functions ######

class RunOptimisation():
    def __init__(self):
        self.info_types = []
        self.cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id, map=map_file)
        self.num_iterations = 1  # Number of iterations for each batch

        # Variables to track dynamic averages
        self.total_w = 0
        self.total_t = 0
        self.num_batches = 0

    def run_simulation(self, selected_info_types, iterations):
        """Run the simulation multiple times and return the average completion time along with individual times."""
        total_coverage = 0
        coverages_per_run = []
        best_coverage = 1
        print(f"\nStarting batch run for the following information types: {selected_info_types}")

        for i in range(iterations):
            # Init simulator
            sim = Simulator(self.cfg_obj, verbose=verbose)

            # Update weights for information types
            for node in sim.Hive_Mind.graph.nodes(data=True):
                node_id, attributes = node
                if 'type' in attributes and attributes['type'] in selected_info_types:
                    sim.Hive_Mind.graph.nodes[node_id]['weight'] = 1

            # Run the simulation and record the area coverage for this run
            run_time = sim.run(iteration=i)
            total_area = (self.cfg_obj.get('warehouse', 'width') * self.cfg_obj.get('warehouse', 'height')) / self.cfg_obj.get('warehouse', 'cell_size') ** 2
            area_coverage = len(sim.warehouse.pheromone_map) / total_area

            if area_coverage < best_coverage:
                self.print_pheromone_map(sim.warehouse.pheromone_map, self.cfg_obj.get('warehouse', 'width'), self.cfg_obj.get('warehouse', 'height'), name=f'P_Map for {selected_info_types}')

            total_coverage += area_coverage
            coverages_per_run.append(area_coverage)

        average_coverage = total_coverage / iterations  # Return average coverage for this batch
        return average_coverage, coverages_per_run  # Also return individual times for each run

    def get_hive_mind_info_types(self):
        """Retrieve all nodes with 'weight' attribute and group them by their 'type' attribute."""
        sim = Simulator(self.cfg_obj, verbose=verbose)
        hive_mind = sim.Hive_Mind.graph

        nodes_with_weight = [n for n, d in hive_mind.nodes(data=True) if 'weight' in d]
        groups = {}
        for node in nodes_with_weight:
            node_type = hive_mind.nodes[node]['type']
            if node_type not in groups:
                groups[node_type] = []
            groups[node_type].append(node)

        return groups

    def calculate_fitness(self, w, w_ave, coverage, coverage_ave):
        """Calculate fitness using the given formula."""
        """Take inverse of area coverage as we want to minimise fitness"""
        if w_ave and coverage_ave:
            fitness = (0.03 * (w / w_ave) + 1.0 * ((1/coverage) / (1/coverage_ave)))
        elif w_ave:
            fitness = 0.03 * (w / w_ave)
        elif coverage_ave:
            fitness = 1.0 * ((1/coverage) / (1/coverage_ave))
        return fitness

    def update_averages(self, w, t):
        """Update the dynamic averages for w and t after each batch."""
        self.total_w += w
        self.total_t += t
        self.num_batches += 1

        # Calculate running averages
        w_ave = self.total_w / self.num_batches
        t_ave = self.total_t / self.num_batches

        return w_ave, t_ave

    def print_pheromone_map(self, data_map, width, height, name):
        """ Print pheromone map for post-analysis"""
        # Extract x, y coordinates and strength values
        x_coords = [key[0] for key in data_map.keys()]
        y_coords = [key[1] for key in data_map.keys()]
        strengths = [data_map[key] for key in data_map.keys()]

        # Normalize strength values to set transparency (between 0 and 1)
        max_strength = max(max(strengths), 60) # To get some consitency in the plots we use 80 as the maximum visits per hash grid cell unless the odd one goes over
        alphas = [(strength / max_strength) for strength in strengths]

        plt.figure(figsize=(width / 100, height / 100))  # Set plot size in inches
        plt.scatter(x_coords, y_coords, s=130, c='blue', alpha=alphas, marker='s')  # Plot with squares

        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.gca().set_aspect('equal', adjustable='box')  # Equal scaling for x and y

        # Save the figure to a file
        plt.savefig(f'{name}.png', format='png', dpi=300)
        plt.close()  # Close the figure to free memory

    def main(self):
        """Main method for greedy optimization."""
        # Step 1: Retrieve groups of nodes with the same 'type' attribute
        groups = self.get_hive_mind_info_types()
        selected_info_types = []
        results = []
        info_type = None

        previous_fitness = float('inf')  # Track previous fitness
        for _ in range(len(groups) + 1):

            # Run the simulation 30 times, record the average completion time and individual times
            batch_run_average_coverage, coverages_per_run = self.run_simulation(selected_info_types, iterations=self.num_iterations)

            # Calculate w (number of types in selected_info_types)
            w = len(selected_info_types)

            # Update dynamic averages for w_ave and cov_ave - this is used for fitness value normalisation
            total_w_ave, total_cov_ave = self.update_averages(w, batch_run_average_coverage)

            # Calculate fitness based on the current batch
            fitness = self.calculate_fitness(w, total_w_ave, batch_run_average_coverage, total_cov_ave)

            # Record results based on fitness
            if 1.2 * fitness <= previous_fitness:  # Update condition based on fitness - aim for 20% improvement or better
                previous_fitness = fitness
                results.append( (selected_info_types.copy(), total_w_ave, batch_run_average_coverage, fitness, coverages_per_run))  # Record successful group
            else:
                results.append((selected_info_types.copy(), total_w_ave, batch_run_average_coverage, fitness, coverages_per_run))  # Record failed group
                selected_info_types.remove(info_type)  # Remove from selected info types

            # Add a randomly selected info type ready for tbe next run
            if groups:
                info_type, nodes = random.choice(list(groups.items()))
                selected_info_types.append(info_type)
                del groups[info_type]

        # Output results to a .txt file
        result_file_path = os.path.join(current_dir, 'greedy_optimization_results.txt')
        with open(result_file_path, 'w') as file:
            file.write('Type\tRun Times\tWeight\tAverage Time\tFitness\n')
            for selected_info_types, run_times, total_w_ave, batch_run_average_coverage, fitness in results:
                file.write(f'{selected_info_types}\t{run_times}\t{total_w_ave}\t{batch_run_average_coverage}\t{fitness}\n')

        print('Optimization complete. Results saved to "greedy_optimization_results.txt".')


###### Run experiment ######

if __name__ == "__main__":
    run_optimisation = RunOptimisation()
    run_optimisation.main()
