import sys
import os
import random
import time

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
ex_id = 'exp_1_logistics'

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['aamas_exps']
map_file = CFG_FILES['map']


###### Functions ######

class RunOptimisation():
    def __init__(self):
        self.info_types = []
        self.cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id, map=map_file)
        self.num_iterations = 30  # Number of iterations for each batch

        # Variables to track dynamic averages
        self.total_w = 0
        self.total_t = 0
        self.num_batches = 0

    def run_simulation(self, selected_info_types, iterations):
        """Run the simulation multiple times and return the average completion time along with individual times."""
        total_time = 0
        run_times = []
        print(f"\nStarting batch run for the following information types: {selected_info_types}")

        for i in range(iterations):
            # Init simulator
            sim = Simulator(self.cfg_obj, verbose=verbose)

            # Update weights for information types
            for node in sim.Hive_Mind.graph.nodes(data=True):
                node_id, attributes = node
                if 'type' in attributes and attributes['type'] in selected_info_types:
                    sim.Hive_Mind.graph.nodes[node_id]['weight'] = 1

            # Run the simulation and record the time for this run
            start_time = time.time()
            run_time = sim.run(iteration=i)
            end_time = time.time()
            elapsed_time = end_time - start_time

            total_time += run_time
            run_times.append(elapsed_time)

        average_time = total_time / iterations  # Return average time for this batch
        return average_time, run_times  # Also return individual times for each run

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

    def calculate_fitness(self, w, w_ave, t, t_ave):
        """Calculate fitness using the given formula."""
        return 0.2 * (w / w_ave) + 1.0 * (t / t_ave)

    def update_averages(self, w, t):
        """Update the dynamic averages for w and t after each batch."""
        self.total_w += w
        self.total_t += t
        self.num_batches += 1

        # Calculate running averages
        w_ave = self.total_w / self.num_batches
        t_ave = self.total_t / self.num_batches

        return w_ave, t_ave

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
            avg_time, run_times = self.run_simulation(selected_info_types, iterations=self.num_iterations)

            # Calculate w (number of types in selected_info_types)
            w = len(selected_info_types)

            # Update dynamic averages for w_ave and t_ave
            w_ave, t_ave = self.update_averages(w, avg_time)

            # Calculate fitness based on the current batch
            fitness = self.calculate_fitness(w, w_ave, avg_time, t_ave)

            # Record results based on fitness
            if fitness <= 1.2 * previous_fitness:  # Update condition based on fitness
                previous_fitness = fitness
                results.append( (selected_info_types.copy(), w_ave, avg_time, fitness, run_times))  # Record successful group
            else:
                results.append((selected_info_types.copy(), w_ave, avg_time, fitness, run_times))  # Record failed group
                selected_info_types.remove(info_type)  # Remove from selected info types

            # Add a randomly selected info type ready for tbe next run
            info_type, nodes = random.choice(list(groups.items()))
            selected_info_types.append(info_type)
            del groups[info_type]

        # Output results to a .txt file
        result_file_path = os.path.join(current_dir, 'greedy_optimization_results.txt')
        with open(result_file_path, 'w') as file:
            file.write('Type\tRun Times\tWeight\tAverage Time\tFitness\n')
            for selected_info_types, run_times, w_ave, avg_time, fitness in results:
                file.write(f'{selected_info_types}\t{run_times}\t{w_ave}\t{avg_time}\t{fitness}\n')

        print('Optimization complete. Results saved to "greedy_optimization_results.txt".')


###### Run experiment ######

if __name__ == "__main__":
    run_optimisation = RunOptimisation()
    run_optimisation.main()
