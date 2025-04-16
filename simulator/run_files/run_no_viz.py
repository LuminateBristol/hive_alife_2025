import sys
import os
import time

# from babel.messages.setuptools_frontend import extract_messages

# We need to setup  parent directories to properly import other modules
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
batch_id = 'test'
ex_id = 'exp_3_traffic'

# Setup config for this experiment
gen_cfg = CFG_FILES['default']
exp_cfg = CFG_FILES['exp_setup']
map_cfg = CFG_FILES['map']
gen_cfg = Config(cfg_path=gen_cfg)
exp_cfg = Config(cfg_path=exp_cfg, ex_id=ex_id)
map_cfg = Config(cfg_path=map_cfg)

# Experiment set from cfg file 'exp_setup' NOTE: change this file to update experimental parameters

###### Functions ######
def run_ex():

    # Create simulator object
    sim = Simulator(gen_cfg, exp_cfg, map_cfg, verbose=verbose)

    counter = sim.run(iteration=0) # Counter is equivalent to the number of times the entire robot_tree is ticked == simulation timesteps

    print(f'TOTAL COUNTS: {counter}')

def run_many_log():
    num_runs = 10
    num_robots = [10, 20, 50]

    # Open file once and write the header
    with open('results/logistics_baseline.txt', 'w') as f:
        f.write('id\texp\ttype\tnum_rob\ttime\n')  # Write header

    for num in num_robots:
        # Set number of robots
        exp_cfg.set('number_of_agents', num)

        # Run
        for i in range(num_runs):
            sim = Simulator(gen_cfg, exp_cfg, map_cfg, verbose=verbose)
            counter = sim.run(iteration=i+20)

            # Append results for this run
            with open('results/logistics_baseline.txt', 'a') as f:  # Append mode
                f.write(f"{i}\tlogistics\tbaseline\t{num}\t{counter}\n")

    print("Results complete and saved - yay!")

def run_many_acov():
    num_runs = 10
    num_robots = [0]

    # Open file once and write the header
    with open('results/logistics_optimised.txt', 'w') as f:
        f.write('id\texp\ttype\tnum_rob\ttimesteps\n')  # Write header

    for num in num_robots:
        # Set number of robots
        exp_cfg.set('number_of_agents', num)

        # Run
        for i in range(num_runs):
            sim = VizSim(gen_cfg, exp_cfg, map_cfg, verbose=verbose)
            counter = sim.run(iteration=i)
            # total_cells = (sim.cfg.get('warehouse', 'width') * sim.cfg.get('warehouse', 'height')) / sim.cfg.get('warehouse', 'cell_size') ** 2
            # percent = (len(sim.warehouse.pheromone_map) / total_cells) * 100

            # Append results for this run
            with open('results/logistics_optimised.txt', 'a') as f:  # Append mode
                f.write(f"{i}\tlogistics\toptimised\t{num}\t{counter}\n")

    print("Results complete and saved - yay!")

def run_many_traf():

    num_runs = 10
    num_robots = [10,20,50]

    # Open file once and write the header
    with open('traffic_dist.txt', 'w') as f:
        f.write('id\texp\ttype\tnum_rob\ttimesteps\n')  # Write header

    for num in num_robots:
        # Set number of robots
        exp_cfg.set('number_of_agents', num)

        # Run
        score_total = 0
        time_total = 0
        for i in range(num_runs):
            sim = VizSim(gen_cfg, exp_cfg, map_cfg, verbose=verbose)
            counter = sim.run(iteration=i)
            score = sim.traffic_score['score']
            score_total += score
            time_total += counter

            # Append results for this run
            with open('traffic_dist.txt', 'a') as f:  # Append mode
                f.write(f"{i}\ttraffic\tdistributed\t{num}\t{counter}\n")

        # Record results
        av_score = score_total / num_runs
        av_time = time_total / num_runs
        print(f"Results complete and saved - num_rob: {num}. Ave score: {av_score} Ave time: {av_time}")

def main():
    t0 = time.time()
    # run_many_log()
    # run_many_acov()
    run_many_traf()
    t1 = time.time()
    dt = t1-t0
    print("Time taken: %s"%str(dt), '\n')

###### Run experiment ######

if __name__ == "__main__":
    main()
