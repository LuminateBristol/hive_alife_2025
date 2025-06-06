import sys
import os
import time
from multiprocessing import Pool

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
    num_runs = 30
    num_robots = [200]

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

    num_runs = 20
    num_robots = [10, 50, 100, 200]

    # Open file once and write the header
    with open('traffic_optimised_dist.txt', 'w') as f:
        f.write('id\texp\ttype\tnum_rob\ttimesteps\n')  # Write header

    for num in num_robots:
        # Set number of robots
        exp_cfg.set('number_of_agents', num)

        # Run
        score_total = 0
        time_total = 0
        for i in range(num_runs):
            sim = Simulator(gen_cfg, exp_cfg, map_cfg, verbose=verbose)
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

### Testing multiprocessing ###

def run_traffic_sim(args):
    num_robots, run_id = args

    # Set up configs locally to avoid shared state issues
    local_gen_cfg = Config(cfg_path=CFG_FILES['default'])
    local_exp_cfg = Config(cfg_path=CFG_FILES['exp_setup'], ex_id=ex_id)
    local_map_cfg = Config(cfg_path=CFG_FILES['map'])

    local_exp_cfg.set('number_of_agents', num_robots)

    sim = Simulator(local_gen_cfg, local_exp_cfg, local_map_cfg, verbose=False)
    counter = sim.run(iteration=run_id)
    score = sim.traffic_score['score']

    return (run_id, num_robots, counter, score)

def run_many_traf_parallel():
    num_runs = 30
    num_robots_list = [10, 20, 50, 100, 200]

    # Open file once and write the header
    with open('traffic_cent.txt', 'w') as f:
        f.write('id\texp\ttype\tnum_rob\ttimesteps\n')  # Write header

    for num_robots in num_robots_list:
        print(f"Running simulations for num_robots: {num_robots}")
        args_list = [(num_robots, i) for i in range(num_runs)]

        with Pool(processes=15) as pool:
            results = pool.map(run_traffic_sim, args_list)

        score_total = 0
        time_total = 0

        with open('traffic_cent.txt', 'a') as f:
            for run_id, num, counter, score in results:
                f.write(f"{run_id}\ttraffic\tcentralised\t{num}\t{counter}\n")
                score_total += score
                time_total += counter

        av_score = score_total / num_runs
        av_time = time_total / num_runs
        print(f"Results complete - num_rob: {num_robots}. Avg score: {av_score}, Avg time: {av_time}")

def main():
    t0 = time.time()
    # run_many_log()
    # run_many_acov()
    # run_many_traf()
    run_many_traf_parallel()
    t1 = time.time()
    dt = t1-t0
    print("Time taken: %s"%str(dt), '\n')

###### Run experiment ######

if __name__ == "__main__":
    main()
