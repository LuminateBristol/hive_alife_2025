import sys
import os
import time

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

verbose = True
ex_id = 'exp_3_traffic'   # Experiment set from cfg file 'exp_setup' NOTE: change this file to update experimental parameters

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['exp_setup']
map_file = CFG_FILES['map']

###### Functions ######

def run_ex():
    """
    Runs a single experiment by initializing the configuration, setting up parameters,
    creating the simulation object, and executing the simulation.

    The function retrieves the number of agents and boxes from the configuration,
    updates the configuration parameters, and runs the simulation.
    """

    # Setup config for this experiment
    cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id, map=map_file)

    agentnum = cfg_obj.get('number_of_agents')
    boxes = cfg_obj.get('boxes')
    
    # Set up config file with parameters for this run
    cfg_obj.set('warehouse.number_of_agents', agentnum)
    cfg_obj.set('boxes', boxes)
    
    # Create simulator object
    sim = VizSim(cfg_obj, verbose=verbose)

    counter = sim.run() # Counter is equivalent to the number of times the entire robot_tree is ticked == simulation timesteps

    print(f' TOTAL COUNTS: {counter}')

def main():
    """
    Main function to execute the experiment.
    It records the start time, runs the experiment, records the end time,
    and prints the total execution time.
    """
    t0 = time.time()
    run_ex()
    t1 = time.time()
    dt = t1-t0
    print("Time taken: %s"%str(dt), '\n')

###### Run experiment ######

if __name__ == "__main__":
    main()
