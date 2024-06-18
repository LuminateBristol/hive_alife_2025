import sys
import os
import time
import copy

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

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['exp_setup']
map_file = CFG_FILES['map']
ex_id = 'exp_1'
task_log = None

###### Functions ######
def run_ex():

    # Setup config for this experiment
    cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id, map=map_file)
    cfg_obj.print()

    agentnum = cfg_obj.get('number_of_agents')
    boxes = cfg_obj.get('boxes')
    task_log = cfg_obj.get('task_log')
    
    # Set up config file with parameters for this run
    cfg_obj.set('warehouse.number_of_agents', agentnum)
    cfg_obj.set('boxes', boxes)
    cfg_obj.set('task_log', copy.deepcopy(task_log))

    # Create simulator object
    sim = VizSim(cfg_obj, verbose=verbose)

    counter = sim.run() # Counter is equivalent to the number of times the entire robot_tree is ticked == simulation timesteps

    print(f' TOTAL COUNTS: {counter}')

def main():
    t0 = time.time()
    run_ex()
    t1 = time.time()
    dt = t1-t0
    print("Time taken: %s"%str(dt), '\n')

###### Run experiment ######

if __name__ == "__main__":
    main()
