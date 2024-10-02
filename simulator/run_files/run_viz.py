import sys
import os
import time

# We need to setup  parent directories to properly import other modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(parent_dir)
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
ex_id = 'exp_1'                             # Experiment set from cfg file 'exp_setup' NOTE: change this file to update experimental parameters

###### Config class ######

default_cfg_file = CFG_FILES['default']     # Config for general parameters in cfg folder NOTE: change this file to update general parameters
cfg_file = CFG_FILES['aamas_exps']           # Config for map parameters in cfg folder     NOTE: change this file to update the map settings
map_file = CFG_FILES['map']                 # Config for map parameters in cfg folder     NOTE: change this file to update the map settings

###### Functions ######
def run_ex():

    # Setup config for this experiment
    cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id, map=map_file)
    #cfg_obj.print()

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
    t0 = time.time()
    run_ex()
    t1 = time.time()
    dt = t1-t0
    print("Time taken: %s"%str(dt), '\n')

###### Run experiment ######

if __name__ == "__main__":
    main()
