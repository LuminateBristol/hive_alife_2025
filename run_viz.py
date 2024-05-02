from simulator.simulator import *
from simulator.lib import Config
from simulator import CFG_FILES
import time
import copy

###### Experiment parameters ######

export_data = False
verbose = True    
batch_id = 'test'

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['exp_setup']
ex_id = 'exp_2'
task_log = None

###### Functions ######
def run_ex():

    # Setup config for this experiment
    cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)

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
