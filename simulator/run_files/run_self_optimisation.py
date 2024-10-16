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

export_data = False
verbose = True    
batch_id = 'test'
ex_id = 'exp_1'                             # Experiment set from cfg file 'exp_setup' NOTE: change this file to update experimental parameters

###### Config class ######

default_cfg_file = CFG_FILES['default']     # Config for general parameters in cfg folder NOTE: change this file to update general parameters
cfg_file = CFG_FILES['exp_setup']          
map_file = CFG_FILES['map']                 # Config for map parameters in cfg folder     NOTE: change this file to update the map settings

###### Functions ######
def run_ex(hive_mind):
    # Setup config for this experiment
    cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id, map=map_file)

    agentnum = cfg_obj.get('number_of_agents')
    boxes = cfg_obj.get('boxes')

    # Set up config file with parameters for this run
    cfg_obj.set('warehouse.number_of_agents', agentnum)
    cfg_obj.set('boxes', boxes)

    # Create simulator object, passing in the HiveMind
    sim = Simulator(cfg_obj, hive_mind=hive_mind, verbose=verbose)

    # TODO: UPDATE WEIGHTS OF SIM.HIVE_MIND

    # Run simulation
    counter, messages = sim.run()  # Counter is the number of timesteps (simulation time)

    return counter, messages

if __name__ == "__main__":
    # Initialize robot observation space, tasks, and entities (define them as per your setup)
    # TODO: this is hardcoded currently but should be updated to use the config file
    entities = [['tasks', None],
                ['robots', None],
                ['boxes', None],
                ['delivery_point', {'id':1, 'coords': [50, 17]}],
                ['delivery_point', {'id':2, 'coords':[150, 17]}],
                ['delivery_point', {'id':3, 'coords':[250, 17]}],
                ['delivery_point', {'id':4, 'coords':[350, 17]}],
                ['delivery_point', {'id':5, 'coords':[450, 17]}],
                ['delivery_point', {'id':6, 'coords':[550, 17]}],
                ['delivery_point', {'id':7, 'coords':[650, 17]}],
                ['delivery_point', {'id':8, 'coords':[750, 17]}],
                ['delivery_point', {'id':9, 'coords':[850, 17]}],
                ['delivery_point', {'id':10, 'coords':[950, 17]}],
                ['box', {'colour':'yellow'}],
                ['box', {'colour':'blue'}],
                ['box', {'colour':'red'}],
                ['box', {'colour':'green'}],
                ['box', {'colour':'tan'}],
                ['box', {'colour':'purple'}],
                ['box', {'colour':'orange'}],
                ['box', {'colour':'peachpuff'}],
                ['box', {'colour':'darkseagreen'}],
                ['box', {'colour':'teal'}]]

    tasks = [[1, 1, 'yellow'],
             [2, 2, 'blue'],
             [3, 3, 'red'],
             [4, 4, 'green'],
             [5, 5, 'tan'],
             [6, 6, 'purple'],
             [7, 7, 'orange'],
             [8, 8, 'peachpuff'],
             [9, 9, 'darkseagreen'],
             [10, 10, 'teal']]

    # TODO: this should be uploaded by the robots at the start of the optimisation run
    # TODO: BUILD HIVE MIND IN SIM.PY
    # TODO: ACCESS HIVE MIND OBJECT AND UPDATE WEIGHTS IN OPTIMISATION ROUTINE IN RUN FILE
    # TODO: RUN USING SIM.RUN()

    # Create the optimizer
    optimiser = OptimiseHiveMind(robot_observation_space, tasks, entities, num_runs=10)

    # Run the greedy optimization
    best_information_sharing, best_cost = optimiser.greedy_optimization(run_ex)

    print(f'Best Information Sharing: {best_information_sharing}')
    print(f'Best Cost: {best_cost}')

###### Run experiment ######

if __name__ == "__main__":
    main()
