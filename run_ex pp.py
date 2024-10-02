from simulator.archive import *
from simulator.lib import Config, SaveSample
from simulator import CFG_FILES


import time
from datetime import datetime
import csv
import copy

###### Experiment parameters ######

iterations = 1
experiments = ['exp_1']
export_data = False
verbose = True    
batch_id = 'test'

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['ex_1']
pp_1 = CFG_FILES['aamas_exps']
task_log = None

###### Functions ######
def create_savefile(exp_name):                              
    current_date = datetime.now().strftime('%Y-%m-%d')
    csvname = f"{current_date}_{exp_name}.csv"
    with open(csvname, 'a', newline='') as csvfile:
        fieldnames = ['num_robots', 'boxes', 'global', 'global_s', 'dropout_rate', 'latency', 'counter']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

         # Write the header if the file is empty
        if csvfile.tell() == 0:
            writer.writeheader()
    return csvname, fieldnames

def save_data(csvname, fieldnames, num_agents, boxes, global_s, dropout, latency, counter): 
    with open(csvname, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({
            'num_robots': num_agents,
            'boxes': boxes,
            'global_s': global_s,
            'global': True,    
            'dropout_rate': dropout, 
            'latency': latency,                                        
            'counter': counter
        })

def iterate_ex(iterations, faults=None):
    for i in range(iterations):
        print("-- %d/%d iterations"%(i, iterations))

        for exp in experiments:
            csvname, fieldnames = create_savefile(exp)
            run_ex(i, exp, faults, csvname, fieldnames)

def run_ex(iteration, pp_id, faults, csvname, fieldnames, st=None):

    # Setup config for this experiment
    cfg_obj = Config(cfg_file, default_cfg_file, pp_ex=pp_1, pp_id=pp_id)

    task_log = copy.deepcopy(cfg_obj.get('task_log'))
    range_of_agents = cfg_obj.get('range_of_agents')
    range_of_boxes = cfg_obj.get('range_of_boxes')
    colour_ratio = cfg_obj.get('colour_ratio')
    use_global_batch = cfg_obj.get('use_global_batch')
    local_comms_range = cfg_obj.get('local_comms_range')
    latency = cfg_obj.get('latency')
    drop_out_rate = cfg_obj.get('drop_out_rate')
    
    # Loop through hive mind and no hive mind
    for i in range(len(use_global_batch)):
        use_global = use_global_batch[i]
        cfg_obj.set('use_global', use_global)

        # Loop though drop out rates
        for dropout in drop_out_rate:
            cfg_obj.set('drop_out_rate', dropout)

            # Loop throgh latencys
            for _latency in latency:
                cfg_obj.set('latency', _latency)
        
                # Loop through number of agents
                for agentnum in range(range_of_agents[0], range_of_agents[1]+1): 

                    # Loop through number of boxes 
                    for boxnum in range(range_of_boxes[0], range_of_boxes[1]+1):
                        boxes = {key: value * boxnum for key, value in colour_ratio.items()}

                        # Set up config file with parameters for this run
                        cfg_obj.set('warehouse.number_of_agents', agentnum)
                        cfg_obj.set('boxes', boxes)
                        cfg_obj.set('task_log', copy.deepcopy(task_log))

                        print(f'Running....num agents: {agentnum}, boxes: {boxes}, task_log: {task_log}. hive_mind: {use_global}')

                        # Create simulator object
                        sim = VizSim(cfg_obj, verbose=verbose)

                        counter, carry_counter = sim.run() # Counter is equivalent to the number of times the entire robot_tree is ticked == simulation timesteps

                        print(f' TOTAL COUNTS: {counter}')

                        # Save data
                        save_data(csvname, fieldnames, agentnum, boxes, use_global, dropout, _latency, counter)                                

###### Run experiment ######

log_time = []
t0 = time.time()
iterate_ex(iterations)
t1 = time.time()
dt = t1-t0
print("Time taken: %s"%str(dt), '\n')
log_time.append(dt)
