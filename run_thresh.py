from simulator.wh_sim import *
from simulator.lib import Config, SaveRes
from simulator import CFG_FILES
import time
import argparse

###### Experiment parameters ######

parser = argparse.ArgumentParser()
parser.add_argument('--ex_id')
parser.add_argument('--n')

args = parser.parse_args()
ex_id = args.ex_id
n = int(args.n)

###### Hardcoded parameters ######

# ex_id = 'e_1'
iterations = 10
export_data = True
verbose = False    
fault_range = range(11) # inject 0-10 faults
batch_id = 'emin_sc'
no_agents = 10
# n = 5
s = 0.15

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['ex_1']
cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)
thresh_file = os.path.join("models", "%s_%s.txt"%(ex_id, batch_id))
stats_file = os.path.join("stats", "%s_%s.txt"%(ex_id, batch_id))

###### Functions ######

def iterate_ex(iterations, faults, no_ag, st, export_data=True):
    for i in range(iterations):
        if i%2 == 0:
            print("-- %d/%d iterations"%(i, iterations))

        run_ex(i, faults, no_ag, st, export_data)

def run_ex(iteration, faults, no_ag, st, export_data=True):
    data_model = DataModel(faults[0], compute_roc=True)
    ad_model = ThresholdModel(no_agents, thresh_file, stats_file, n, s, store_internal_res=True)        

    # Create simulator object
    sim = Simulator(cfg_obj, 
        verbose=verbose,
        data_model=data_model,
        ad_model=ad_model,
        fault_count=faults)

    sim.run()

    # Save data
    data = ad_model.get_df_res()
    st.export_data(data, faults[0], sim.random_seed)

###### Run experiment ######

st = SaveRes(ex_id, batch_id)
t0 = time.time()
for it in fault_range:
    print("Running thresh model with %d faulty robots"%it)
    faults = [it]
    iterate_ex(iterations, faults, no_agents, st, export_data=export_data)

t1 = time.time()
dt = t1-t0
print("Time taken: %s"%str(dt), '\n')