from simulator.wh_sim import *
from simulator.lib import Config, SaveTo
from simulator import CFG_FILES

###### Experiment parameters ######

ex_id = 'e_1'
iterations = 200
export_data = True
verbose = False    
fault_range = range(1,2) # inject 0-10 faults
batch_id = 'test'

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['ex_1']
cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)

###### Functions ######

def gen_random_seed(iteration):
    P1 = 33331
    P2 = 73
    a = 1
    b = int(ex_id.split("_")[1])
    c = iteration
    return (a*P1 + b)*P2 + c

def iterate_ex(iterations, faults, st, export_data=True):
    for i in range(iterations):
        if i%10 == 0:
            print("-- %d/%d iterations"%(i, iterations))

        run_ex(i, faults, st, export_data)

def run_ex(iteration, faults, st, export_data=True):
    random_seed = gen_random_seed(iteration)

    if export_data:
        data_model = MinimalDataModel(faults[0], store_internal=True, compute_roc=True)
    else:
        data_model = None

    # Create simulator object
    sim = Simulator(cfg_obj, 
        verbose=verbose,
        data_model=data_model,
        random_seed=random_seed,
        fault_count=faults)

    sim.run()

    # Save data
    if export_data:
        st.export_data(data_model, ex_id, faults[0], random_seed)

###### Run experiment ######

st = SaveTo(batch_id)
for it in fault_range:
    print("Running simulation with %d faulty robots"%it)
    faults = [it]
    iterate_ex(iterations, faults, st, export_data=export_data)
