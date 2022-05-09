from simulator.wh_sim import *
from simulator.lib import Config, SaveTo
from simulator import CFG_FILES
import multiprocessing as mp

###### Experiment parameters ######

ex_id = 'e_2'
iterations = 200
export_data = True
verbose = False    
fault_range = range(7,8) # inject 0-10 faults
batch_id = '8may'
log_dir = 'logs'
cores = mp.cpu_count()
global_var = {'log_lock': False}

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

def gen_batches(iterations, no_procs):
    per_batch = int(iterations/no_procs)
    batches = []
    for i in range(no_procs):
        start = i*per_batch
        end = min((i+1)*per_batch, iterations)
        batch = list(range(start, end))
        batches.append(batch)
        
    if end < iterations:
        remainder = list(range(end, iterations))
        b_idx = 0
        while len(remainder):
            it = remainder.pop(0)
            batches[b_idx].append(it)
            b_idx = (b_idx+1)%len(batches)
            
    return batches

def create_procs(iterations, faults, st, export_data=True, cores_available=1):
    procs = []
    batches = gen_batches(iterations, cores_available)
    for i in range(cores_available):
        p = mp.Process(target=iterate_ex, args=(batches[i], faults, st, export_data))
        p.start()
        procs.append(p)
    
    return procs

def iterate_ex(iteration_list, faults, st, export_data=True):
    for idx, it in enumerate(iteration_list):
        if idx%10 == 0:
            print("-- %d/%d iterations"%(idx, len(iteration_list)))

        # _log(iteration, faults[0])
        run_ex(it, faults, st, export_data)

def run_ex(iteration, faults, st, export_data=True):
    random_seed = gen_random_seed(iteration)
    if export_data:
        data_model = DataModel(store_internal=True, compute_roc=True)
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

def _log(iteration, faults):
    while global_var['log_lock']:
        continue

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    global_var['log_lock'] = True
    log_file = os.path.join(log_dir, '%s.log'%batch_id)
    with open(log_file, 'a') as f:
        f.write("Running ex iteration %d, faults %d\n"%(iteration, faults))    
    global_var['log_lock'] = False
    
###### Run experiment ######

st = SaveTo(batch_id)
for it in fault_range:
    print("Running simulation with %d faulty robots"%it)
    faults = [it]
    procs = create_procs(iterations, faults, st, export_data, cores)
    for p in procs:
        p.join()
