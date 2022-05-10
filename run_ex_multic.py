from simulator.wh_sim import *
from simulator.lib import Config, SaveSample
from simulator import CFG_FILES
import multiprocessing as mp
import argparse

###### Experiment parameters ######

parser = argparse.ArgumentParser()
parser.add_argument('--ex_id')
parser.add_argument('--iterations')
parser.add_argument('--it_offset')
parser.add_argument('--export_data')
parser.add_argument('--verbose')
parser.add_argument('--faults')
parser.add_argument('--batch_id')
parser.add_argument('--cores')

args = parser.parse_args()
ex_id = args.ex_id
iterations = int(args.iterations)
it_offset = int(args.it_offset)
export_data = bool(args.export_data)
verbose = bool(args.verbose)
faults = int(args.faults)
batch_id = args.batch_id
cores = int(args.cores)

###### Hardcode parameters ######

# ex_id = 'e_2'
# iterations = 200
# it_offset = 0
# export_data = True
# verbose = False    
# faults = [1] # inject 0-10 faults
# batch_id = '10may'
# cores = 1

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['ex_1']
cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)

###### Functions ######

def gen_random_seed(iteration):
    global it_offset
    P1 = 33331
    P2 = 73
    a = 1
    b = int(ex_id.split("_")[1])
    c = iteration + it_offset
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
            print("-- %d/%d iterations"%(idx+1, len(iteration_list)))

        run_ex(it, faults, st, export_data)

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

st = SaveSample(batch_id)
print("Running simulation with %d faulty robots"%faults[0])
procs = create_procs(iterations, faults, st, export_data, cores)
for p in procs:
    p.join()
