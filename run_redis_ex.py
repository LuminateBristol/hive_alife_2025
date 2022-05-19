from simulator.wh_sim import *
from simulator.lib import Config, SaveTo
from simulator import CFG_FILES
import argparse

###### Experiment parameters ######

parser = argparse.ArgumentParser()
parser.add_argument('--ex_id')
parser.add_argument('--export_data')
parser.add_argument('--verbose')
parser.add_argument('--faults')

args = parser.parse_args()
ex_id = args.ex_id
export_data = bool(args.export_data)
verbose = bool(int(args.verbose))
faults = [int(args.faults)]

###### Hardcoded parameters ######

# ex_id = 'e_3'
# export_data = True
# verbose = False
# faults = [4]

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['ex_1']
cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)

###### Run experiment ######

if export_data:
    data_model = ExportRedisData(export_vis_data=True)
else:
    data_model = None

# Create simulator object
sim = Simulator(cfg_obj, 
    verbose=verbose,
    data_model=data_model,
    fault_count=faults)

print("Running...")
sim.run()
print("Complete!")

