from simulator.wh_sim import *
from simulator.lib import Config, SaveSample
from simulator import CFG_FILES, MODEL_ROOT, STATS_ROOT
import time

###### Experiment parameters ######

ex_id = 'e_4'
verbose = False    
faults = [0]

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['ex_1']
cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)

###### Functions ######

data_model = ExportRedisData(export_vis_data=True, compute_roc=True)
thresh_file = os.path.join(MODEL_ROOT, "%s_%s.txt"%(ex_id, "emin_sc"))
stats_file = os.path.join(STATS_ROOT, "%s_%s.txt"%(ex_id, "emin_sc"))
ad_model = ExportThresholdModel(10, thresh_file, stats_file, 3, 0.15, 2)

# Create simulator object
sim = VizSim(cfg_obj,
    data_model=data_model,
    fault_count=faults,
    ad_model=ad_model,
    random_seed=66764970)

sim.run()
