from simulator.behaviour_tree_sim.bt_random_walk import * # TODO: update this so the config file is used to launch the correct simulator for the chosen project
from simulator.behaviour_tree_sim import*
from simulator.lib import Config, SaveSample
from simulator import CFG_FILES, MODEL_ROOT, STATS_ROOT
import time

###### Experiment parameters ######

ex_id = 'e_4'
verbose = True    

#HH faults = [0]

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['ex_1']
pp_1 = CFG_FILES['pp_1']
cfg_obj = Config(cfg_file, default_cfg_file, pp_ex=pp_1, pp_id='e_1', ex_id=ex_id)

###### Faults Setup ######

#data_model = ExportRedisData(export_vis_data=True, compute_roc=True)
#thresh_file = os.path.join(MODEL_ROOT, "%s_%s.txt"%(ex_id, "emin_sc")) # Thresholds for fault detection
#stats_file = os.path.join(STATS_ROOT, "%s_%s.txt"%(ex_id, "emin_sc"))  # Fault detection pipeline related
#ad_model = ExportThresholdModel(10, thresh_file, stats_file, 3, 0.15, 2)

###### Simulator Object - All Analysis is Run From Here ######

sim = VizSim(cfg_obj,
    #data_model=data_model,
    #fault_count=faults,
    #ad_model=ad_model,
    random_seed=None) # If seed is set, the random class in sim.py will inherit this seed and all randomisations will be linked to this seed - ie start robot and box positions will be the same

sim.run()