# import os
# import pandas as pd
# import numpy as np
# import multiprocessing as mp

from simulator.wh_proc import *

batch_id = '7may'
ex_id = 'e_1'
data_dir = "data/samples"
export_dir = "stats"
export_thresh = "models"

###### Run stats ######

# sp = SigProc(batch_id, ex_id, data_dir, export_dir)
# sp.compute_sig()
# sp.export_sig()

tp = ThreshProc(data_dir, batch_id, ex_id, export_thresh)
tp.process_data()
tp.export_thresh()