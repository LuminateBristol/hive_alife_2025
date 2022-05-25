# import os
# import pandas as pd
# import numpy as np
# import multiprocessing as mp

from simulator.wh_proc import *

batch_id = 'test'
ex_ids = ['e_1', 'e_2', 'e_3', 'e_4', 'e_5', 'e_6']
data_dir = "data/samples"
export_dir = "stats"
export_thresh = "models"

###### Run stats ######

for ex_id in ex_ids:
    print("Running stats: %s"%ex_id)
    sp = SigProc(batch_id, ex_id, data_dir, export_dir)
    sp.compute_sig()
    sp.export_sig()

    tp = ThreshProc(data_dir, batch_id, ex_id, export_thresh)
    tp.process_data()
    tp.export_thresh()