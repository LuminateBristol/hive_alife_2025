import os
import pandas as pd
import numpy as np
import multiprocessing as mp

from simulator.wh_proc import *
from simulator.wh_sim import *

batch_id = '7may'
ex_id = 'e_1'
data_dir = "data"
cores = 8

###### Functions ######

def gen_batches(metrics, no_procs):
    per_batch = int(len(metrics)/no_procs)
    batches = []
    while True:
        for i in range(no_procs):
            if len(metrics) == 0:
                return batches 

            metric = metrics.pop(0)
            if len(batches) > i:
                batches[i].append(metric)
            else:
                batches.append([metric])

    return batches

def create_procs(data_sampler, metrics, export_data=True, np_seed=None, cores_available=1):
    procs = []
    batches = gen_batches(metrics, cores_available)
    for i in range(cores_available):
        p = mp.Process(target=iterate, args=(data_sampler, batches[i], export_data, np_seed))
        p.start()
        procs.append(p)
    
    return procs

def iterate(data_sampler, metrics, export_data=False, np_seed=None):
    for metric in metrics:
        data_sampler.sample_metric(metric, export_data, np_seed)

###### Run script ######

metrics = DataModel.metrics + DataModel.roc_metrics
ds = DataSampler(batch_id, ex_id, data_dir, metrics)
procs = create_procs(ds, metrics, export_data=True, cores_available=cores)
for p in procs:
    p.join()