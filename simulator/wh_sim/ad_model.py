import os
from os.path import dirname, realpath
import json
# try:
#     import tensorflow.keras as keras
# except:
#     print("WARN: Tensorflow module not available")
import pandas as pd
import numpy as np

class ThresholdModel:

    def __init__(self, number_of_agents, threshold_file, 
        time_window=50, store_internal=False):
        self.no_ag = number_of_agents
        self.time_w = time_window
        self.counter = 1

        # in the form of {metric: [t value, direction]}
        # negative direction means greater than threshold is unhealthy
        self._load_thresh(threshold_file)
        self.store_internal = store_internal
        self.df = None        
        self.data = {}

    def _load_thresh(self, file_path):
        with open(file_path, 'r') as f:
            t = f.readline()
            t = t.replace('\'', '\"').replace('None', 'null')
        
        self.thresholds = json.loads(t)

    # k: number of thresholds to pass to flag a fault
    # def predict(self, metrics, k, data):
    #     res = np.zeros(self.no_ag)
    #     if self.counter < self.time_w:
    #         return res, res

    #     for m in metrics:
    #         res = self.check_threshold_passed(metric, data)

    def check_thresholds(self, data):
        self.counter += 1
        total_passed = np.array([]) # a row of all thresholds passed (1) or not (0) for all agents
        for metric, it in self.thresholds.items():
            passed = self.check_threshold_passed(metric, data)
            total_passed = np.concatenate((total_passed, passed))
            
        if self.store_internal:
            if self.df is None:
                self.df = np.array([])

            self.df = np.concatenate([self.df, total_passed])
        
        return total_passed

    def check_threshold_passed(self, metric, data):
        if self.counter < self.time_w:
            return np.array([])

        t = self.counter%self.time_w
        threshold = self.thresholds[metric]
        if metric in ['nearest_agent_distance', 'nearest_box_distance', 'nearest_wall_distance', 'nearest_combined_distance']:
            d = np.nan_to_num(data[metric], nan=100)
        else:
            d = data[metric]
        
        d = d.reshape(1, self.no_ag)
        # self.data stores all the data over time window
        if self.counter == self.time_w:
            self.data[metric] = d
        elif self.data[metric].shape[0] <= t:
            self.data[metric] = np.concatenate([self.data[metric], d], axis=0)            
        else:
            self.data[metric][t] = d
        
        # print(metric)
        # print(self.data[metric].shape)
        mean_w = sum(self.data[metric])/self.time_w
        thresh = threshold[0]
        direction = threshold[1]
        t_arr = np.ones(self.no_ag)*thresh
        if direction > 0: # healthy is above threshold for dir > 0
            passed = mean_w < t_arr
        else:
            passed = mean_w > t_arr
        
        return passed.astype(int)
