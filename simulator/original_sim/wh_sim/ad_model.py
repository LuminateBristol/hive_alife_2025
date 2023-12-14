import os
from os.path import dirname, realpath
import json
from pathlib import Path
import sys
import numpy as np

try:
    dir_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(dir_root))
    from simulator.lib import RedisConn, RedisKeys
except Exception as e:
    pass

class ThresholdModel:

    def __init__(self, number_of_agents, threshold_file, stats_file, n, s, k=None,
        time_window=50, store_internal_res=False, store_internal_thresh=False):
        self.no_ag = number_of_agents
        self.n = n
        if k is None:
            k = n
        self.k = k
        self.s = s
        self.time_w = time_window
        self.counter = 1

        # in the form of {metric: [t value, direction]}
        # negative direction means greater than threshold is unhealthy
        self._load_thresh(threshold_file, stats_file)
        self.store_internal_res = store_internal_res
        self.store_internal_thresh = store_internal_thresh
        self.df_res = None
        self.df_thresh = None
        self.data = {}
        self.pred = [0]*self.no_ag

    def _load_thresh(self, t_path, s_path):
        with open(t_path, 'r') as f:
            t = f.readline()
            t = t.replace('\'', '\"').replace('None', 'null')
        
        tmp = json.loads(t)

        with open(s_path, 'r') as f:
            s = f.readline()
            s = s.replace('\'', '\"').replace('None', 'null')

        self.stats = json.loads(s)
        # order metrics by ES
        val = list(self.stats.values())
        val.sort(reverse=True)
        val_ = val[:self.n]
        thresh = {}
        for m, es in self.stats.items():
            if es in val_ and es >= self.s:
                thresh[m] = tmp[m]
                
        self.thresholds = thresh        

    # k: number of thresholds to pass to flag a fault
    def predict(self, data, counter):
        self.counter = counter
        res = np.zeros(self.no_ag)
        if self.counter < self.time_w:
            return res

        res = self.check_thresholds(data)
        no_thresh = int(len(res)/self.no_ag)
        res_ = res.reshape((no_thresh, self.no_ag))
        res_ = res_.sum(axis=0)
        
        if self.store_internal_res:
            if self.df_res is None:
                self.df_res = np.array([])
 
            self.df_res = np.concatenate([self.df_res, res_])
        
        pred = res_ >= self.k
        self.pred = pred.tolist()
        return pred

    def check_thresholds(self, data):
        self.counter += 1
        # a row of all thresholds passed (1) or not (0) for all agents
        # order: t0_a0, ..., t0_an, t1_a0, ..., t1_an, ..., tn_an
        total_passed = np.array([])         
        for metric, it in self.thresholds.items():
            passed = self.check_threshold_passed(metric, data)
            total_passed = np.concatenate((total_passed, passed))
            
        if self.store_internal_thresh:
            if self.df_thresh is None:
                self.df_thresh = np.array([])
            
            self.df_thresh = np.concatenate([self.df_thresh, total_passed])
        
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
        if metric not in self.data:
            self.data[metric] = d
        elif self.data[metric].shape[0] <= t:
            self.data[metric] = np.concatenate([self.data[metric], d], axis=0)            
        else:
            self.data[metric][t] = d
            
        mean_w = sum(self.data[metric])/self.time_w
        thresh = threshold[0]
        direction = threshold[1]
        t_arr = np.ones(self.no_ag)*thresh
        if direction > 0: # healthy is above threshold for dir > 0
            passed = mean_w < t_arr
        else:
            passed = mean_w > t_arr
            
        return passed.astype(int)

    def get_df_thresh(self):
        rows = self.counter-1-self.time_w
        t_len = int(len(self.df_thresh)/rows)
        return self.df_thresh.reshape((rows, t_len))

    def get_df_res(self):
        rows = int(len(self.df_res)/self.no_ag)
        return self.df_res.reshape((rows, self.no_ag))


class ExportThresholdModel(ThresholdModel):

    KEY = 'pred'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rconn = RedisConn()
        self.rkeys = RedisKeys()

    def predict(self, *args, **kwargs):
        pred = super().predict(*args, **kwargs)
        if not self.rconn.is_connected():
            self.rconn.reconnect()

        key = self.rkeys.gen_timestep_key(self.counter, self.KEY)
        self.rconn.set(key, json.dumps(pred.tolist()))
        
