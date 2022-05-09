import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import traceback
import json

parent_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(parent_dir))

from wh_sim import Config, DataModel

class ThreshProc:

    def __init__(self, ex_id, ex_cfg, data_dir, metrics=DataModel.metrics, sample_size=1, roc_prefix="#_"):
        self.dir_root = Path(__file__).resolve().parents[1]
        self.ex_id = ex_id
        self.data_dir = os.path.join(self.dir_root, "data", data_dir, self.ex_id, "compiled") 
        self.metrics = metrics
        self.roc_metrics = []
        self.sample_size = sample_size
        self.roc_prefix = roc_prefix
        self.data_uh = {}
        self.data_h = {}
        self.sig = {}

        self.ex_cfg = Config(setup_default=False, filename=ex_cfg).get_key(self.ex_id)
        self.no_agents = self.ex_cfg['cfg']['warehouse']['number_of_agents']
    
        self._load_data()

    def _load_data(self):
        files = os.listdir(self.data_dir)
        
        for i, f in enumerate(files):
            file_path = os.path.join(self.data_dir, f)
            if not os.path.isfile(file_path):
                print('Skipping file')
                continue

            try:
                meta = f.split('.')[0].split('_')
                no_faulty = int(meta[0][:-1])
                healthy = meta[1][:-1] == 'h'
                partition = meta[1][-1]
                sample_size = int(meta[2][1:])
            except:
                continue

            if sample_size != self.sample_size:
                continue
            
            df = pd.read_csv(file_path, dtype=float)            
            if healthy:
                data = self.data_h
            else:
                data = self.data_uh

            for metric in self.metrics:
                key = "%s:%s"%(metric, partition)
                try:
                    d = df[metric]
                    if key not in data:
                        data[key] = {}
                 
                    data[key][no_faulty] = d.to_numpy()
                except Exception as e:
                    # print(e)
                    pass

                roc_metric = self._gen_roc_metric(metric)
                roc_key = "%s:%s"%(roc_metric, partition)
                try:
                    d = df[roc_metric]
                    if roc_key not in data:
                        data[roc_key] = {}
                 
                    data[roc_key][no_faulty] = d.to_numpy()
                    self.roc_metrics.append(roc_metric)
                except Exception as e:
                    # print(e)
                    pass

    def _gen_roc_metric(self, metric):
        return self.roc_prefix + metric

    def _compile_metric_data(self, metric):
        p1h = self.data_h["%s:1"%metric]
        p2h = self.data_h["%s:2"%metric]
        p1uh = self.data_uh["%s:1"%metric]
        p2uh = self.data_uh["%s:2"%metric]
        
        dh = np.array([])
        duh = np.array([])
        for f, v in p1h.items():
            dh = np.concatenate([dh, v, p2h[f]])
            if f > 0:
                duh = np.concatenate([duh, p1uh[f], p2uh[f]])

        duh = np.concatenate([duh, p1uh[self.no_agents], p2uh[self.no_agents]])
        return dh, duh

    def plot_distribution(self, metric, legend=False, bins=10, kde=True, alpha=0.4):
        dh, duh = self._compile_metric_data(metric)
        # fig, ax = plt.subplots()
        # ax.boxplot([dh, duh])
        # plt.show()
        data = pd.DataFrame({'H':dh, 'F':duh})
        ax = sb.histplot(data, legend=legend, bins=bins, edgecolor="k", linewidth=0, 
            alpha=alpha, kde=kde)
        
        # h,l = ax.get_legend_handles_labels()
        # l = data.columns.to_list()
        # plt.legend(loc='center right', bbox_to_anchor=(-1.25, 0.5), ncol=1)
        # plt.boxplot(duh)

    def compute_threshold(self, metric, learning_rate=0.1, max_iterations=100):
        dh, duh = self._compile_metric_data(metric)
        h_m = np.mean(dh)
        uh_m = np.mean(duh)
        t_arr = [(h_m+uh_m)/2]
        w_arr = [1]
        delta = [0]
        df = pd.DataFrame([dh.tolist() + (-1*duh).tolist()])
        
        print("H-mean %f "%(h_m))
        print("UH-mean %f "%(uh_m))
        print("Intial threshold set to %f"%(t_arr[0]))
        for i in range(max_iterations):
            t = t_arr[i]
            w = w_arr[i]
            d0 = delta[i]
            print("Threshold: %f    Weight: %f  Delta: %f"%(t,w,d0))
            
            if i > 5 and np.mean(delta[i:i+5])-d0 < 0.01:
                print("Little delta change: stop")
                break

            diff = (w*df - t)
            misclassed = diff < 0
            total_miss = np.sum(misclassed, axis=1)
            # print("Misclassed: %d"%total_miss)
            d = learning_rate * float(np.sum((misclassed*df), axis=1))
            t_arr.append(d*t)
            w_arr.append(d*w)
            delta.append(d)
            # print("\n-----------------------------------\n")

        thresh = [t_arr[i]/w_arr[i] for i in range(len(t_arr))]
        return thresh
            
            

            


