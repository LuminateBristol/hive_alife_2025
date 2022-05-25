import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ModelEv:

    def __init__(self, data_dir, batch_id, ex_id):
        load_path = os.path.join(data_dir, batch_id, ex_id)
        files = os.listdir(load_path)
        files_ = {}
        for f in files:
            filepath = os.path.join(load_path, f)
            meta = f.split('_')
            files_[filepath] = int(meta[0][1:])

        self.files = files_
        self.results = {}

    # Compute accuracy per timestep for each simulation run
    def evaluate(self, k, compute_sd=True):
        a_arr = None
        se_arr = None
        sp_arr = None

        for fpath, faults in self.files.items():
            a,se,sp = self.evaluate_file(fpath, faults, k)
            if a_arr is None:
                a_arr = a
                se_arr = se
                sp_arr = sp
            else:
                a_arr = pd.concat([a_arr, a], axis=1)
                se_arr = pd.concat([se_arr, se], axis=1)
                sp_arr = pd.concat([sp_arr, sp], axis=1)

        a_mean = a_arr.mean(axis=1)
        se_mean = se_arr.mean(axis=1)
        sp_mean = sp_arr.mean(axis=1)

        if compute_sd:
            x_range = a_mean.shape[0]
            a_std = a_arr.std(axis=1)
            se_std = se_arr.std(axis=1)
            sp_std = sp_arr.std(axis=1)
        else:
            a_std = None
            se_std = None
            sp_std = None

        results = {
            'a': [a_mean, a_std],
            'se': [se_mean, se_std],
            'sp': [sp_mean, sp_std]
        }

        self.results[k] = results
        return results


    def evaluate_file(self, fpath, faults, k):
        df = pd.read_csv(fpath)
        rows = df.shape[0]
        no_agents = df.shape[1]
        is_faulty = df >= k
        truth = [1]*faults+[0]*(no_agents-faults)
        truth = np.array(truth)
        truth_arr = np.tile(truth, (rows, 1))
        match = is_faulty + truth_arr

        TP = (match == 2).sum(axis=1)
        TN = (match == 0).sum(axis=1)
        FP = ((truth_arr == 0) & (is_faulty == 1)).sum(axis=1)
        FN = ((truth_arr == 1) & (is_faulty == 0)).sum(axis=1)        
        a, se, sp = self._compute_scores(TP,TN,FP,FN)
        return a,se,sp

    def _compute_scores(self,TP,TN,FP,FN):
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        sensitivity = 1 - (FN/(TP+FN))
        specificity = 1 - (FP/(TN+FP))
        return accuracy, sensitivity, specificity

    def plot(self, k, compute_sd=True, plot_sd=True):
        if k not in self.results:
            self.evaluate(k, compute_sd)

        results = self.results[k]
        a, a_std = results['a']
        se, se_std = results['se']
        sp, sp_std = results['sp']
        a_m1 = a+a_std
        se_m1 = se+se_std
        sp_m1 = sp+sp_std
        a_m1[a_m1>1] = 1
        se_m1[se_m1>1] = 1
        sp_m1[sp_m1>1] = 1
        a_m0 = a-a_std
        se_m0 = se-se_std
        sp_m0 = sp-sp_std
        x_range = range(a.shape[0])
        
        fig, ax = plt.subplots()
        ax.fill_between(x_range, a_m0, a_m1, alpha=0.05,color='tab:blue')
        ax.fill_between(x_range, se_m0, se_m1, alpha=0.05,color='tab:green')
        ax.fill_between(x_range, sp_m0, sp_m1, alpha=0.05,color='tab:orange')
        
        ax.plot(a,color='tab:blue')
        ax.plot(se,color='tab:green')
        ax.plot(sp,color='tab:orange')
        ax.legend(['A','SE','SP', 'Std A', 'Std SE', 'Std SP'])
        ax.set_xlabel("Simulation timestep")
        ax.set_ylabel("Score")
        # plt.show()
        # plt.rcParams.update({'font.size': 15})

class ROC_Ev:

    # emin_sc_3, emin_sc_5
    def __init__(self, data_dir, batch_id):
        self.n_range = [3,5]
        self.ex_ids = ["e_%d"%x for x in range(1,7)]
        self.data_dir = data_dir
        self.batch_id = batch_id
        
    def comp_roc(self, n, init_ts=0, end_ts=None):
        data = {}
        k_range = range(1,n+1)
        for ex_id in self.ex_ids:
            if ex_id not in data:
                data[ex_id] = {}
            
            me = ModelEv(self.data_dir, "%s_%d"%(self.batch_id,n), ex_id)
            for k in k_range:
                results = me.evaluate(k)
                if end_ts is None:
                    end_ts = len(results['se'][0])

                se_mean = results['se'][0][init_ts:end_ts].mean()
                sp_mean = results['sp'][0][init_ts:end_ts].mean()
                lab = "n%d_k%d"%(n,k)
                data[ex_id][lab] = [se_mean, 1-sp_mean]

        self.data = data
        return data


