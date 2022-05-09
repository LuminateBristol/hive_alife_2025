from scipy.stats import mannwhitneyu as mwu
import pandas as pd
import os
import json
import seaborn as sns

class SigProc:

    def __init__(self, batch_id, ex_id, data_dir, export_dir):
        self.ex_dir = "%s_%s"%(ex_id, batch_id)
        self.data_dir = os.path.join(data_dir, self.ex_dir)
        self.export_dir = export_dir
        dir_list = os.listdir(self.data_dir)
        
        files = {}
        for f in dir_list:
            f_path = os.path.join(self.data_dir, f)
            metric = f.split('.')[0]
            files[metric] = f_path

        self.files = files
        self.sig = {}

    def sig_measure(self, dataset_1, dataset_2):
        return mwu(dataset_1, dataset_2)

    # U: MWU statistic
    # n: size of sample 1
    # m: size of sample 2
    def comp_effect_size(self, U, n, m, transformed=True):
        base = U/(n*m)

        if not transformed:
            return base

        return 2*abs(0.5-base)

    def compute_sig(self):
        sig = {}
        for metric, f in self.files.items():
            df = pd.read_csv(f, header=0)
            n = df['n'].tolist()
            f = df['f'].tolist()
            mwu_result = self.sig_measure(n, f)
            U = mwu_result.statistic
            es = self.comp_effect_size(U, len(n), len(f))
            sig[metric] = es
            del(df)

        self.sig = sig

    def _mkdir(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def export_sig(self):
        if len(self.sig) == 0:
            raise Exception("No sig values to export")

        export_path = os.path.join(self.export_dir, "%s.txt"%self.ex_dir)
        self._mkdir(self.export_dir)
        with open(export_path, 'w') as f:
            f.write(json.dumps(self.sig))

class DistPlot:

    def __init__(self, data_dir, batch_id, ex_id):
        self.ex_dir = "%s_%s"%(ex_id, batch_id)
        self.data_dir = os.path.join(data_dir, self.ex_dir)
        dir_list = os.listdir(self.data_dir)
        
        samples = {}
        metrics = []
        for f in dir_list:
            metric = f.split('.')[0]
            filepath = os.path.join(self.data_dir, f)
            df = pd.read_csv(filepath, header=0)
            samples[metric] = df
            metrics.append(metric)
            del(df)
        
        self.samples = samples
        self.metrics = metrics

    def plot(self, metric):
        if metric not in self.metrics:
            raise Exception("Invalid metric")

        df = self.samples[metric]
        sns.histplot(df)

    
class SigPlot:

    def __init__(self, stats_dir, config=None):
        stats = os.listdir(stats_dir)
        data = {}
        for f in stats:
            ex_id = f.split('_')[1]
            f_path = os.path.join(stats_dir, f)
            with open(f_path, 'r') as f:
                line = f.readline()
                line_ = line.strip().replace('\'', '\"').replace('None', 'null')
                sig = json.loads(line_)
            
            data[ex_id] = sig

        self.data = data

    def plot(self):
        df = pd.DataFrame(self.data)
        # _r reverses the normal order of the color map 'RdYlGn'
        sns.set(rc={'figure.figsize':(18, 8.27)})
        sns.heatmap(df.transpose(), cmap='RdYlGn_r', linewidths=0.5, annot=True)

class ThreshProc:

    def __init__(self, data_dir, batch_id, ex_id, export_dir):
        self.ex_path = "%s_%s"%(ex_id, batch_id)
        self.load_dir = os.path.join(data_dir, self.ex_path)
        self.export_dir = export_dir
        self.process_data()
        self.data = {}

    def process_data(self):
        data = {}
        for sample_file in os.listdir(self.load_dir):
            filepath = os.path.join(self.load_dir, sample_file)
            df = pd.read_csv(filepath)
            n = df['n'].mean()
            f = df['f'].mean()
            t = (n+f)/2
            if n > t:
                dirn = 1
            else:
                dirn = -1

            metric = sample_file.split('.')[0]
            data[metric] = [t, dirn]
            del(df)
            
        self.data = data

    def _mkdir(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def export_thresh(self):
        if len(self.data) == 0:
            raise Exception("No data to export")

        self._mkdir(self.export_dir)
        self.export_path = os.path.join(self.export_dir, "%s.txt"%self.ex_path)
        with open(self.export_path, 'w') as f:
            f.write(json.dumps(self.data))

