from scipy.stats import mannwhitneyu as mwu
import pandas as pd
import os
import json
import seaborn as sns

# @TODO Move load/export path configuration to lib

class SigProc:

    def __init__(self, batch_id, ex_id, data_dir, export_dir):
        self.ex_dir = "%s_%s"%(ex_id, batch_id)
        self.data_dir = os.path.join(data_dir, batch_id, ex_id)
        self.export_dir = export_dir
        dir_list = os.listdir(self.data_dir)
        
        files = {}
        for f in dir_list:
            f_path = os.path.join(self.data_dir, f)
            metric = f.split('.')[0]
            if metric in ['nearest_agent_id', 'nearest_wall_id', 'nearest_box_id', 'nearest_id', 'heading']:
                continue

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
            n = df['n']
            f = df['f']
            if metric in ['nearest_agent_distance', 'nearest_box_distance', 'nearest_wall_distance', 'nearest_combined_distance']:
                n = n.fillna(100)
                f = f.fillna(100)
                
            n = n.tolist()
            f = f.tolist()                
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
        # self.ex_dir = "%s_%s"%(ex_id, batch_id)
        self.data_dir = os.path.join(data_dir, batch_id, ex_id)
        dir_list = os.listdir(self.data_dir)
        
        samples = {}
        metrics = []
        for f in dir_list:
            metric = f.split('.')[0]
            filepath = os.path.join(self.data_dir, f)
            df = pd.read_csv(filepath, header=0)
            if metric in ['nearest_agent_distance', 'nearest_box_distance', 'nearest_wall_distance', 'nearest_combined_distance']:
                df = df.fillna(100)

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

    def __init__(self, stats_dir, batch_id, config=None):
        stats = os.listdir(stats_dir)
        data = {}
        for f in stats:
            meta = f.split('_')
            ex_id = meta[1]
            bid = '_'.join(meta[2:]).split('.')[0]
            if bid != batch_id:
                continue

            f_path = os.path.join(stats_dir, f)
            with open(f_path, 'r') as f:
                line = f.readline()
                line_ = line.strip().replace('\'', '\"').replace('None', 'null')
                sig = json.loads(line_)
            
            if config is not None:
                ylab = config[int(ex_id)]
            else:
                ylab = ex_id

            data[ylab] = sig

        self.data = data

    def plot(self):
        df = pd.DataFrame(self.data)
        df = df.reindex(sorted(df.columns), axis=1)
        df = df.reindex(sorted(df.index), axis=0)
        # _r reverses the normal order of the color map 'RdYlGn'
        sns.set(rc={'figure.figsize':(22, 6)})
        sns.heatmap(df.transpose(), linewidths=0.5, annot=True)

class ThreshProc:

    def __init__(self, data_dir, batch_id, ex_id, export_dir):
        self.ex_path = "%s_%s"%(ex_id, batch_id)
        self.load_dir = os.path.join(data_dir, batch_id, ex_id)
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

