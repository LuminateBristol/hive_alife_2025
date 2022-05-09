import os
import numpy as np
import pandas as pd

class FileList:

    def __init__(self, batch_id, ex_id, data_dir):
        dir_list = os.listdir(data_dir) # root data dir
        subdir_list = {}
        eid = ex_id.split('_')[1]
        
        for d in dir_list:
            d_path = os.path.join(data_dir, d)
            if os.path.isfile(d_path):
                continue

            meta = d.split('_')
            if meta[0] != 'e' or meta[1] != eid or meta[3] != str(batch_id):
                continue

            faults = int(meta[2][1:])
            subdir_list[faults] = d_path
        
        self.dir_list = subdir_list            
        
class DataSampler:

    SAMPLE_DIR = "samples"

    def __init__(self, batch_id, ex_id, data_dir, metrics, total_agents=10, np_seed=None):
        self.file_list = FileList(batch_id, ex_id, data_dir)
        self.metrics = metrics
        self.total_agents = total_agents
        self.data_dir = data_dir
        self.batch_id = batch_id
        self.ex_id = ex_id
        self.total_agents = total_agents
        self.export_dir = os.path.join(self.data_dir, self.SAMPLE_DIR, 
            "%s_%s"%(ex_id, str(batch_id)))

        if np_seed is not None:
            np.random.seed(int(np_seed))

    def process_all_metrics(self, export=False, np_seed=None):
        for metric in self.metrics:
            print("Sampling %s"%metric)
            self.sample_metric(metric, export, np_seed)

    def export_samples(self, metric, samples):
        try:
            self._mkdir(self.export_dir)
            export_path = os.path.join(self.export_dir, "%s.csv"%metric)
            df = pd.DataFrame(samples)
            df.to_csv(export_path, index=None)
            return True
        except Exception as e:
            raise Exception("Error in export: %s"%e)

    def _mkdir(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    # Take equal N/F samples from each fault number
    def sample_metric(self, metric, export=False, np_seed=None):
        if np_seed is not None:
            np.random.seed(int(np_seed))

        if metric not in self.metrics:
            raise Exception("Metric not in metric list")

        metric_id = self.metrics.index(metric)
        n_samples = []
        f_samples = []
        for faults, d_path in self.file_list.dir_list.items():
            print("-- Sampling %d faults for %s"%(faults, metric))
            pt = self._partition_files(d_path, faults)
            for f_path in pt['n']:
                n_sample = self._sample_from_file(faults, f_path, 1, metric_id)
                n_samples += n_sample.tolist()
            for f_path in pt['f']:
                f_sample = self._sample_from_file(faults, f_path, 0, metric_id)
                f_samples += f_sample.tolist()

        samples = {
            'n': n_samples,
            'f': f_samples
        }

        if export:
            self.export_samples(metric, samples)

        return samples

    def _partition_files(self, dir_path, faults):
        files = os.listdir(dir_path)
        if faults == 0: # take all normal samples
            n_sample_range = range(len(files))
            f_sample_range = []
        elif faults == self.total_agents:
            f_sample_range = range(len(files))
            n_sample_range = []
        else:  
            half_sample_size = int(len(files)/2)
            n_sample_range = range(half_sample_size)
            f_sample_range = range(half_sample_size, 2*half_sample_size)

        partition = {'n':[], 'f':[]}
        for i in n_sample_range:
            f_path = os.path.join(dir_path, files[i])
            partition['n'].append(f_path)

        for i in f_sample_range:
            f_path = os.path.join(dir_path, files[i])
            partition['f'].append(f_path)
        
        return partition

    # Takes a single sample from a file, given agent_state
    # agent_state 1: normal, 0: faulty
    def _sample_from_file(self, no_faults, filename, agent_state, metric_id):
        df = pd.read_csv(filename, index_col='ts')
        if agent_state == 0:
            ag_id = 0
        elif agent_state == 1:
            ag_id = no_faults
        else:
            raise Exception("Invalid agent state")

        col = "m%d_a%d"%(metric_id, ag_id)
        return df[col].sample()


    # # Returns an agent_id to sample
    # # state 0: faulty, state 1: normal
    # def _random_agent(self, faults, total_agents, state=0):
    #     if state == 0:
    #         ag_range = range(faults)
    #     else:
    #         ag_range = range(faults, total_agents)

    #     return 
        
