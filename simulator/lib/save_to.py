import os
import datetime
import pandas as pd

class GenDir:

    BASE = 'data'

    def __init__(self, batch_id=None):
        if batch_id is None:
            self.batch = datetime.datetime.now().timestamp()
        else:
            self.batch = batch_id

    def mkdir(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def gen_save_dirname(self, ex_id, faults, makedir=True):
        dirname = '%s_f%d'%(ex_id, faults)
        target = os.path.join(self.BASE, str(self.batch), dirname)
        if makedir:
            self.mkdir(target)
        return target

class SaveTo(GenDir):

    def export_data(self, data_model, ex_id, faults, random_seed=None):
        ts = datetime.datetime.now().timestamp()
        data = data_model.get_dataframe()
        dirname = self.gen_save_dirname(ex_id, faults)
        if random_seed is None:
            filename = '%d.csv'%ts
        else:
            filename = '%d.csv'%random_seed

        save_path = os.path.join(dirname, filename)
        try:
            data.to_csv(save_path, index=False)
        except Exception as e:
            print(e)

# Use with MinimalDataModel and ExtremeMinModel
class SaveSample(SaveTo):

    def export_data(self, data_model, ex_id, faults, random_seed=None):
        ts = datetime.datetime.now().timestamp()
        data = data_model.sample_data()
        dirname = self.gen_save_dirname(ex_id, faults)
        if random_seed is None:
            filename = '%d.csv'%ts
        else:
            filename = '%d.csv'%random_seed

        save_path = os.path.join(dirname, filename)
        try:
            data.to_csv(save_path, index=False)
        except Exception as e:
            print(e)

class SaveRes(GenDir):

    BASE = 'res'

    def __init__(self, ex_id, batch_id=None):
        if batch_id is None:
            self.batch = datetime.datetime.now().timestamp()
        else:
            self.batch = batch_id
        
        self.save_dir = self.gen_save_dirname(ex_id)

    def mkdir(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def gen_save_dirname(self, ex_id, makedir=True):
        target = os.path.join(self.BASE, self.batch, ex_id)
        if makedir:
            self.mkdir(target)
        return target

    def export_data(self, data, faults, seed):
        filepath = "f%d_%d.csv"%(faults, seed)
        save_path = os.path.join(self.save_dir, filepath)
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False, header=False)


# class LoadSample(GenDir):

#     def load_data(self, ex_id):
#         load_path = os.path.join(self.BASE, str(self.batch))
