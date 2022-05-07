import os
import datetime
import pandas as pd

class SaveTo:

    BASE = 'data'

    def __init__(self, batch_id=None):
        if batch_id is None:
            self.batch = datetime.datetime.now().timestamp()
        else:
            self.batch = batch_id

    def mkdir(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def gen_save_dirname(self, ex_id, faults):
        dirname = '%s_f%d_%s'%(ex_id, faults, str(self.batch))
        save_to = os.path.join(self.BASE, dirname)
        self.mkdir(save_to)
        return save_to

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