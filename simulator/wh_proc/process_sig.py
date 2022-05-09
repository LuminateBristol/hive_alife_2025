import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import traceback
import json

parent_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(parent_dir))

from wh_sim import Config, DataModel
from scipy.stats import mannwhitneyu as mwu


# significance processor
class SigProc:

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

            if sample_size != self.sample_size:# or no_faulty in [0, self.no_agents]:
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

                if metric == "heading":
                    continue

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

    def sig_measure(self, dataset_1, dataset_2):
        return mwu(dataset_1, dataset_2)

    # effect size
    def comp_effect_size(self, U, n, m):
        return U/(n*m)

    # performs mann-whitney to compare healthy and unhealthy datasets 
    # across range of faulty robots in the scenario
    # also compares same health sets as baseline
    def compute_sig_for_metric(self, metric):
        sig = {}
        a_measure = {}
        p1 = "%s:%s"%(metric, 1)
        p2 = "%s:%s"%(metric, 2)
        h1 = self.data_h[p1]
        h2 = self.data_h[p2]
        uh1 = self.data_uh[p1]
        uh2 = self.data_uh[p2]
        datasets = {1: [h1, uh2], 0: [h1, h2]} # 0 corresponds to baseline
        
        for i in range(1, self.no_agents):
            for key, data in datasets.items():
                try:
                    d1 = data[0][i]
                    d2 = data[1][i]
                    if metric in ['nearest_agent_id', 'nearest_box_id', 'nearest_wall_id', 'nearest_id',]:
                        np.nan_to_num(d1, nan=-1, copy=False)
                        np.nan_to_num(d2, nan=-1, copy=False)
                    
                    sig_m = self.sig_measure(d1, d2)
                    sig["%d_%s"%(i,key)] = {'statistic': sig_m.statistic, 'pvalue': sig_m.pvalue}
                    a_measure["%d_%s"%(i,key)] = self.comp_effect_size(sig_m.statistic, len(d1), len(d2))
                except ValueError as e:
                    if str(e) == 'All numbers are identical in mannwhitneyu':
                        sig["%d_%s"%(i,key)] = {'statistic': None, 'pvalue': None}
                        a_measure["%d_%s"%(i,key)] = 0.5 # no difference in distns
                        # print(e)
                    else:
                        print(traceback.format_exc())
                except Exception as e:
                    print(i, traceback.format_exc())
                    pass

        return sig, a_measure

    
    # performs mann-whitney to compare healthy and unhealthy datasets 
    # across range of faulty robots in the scenario
    # also compares same health sets as baseline
    def compute_sig_for_metric_collapse(self, metric):
        sig = {}
        a_measure = {}
        p1 = "%s:%s"%(metric, 1)
        p2 = "%s:%s"%(metric, 2)
        h1 = self.data_h[p1]
        h2 = self.data_h[p2]
        uh1 = self.data_uh[p1]
        uh2 = self.data_uh[p2]
        datasets = {1: [h1, uh2], 0: [h1, h2]} # 0 corresponds to baseline
        
        for key, data in datasets.items():
            d1 = []
            d2 = []
            for i in range(0, self.no_agents+1):
                if key == 0 and i == self.no_agents:
                    continue
                
                if i == self.no_agents:
                    d1_ = np.array([])
                    d2_ = uh1[i]
                elif key == 0 and i == 0 or i != 0: # include 0 faulty in baseline (h/h)
                    d1_ = data[0][i]
                    d2_ = data[1][i]                
                elif i == 0:
                    d1_ = data[0][i]
                    d2_ = np.array([])

                if metric in ['nearest_agent_id', 'nearest_box_id', 'nearest_wall_id', 'nearest_id',]:
                    np.nan_to_num(d1_, nan=-1, copy=False)
                    np.nan_to_num(d2_, nan=-1, copy=False)
                
                d1 += d1_.tolist()
                d2 += d2_.tolist()
            
            try:
                sig_m = self.sig_measure(d1, d2)
                sig[str(key)] = {'statistic': sig_m.statistic, 'pvalue': sig_m.pvalue}
                a_measure[str(key)] = self.comp_effect_size(sig_m.statistic, len(d1), len(d2))
            except ValueError as e:
                if str(e) == 'All numbers are identical in mannwhitneyu':
                    sig[str(key)] = {'statistic': None, 'pvalue': None}
                    a_measure[str(key)] = 0.5 # no difference in distns
                    # print(e)
                else:
                    print(traceback.format_exc())
            except Exception as e:
                print(traceback.format_exc())
                pass

        return sig, a_measure

    def compute_sig_for_metric_collapse_limit0nf(self, metric, faulty_range_end=1):
        sig = {}
        a_measure = {}
        p1 = "%s:%s"%(metric, 1)
        p2 = "%s:%s"%(metric, 2)
        h1 = self.data_h[p1]
        h2 = self.data_h[p2]
        uh1 = self.data_uh[p1]
        uh2 = self.data_uh[p2]
        datasets = {1: [h1, uh2], 0: [h1, h2]} # 0 corresponds to baseline
        
        # compute number of samples to take from each faulty set so that h/uh sets have same size for comparison
        sample_size = len(h1[0]) # assume all sizes equal
        size_uh = faulty_range_end*sample_size # assume 0 faulty always included
        number_healthy_samples_to_take = int(size_uh/(faulty_range_end+1))
        
        for key, data in datasets.items():
            d1 = []
            d2 = []
            for i in range(0, faulty_range_end+1):
                if key == 0 and i == self.no_agents:
                    continue
                
                if i == self.no_agents:
                    d1_ = np.array([])
                    d2_ = uh1[i]
                elif key == 0 and i == 0 or i != 0: # include 0 faulty in baseline (h/h)
                    d1_ = data[0][i]
                    d2_ = data[1][i]                
                elif i == 0:
                    d1_ = data[0][i]
                    d2_ = np.array([])

                if metric in ['nearest_agent_id', 'nearest_box_id', 'nearest_wall_id', 'nearest_id',]:
                    np.nan_to_num(d1_, nan=-1, copy=False)
                    np.nan_to_num(d2_, nan=-1, copy=False)
                
                if key == 1:
                    d1_ = d1_[:number_healthy_samples_to_take]

                d1 += d1_.tolist()
                d2 += d2_.tolist()
            
            try:
                sig_m = self.sig_measure(d1, d2)
                sig[str(key)] = {'statistic': sig_m.statistic, 'pvalue': sig_m.pvalue}
                a_measure[str(key)] = self.comp_effect_size(sig_m.statistic, len(d1), len(d2))
            except ValueError as e:
                if str(e) == 'All numbers are identical in mannwhitneyu':
                    sig[str(key)] = {'statistic': None, 'pvalue': None}
                    a_measure[str(key)] = 0.5 # no difference in distns
                    # print(e)
                else:
                    print(traceback.format_exc())
            except Exception as e:
                print(traceback.format_exc())
                pass

        return sig, a_measure

    def export_sig(self, export_dir=None):
        if export_dir is None:
            export_dir = os.path.join(self.dir_root, "plot", "sig")

        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        filename = "%s.csv"%self.ex_id
        export_file = os.path.join(export_dir, filename)
        result = {}
        metrics = self.metrics + self.roc_metrics
        for metric in metrics:
            try:
                sig, a = self.compute_sig_for_metric(metric)
                sig_c, a_c = self.compute_sig_for_metric_collapse(metric)
                sig_c_01, a_c_01 = self.compute_sig_for_metric_collapse_limit0nf(metric, 1)
                result[metric] = {
                    'sig': sig, 
                    'a_measure': a,
                    'sig_c': sig_c, 
                    'a_measure_c': a_c,
                    'sig_c_01': sig_c_01, 
                    'a_measure_c_01': a_c_01
                }
            except Exception as e:
                print(traceback.format_exc())
                # pass
        
        df = pd.DataFrame(result, 
            index=['sig', 'a_measure', 'sig_c', 'a_measure_c', 'sig_c_01', 'a_measure_c_01'])
        df.to_csv(export_file)
        del df


class SigResult:

    def __init__(self, load_dir=None):
        if load_dir is None:
            dir_root = Path(__file__).resolve().parents[0]
            load_dir = os.path.join(dir_root, "sig")

        files = os.listdir(load_dir)
        self.result = {}
        self.metrics = {}
        for i, f in enumerate(files):
            file_path = os.path.join(load_dir, f)
            ex_id = f.split('.')[0]
            df = pd.read_csv(file_path, index_col=0)
            self.result[ex_id] = df
            self.metrics[ex_id] = df.columns
            del df

    def get_metrics(self, ex_id):
        return self.metrics[ex_id]

    class Result:

        SIG = 0
        A_MEASURE = 1

        def __init__(self, data, metric, mode=SIG):
            self.comp = {}
            self.base = {}
            self.mode = mode
            self._load(data, metric)

        def _load(self, data, metric):
            if metric not in data:
                return

            if self.mode == self.SIG:
                data = data[metric]['sig']
            else:
                data = data[metric]['a_measure']
            
            data = json.loads(data.replace('\'', '\"').replace('None', 'null'))
            for key, it in data.items():
                meta = key.split('_')
                no_faulty = meta[0]
                base = meta[1] == '0'

                if self.mode == self.SIG:
                    it = it['pvalue']

                if base:
                    self.base[no_faulty] = it
                else:
                    self.comp[no_faulty] = it

    class CollapsedResult:

        SIG = 0
        A_MEASURE = 1

        def __init__(self, data, metric, mode=SIG, limit_01=False):
            self.comp = None
            self.base = None
            self.mode = mode
            self.limit_01 = limit_01
            self._load(data, metric)

        def _load(self, data, metric):
            if metric not in data:
                return

            suffix = "_c"
            if self.limit_01:
                suffix = suffix + "_01"
            
            if self.mode == self.SIG:
                key = "sig"
            else:
                key = "a_measure"
            
            data = data[metric][key+suffix]
            data = json.loads(data.replace('\'', '\"').replace('None', 'null'))
            
            # key 0 is base comparison
            if self.mode == self.SIG:
                self.comp = data["1"]['pvalue']
                self.base = data["0"]['pvalue']
            else:
                self.comp = data["1"]
                self.base = data["0"]


    # mode=0: significance
    # mode=1: effect size
    def _load_result(self, data, metric, mode=0):
        result = self.Result(data, metric, mode)
        base = result.base
        comp = result.comp
        
        x = []
        y = []
        colour = []
        for key, it in base.items():
            c_val = comp[key]
            if mode == 0:
                if it is None:
                    it = 0.5
                if c_val is None:
                    c_val = 0.5
            else:
                it = abs(0.5-it)
                c_val = abs(0.5-c_val)

            x.append(key)
            y.append(it)
            x.append("%sc"%key)
            y.append(c_val)            
            colour += ['red', 'blue']
        
        return x, y, colour

    # mode=0: significance
    # mode=1: effect size
    def _load_collapsed_result(self, data, metric, mode=0, limit_01=False):
        result = self.CollapsedResult(data, metric, mode, limit_01)
        base = result.base
        comp = result.comp
        
        x = []
        y = []
        colour = []

        if mode == 0:
            if base is None:
                base = 0.5
            if comp is None:
                comp = 0.5
        else:
            base = abs(0.5-base)
            comp = abs(0.5-comp)

        x = ["Same state", "Different state"]
        y = [base, comp]          
        colour = ['red', 'blue']
        
        return x, y, colour

    # mode=0: significance
    # mode=1: effect size
    def plot_comp_chart(self, ex_id, mode=0,
        no_cols=4, figsize=(15,30), title=None, metrics=[], show_legend=True,
        y_max=[0.5,0.5], collapse_result=False, limit_01=False):
        
        if len(metrics) == 0:
            metrics = self.get_metrics(ex_id)
        
        rows = int(np.ceil(len(metrics)/no_cols))
        fig, axs = plt.subplots(rows, no_cols, figsize=figsize)

        if mode == 0:
            yaxis_lab = "p value"
        else:
            yaxis_lab = "effect size"

        data = self.result[ex_id]
        for idx, metric in enumerate(metrics):
            if metric not in data:
                continue
            
            if collapse_result:
                x, y, colour = self._load_collapsed_result(data, metric, mode, limit_01)
            else:
                x, y, colour = self._load_result(data, metric, mode)

            i = int(np.floor(idx/no_cols))
            j = idx%no_cols
            if rows == 1:
                ax = axs[j]
            else:
                ax = axs[i, j]
            
            # bars = ax.bar(x, y, alpha=0.5)
            bars = ax.bar(x, y, color=colour, alpha=0.5)
            ax.set_xticks(x)
            ax.set_ylim([0, y_max[mode]])
            if mode == 0:
                vline = ax.axhline(y=0.05, xmin=-0.5, xmax=10.5, linewidth=1, linestyle='--', color='purple')
            ax.set_title(metric)                    

        
        fig.text(0.5, -0.04, 'no. faulty robots', ha='center', fontsize='xx-large')
        fig.text(-0.04, 0.5, yaxis_lab, va='center', rotation='vertical', fontsize='xx-large')
        if title is not None:
            fig.suptitle(title, fontsize=16, y=1.0)

        if show_legend:
            # Create the legend
            if mode == 0:
                handles = [vline]+bars.patches
                labels = [
                    '0.05 pvalue threshold for significance',
                    'Same state (H-H or H-UH) comparison',
                    'Different state (H-UH) comparison'
                ]
            else:
                handles = bars.patches
                labels = [ 
                    'Same state (H-H or H-UH) comparison',
                    'Different state (H-UH) comparison',
                ]

            fig.legend(
                    handles=handles,     # The line objects
                    labels=labels,   # The labels for each line
                    loc="lower right",   # Position of legend
                    borderaxespad=0.5,    # Small spacing around legend box
                    title="Key"  # Title for the legend
                    )

            # Adjust the scaling factor to fit your legend text completely outside the plot
            # (smaller value results in more space being made for the legend)
            # plt.subplots_adjust(bottom=-0.95)
                            
        fig.tight_layout()
        return

    # mode=0: linear
    # mode=1: quadratic
    def _weight(self, es, mode=0):
        if mode == 0:
            return es
        elif mode == 1:
            return es*es

    def sum_metric_effect_size(self, ex_id, metric, weight=0):
        data = self.result[ex_id]
        result = self.Result(data, metric, self.Result.A_MEASURE)
        b = list(result.base.values())
        c = list(result.comp.values())
        b = abs(0.5-np.array(b))
        c = abs(0.5-np.array(c))
        diff = self._weight(c, weight) - self._weight(b, weight)
        diff[diff<0] = 0
        return sum(diff)

    def sum_metric_effect_size_collapsed(self, ex_id, metric, weight=0, limit_01=False):
        data = self.result[ex_id]
        result = self.CollapsedResult(data, metric, self.CollapsedResult.A_MEASURE, limit_01)
        b = abs(0.5-result.base)
        c = abs(0.5-result.comp)
        return 2*c
        diff = self._weight(c, weight) - self._weight(b, weight)
        if diff < 0:
            diff = 0
        return diff

    def plot_effect_size_dotplot(self):
        ex_ids = list(self.result.keys())
        metrics = self.metrics[ex_ids[0]] # assume same metrics
        
        plot_w = len(ex_ids)
        plot_h = len(metrics)+1
        fig, axs = plt.subplots(figsize=(plot_w,plot_h))
        dot_size_range = [0.1, 1]
        dot_spacing = 0.5 # each dot has max radius 1 with border padding 0.5, so dot lies in 1.5 wide square
        
        col = 1
        for ex_id in ex_ids:  
            for row, metric in enumerate(metrics):
                result = self.sum_metric_effect_size(ex_id, metric)

                if result < 0.1:
                    dot_size = 0.1
                else:
                    dot_size = result

                y = row+1#(row+1)*(dot_size_range[1]+dot_spacing)/2
                x = col#col*(dot_size_range[1]+dot_spacing)/2
                c = plt.Circle((x, y), dot_size, color='red', alpha=0.5)

                axs.set_aspect('equal')
                axs.add_patch(c)

            col += 1
            plt.xlim([0,plot_w])
            plt.ylim([0,plot_h])
            plt.xticks(range(1,len(ex_ids)+1))
            plt.yticks(np.arange(1, plot_h), metrics)
            axs.xaxis.tick_top()

            fig.text(0.5, 0.9, 'no. faulty robots', ha='center', fontsize='xx-large')
            fig.text(-0.095, 0.5, 'metric', va='center', rotation='vertical', fontsize='xx-large')
    
    def get_combined_metrics(self):
        return list(self.metrics.values())[0]
        metrics = []
        for ex_id, it in self.metrics.items():
            metrics += it.tolist()

        return list(set(metrics))

    def _order_metrics(self, metrics, metric_order=0):
        roc = []
        default = []
        for metric in metrics:
            if metric[:2] == "#_":
                roc.append(metric)
            else:
                default.append(metric)

        if metric_order == 1:
            return default + roc

        ordered = []
        ordered_2 = []
        for metric in default:
            roc_metric = "#_%s"%metric
            if roc_metric in roc:
                ordered += [metric, roc_metric]
            else:
                ordered_2.append(metric)

        return ordered + ordered_2

    def _filter_metrics(self, metrics, skip_metrics=[], show_default=True, show_ROC=True):
        filtered = []
        for metric in metrics:
            is_roc = metric[:2] == "#_"

            if metric in skip_metrics:
                continue

            if not show_ROC and is_roc:
                continue

            if not show_default and not is_roc:
                continue

            filtered.append(metric) 
        
        return filtered

    # mode=0: remove underscores 
    # mode=1: replace #_ with ROC
    def _generate_metric_labels(self, metrics, mode=0):
        labels = []
        for metric in metrics:
            metric = str.capitalize(metric)
            label = metric.replace("#_", "ROC ")
            label = label.replace("_", " ")
            label = label.replace("id", "ID")
            labels.append(label)
        
        return labels

    # metric_order=0: alphabetical (metric, roc_metric) pairs
    # metric_order=1: alphabetical metrics followed by alphabetic roc_metrics
    # ex_ids: in format (ex_id, label, description)
    def plot_effect_size_heatmap(self, figsize=(15,15), skip_metrics=[], 
        show_default=True, show_ROC=True, metric_order=0, custom_metric_order=[], 
        ex_ids=[], ex_id_labels={}, metric_labels={}, max_value=None, dark_label_t=0.7, title=None,
        effect_size_weighting=0, flip_x_y=False, collapse_result=False, limit_01=False,
        fault_label_short=True, save_as=None):
        
        dir_main = Path(__file__).resolve().parents[1]
        
        if len(custom_metric_order):
            metrics = custom_metric_order
        else:
            metrics = self._order_metrics(self.get_combined_metrics(), metric_order)

        metrics = self._filter_metrics(metrics, skip_metrics, show_default, show_ROC)
        if len(ex_ids) == 0:
            ex_ids = self.result.keys()
        
        y_ticks = metrics
        cfg = {}
        # if len(ex_id_labels) == 0:
        for i, it in enumerate(ex_ids):
            if it in ex_id_labels:
                continue
            spl = it.split("_")
            cfg_no = spl[0][1:]
            if cfg_no not in cfg:
                cfg_file = "ex_%s.yaml"%cfg_no
                cfg_path = os.path.join(dir_main, "cfg", cfg_file)
                cfg[cfg_no] = Config(setup_default=False, filename=cfg_path)
            
            cfg_ = cfg[cfg_no].get_key(it)
            if "description" in cfg_:
                desc = cfg_["description"]
            else:
                desc = it
            
            if fault_label_short:
                ex_id_labels[it] = "F%d"%(i+1)
            else:
                ex_id_labels[it] = "F%d: %s"%(i+1, desc)

        x_ticks = []
        for e_id in ex_ids:
            x_ticks.append(ex_id_labels[e_id])
        
        result_matrix = {}
        for i, ex_id in enumerate(ex_ids):
            result_matrix[ex_id] = {}

            for j, metric in enumerate(metrics):
                # if metric in skip_metrics:
                #     continue

                if collapse_result:
                    result = self.sum_metric_effect_size_collapsed(ex_id, metric, effect_size_weighting, limit_01)
                else:
                    result = self.sum_metric_effect_size(ex_id, metric, effect_size_weighting)
                
                if max_value is not None and result > max_value:
                    result = max_value

                result_matrix[ex_id][metric] = result
        
        values = pd.DataFrame(result_matrix)
        if len(metric_labels):
            y_labels = []
            for m in y_ticks:
                if m in metric_labels:
                    y_labels.append(metric_labels[m])
                else:
                    y_labels.append(m)
        else:
            y_labels = self._generate_metric_labels(y_ticks, 1)

        if flip_x_y:
            values = values.transpose(copy=False)
            y_ticks = x_ticks
            x_ticks = y_labels

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(values, vmax=max_value)
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.57)
        cbar.ax.tick_params(labelsize='x-large') 
        cbar.ax.set_ylabel("Effect size", rotation=-90, va="bottom", labelpad=5)
        # if max_value is not None:
        #     cbar_ticks = cbar.get_ticks()
        #     cap = "%s+"%cbar_ticks[-1]
        #     new_ticks = tuple(cbar_ticks[:-1]) + (cap]
        #     cbar.ax.set_yticklabels(new_ticks)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(x_ticks)))
        ax.set_yticks(np.arange(len(y_ticks)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(x_ticks, fontdict={'fontsize':'x-large'})
        ax.set_yticklabels(y_ticks, fontdict={'fontsize':'x-large'})

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right",
                rotation_mode="anchor")
        # plt.setp(ax.get_xticklabels(), ha="right",
        #         rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i, ex_id in enumerate(ex_ids):
            for j, metric in enumerate(metrics):
                # if metric in skip_metrics:
                #     continue
                if flip_x_y:
                    value = values[metric][ex_id]
                    x = j
                    y = i
                else:
                    value = values[ex_id][metric]
                    x = i
                    y = j
                
                if value > dark_label_t:
                    colour = '#0485d1'
                else:
                    colour = 'w'
                
                value = np.around(value, 2)
                # if value == max_value:
                #     value = "%s+"%str(value)
                
                text = ax.text(x, y, value, ha="center", va="center", color=colour, fontdict={'fontsize':'x-large'})
        
        if title is not None:
            fig.suptitle(title, fontsize=16, y=0.7)
        
        fig.tight_layout()
        plt.rcParams.update({'axes.labelsize': 'xx-large'})
        plt.show()
        if save_as is not None:
            form = "svg"
            save_path = os.path.join(dir_main, "plot", "%s.%s"%(save_as,form))
            fig.savefig(save_path, format=form, dpi=1200)

    def order_metric_significance(self, ex_id, skip_metrics=[], effect_size_weighting=0, 
        collapse_result=False, limit_01=False, count_limit=None, metrics_only=False):
        
        dir_main = Path(__file__).resolve().parents[1]
        metrics = self.get_combined_metrics()
        if count_limit is None:
            count_limit = len(metrics)
        results = {}
        
        for j, metric in enumerate(metrics):
            if metric in skip_metrics:
                continue

            if collapse_result:
                result = self.sum_metric_effect_size_collapsed(ex_id, metric, effect_size_weighting, limit_01)
            else:
                result = self.sum_metric_effect_size(ex_id, metric, effect_size_weighting)

            results[metric] = result

        ordered = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
        top = {}
        count = 1
        for metric, sig in ordered.items():
            if count > count_limit:
                break

            if metrics_only:
                top[count] = metric
            else:
                top[count] = {'metric': metric, 'sig': sig}
                
            count += 1

        return top

    def prettify_label(self, metric):
        if metric[:3] == "fov":
            return "FOV" + metric[3:]

        return metric.replace("_", " ").capitalize()

    def plot_ordered(self, ex_id, skip_metrics=[], effect_size_weighting=0, 
        collapse_result=False, limit_01=False, count_limit=None):

        dir_main = Path(__file__).resolve().parents[1]
        metrics = self.get_combined_metrics()
        if count_limit is None:
            count_limit = len(metrics)
        results = {}
        
        for j, metric in enumerate(metrics):
            if metric in skip_metrics:
                continue

            if collapse_result:
                result = self.sum_metric_effect_size_collapsed(ex_id, metric, effect_size_weighting, limit_01)
            else:
                result = self.sum_metric_effect_size(ex_id, metric, effect_size_weighting)

            results[metric] = result

        ordered = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
        y = []
        x = []
        color = []
        count = 1
        lim = 0.1
        for metric, sig in ordered.items():
            if count > count_limit:
                break
                
            y.append(sig)
            x.append(self.prettify_label(metric))
            if sig < lim:
                color.append('sandybrown')
            else:
                color.append('tab:blue')

            count += 1

        plt.barh(x, y, color=color, alpha=0.6)
        plt.xlim(0,1)
        h = plt.vlines(x=lim, ymin=[-0.5], ymax=[len(x)-0.5], 
                color='salmon', linestyle="--")
        plt.legend([h], ['S = 0.1'])
        plt.xlabel('Significance')


