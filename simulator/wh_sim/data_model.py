import csv
import datetime
import os
from os.path import dirname, realpath
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

try:
    dir_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(dir_root))
    from simulator.lib import RedisConn, RedisKeys
except Exception as e:
    pass

class DataModel:

    velocity_d_timestep = 10
    roc_window = 50
    metrics = [
        'agents_in_range',
        'boxes_in_range',
        'walls_in_range',
        'combined_in_range',
        'velocity',
        'agent_delivery_count',
        'action_state',
        'nearest_agent_distance',
        'nearest_box_distance',
        'nearest_wall_distance',
        'nearest_combined_distance',
        'nearest_agent_id',
        'nearest_box_id',
        'nearest_wall_id',
        'nearest_id',
        'heading',
        'angular_velocity'
    ]

    roc_metrics = [
        '#_nearest_agent_id',
        '#_nearest_box_id',
        '#_nearest_wall_id',
        '#_nearest_id',
    ]

    def __init__(self, store_internal=False, compute_roc=False):
        self.number_of_agents = None
        self.number_of_boxes = None
        self.metric_data = {}
        self.roc_data = {}
        self.store_internal = store_internal
        self.compute_roc = compute_roc
        self.df = None

    # run this after every iteration of simulation
    def get_metric_data(self, warehouse):
        self.swarm = warehouse.swarm
        self.warehouse = warehouse      

        if self.number_of_agents is None:
            self.number_of_agents = self.swarm.number_of_agents
        if self.number_of_boxes is None:
            self.number_of_boxes = self.warehouse.number_of_boxes          
            
        a_ir = self.count_in_range(1)
        b_ir = self.count_in_range(0)
        w_ir = self.count_in_range(2)
        comb_ir = np.stack([a_ir, b_ir, w_ir]).sum(axis=0)

        rob_v = self.compute_velocity()
        agent_delivery_count = self.warehouse.rob_delivery_count
        action_state = self.swarm.agent_has_box
        
        nearest_agent = self.get_nearest_object(1)
        nearest_box = self.get_nearest_object(0)
        nearest_wall = self.get_nearest_object(2)
        nearest_dist = self.get_single_nearest([nearest_box, nearest_agent, nearest_wall])
        heading = self.swarm.computed_heading
        angular_velocity = self.compute_velocity(angular=True)
        
        self.metric_data = {
            'agents_in_range': a_ir,
            'boxes_in_range': b_ir,
            'walls_in_range': w_ir,
            'combined_in_range': comb_ir,
            'velocity': rob_v,
            'agent_delivery_count': agent_delivery_count,
            'action_state': action_state,
            'nearest_agent_distance': nearest_agent[0],
            'nearest_box_distance': nearest_box[0],
            'nearest_wall_distance': nearest_wall[0],
            'nearest_combined_distance': nearest_dist[0],
            'nearest_agent_id': nearest_agent[1],
            'nearest_box_id': nearest_box[1],
            'nearest_wall_id': nearest_wall[1],
            'nearest_id': nearest_dist[1],
            'heading': heading,
            'angular_velocity': angular_velocity
        }
        
        if self.compute_roc:
            self.metric_data['#_nearest_agent_id'] = self.compute_roc_qual('nearest_agent_id')
            self.metric_data['#_nearest_box_id'] = self.compute_roc_qual('nearest_box_id')
            self.metric_data['#_nearest_wall_id'] = self.compute_roc_qual('nearest_wall_id')
            self.metric_data['#_nearest_id'] = self.compute_roc_qual('nearest_id')

        if self.store_internal:
            if self.df is None:
                self.df = np.array([])

            self.df = np.concatenate([self.df, self.get_write_data()])

        return self.metric_data

    def get_write_data(self):
        row = np.array(list(self.metric_data.values())).flatten()
        row = np.concatenate([[self.warehouse.counter], row, [self.warehouse.delivered]])
        return row

    def get_model_data(self):
        df = pd.DataFrame.from_dict(self.metric_data)
        return df

    def get_dataframe(self):
        rows = self.warehouse.counter
        cols = int(self.df.shape[0]/rows)
        data = self.df.reshape((rows, cols))
        head = self.generate_df_head()
        df = pd.DataFrame(data, columns=head)
        return df
        
    def generate_df_head(self):
        head = ['ts']
        metric_count = len(self.metrics)

        if self.compute_roc:
            metric_count += len(self.roc_metrics)

        for i in range(metric_count):
            for j in range(self.number_of_agents):
                col = 'm%d_a%d'%(i,j)
                head += [col]

        head += ['d']
        return head

    # obj_type: 0 box, 1 agent, 2 wall
    def get_object_distance(self, obj=0, in_range=True):
        rob_c = self.warehouse.rob_c
        if obj == 0:
            total_objs = self.warehouse.number_of_boxes
            obj_dist = self.swarm.box_dist
        elif obj == 1:
            total_objs = self.swarm.number_of_agents
            obj_dist = self.swarm.agent_dist
        elif obj == 2:
            total_objs = len(self.warehouse.map.wall_divisions)
            obj_dist = np.transpose(self.swarm.wall_dist)
        else:
            raise Exception("Invalid obj parameter")

        if in_range:
            cam_range = self.swarm.camera_sensor_range_V
            cam_range_ = np.tile(cam_range, (total_objs, 1))
            ir = (obj_dist < cam_range_).astype(float)
            ir[ir == 0] = np.nan
            if obj == 1:
                ir = np.transpose(ir)
            obj_dist = np.multiply(obj_dist, ir)

        if obj ==0 or obj==2:
            obj_dist = np.transpose(obj_dist)
        
        if obj == 0:
            for idx, r_id in enumerate(self.warehouse.robot_carrier):
                if r_id != -1:
                    obj_dist[r_id][idx] = np.nan

        return obj_dist

    def count_in_range(self, obj_type=0):
        dist = self.get_object_distance(obj_type)
        count = np.count_nonzero(~np.isnan(dist), axis=1)
        if obj_type == 1:
            count = count-1 # don't count self
        return count

    # obj_type: 0 box, 1 agent, 2 wall
    def get_nearest_object(self, obj_type=0):
        dist = self.get_object_distance(obj_type)
        foo = 99999
        dist[np.isnan(dist)] = foo
        # don't count self or any box currently lifted
        # if obj_type in [0, 1]:
        if obj_type == 1:
            min_dist = np.partition(dist,kth=1,axis=1)[:,1]
            min_dist_idx = np.argpartition(dist,kth=1,axis=1)[:,1]
        else:
            min_dist = np.min(dist, axis=1)
            min_dist_idx = np.argmin(dist, axis=1)

        md = min_dist.astype(float)
        mdi = min_dist_idx.astype(float)
        nan_val = min_dist == foo
        md[nan_val] = np.nan
        mdi[nan_val] = np.nan
        mdi = self._map_obj_id(mdi, obj_type)
        return (md, mdi)

    def _map_obj_id(self, obj_id, obj_type=0):
        if obj_type == 0:
            offset = 1
        elif obj_type == 1:
            offset = self.warehouse.number_of_boxes + 1
        elif obj_type == 2:
            offset = self.warehouse.number_of_boxes + self.swarm.number_of_agents + 1
        
        return obj_id+offset

    def get_single_nearest(self, nearest_arr=None):
        if nearest_arr is None:
            nearest_box = self.get_nearest_object(0)
            nearest_agent = self.get_nearest_object(1)
            nearest_wall = self.get_nearest_object(2)
        else:
            nearest_box = nearest_arr[0]
            nearest_agent = nearest_arr[1]
            nearest_wall = nearest_arr[2]

        dist = np.array([nearest_box[0], nearest_agent[0], nearest_wall[0]])
        foo = 99999
        dist[np.isnan(dist)] = foo
        md = np.min(dist, axis=0)
        md[md==foo] = np.nan

        dist_idx = np.array([nearest_box[1], nearest_agent[1], nearest_wall[1]])
        mdi = []
        for r_id, d in enumerate(md):
            if np.isnan(d):
                mdi.append(0)
            else:
                idxs = np.where(dist[:,r_id]==d)[0]
                idx = np.random.choice(idxs, 1) # break a tie in more than one nearest distance
                mdi += dist_idx[idx,r_id].flatten().tolist()
        
        return (md, np.array(mdi))
        
    def compute_velocity(self, angular=False):
        # Find change in distance over 5 timesteps
        t_d = self.velocity_d_timestep
        t = (self.warehouse.counter - t_d)%t_d
        if self.warehouse.counter > t_d and angular:
            return (self.swarm.computed_heading - self.swarm.computed_heading_prev[t])/t_d
        elif self.warehouse.counter > t_d:
            change_v = (self.warehouse.rob_c - self.warehouse.rob_c_prev[t])/t_d
            c_ = np.sum(np.multiply(change_v, change_v), axis=1)
            return np.sqrt(c_)
        else:
            return np.zeros(self.swarm.number_of_agents)

    def compute_roc_qual(self, metric):
        t_d = self.roc_window
        t = (self.warehouse.counter - t_d)%t_d

        data = np.array(self.metric_data[metric])
        np.nan_to_num(data, copy=False)
        data = data.reshape(1, self.number_of_agents)

        if metric not in self.roc_data:
            self.roc_data[metric] = data
        elif self.roc_data[metric].shape[0] < self.roc_window: # not fully filled
            self.roc_data[metric] = np.concatenate((self.roc_data[metric], data), axis=0)
        else:
            self.roc_data[metric][t] = data
        
        counts = []
        for i in range(self.number_of_agents):
            col = self.roc_data[metric][:, i]
            u, c = np.unique(col, return_counts=True)
            val = len(c)/self.roc_window
            counts.append(val)
        
        return np.array(counts)

class ExportRedisData(DataModel):
    
    vis_keys = [
        'no_boxes',
        'no_robots',
        'box_coords',
        'robot_coords',
        'boxes_to_be_delivered',
        'camera_range'
        # 'ad_prediction'
    ]

    setup_keys = [
        'fault_count',
        'metrics'
    ]

    def __init__(self, export_vis_data=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rconn = RedisConn()
        self.rkeys = RedisKeys()
        self.export_vis_data = export_vis_data
        
    def get_metric_data(self, warehouse):
        metric_data = super().get_metric_data(warehouse)
        ts = warehouse.counter

        if not self.rconn.is_connected():
            self.rconn.reconnect()

        self._export_metric_data(ts, metric_data)

        if self.export_vis_data:
            self._export_vis_data(ts, warehouse)

        if ts == 1:
            self._export_metadata(warehouse)

    def _export_metric_data(self, timestep, metric_data):                
        count = 0
        for metric, data in metric_data.items():
            key = self.rkeys.gen_metric_timestep_key(timestep, count)
            try:
                data = data.tolist()
            except:
                data = data
            finally:
                self.rconn.set(key, json.dumps(data))
                count += 1

    def _export_vis_data(self, timestep, warehouse):
        swarm = warehouse.swarm
        vis_data = {
            'no_boxes': warehouse.number_of_boxes,
            'no_robots': swarm.number_of_agents,
            'box_coords': json.dumps(warehouse.box_c.tolist()),
            'robot_coords': json.dumps(warehouse.rob_c.tolist()),
            'boxes_to_be_delivered': json.dumps(warehouse.to_be_delivered.tolist()),
            'fault_count': json.dumps(swarm.fault_count),
            'camera_range': json.dumps(swarm.camera_sensor_range_V.tolist())
        }

        for key, val in vis_data.items():
            self.rconn.set(self.rkeys.gen_timestep_key(timestep, key), val)

    def _export_metadata(self, warehouse):
        swarm = warehouse.swarm
        metrics = self.metrics
        if self.compute_roc:
            metrics += self.roc_metrics
        data = {
            'fault_count': json.dumps(swarm.fault_count),
            'metrics': json.dumps(metrics)
        }

        for key, val in data.items():
            self.rconn.set(self.rkeys.gen_metadata_key(key), val)


class MinimalDataModel(DataModel):

    def __init__(self, faults, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.faults = faults
        self.export_cols = []
        self.metric_data = {}
        self.ag_ids = [0] # 0th index always captured   
        self.export_cols = []         

    def get_metric_data(self, warehouse):
        self.swarm = warehouse.swarm
        self.warehouse = warehouse      
        
        if self.number_of_agents is None:
            self.number_of_agents = self.swarm.number_of_agents
            self.number_of_boxes = self.warehouse.number_of_boxes
        
        if len(self.export_cols) == 0:
            if self.number_of_agents == self.faults:
                self.export_cols = ['f']
            elif self.faults == 0:
                self.export_cols = ['n']
            else:
                self.ag_ids = [0, self.faults] # capture F and N
                self.export_cols = ['f', 'n']
        
        data = []
        for aid in self.ag_ids:
            d = self._get(aid)
            data.append(d)

        data = np.array(data)
        for i,m in enumerate(self.metrics):
            self.metric_data[m] = data[:,i]
        
        if self.compute_roc:
            self.metric_data['#_nearest_agent_id'] = self.compute_roc_qual('nearest_agent_id')
            self.metric_data['#_nearest_box_id'] = self.compute_roc_qual('nearest_box_id')
            self.metric_data['#_nearest_wall_id'] = self.compute_roc_qual('nearest_wall_id')
            self.metric_data['#_nearest_id'] = self.compute_roc_qual('nearest_id')
            
        if self.store_internal:
            self._store_internal()
        
        return self.metric_data

    def _get(self, ag_id):
        a_ir = self.count_in_range(ag_id, 1)
        b_ir = self.count_in_range(ag_id, 0)
        w_ir = self.count_in_range(ag_id, 2)
        comb_ir = a_ir + b_ir + w_ir

        rob_v = self.compute_velocity(ag_id)
        agent_delivery_count = self.warehouse.rob_delivery_count[ag_id]
        action_state = self.swarm.agent_has_box[ag_id]
        
        nearest_agent = self.get_nearest_object(ag_id, 1)
        nearest_box = self.get_nearest_object(ag_id, 0)
        nearest_wall = self.get_nearest_object(ag_id, 2)
        nearest_dist = self.get_single_nearest(ag_id, [nearest_box, nearest_agent, nearest_wall])
        heading = self.swarm.computed_heading[ag_id]
        angular_velocity = self.compute_velocity(ag_id, angular=True)
        
        return (
            a_ir, b_ir, w_ir, comb_ir, rob_v, agent_delivery_count, action_state, 
            nearest_agent[0], nearest_box[0], nearest_wall[0], nearest_dist[0],
            nearest_agent[1], nearest_box[1], nearest_wall[1], nearest_dist[1],
            heading, angular_velocity
        )

    def _store_internal(self):
        if self.df is None:
            self.df = np.array([])

        self.df = np.concatenate([self.df, self.get_write_data()])

    # obj_type: 0 box, 1 agent, 2 wall
    def get_object_distance(self, ag_id, obj=0, in_range=True):
        rob_c = self.warehouse.rob_c[ag_id]
        if obj == 0:
            total_objs = self.warehouse.number_of_boxes
            obj_dist = self.swarm.box_dist[:, ag_id]
        elif obj == 1:
            total_objs = self.swarm.number_of_agents
            obj_dist = self.swarm.agent_dist[ag_id]
        elif obj == 2:
            total_objs = len(self.warehouse.map.wall_divisions)
            obj_dist = self.swarm.wall_dist[ag_id, :]
        else:
            raise Exception("Invalid obj parameter")

        if in_range:
            cam_range = self.swarm.camera_sensor_range_V[ag_id]
            # cam_range_ = np.tile(cam_range, (total_objs, 1))
            ir = (obj_dist < cam_range).astype(float)
            ir[ir == 0] = np.nan
            obj_dist = np.multiply(obj_dist, ir)

        if obj ==0 or obj==2:
            obj_dist = np.transpose(obj_dist)

        if obj == 0 and self.swarm.agent_has_box[ag_id]:
            # Get the box ID the robot is carrying
            try:
                box_id = self.warehouse.robot_carrier.tolist().index(ag_id)
            except:
                print("no carrier")
            obj_dist[box_id] = np.nan

        return obj_dist

    def count_in_range(self, ag_id, obj_type=0):
        dist = self.get_object_distance(ag_id, obj_type)
        count = np.count_nonzero(~np.isnan(dist))
        if obj_type == 1:
            count = count-1 # don't count self
        return count
    
    # obj_type: 0 box, 1 agent, 2 wall
    def get_nearest_object(self, ag_id, obj_type=0):
        dist = self.get_object_distance(ag_id, obj_type)
        foo = 99999
        dist[np.isnan(dist)] = foo
        # don't count self or any box currently lifted
        # if obj_type in [0, 1]:
        if obj_type == 1:
            min_dist = np.partition(dist,kth=1)[1]
            min_dist_idx = np.argpartition(dist,kth=1)[1]
        else:
            min_dist = np.min(dist)
            min_dist_idx = np.argmin(dist)

        md = min_dist.astype(float)
        mdi = min_dist_idx.astype(float)
        if md == foo:
            md = np.nan
            mdi = np.nan
        
        mdi = self._map_obj_id(mdi, obj_type)
        return (md, mdi)

    def get_single_nearest(self, ag_id, nearest_arr=None):
        if nearest_arr is None:
            nearest_box = self.get_nearest_object(ag_id, 0)
            nearest_agent = self.get_nearest_object(ag_id, 1)
            nearest_wall = self.get_nearest_object(ag_id, 2)
        else:
            nearest_box = nearest_arr[0]
            nearest_agent = nearest_arr[1]
            nearest_wall = nearest_arr[2]

        dist = np.array([nearest_box[0], nearest_agent[0], nearest_wall[0]])
        foo = 99999
        dist[np.isnan(dist)] = foo
        md = np.min(dist)
        dist_idx = np.array([nearest_box[1], nearest_agent[1], nearest_wall[1]])
        if md == foo:
            md = np.nan
            mdi = 0
        else:
            mdi_ = np.argmin(dist)
            mdi = dist_idx[mdi_]

        return (md, mdi)

    def compute_velocity(self, ag_id, angular=False):
        # Find change in distance over 5 timesteps
        t_d = self.velocity_d_timestep
        t = (self.warehouse.counter - t_d)%t_d
        if self.warehouse.counter > t_d and angular:
            c = (self.swarm.computed_heading - self.swarm.computed_heading_prev[t])/t_d
            return c[ag_id]
        elif self.warehouse.counter > t_d:
            change_v = (self.warehouse.rob_c - self.warehouse.rob_c_prev[t])/t_d
            c_ = np.sum(np.multiply(change_v, change_v), axis=1)
            c = np.sqrt(c_)
            return c[ag_id]
        else:
            return 0

    def compute_roc_qual(self, metric):
        t_d = self.roc_window
        t = (self.warehouse.counter - t_d)%t_d

        no_ag = len(self.ag_ids)
        data = self.metric_data[metric]
        np.nan_to_num(data, copy=False)
        data = data.reshape(1, no_ag)

        if metric not in self.roc_data:
            self.roc_data[metric] = data
        elif self.roc_data[metric].shape[0] < self.roc_window: # not fully filled
            self.roc_data[metric] = np.concatenate((self.roc_data[metric], data), axis=0)
        else:
            self.roc_data[metric][t] = data
        
        counts = []
        for i in range(no_ag):
            col = self.roc_data[metric][:, i]
            u, c = np.unique(col, return_counts=True)
            val = len(c)/self.roc_window
            counts.append(val)
        
        return counts

    def generate_df_head(self):
        head = ['ts']
        metric_count = len(self.metrics)

        if self.compute_roc:
            metric_count += len(self.roc_metrics)

        for i in range(metric_count):
            for j in self.export_cols:
                col = 'm%d_%s'%(i,j)
                head += [col]

        head += ['d']
        return head

    def sample_data(self):
        df = self.get_dataframe()
        time = df['ts'].shape[0]
        df.drop('ts', axis=1, inplace=True)
        df.drop('d', axis=1, inplace=True)
        n = []
        f = []
        for col in df.columns:
            meta = col.split('_')
            metric_id = int(meta[0][1:])
            state = meta[1]
            s = df[col].sample().tolist()
            if state == 'n':
                n += s
            else:
                f += s

        export_d = {}
        if len(n):
            export_d['n'] = n
        if len(f):
            export_d['f'] = f
        
        return pd.DataFrame(export_d)
        
# Can minimalize even further:
# Generate an array of random timesteps, t in [T_start, T_end]
# Only sample data when warehouse.counter == t
# Sampling on the fly :)

class ExtremeMinDataModel(MinimalDataModel):

    def __init__(self, faults, max_time, *args, **kwargs):
        super().__init__(faults, *args, **kwargs)

        # hardcode for now!
        self.number_of_agents = 10#self.swarm.number_of_agents
        self.number_of_boxes = 10#self.warehouse.number_of_boxes
    
        if self.number_of_agents == self.faults:
            self.export_cols = ['f']
        elif self.faults == 0:
            self.export_cols = ['n']
        else:
            self.ag_ids = [0, self.faults] # capture F and N
            self.export_cols = ['f', 'n']

        self._generate_comp_timesteps(max_time)

    def _generate_comp_timesteps(self, max_time):
        sample_ts = {}
        comp_ts = []
        for j, ag_id in enumerate(self.ag_ids):
            state = self.export_cols[j]
            for i, m in enumerate(self.metrics):
                if m == 'velocity':
                    rchoice = np.arange(self.velocity_d_timestep+1,max_time)
                else:
                    rchoice = np.arange(1,max_time)
                
                x = np.random.choice(rchoice)
                sample_ts["m%d_%s"%(i,state)] = x
                comp_ts.append(x)
                if m == 'velocity':
                    maxx = max(0, x-self.velocity_d_timestep)
                    comp_ts += range(maxx, x)
                
            if self.compute_roc:
                for k, m in enumerate(self.roc_metrics):
                    x = np.random.choice(np.arange(self.roc_window+1,max_time))
                    sample_ts["m%d_%s"%(i+k+1,state)] = x
                    maxx = max(0, x-self.roc_window)
                    comp_ts += range(maxx, x+1)
        
        self.sample_ts = sample_ts
        self.comp_ts = list(set(comp_ts))
        self.comp_ts.sort()
        self.no_comp = len(self.comp_ts)
        self.c0 = 0
        self.comp_ts_cur = self.comp_ts[self.c0]

    def get_metric_data(self, warehouse):
        c = warehouse.counter
        if c == self.comp_ts_cur:
            super().get_metric_data(warehouse)
            
            self.c0 += 1
            if self.c0 < self.no_comp:
                self.comp_ts_cur = self.comp_ts[self.c0]

    def get_dataframe(self):
        rows = self.no_comp
        try:
            cols = int(self.df.shape[0]/rows)
            data = self.df.reshape((rows, cols))
            head = self.generate_df_head()
            df = pd.DataFrame(data, columns=head)
            return df
        except Exception as e:
            print('Error, f%d'%self.faults, e)
        
    def sample_data(self):
        df = self.get_dataframe()
        time = df['ts'].shape[0] # these correspond to comp_ts list
        df.drop('ts', axis=1, inplace=True)
        df.drop('d', axis=1, inplace=True)
        n = []
        f = []
        for col in df.columns:
            meta = col.split('_')
            metric_id = int(meta[0][1:])
            state = meta[1]
            ts = self.sample_ts[col]
            ts_idx = self.comp_ts.index(ts)
            s = df[col][ts_idx]
            
            if state == 'n':
                n.append(s)
            else:
                f.append(s)
        
        export_d = {}
        if len(n):
            export_d['n'] = n
        if len(f):
            export_d['f'] = f
        
        return pd.DataFrame(export_d)