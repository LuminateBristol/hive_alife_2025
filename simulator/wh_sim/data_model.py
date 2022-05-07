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
        'FOV_agent_density_forward',
        'FOV_agent_density_behind',
        'FOV_object_density_forward',
        'FOV_object_density_behind',
        'FOV_wall_density_forward',
        'FOV_wall_density_behind',
        'FOV_combined_density_forward',
        'FOV_combined_density_behind',
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

        a_dir_ir = self.count_directional_in_range(1)
        b_dir_ir = self.count_directional_in_range(0)
        w_dir_ir = self.count_directional_in_range(2)
        comb_dir_f = np.stack([a_dir_ir[0], b_dir_ir[0], w_dir_ir[0]]).sum(axis=0)
        comb_dir_b = np.stack([a_dir_ir[1], b_dir_ir[1], w_dir_ir[1]]).sum(axis=0)
        
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
            'FOV_agent_density_forward': a_dir_ir[0],
            'FOV_agent_density_behind': a_dir_ir[1],
            'FOV_object_density_forward': b_dir_ir[0],
            'FOV_object_density_behind': b_dir_ir[1],
            'FOV_wall_density_forward': w_dir_ir[0],
            'FOV_wall_density_behind': w_dir_ir[1],
            'FOV_combined_density_forward': comb_dir_f,
            'FOV_combined_density_behind': comb_dir_b,
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
            coord = self.warehouse.box_c
            sum_axis = 1
        elif obj == 1:
            total_objs = self.swarm.number_of_agents
            obj_dist = self.swarm.agent_dist
            coord = rob_c
            sum_axis = 0
        elif obj == 2:
            total_objs = len(self.warehouse.map.wall_divisions)
            obj_dist = np.transpose(self.swarm.wall_dist)
            coord = self.warehouse.map.wall_divisions
        else:
            raise Exception("Invalid obj parameter")

        if in_range:
            cam_range = self.swarm.camera_sensor_range_V
            cam_range_ = np.tile(cam_range, (total_objs, 1))
            ir = (obj_dist < cam_range_).astype(float)
            ir[ir == 0] = np.nan
            obj_dist = np.multiply(obj_dist, ir)

        if obj ==0 or obj==2:
            obj_dist = np.transpose(obj_dist)

        # compute angle to object
        n_r = self.swarm.number_of_agents
        n_c = len(coord)
        
        coord_norm_x = np.tile(coord[:,0], (n_r,1)) - np.transpose(np.tile(rob_c[:,0], (n_c,1)))
        coord_norm_y = np.tile(coord[:,1], (n_r,1)) - np.transpose(np.tile(rob_c[:,1], (n_c,1)))
        
        obj_angles = np.arctan2(coord_norm_y.flatten(), coord_norm_x.flatten())
        obj_angles_M = np.reshape(obj_angles, (n_r, n_c))

        pi_2_arr = np.empty(n_r)
        pi_2_arr.fill(np.pi*2)
        pi_h_arr = np.empty(n_r)
        pi_h_arr.fill(np.pi/2)

        heading_angles = (self.swarm.computed_heading - np.pi)%(2*np.pi)
        heading_angles_M = np.transpose(np.tile(heading_angles, (n_c,1)))
        ha_norm = heading_angles_M%(2*np.pi) # Values: 0 <= x < 2*pi
        oa_norm = obj_angles_M%(2*np.pi) # Values: 0 <= x < 2*pi
        obj_arg = ha_norm - oa_norm

        # obj_arg values: -2*pi < x < 2*pi
        # We would like the arg to be between -pi and pi
        le_idx = np.less_equal(obj_arg, -np.pi)
        g_idx = np.greater(obj_arg, np.pi)
        obj_arg[le_idx] += 2*np.pi
        obj_arg[g_idx] -= 2*np.pi

        if obj == 0:
            for idx, r_id in enumerate(self.warehouse.robot_carrier):
                if r_id != -1:
                    obj_arg[r_id][idx] = np.nan # if robot is carrying box, set angle to box as 0
                    obj_dist[r_id][idx] = np.nan

        if obj == 1:
            np.fill_diagonal(obj_arg, np.nan)
        
        if in_range:
            ir = np.transpose(ir)
            obj_arg = np.multiply(obj_arg, ir)
            
        return (obj_dist, obj_arg)

    def count_in_range(self, obj_type=0):
        dist, arg = self.get_object_distance(obj_type)
        count = np.count_nonzero(~np.isnan(dist), axis=1)
        if obj_type == 1:
            count = count-1 # don't count self
        return count

    def count_directional_in_range(self, obj_type=0):
        dist, arg = self.get_object_distance(obj_type)
        in_front = np.logical_and(arg <= np.pi/2, arg >= -np.pi/2)
        front_c = np.sum(in_front, axis=1)
        total_c = self.count_in_range(obj_type)
        behind_c = total_c - front_c
        return (front_c, behind_c)

    # obj_type: 0 box, 1 agent, 2 wall
    def get_nearest_object(self, obj_type=0):
        dist, _ = self.get_object_distance(obj_type)
        foo = 99999
        dist[np.isnan(dist)] = foo
        # don't count self or any box currently lifted
        # if obj_type in [0, 1]:
        if obj_type == 1:
            axis = (obj_type-1)%2
            min_dist = np.partition(dist,kth=1,axis=axis)[1]
            min_dist_idx = np.argpartition(dist,kth=1,axis=axis)[1]
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
                mdi.append(np.nan)
            else:
                idxs = np.where(dist[:,r_id]==d)[0]
                idx = np.random.choice(idxs, 1) # break a tie in more than one nearest distance
                mdi += dist_idx[idx,r_id].flatten().tolist()
        
        return (md, mdi)
        
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
 
        return counts

class ExportRedisData(DataModel):
    
    vis_keys = [
        'no_boxes',
        'no_robots',
        'box_coords',
        'robot_coords',
        'boxes_to_be_delivered',
        'metrics',
        # 'ad_prediction'
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
            'boxes_to_be_delivered': json.dumps(warehouse.to_be_delivered.tolist())
        }

        for key, val in vis_data.items():
            self.rconn.set(self.rkeys.gen_timestep_key(timestep, key), val)

        # self.rconn.set(self.rkeys.gen_timestep_key(timestep, 'no_boxes'), warehouse.number_of_boxes)
        # self.rconn.set(self.rkeys.gen_timestep_key(timestep, 'no_robots'), swarm.number_of_agents)
        # self.rconn.set(self.rkeys.gen_timestep_key(timestep, 'box_coords'), json.dumps(warehouse.box_c.tolist()))
        # self.rconn.set(self.rkeys.gen_timestep_key(timestep, 'robot_coords'), json.dumps(warehouse.rob_c.tolist()))
        # self.rconn.set(self.rkeys.gen_timestep_key(timestep, 'boxes_to_be_delivered'), json.dumps(warehouse.to_be_delivered.tolist()))
