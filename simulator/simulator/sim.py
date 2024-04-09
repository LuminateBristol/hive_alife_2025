from pathlib import Path
import sys
from .objects import Swarm

dir_root = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import random
import threading
import os
from os.path import dirname, realpath
import datetime
import time
import json

from . import Swarm, Warehouse, Robot, FaultySwarm

class Simulator:

    def __init__(self, config,
        verbose=False,              # verbosity
        check_collisions=False,
        data_model=None,
        random_seed=None,
        fault_count=[0],
        ad_model=None):

        self.cfg = config
        self.verbose = verbose
        # self.state_changes = 0 # intended to log changes in the system from normal (if faults are injected mid-runtime for example)
        self.exit_threads = False
        self.exit_run = False
        self.delivered_in = None
        self.fault_count = fault_count
        self.carry_count = [] # Used to count the amount of time that robots spend carrying - this same object is references by each robot's behaviour tree to keep track
        self.data_model = data_model
        self.ad_model = ad_model
        self.task_log = self.cfg.get('task_log')
        
        if random_seed is None:
            self.random_seed = random.randint(0,100000000)
        else:
            self.random_seed = random_seed

        np.random.seed(int(self.random_seed))

        try:
            self.swarm = self.build_swarm(self.cfg)
        except Exception as e:
            raise e

        self.warehouse = Warehouse(
            self.cfg.get('warehouse', 'width'),
            self.cfg.get('warehouse', 'height'), 
            self.cfg.get('boxes'), 
            self.cfg.get('warehouse', 'box_radius'), 
            self.swarm, 
            self.cfg.get('warehouse', 'exit_width'),
            self.cfg.get('warehouse', 'object_position'),
            check_collisions=check_collisions)            

    def build_swarm(self, cfg):
        robot_obj = Robot(
            cfg.get('robot', 'radius'), 
            cfg.get('robot', 'max_v'),
            camera_sensor_range=cfg.get('robot', 'camera_sensor_range'),
            place_tol = cfg.get('tolerances', 'place_tol'),
            abandon_tol = cfg.get('tolerances', 'abandon_tol'),
            pre_place_delta = cfg.get('tolerances', 'pre_place_delta')
        )
        
        swarm = Swarm(
            repulsion_o=cfg.get('warehouse', 'repulsion_object'), 
            repulsion_w=cfg.get('warehouse', 'repulsion_wall'),
            heading_change_rate=cfg.get('heading_change_rate')
        )

        self.carry_count = [0] * cfg.get('warehouse', 'number_of_agents')

        swarm.add_agents(robot_obj, cfg.get('warehouse', 'number_of_agents'),
                         width=self.cfg.get('warehouse', 'width'),
                         height=self.cfg.get('warehouse', 'height'),
                         bt_controller=self.cfg.get('behaviour_tree'),
                         task_log=self.task_log,
                         use_hm=self.cfg.get('use_hm'),
                         hm_mode=self.cfg.get('hm_mode'),
                         carry_count = self.carry_count,
                         print_bt = cfg.get('print_bt'))
        
        return swarm

    def generate_fault_type(self, fault, faulty_agents_range):
        fault_type = fault['type']
        
        if fault_type == FaultySwarm.ALTER_AGENT_SPEED:
            speed = fault['cfg']['speed_at_fault']
            lookup = []
            for i in faulty_agents_range:
                lookup.append((0, i, speed))
            return {'ftype': FaultySwarm.ALTER_AGENT_SPEED, 'lookup': lookup}

        if fault_type == FaultySwarm.FAILED_BOX_PICKUP:
            lookup = {}
            for i in faulty_agents_range:
                lookup[i] = 0
            return {'ftype': FaultySwarm.FAILED_BOX_PICKUP, 'lookup': lookup}

        if fault_type == FaultySwarm.FAILED_BOX_DROPOFF:
            lookup = {}
            for i in faulty_agents_range:
                lookup[i] = 0
            return {'ftype': FaultySwarm.FAILED_BOX_DROPOFF, 'lookup': lookup}

        if fault_type == FaultySwarm.REDUCED_CAMERA_RANGE:
            r_range = fault['cfg']['reduced_range']
            lookup = {}
            for i in faulty_agents_range:
                lookup[i] = r_range
            return {'ftype': FaultySwarm.REDUCED_CAMERA_RANGE, 'lookup': lookup}
   
    # iterate method called once per timestep
    def iterate(self):
        self.warehouse.iterate(self.cfg.get('heading_bias'), self.cfg.get('box_attraction'))
        
        self.exit_sim(delivered=self.warehouse.delivered, counter=self.warehouse.counter, global_task_log=self.task_log)

    def exit_sim(self, delivered=None, counter=None, global_task_log=None):
        if self.cfg.get('exit_criteria') == 'delivered' and delivered == self.cfg.get('warehouse', 'number_of_boxes'):
            if self.verbose:
                print("in", counter, "seconds")
            sr = float(delivered/self.cfg.get('warehouse', 'number_of_boxes'))
            if self.verbose:
                print(delivered, "of", self.cfg.get('warehouse', 'number_of_boxes'), "collected =", sr*100, "%")

        elif self.cfg.get('exit_criteria') == 'global_task_log':
            for task in global_task_log:
                if global_task_log[task]['status'] == 0:
                    break
            else:
                print('All boxes placed - sim completed')    
                self.exit_threads = True
                self.exit_run = True
        elif counter > self.cfg.get('time_limit'):
            print('{counter} counts reached - Time limit expired')
            self.exit_threads = True
            self.exit_run = True

    def run(self):
        if self.verbose:
            print("Running with seed: %d"%self.random_seed)

        while self.warehouse.counter <= self.cfg.get('time_limit') and self.exit_run is False:
            self.iterate()
        
        if self.delivered_in is None:
            self.delivered_in = self.warehouse.counter
        
        if self.verbose:
            print("\n")
        
        return self.warehouse.counter, self.carry_count

class SimTest(Simulator):

    def run(self, testID=0):
        self.testID = testID
        if self.verbose:
            print("Running with seed: %d"%self.random_seed)

        while self.warehouse.counter <= self.cfg.get('time_limit'):
            self.test_hook()
            self.iterate()
        
        if self.delivered_in is None:
            self.delivered_in = self.warehouse.counter
        
        if self.verbose:
            print("\n")

    def test_hook(self):
        if self.testID == 0:
            self.test_count_lifted_box()
        if self.testID == 1:
            self.test_walls_in_range()
        if self.testID == 2:
            self.test_agents_in_range()
            

    # shouldn't include lifted box in count
    # 3 robots, 2 boxes
    # one robot in the center of the arena, two robots in the bottom left corner (out of action)
    # center robot has first box, is in 25cm range of second box
    # boxes_in_range should be 1 (shouldn't count the lifted box)
    # nearest box distance should be 25cm
    # nearest ID should be ID of second box (should be 2)
    # box1 = ID 1, box2 = ID2, agent1 = ID3 etc. >> IDs start at 1, with boxes first
    def test_count_lifted_box(self):
        no_ag = self.swarm.number_of_agents
        self.swarm.heading = np.array([0.]*no_ag)
        self.swarm.robot_v *= 0
        rob_test = [self.warehouse.width/2, self.warehouse.height/2]
        self.swarm.repulsion_o = 0
        self.warehouse.rob_c = np.array([rob_test, [0,0], [30,0]])
        box_test = rob_test
        self.warehouse.box_c = np.array([box_test, np.add(rob_test,[25,0])])

        data = self.data_model.get_model_data()
        if self.warehouse.counter%10 == 1:
            print("In range: %s / Nearest dist: %s / Nearest id: %s"%(
                str(data['boxes_in_range'][0]),
                str(data['nearest_box_distance'][0]),
                str(data['nearest_box_id'][0]),
            ))

    def test_walls_in_range(self):
        no_ag = self.swarm.number_of_agents
        self.swarm.heading = np.array([0.]*no_ag)
        self.swarm.robot_v *= 0
        rob_test = [self.warehouse.width/2, self.warehouse.height/2]
        self.swarm.repulsion_o = 0
        self.warehouse.rob_c = np.array([rob_test, [0,0], [30,0]])
        box_test = rob_test
        self.warehouse.box_c = np.array([box_test, np.add(rob_test,[25,0])])

        data = self.data_model.get_model_data()
        if self.warehouse.counter%10 == 1:
            print("Walls in range: %s / Nearest dist: %s / Nearest id: %s"%(
                str(data['walls_in_range']),
                str(data['nearest_wall_distance']),
                str(data['nearest_wall_id'])
            ))

    def test_agents_in_range(self):
        no_ag = self.swarm.number_of_agents
        self.swarm.heading = np.array([0.]*no_ag)
        self.swarm.robot_v *= 0
        rob_test = [self.warehouse.width/2, self.warehouse.height/2]
        self.swarm.repulsion_o = 0
        self.warehouse.rob_c = np.array([rob_test, [0,0], [30,0]])
        box_test = rob_test
        self.warehouse.box_c = np.array([box_test, np.add(rob_test,[25,0])])

        data = self.data_model.get_model_data()
        if self.warehouse.counter%10 == 1:
            print("Agents in range: %s / Nearest dist: %s / Nearest id: %s"%(
                str(data['agents_in_range'].tolist()),
                str(data['nearest_agent_distance'].tolist()),
                str(data['nearest_agent_id'].tolist())
            ))