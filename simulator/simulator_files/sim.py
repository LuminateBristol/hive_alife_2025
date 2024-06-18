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

from . import Swarm, Warehouse, Robot

class Simulator:

    def __init__(self, config,
        verbose=False,
        random_seed=None):

        self.cfg = config
        self.verbose = verbose
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
            self.cfg.get('wallsh'),
            self.cfg.get('wallsv'),
            self.cfg.get('warehouse', 'object_position'))            

    def build_swarm(self, cfg):
        robot_obj = Robot(
            cfg.get('robot', 'radius'), 
            cfg.get('robot', 'max_v'),
            camera_sensor_range=cfg.get('robot', 'camera_sensor_range'),
            place_tol = cfg.get('tolerances', 'place_tol'),
            abandon_tol = cfg.get('tolerances', 'abandon_tol')
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
                         print_bt = cfg.get('print_bt'))
        
        return swarm

   
    # iterate method called once per timestep
    def iterate(self):
        self.warehouse.iterate(self.cfg.get('heading_bias'))
        
        self.exit_sim(counter=self.warehouse.counter)

    def exit_sim(self, counter=None):
        if counter > self.cfg.get('time_limit'):
            print('{counter} counts reached - Time limit expired')
            self.exit_threads = True
            self.exit_run = True

    def run(self):
        if self.verbose:
            print("Running with seed: %d"%self.random_seed)

        while self.warehouse.counter <= self.cfg.get('time_limit') and self.exit_run is False:
            self.iterate()
        
        if self.verbose:
            print("\n")
        
        return self.warehouse.counter

class SimTest(Simulator):

    def run(self, testID=0):
        self.testID = testID
        if self.verbose:
            print("Running with seed: %d"%self.random_seed)

        while self.warehouse.counter <= self.cfg.get('time_limit'):
            self.test_hook()
            self.iterate()
        
        if self.verbose:
            print("\n")

    def test_hook(self):
        if self.testID == 0:
            self.test_count_lifted_box()
        if self.testID == 1:
            self.test_walls_in_range()
        if self.testID == 2:
            self.test_agents_in_range()

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