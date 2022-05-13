from pathlib import Path
import sys

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
        self.delivered_in = None
        self.fault_count = fault_count
        self.data_model = data_model
        self.ad_model = ad_model

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
            self.cfg.get('warehouse', 'number_of_boxes'), 
            self.cfg.get('warehouse', 'box_radius'), 
            self.swarm, 
            self.cfg.get('warehouse', 'exit_width'),
            self.cfg.get('warehouse', 'object_position'),
            check_collisions=check_collisions)            

    def build_swarm(self, cfg):
        robot_obj = Robot(
            cfg.get('robot', 'radius'), 
            cfg.get('robot', 'max_v'),
            camera_sensor_range=cfg.get('robot', 'camera_sensor_range')
        )
        
        swarm = FaultySwarm(
            repulsion_o=cfg.get('warehouse', 'repulsion_object'), 
            repulsion_w=cfg.get('warehouse', 'repulsion_wall'),
            heading_change_rate=cfg.get('heading_change_rate')
        )

        swarm.add_agents(robot_obj, cfg.get('warehouse', 'number_of_agents'))
        swarm.generate()        
        fault_types = cfg.get('faults')
        count = 0

        for i, fault in enumerate(fault_types):
            end_count = self.fault_count[i]+count
            faulty_agents_range = range(count, end_count)
            fault_cfg = self.generate_fault_type(fault, faulty_agents_range)
            swarm.add_fault(**fault_cfg)    
            count = end_count

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
        delivered = self.warehouse.delivered
        counter = self.warehouse.counter

        if self.delivered_in is None and delivered == self.warehouse.number_of_boxes:
            self.delivered_in = counter

        if self.data_model is not None:
            self.data_model.get_metric_data(self.warehouse) # updates metric data for timestep
        
            if self.ad_model is not None:
                self.ad_model.check_thresholds(self.data_model.metric_data)
                # print(self.ad_model.)
        if self.verbose:
            if self.warehouse.counter == 1:
                print("Progress |", end="", flush=True)
            if self.warehouse.counter%100 == 0:
                print("=", end="", flush=True)

        self.exit_sim(delivered, counter)

    def exit_sim(self, delivered, counter):
        if self.cfg.get('exit_on_completion') and delivered == self.cfg.get('warehouse', 'number_of_boxes') or counter > self.cfg.get('time_limit'):
            if self.verbose:
                print("in", counter, "seconds")
            sr = float(delivered/self.cfg.get('warehouse', 'number_of_boxes'))
            if self.verbose:
                print(delivered, "of", self.cfg.get('warehouse', 'number_of_boxes'), "collected =", sr*100, "%")

            self.exit_threads = True

    def run(self):
        if self.verbose:
            print("Running with seed: %d"%self.random_seed)

        while self.warehouse.counter <= self.cfg.get('time_limit'):
            self.iterate()
        
        if self.delivered_in is None:
            self.delivered_in = self.warehouse.counter
        
        if self.verbose:
            print("\n")



class SimTest(Simulator):

    def run(self):
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
        # self.test_FOV_metric_box()
        # self.test_FOV_metric_agent()
        # self.test_FOV_metric_wall()
        self.test_count_lifted_box()

    def test_FOV_metric_box(self, heading_0=0.):
        no_ag = self.swarm.number_of_agents
        self.swarm.heading = np.array([float(heading_0)]*no_ag)
        centre = [self.warehouse.width/2, self.warehouse.height/2]
        self.swarm.repulsion_o = 0
        self.warehouse.rob_c = np.array([centre, [0,0], [30,0]])
        rob_test = self.warehouse.rob_c[0]

        step = 50
        it = int(self.warehouse.counter/step)
        r = 30
        theta = heading_0 + (np.pi/4)*it
        x = r*np.sin(theta)
        y = r*np.cos(theta)
        box_test = rob_test + [x, y]

        self.warehouse.box_c = np.array([box_test, [0,0]])
        data = self.data_model.get_model_data()
        if self.warehouse.counter%10 == 1:
            print("F: %d / B: %d"%(
                data['FOV_object_density_forward'][0], 
                data['FOV_object_density_behind'][0]))

    def test_FOV_metric_agent(self, heading_0=0.):
        no_ag = self.swarm.number_of_agents
        self.swarm.heading = np.array([float(heading_0)]*no_ag)
        centre = [self.warehouse.width/2, self.warehouse.height/2]
        self.swarm.repulsion_o = 0
        self.warehouse.rob_c = np.array([centre, [0,0], [30,0]])
        rob_test = self.warehouse.rob_c[0]
        self.warehouse.box_c = np.array([[30,0], [0,0]])

        step = 50
        it = int(self.warehouse.counter/step)
        r = 30
        theta = heading_0 + (np.pi/4)*it
        x = r*np.sin(theta)
        y = r*np.cos(theta)
        rob_test_2 = rob_test + [x, y]
        self.warehouse.rob_c[1] = rob_test_2

        data = self.data_model.get_model_data()
        if self.warehouse.counter%10 == 1:
            print("F: %d / B: %d"%(
                data['FOV_agent_density_forward'][0], 
                data['FOV_agent_density_behind'][0]))
            
    def test_FOV_metric_wall(self, heading_0=0.):
        no_ag = self.swarm.number_of_agents
        self.swarm.heading = np.array([float(heading_0)]*no_ag)
        rob_test = [self.warehouse.width-65.0, self.warehouse.height-25.0]
        self.swarm.repulsion_o = 0
        self.warehouse.rob_c = np.array([rob_test, [0,0], [30,0]])
        self.warehouse.box_c = np.array([[30,0], [0,0]])

        step = 50
        it = int(self.warehouse.counter/step)
        r = 30
        theta = heading_0 + (np.pi/4)*it
        self.swarm.heading[0] = float(theta)

        data = self.data_model.get_model_data()
        if self.warehouse.counter%10 == 1:
            print("F: %d / B: %d"%(
                data['FOV_wall_density_forward'][0], 
                data['FOV_wall_density_behind'][0]))

    # shouldn't include lifted box in count
    def test_count_lifted_box(self):
        no_ag = self.swarm.number_of_agents
        self.swarm.heading = np.array([0.]*no_ag)
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