from pathlib import Path
dir_root = Path(__file__).resolve().parents[1]

import numpy as np
import random
from . import Swarm, Warehouse, Robot, DeliveryPoint, GraphMind

class Simulator:

    def __init__(self, config, verbose=False):

        self.verbose = verbose
        self.exit_run = False
        self.cfg = config
        self.exit_criteria = self.cfg.get('exit_criteria')
        self.drop_zone_limit = self.cfg.get('drop_zone_limit')

        # Init delivery points
        self.deliverypoints = []
        delivery_points = self.cfg.get('delivery_points')

        # Convert task_log to DeliveryPoint objects and append to the list
        if delivery_points is not None:
            for dp_id, dp_info in delivery_points.items():
                x, y = dp_info['target_c']
                colour = dp_info['colour']
                delivered = dp_info['status']  # Assuming 'status' indicates if delivered or not
                # Append the DeliveryPoint instance
                self.deliverypoints.append(DeliveryPoint(x, y, colour, delivered, dp_id))

        # Init swarm
        try:
            self.swarm = self.build_swarm(self.cfg)
        except Exception as e:
            raise e

        # Init Hive Mind
        self.Hive_Mind = GraphMind()

        # Add entities and tasks from config
        entities = self.cfg.get('entities')
        tasks = self.cfg.get('tasks')
        for entity in entities:
            self.Hive_Mind.add_information_node(entity[0], entity[1], entity[2], **entity[3])
        for task in tasks:
            self.Hive_Mind.add_information_node(task[0], task[1], task[2], **task[3])

        # Build robo_mind and add observation space to Hive Mind for each agent
        for agent in self.swarm.agents:
            agent.build_robo_mind(entities, tasks)
            self.Hive_Mind.add_robot_observation_space(agent.observation_space)

        if self.cfg.get('print_kg') == True:
            self.Hive_Mind.print_graph_mind()
            self.Hive_Mind.plot_node_tree('robot_1')
            self.Hive_Mind.print_hive_mind(attribute_filter={'in_need': 0})

        # Init warehouse
        self.warehouse = Warehouse(
            self.cfg.get('warehouse', 'width'),
            self.cfg.get('warehouse', 'height'), 
            self.cfg.get('warehouse', 'boxes'),
            self.cfg.get('warehouse', 'box_radius'),
            self.swarm, 
            self.cfg.get('warehouse', 'exit_width'),
            self.cfg.get('wallsh'),
            self.cfg.get('wallsv'),
            self.cfg.get('warehouse', 'depot'),
            self.cfg.get('warehouse', 'drop_zone_limit'),
            self.cfg.get('warehouse', 'object_position'),
            hive_mind=self.Hive_Mind)

    def build_swarm(self, cfg):
        robot_obj = Robot(
            cfg.get('robot', 'radius'), 
            cfg.get('robot', 'max_v'),
            camera_sensor_range=cfg.get('robot', 'camera_sensor_range'),
            place_tol = cfg.get('tolerances', 'place_tol')
        )
        
        swarm = Swarm(
            repulsion_o=cfg.get('warehouse', 'repulsion_object'), 
            repulsion_w=cfg.get('warehouse', 'repulsion_wall'),
            heading_change_rate=cfg.get('heading_change_rate')
        )

        swarm.add_agents(robot_obj, cfg.get('number_of_agents'),
                         width=self.cfg.get('warehouse', 'width'),
                         height=self.cfg.get('warehouse', 'height'),
                         bt_controller=self.cfg.get('behaviour_tree'),
                         print_bt = cfg.get('robot', 'print_bt'),
                         task_log = cfg.get('task_log'),
                         delivery_points = self.deliverypoints
                         )
        return swarm

    # iterate method called once per timestep
    def iterate(self):
        self.warehouse.iterate(self.cfg.get('heading_bias'))
        
        self.exit_sim(counter=self.warehouse.counter)

    def exit_sim(self, counter=None):
        if self.exit_criteria == 'counter':
            if counter > self.cfg.get('time_limit'):
                print('{counter} counts reached - Time limit expired')
                self.exit_threads = True
                self.exit_run = True

        elif self.exit_criteria == 'logistics':
            if all(dp.delivered for dp in self.deliverypoints):
                print(f'All deliveries complete in {counter} timesteps - Exit sim.')
                self.exit_threads = True
                self.exit_run = True

            if counter > self.cfg.get('time_limit'):
                print(f'{counter} counts reached - Time limit expired')
                self.exit_threads = True
                self.exit_run = True

    def run(self, iteration=0):
        if self.verbose:
            if iteration:
                print(f"Running simulation iteration: {iteration}")
            else:
                print("Running simulation iteration: 0")

        while self.warehouse.counter <= self.cfg.get('time_limit') and self.exit_run is False:
            self.iterate()
        
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