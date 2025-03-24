from pathlib import Path
dir_root = Path(__file__).resolve().parents[1]

import numpy as np
import random
import matplotlib.pyplot as plt
from . import Swarm, Warehouse, Robot, DeliveryPoint, GraphMind
import copy

class  Simulator:
    """
    The Simulator class serves as the main overarching simulation manager, orchestrating all components of the simulation.
    It initializes and manages the swarm, warehouse, delivery points, and Hive Mind, ensuring the execution of the simulation.
    """

    def __init__(self, config, verbose=False):
        """
        Initialize the simulation environment.

        Args:
            config (class object): Configuration dictionary containing simulation parameters - see lib folder
            verbose (bool, optional): If True, prints verbose output. Defaults to False.
        """
        self.verbose = verbose
        self.exit_run = False
        self.cfg = config
        self.exit_criteria = self.cfg.get('exit_criteria')
        self.drop_zone_limit = self.cfg.get('drop_zone_limit')

        # Init task
        self.task_setup()

        # Init swarm
        try:
            self.swarm = self.build_swarm(self.cfg)
        except Exception as e:
            raise e

        # Init Hive and Robot Knowledge Graph
        self.hive_setup()

        # Init warehouse
        self.warehouse = Warehouse(
            self.cfg,
            self.swarm,
            hive_mind=self.Hive_Mind)

    def task_setup(self):
        """
        Setup task parameters depending on the selected task
        - see lib and exp_setup.yaml - for specification
        """

        # LOGISTICS SETUP
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

        # AREA_COVERAGE SETUP
        # na

        # TRAFFIC SETUP
        self.traffic_score = {'score': 0}

    def hive_setup(self):
        """
        Setup the Hive and Robot knowledge graphs from the following:
        - task / entities: config file - exp_setup.yaml
        - robot observations: objects.py - add_observation_space
        """

        # Build Hive Mind
        self.Hive_Mind = GraphMind()

        # Add entities from config
        try:
            entities = self.cfg.get('entities')
        except:
            print('No entities defined in config')
            entities = None
        else:
            for entity in entities:
                self.Hive_Mind.add_information_node(entity[0], entity[1], entity[2], **entity[3])

        # Add tasks from config
        try:
            tasks = self.cfg.get('tasks')
        except:
            print('No tasks defined in config')
            tasks = None
        else:
            for task in tasks:
                self.Hive_Mind.add_information_node(task[0], task[1], task[2], **task[3])

        # Add a optimistion version on which the cleanup (see below) is not ran
        # This keeps the Hive full size ready to optimise on
        self.optimisation_hive_mind = copy.deepcopy(self.Hive_Mind)

        # Build robo_mind and add observation space to Hive Mind for each agent
        for agent in self.swarm.agents:
            agent.build_robo_mind(entities, tasks)
            self.Hive_Mind.add_robot_observation_space(agent.observation_space)
            self.optimisation_hive_mind.add_robot_observation_space(agent.observation_space)
            self.Hive_Mind.cleanup_hive_mind()

        # Handle graph printing
        if self.cfg.get('print_kg') == True:
            self.Hive_Mind.print_graph_mind()
            self.Hive_Mind.plot_node_tree('robot_1')
            self.Hive_Mind.print_hive_mind()

    def build_swarm(self, cfg):
        """
        Construct the swarm by creating robot agents and adding them to the swarm.

        Args:
            cfg (dict): Configuration dictionary containing swarm and robot parameters.

        Returns:
            Swarm: The initialized swarm object.
        """

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
                         task_log = cfg.get('task_log'),  # TODO: remove - tasks come straight from the Hive Mind
                         delivery_points = self.deliverypoints,
                         traffic_score = self.traffic_score,
                         latency = cfg.get('latency') # TODO: could we just pass the config file rather than endless passing of variables?
                         )
        return swarm

    def iterate(self):
        """
        Perform a single iteration of the simulation by updating the warehouse and checking exit conditions.
        """
        self.warehouse.iterate(self.cfg.get('warehouse', 'pheromones'))
        
        self.exit_sim(counter=self.warehouse.counter)

    def exit_sim(self, counter=None):
        """
        Determine whether the simulation should terminate based on predefined exit criteria.

        Args:
            counter (int, optional): The current time step of the simulation. Defaults to None.
        """
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

        elif self.exit_criteria == 'area_coverage':
            total_cells = (self.cfg.get('warehouse', 'width') * self.cfg.get('warehouse', 'height')) / self.cfg.get('warehouse', 'cell_size') ** 2
            percent_explored = (len(self.warehouse.pheromone_map) / total_cells) * 100
            if counter > self.cfg.get('time_limit') or percent_explored >= 80:
                print(f'{counter} counts reached - percentage explored: {percent_explored}%')
                # self.print_pheromone_map(self.warehouse.pheromone_map, self.cfg.get('warehouse', 'width'), self.cfg.get('warehouse', 'height'), 'test')
                self.exit_threads = True
                self.exit_run = True

        elif self.exit_criteria == 'traffic':
            # print(counter, self.traffic_score)
            if self.traffic_score['score'] >= 200 or counter > self.cfg.get('time_limit'):
                print(f"Counts: {counter}. Traffic score: {self.traffic_score['score']}")
                self.exit_threads = True
                self.exit_run = True

    def print_pheromone_map(self, data_map, width, height, name):
        """
        Generate and save a pheromone map visualization for post-analysis.

        Args:
            data_map (dict): A dictionary mapping grid coordinates to pheromone values.
            width (int): The width of the warehouse grid.
            height (int): The height of the warehouse grid.
            name (str): The filename for saving the output image.
        """

        # Extract x, y coordinates and strength values
        x_coords = [key[0] for key in data_map.keys()]
        y_coords = [key[1] for key in data_map.keys()]
        strengths = [data_map[key] for key in data_map.keys()]

        # Normalize strength values to set transparency (between 0 and 1)
        max_strength = max(max(strengths), 60) # To get some consitency in the plots we use 80 as the maximum visits per hash grid cell unless the odd one goes over
        alphas = [(strength / max_strength) for strength in strengths]

        plt.figure(figsize=(width / 100, height / 100))  # Set plot size in inches
        plt.scatter(x_coords, y_coords, s=130, c='blue', alpha=alphas, marker='s')  # Plot with squares

        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.gca().set_aspect('equal', adjustable='box')  # Equal scaling for x and y

        # Save the figure to a file
        plt.savefig(f'{name}.png', format='png', dpi=300)
        plt.close()  # Close the figure to free memory

    def run(self, iteration=0):
        """
        Run the simulation loop until the exit criteria are met.

        Args:
            iteration (int, optional): The starting iteration number. Defaults to 0.

        Returns:
            int: The total number of iterations completed before termination.
        """

        if self.verbose:
            if iteration:
                print(f"Running simulation iteration: {iteration}")
                pass
            else:
                print(f"Running simulation.")

        while self.warehouse.counter <= self.cfg.get('time_limit') and self.exit_run is False:
            self.iterate()

        return self.warehouse.counter
