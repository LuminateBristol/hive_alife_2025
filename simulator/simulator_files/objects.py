import numpy as np
import py_trees
import copy

from . import bt_setup, GraphMind

class Robot:
    """
    Represents a robot in the simulation with movement, sensors, and interaction capabilities.
    """
    def __init__(self, radius, max_v, camera_sensor_range, place_tol=None):
        """
        Initializes the robot with given physical and sensor properties.
        See default.yaml and exp_setup.yaml for config info

        Args:
            radius (float): Radius of the robot.
            max_v (float): Maximum velocity of the robot.
            camera_sensor_range (float): Range of the camera sensor.
            place_tol (float, optional): Placement tolerance for objects. Defaults to None.
        """
        self.robot_index = None # String index
        self.root = None
        self.robot_tree = None
        self.blackboard = None

        self.radius = radius
        self.max_v = max_v
        self.camera_sensor_range = camera_sensor_range
        self.place_tol = place_tol

        # Set observations ready for sending to the Hive Mind
        self.observation_space = []

    def add_observations(self):
        """
        Adds self-observable information of the robot to the observation space.
        These observations include position, heading, speed, and various status attributes.
        """
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_position',      'has_status',   {'type':'robot_position_status', 'data':np.array([999, 999, 999]),   'weight':0, 'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_led_status',    'has_status',   {'type':'robot_led_status',      'data':[],                          'weight':0, 'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_lifter_status', 'has_status',   {'type':'robot_lifter_status',   'data':False,                       'weight':0, 'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_heading',       'has_status',   {'type':'robot_heading_status',  'data':0,                           'weight':0, 'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_speed',         'has_status',   {'type':'robot_speed_status',    'data':0,                           'weight':0, 'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_task_id',       'in_progress',  {'type':'robot_task_id_status',  'data':0,                           'weight':0, 'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_pheromone_map', 'has_status',   {'type':'robot_pheromone_map',   'data':0,                          'weight':0, 'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_chosen_door',   'has_status',   {'type':'chosen_door',           'data':0,                           'weight':0, 'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_waypoint',      'in_progress',  {'type':'waypoint',              'data':0,                          'weight':0, 'time':0} ])
    
    def setup_bb(self, width, height, heading_change_rate, repulsion_o, repulsion_w, task_log, delivery_points, traffic_score):
        """
        Sets up the robot's blackboard, registering keys and initializing parameters.

        Args:
            width: width of warehouse
            height: height of warehouse
            heading_change_rate (float): Heading change rate of the robot
            repulsion_o (float): Repulsion odometer from other objects
            repulsion_w (float): Repulsion wall from walls
            task_log (float): Task log
            delivery_points (dict): Delivery points dictionary
            traffic_score (float): Traffic score
        """
        self.blackboard.register_key(key="w_rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="w_boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="carrying_box", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="radius", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="max_v", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="heading_change_rate", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="camera_sensor_range", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="place_tol", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="arena_size", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="repulsion_w", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="repulsion_o", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="local_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="delivery_points", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="traffic_score", access=py_trees.common.Access.WRITE)

        self.blackboard.w_rob_c = []
        self.blackboard.w_boxes = []
        self.blackboard.carrying_box = False
        self.blackboard.radius = self.radius
        self.blackboard.max_v = self.max_v
        self.blackboard.heading_change_rate = heading_change_rate
        self.blackboard.camera_sensor_range = self.camera_sensor_range
        self.blackboard.place_tol = self.place_tol
        self.blackboard.arena_size = [width, height]
        self.blackboard.repulsion_w = repulsion_w
        self.blackboard.repulsion_o = repulsion_o
        self.blackboard.local_task_log = copy.deepcopy(task_log) # Use deepcopy so that each robot has a unique copy
        self.blackboard.delivery_points = delivery_points
        self.blackboard.traffic_score = traffic_score
        
    def add_map(self, map):
        """
        Adds the map to the robots blackboard.
        This is done seperately as it needs to be done after the warehouse is fully initialized.

        Args:
            map (class object): Map object - see objects.py
        """
        self.blackboard.register_key(key="map", access=py_trees.common.Access.WRITE)
        self.blackboard.map = map

    def build_robo_mind(self, entities, tasks, map=None):
        """
        Sets up the robot's internal knowledge graph.

        Args:
            entities (dict): Entities - see exp_setup.yaml
            tasks (dict): Tasks - see exp_setup.yaml
            map (class object): Map object
        """
        self.add_observations()
        self.robo_mind = GraphMind()

        if entities is not None:
            for entity in entities:
                self.robo_mind.add_information_node(entity[0], entity[1], entity[2], **entity[3])
        if tasks is not None:
            for task in tasks:
                self.robo_mind.add_information_node(task[0], task[1], task[2], **task[3])
        if map is not None:
            for wp in map:
                self.robo_mind.add_information_node(wp[0], wp[1], wp[2], **wp[3])

        for observation in self.observation_space:
            self.robo_mind.add_information_node(observation[0], observation[1], observation[2], **observation[3])

        self.blackboard.register_key(key="robo_mind", access=py_trees.common.Access.WRITE)
        self.blackboard.robo_mind = self.robo_mind

    def add_hive_mind(self, hive_mind):
        """
        Sets up the link to the Hive knowledge graph within the robot's blackboard

        Args:
            hive_mind (class object): Hive mind object - see hive_mind.py and initalised in sim.py
        """
        self.blackboard.register_key(key="hive_mind", access=py_trees.common.Access.WRITE)
        self.blackboard.hive_mind = hive_mind

class Box:
    """
    Represents a box object in the warehouse with attributes like position, color, and status.
    """

    def __init__(self, colour=None, id=None):
        """
        Initializes the box with optional color and ID attributes.

        Args:
            colour (str, optional): Color of the box. Defaults to None.
            id (int, optional): Unique identifier for the box. Defaults to None.
        """
        self.x = None
        self.y = None
        self.colour = colour
        self.id = id
        self.action_status = 0 # Set to 1 if being carried or if placed so other robots ignore
        
class Swarm:
    """
    Represents a swarm of robots in the simulation, handling coordination and movement.
    """
    def __init__(self, repulsion_o, repulsion_w, heading_change_rate=1):
        """
        Initializes the swarm with repulsion parameters and heading change rate.

        Args:
            repulsion_o (class object): Repulsion object - see default.yaml
            repulsion_w (class object): Repulsion object - see default.yaml
            heading_change_rate (int): Heading change rate - see default.yaml
        """
        self.agents = [] # turn this into a dictionary to make it accessible later for heterogeneous swarms?
        self.number_of_agents = 0
        self.repulsion_o = repulsion_o # repulsion distance between agents-objects
        self.repulsion_w = repulsion_w # repulsion distance between agents-walls
        self.heading_change_rate = heading_change_rate
        self.F_heading = None
        self.agent_dist = None
    
    def add_agents(self, agent_obj, number, width, height, bt_controller, print_bt = False, task_log=None, delivery_points=None, traffic_score=None):
        """
        Adds a specified number of robot agents to the swarm, initializing their behavior trees and blackboards.

        Args:
            agent_obj (class object): Agent object - see objects.py
            number (int): Number of agents to add - defined per experiment - see exp_setup.yaml
            width (int): Width of the warehouse
            height (int): Height of the warehouse
            bt_controller (class object): Chosen behaviour tree controller object - see exp_setup.yaml
            print_bt (bool): print the behaviour tree for each agent if true
            task_log (class object): Task object - see exp_setup.yaml
            delivery_points (dict): Delivery points dictionary - see exp_setup.yaml
            traffic_score (dict): Traffic score object - only used for traffic scenario experiments
        """
        for num in range(number):
            ag = copy.deepcopy(agent_obj) # Use deepcopy soy that each robot is a unique agent object
            self.agents.append(ag)

        num = 0
        # Each agent obj is of class 'Robot' above
        for ag in self.agents:
            # Add robot to swarm
            ag.robot_index = num
            str_index = 'robot_' + str(ag.robot_index)
            num += 1

            # Set up behaviour tree
            bt_module = bt_setup.behaviour_trees[bt_controller] # Get controller from the setup file
            ag.root = bt_module.create_root(robot_index = ag.robot_index)
            if print_bt:
                py_trees.display.render_dot_tree(ag.root) # Uncomment to print png of behaviour tree
            ag.robot_tree = py_trees.trees.BehaviourTree(ag.root)

            # Set up blackboard
            name    = f'Pick Place DOTS: {str_index}'
            namespace = str_index
            ag.blackboard = py_trees.blackboard.Client(name=name, namespace=namespace)
            ag.setup_bb(width, height, self.heading_change_rate, self.repulsion_o, self.repulsion_w, task_log, delivery_points, traffic_score)

            self.number_of_agents += 1

    def add_map(self, map):
        """
        Adds the map to each agent in the swarm - see Robot class object

        Args:
            map (class object): Map object
        """
        for ag in self.agents:
            ag.add_map(map)

    def add_hive_mind(self, hive_mind):
        """
        Adds the Hive to each robot in the swarm - see Robot class object

        Args:
            hive_mind (class object): Hive mind object
        """
        for ag in self.agents:
            ag.add_hive_mind(hive_mind)

    def iterate(self, rob_c, boxes, init=0):
        """
        Advances the simulation by one step, updating robot and box positions.

        Args:
            rob_c (list): Current positions of robots.
            boxes (list): Current positions of boxes.
            init (int, optional): If set to 1, runs initial setup for behavior trees. Defaults to 0.

        Returns:
            tuple: Updated positions of robots and boxes.
        """
        rob_c_new = rob_c
        boxes_new = boxes

        for ag in self.agents:
            # Run setup() method for all behaviours on first run of the behaviour tree
            if init:
                ag.robot_tree.setup()

            # Update behaviour tree robot positions and boxes after last tick of all robots
            ag.blackboard.w_rob_c = rob_c_new
            ag.blackboard.w_boxes = boxes_new
            # Tick behaviour tree
            ag.robot_tree.tick()
            # py_trees.display.unicode_tree(ag.robot_tree.root)
            # Update robot and box positions
            rob_c_new = ag.blackboard.w_rob_c
            boxes_new = ag.blackboard.w_boxes

        return rob_c_new, boxes_new

class DeliveryPoint:
    """
    Represents a delivery point where boxes are delivered in the warehouse.
    Standard format - only used for certain experiments.
    """
    def __init__(self, x, y, colour, delivered, id):
        """
        Initializes a delivery point with coordinates, color, and delivery status.

        Args:
            x (float): X-coordinate of the delivery point.
            y (float): Y-coordinate of the delivery point.
            colour (str): Color associated with the delivery point.
            delivered (bool): Whether the delivery point has been served.
            id (int): Unique identifier for the delivery point.
        """
        self.x = x
        self.y = y
        self.colour = colour
        self.id = id
        self.delivered = delivered