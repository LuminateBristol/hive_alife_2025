import numpy as np
import py_trees
import copy

from . import bt_setup, GraphMind

class Robot:
    # max_v: max speed, assume robot moves at max speed if healthy
    # camera_sensor: assume camera range is 360deg (may be multiple cameras)
    def __init__(self, radius, max_v, camera_sensor_range, place_tol=None):
        # Setup behaviour tree and share variables
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
        # Add robot self-observable information to the observation space
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_position',      'has_status',   {'type':'robot_position_status', 'data':np.array([999, 999, 999]),   'weight':0, 'time':0} ])
        # self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_led_status',    'has_status',   {'type':'robot_led_status',      'data':[],                          'weight':0, 'time':0} ])
        # self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_lifter_status', 'has_status',   {'type':'robot_lifter_status',   'data':False,                       'weight':0, 'time':0} ])
        # self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_heading',       'has_status',   {'type':'robot_heading_status',  'data':0,                           'weight':0, 'time':0} ])
        # self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_speed',         'has_status',   {'type':'robot_speed_status',    'data':0,                           'weight':0, 'time':0} ])
        # self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_task_id',       'in_progress',  {'type':'robot_task_id_status',  'data':0,                           'weight':0, 'time':0} ])
        # self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_pheromone_map', 'has_status',   {'type':'robot_pheromone_map',   'data':{},                          'weight':0, 'time':0} ])
        # self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_chosen_door',   'has_status',   {'type':'chosen_door',           'data':0,                           'weight':0, 'time':0} ])
        # self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_waypoint',      'in_progress',  {'type':'waypoint',              'data':{},                          'weight':0, 'time':0} ])
    
    def setup_bb(self, width, height, heading_change_rate, repulsion_o, repulsion_w, task_log, delivery_points, traffic_score):
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
        self.blackboard.register_key(key="map", access=py_trees.common.Access.WRITE)
        self.blackboard.map = map

    def build_robo_mind(self, entities, tasks, map):
        self.add_observations()
        self.robo_mind = GraphMind()

        if entities:
            for entity in entities:
                self.robo_mind.add_information_node(entity[0], entity[1], entity[2], **entity[3])
        if tasks:
            for task in tasks:
                self.robo_mind.add_information_node(task[0], task[1], task[2], **task[3])
        if map:
            for wp in map:
                self.robo_mind.add_information_node(wp[0], wp[1], wp[2], **wp[3])

        for observation in self.observation_space:
            self.robo_mind.add_information_node(observation[0], observation[1], observation[2], **observation[3])

        self.blackboard.register_key(key="robo_mind", access=py_trees.common.Access.WRITE)
        self.blackboard.robo_mind = self.robo_mind

    def add_hive_mind(self, hive_mind):
        self.blackboard.register_key(key="hive_mind", access=py_trees.common.Access.WRITE)
        self.blackboard.hive_mind = hive_mind

class Box:
    def __init__(self, colour=None, id=None):
        self.x = None
        self.y = None
        self.colour = colour
        self.id = id
        self.action_status = 0 # Set to 1 if being carried or if placed so other robots ignore
        
class Swarm:
    def __init__(self, repulsion_o, repulsion_w, heading_change_rate=1):
        self.agents = [] # turn this into a dictionary to make it accessible later for heterogeneous swarms?
        self.number_of_agents = 0
        self.repulsion_o = repulsion_o # repulsion distance between agents-objects
        self.repulsion_w = repulsion_w # repulsion distance between agents-walls
        self.heading_change_rate = heading_change_rate
        self.F_heading = None
        self.agent_dist = None
    
    def add_agents(self, agent_obj, number, width, height, bt_controller, print_bt = False, task_log=None, delivery_points=None, traffic_score=None):
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
        for ag in self.agents:
            ag.add_map(map)

    def add_hive_mind(self, hive_mind):
        for ag in self.agents:
            ag.add_hive_mind(hive_mind)

    def iterate(self, rob_c, boxes, init=0):
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
    def __init__(self, x, y, colour, delivered, id):
        self.x = x
        self.y = y
        self.colour = colour
        self.id = id
        self.delivered = delivered