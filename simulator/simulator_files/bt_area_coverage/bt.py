import copy
import random
import math
from lib2to3.fixes.fix_metaclass import remove_trailing_newline
from turtledemo.nim import computerzug

import py_trees
import numpy as np
import operator
import time
import random
import httplib2
import json_tricks as json
import argparse
import csv
import time

import networkx as nx

def distance_to_wall(robot_c, wall):
    # Find distance to wall using the assumption that the closest point shall be perpendicular to the robot:
    # https://stackoverflow.com/questions/5204619/how-to-find-the-point-on-an-edge-which-is-the-closest-point-to-another-point

    # Check within wall limits
    wall_limits = (min(wall.start[0], wall.end[0]) <= robot_c[0] <= max(wall.start[0], wall.end[0]) or min(wall.start[1], wall.end[1]) <= robot_c[1] <= max(wall.start[1], wall.end[1]))
    if not wall_limits:
        return [0,0]

    # Check for vertical wall
    if wall.start[0] == wall.end[0]:
        x_closest, y_closest=wall.start[0], robot_c[1]
    # Check for horizontal wall
    elif wall.start[1] == wall.end[1]: 
        x_closest, y_closest=robot_c[0], wall.start[1]
    else:
        # Find gradient of the wall (m1) and of the line between the robot and the wall (m2)
        m1 = (wall.end[1]-wall.start[0]) / (wall.end[0]-wall.start[0])
        m2 = -1/m1

        # Find the point on the line that is closest (ie perpendicular) to the robot
        x_closest = (m1*wall.start[0] - m2*robot_c[0] - wall.start[1]) / (m1-m2)
        y_closest = m2* (x_closest-robot_c[0]) + robot_c[1]

    # Find distance to wall in each direction
    dist_to_wall = [robot_c[0]-x_closest, robot_c[1]-y_closest]
    
    return dist_to_wall

def euclidean_agents(agent1, agent2):
    x1, y1 = agent1[0], agent1[1]
    x2, y2 = agent2[0], agent2[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def euclidean_objects(agent, object):
    x1, y1 = agent[0], agent[1]
    x2, y2 = object.x, object.y
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def _generate_wall_avoidance_force(w_rob_c, map, robot_index, repulsion_w): # input the warehouse map
    '''
    Function to compute the force vector for wall-based collisions.
    1) compute robot distance from walls
    2) Use inverse square explonent to calculate force based on distance from wall
    '''
    Fy = 0
    Fx = 0

    # For each wall - compute a force vector based on an inverse square exponential - ie very close walls = very high repulsion force
    for wall in map.walls:
        # Get distance to wall
        distance_vec = distance_to_wall(w_rob_c[robot_index], wall)

        # Output force from wall using inverse exponent and repulsion factor
        Fx += np.exp(-2*abs(distance_vec[0]) + repulsion_w) * np.sign(distance_vec[0])
        Fy += np.exp(-2*abs(distance_vec[1]) + repulsion_w) * np.sign(distance_vec[1])

    # Combine to one vector variable
    F = np.array([Fx, Fy])
    
    return F

def _generate_heading_force(heading):
    # Force for movement according to new chosen heading 
    heading_x = 1*np.cos(heading) # move in x
    heading_y = 1*np.sin(heading) # move in y
    return np.array([heading_x, heading_y])   

def _generate_interobject_force(w_boxes, w_rob_c, robot_index, task, repulsion_o, box_attraction=False):
    repulsion = repulsion_o

    # Agents - always avoid
    agent_dist = [euclidean_agents(w_rob_c[robot_index], coord) for coord in w_rob_c]
    too_close_agents = np.array([dist > 0 and dist <= repulsion for dist in agent_dist]) # TRUE if agent is too close to another agent (enable collision avoidance)
    proximity_to_agents = w_rob_c[robot_index] - w_rob_c
    F_agent = proximity_to_agents[too_close_agents, :2]                                  # Calc repulsion vector on agent due to proximity to other agents
    F_agent = np.sum(F_agent, axis =0)
    F_box = 0

    # Boxes - only avoid if task != pick
    if w_boxes and task != 'pick':
        box_dist = [euclidean_objects(w_rob_c[robot_index], coord) for coord in w_boxes]
        too_close_boxes = np.array([dist > 0 and dist <= repulsion for dist in box_dist]) # TRUE if agent is too close to a box (enable collision avoidance). Does not avoid box if agent does not have a box but this is considered later in the code (not_free*F_box)
        proximity_to_boxes = np.array([w_rob_c[robot_index] - [box.x, box.y, 0] for box in w_boxes])
        F_box = proximity_to_boxes[too_close_boxes, :2]                                      # Find which box vectors exhibit forces on the agents due to proximity 
        F_box = np.sum(F_box, axis=0)                                                    # Sum the vectors due to boxes on the agents 
   
    return F_box, F_agent

def _generate_pheromone_force(w_rob_c, robot_index, pheromone_map, search_radius):
    '''
    Checks the pheromone levels in the three cells directly in front of the robot and returns a float direction indicator.
    The direction value is calculated as a continuous value between -1 and 1, representing the relative pheromone strength on the left vs. right.
    A value close to:
    - 1 indicates stronger pheromone on the left
    - -1 indicates stronger pheromone on the right
    - 0 indicates equal pheromone levels
    :param w_rob_c: Robot coordinates for all robots in sim - list
    :param robot_index: Robot index - integer
    :param pheromone_map: Hash grid map of pheromones - dictionary - {cell_id: pheromone level, ...}
    :param cell_size: Size of each cell in the pheromone map grid - int
    :return: Direction indicator - float between -1 and 1
    '''
    # Robot's current position and heading
    x = w_rob_c[robot_index][0]
    y = w_rob_c[robot_index][1]
    heading = w_rob_c[robot_index][2]  # Robot's heading in radians
    cell_size = 25

    # Offsets to check the three cells in front of the robot
    half_cell = cell_size / 2
    offsets = [
        (math.cos(heading) * cell_size, math.sin(heading) * cell_size),  # Center cell directly ahead
        (math.cos(heading + np.pi / 6) * cell_size, math.sin(heading + np.pi / 6) * cell_size),  # Left cell
        (math.cos(heading - np.pi / 6) * cell_size, math.sin(heading - np.pi / 6) * cell_size)   # Right cell
    ]

    # Get pheromone levels in each of the three cells
    pheromone_levels = []
    for dx, dy in offsets:
        cell_x = (math.floor((x + dx) / cell_size) * cell_size) + half_cell
        cell_y = (math.floor((y + dy) / cell_size) * cell_size) + half_cell
        cell_id = (cell_x, cell_y)
        pheromone_levels.append(pheromone_map.get(cell_id, 0))  # Get pheromone level, default to 0 if none

    # Assign pheromone levels to left, center, and right
    left_level, center_level, right_level = pheromone_levels

    # Calculate a weighted bias based on pheromone levels
    # If left has higher pheromones, output will lean toward 1; if right has higher, it will lean toward -1
    # Center cell acts as a baseline for comparison
    if left_level + right_level > 0:  # Avoid division by zero if both are zero
        direction_bias = (left_level - right_level) / (left_level + right_level)
    else:
        direction_bias = 0

    # Clamp the output to be between -1 and 1, ensuring it remains within this range
    direction_bias = max(-1, min(direction_bias, 1))

    return direction_bias

class Sense(py_trees.behaviour.Behaviour):
    '''
    A behaviour to process simulated camera sensor information within the prespecified sensing range.
    Cameras are used to detect objects, in this case boxes and delivery locations.
    Sensing uses items on blackboard that begin with 'w_', this denotes they are properties of the warehouse.
    These warehouse properties are used only for artificial sensing purposes.
    '''
    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key='camera_sensor_range', access=py_trees.common.Access.READ)
        self.blackboard.register_key(key='place_tol', access=py_trees.common.Access.READ)
        self.blackboard.register_key(key='w_rob_c', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='w_boxes', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='boxes_seen', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='dps_seen', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='tick_clock', access=py_trees.common.Access.WRITE)
        self.setup()

    def setup(self):
        self.logger.debug(f"Sense::setup {self.name}")
        self.blackboard.tick_clock = 0

    def initalise(self):
        self.logger.debug(f"Sense::initialise {self.name}")

    def sense_boxes(self):
        '''
        Look for boxes, if any boxes are within the camera sensor range, add them to the boxes seen in this tick.
        Output: Update blackboard.boxes_seen.
        '''
        self.blackboard.boxes_seen = []
        for box in self.blackboard.w_boxes:
            dis = euclidean_objects(self.blackboard.w_rob_c[self.robot_index], box)
            if dis < self.blackboard.camera_sensor_range:
                    self.blackboard.boxes_seen.append(box)

    def update_tick_clock(self):
        self.blackboard.tick_clock += 1

    def update(self):
        self.sense_boxes()
        self.update_tick_clock()
        return py_trees.common.Status.SUCCESS

class Connect_To_Hive_Mind(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        self.str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(self.str_index))
        self.blackboard.register_key(key='hive_mind', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='robo_mind', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_task_id', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='w_rob_c', access=py_trees.common.Access.WRITE)

    def setup(self):
        self.logger.debug(f"Connect to Hive Mind::setup {self.name}")

    def compare_robot_hive_graphs(self):
        hive_mind_graph = self.blackboard.hive_mind.graph
        robot_graph = self.blackboard.robo_mind.graph

        # Extract nodes where weight > 0 from the Hive Mind graph
        weight_nodes = {node: hive_mind_graph.nodes[node] for node in hive_mind_graph.nodes if hive_mind_graph.nodes[node].get('weight', 0) > 0}

        # Loop through nodes in the weight_nodes dictionary
        for node_name, hive_mind_attributes in weight_nodes.items():
            # Check if the node exists in the robot's knowledge graph
            if node_name in robot_graph.nodes:
                robot_attributes = robot_graph.nodes[node_name]

                # Compare attributes of the two nodes
                data = {}
                for key in hive_mind_attributes:
                    if key == 'data': # We only update data attributes as part of the Hive Mind standard architecture
                        hive_data = hive_mind_attributes[key]
                        robot_data = robot_attributes[key]

                        # Handle comparison for different types
                        if isinstance(hive_data, np.ndarray) or isinstance(hive_data, list):
                            # Compare arrays or lists (ensure same shape/length for arrays)
                            if len(hive_data) == len(robot_data):  # Length check for lists or arrays
                                if not np.array_equal(hive_data, robot_data):
                                    data['hive'] = hive_mind_attributes[key]
                                    data['robot'] = robot_attributes.get(key)
                                    data['attribute'] = key

                        elif isinstance(hive_data, str):
                            # Compare strings
                            if hive_data != robot_data:
                                data['hive'] = hive_mind_attributes[key]
                                data['robot'] = robot_attributes.get(key)
                                data['attribute'] = key

                        elif isinstance(hive_data, (int, float, bool)):
                            # Compare integers or floats
                            if hive_data != robot_data:
                                data['hive'] = hive_mind_attributes[key]
                                data['robot'] = robot_attributes.get(key)
                                data['attribute'] = key

                        elif isinstance(hive_data, dict):
                            if hive_data.keys() != robot_data.keys():
                                data['hive'] = hive_mind_attributes[key]
                                data['robot'] = robot_attributes.get(key)
                                data['attribute'] = key

                        else:
                            # Handle any other types as needed
                            print(f"Unsupported type for {key}: {type(hive_data)}")

                # If there are differences, perform actions based on node type
                if data:
                    self.handle_update(node_name, data)

    def handle_update(self, node_name, data):
        '''
        Handles the update of information based on the node name.
        Each node name has a different pre=programmed update protocol.
        We either update the Hive or the Robo Mind based on the data.
        We use deepcopy because we do not want to intrinsically link the two data objects in Hive and Robo minds
        :param node_name: name of node containing information
        :param data: data held by attributes in the hive and robot (see compare_robot_hive_graphs)
        :return:
        '''
        robo_mind = self.blackboard.robo_mind.graph
        hive_mind = self.blackboard.hive_mind.graph
        robo_mind_attr = self.blackboard.robo_mind.graph.nodes[node_name]
        hive_mind_attr = self.blackboard.hive_mind.graph.nodes[node_name]

        if node_name.endswith('completion'):
            # If robot completion status is 1, update Hive Mind
            if data['robot'] == 1:
                hive_mind.nodes[node_name]['data'] = copy.deepcopy(robo_mind_attr['data'])
            # If Hive completion status is 1, update the robot
            elif data['hive'] == 1:
                robo_mind.nodes[node_name]['data'] = copy.deepcopy(hive_mind_attr['data'])

        elif node_name.endswith('progress'):
            # 1) A robot is progressing the task
            # If we are progressing the task, update Hive Mind
            if data['robot'] == self.str_index and data['hive'] == 0:
                hive_mind.nodes[node_name]['data'] = copy.deepcopy(robo_mind_attr['data'])

            # If Hive progress status is a different robot name, update the robot
            elif data['robot'] == 0 and data['hive'] != self.str_index:
                robo_mind.nodes[node_name]['data'] = copy.deepcopy(hive_mind_attr['data'])

            # 2) A robot has completed the task and is no longer progressing
            # If we have completed the task, update Hive Mind
            elif data['robot'] == 0 and data['hive'] == self.str_index:
                hive_mind.nodes[node_name]['data'] = copy.deepcopy(robo_mind_attr['data'])

            # If local data is another robot and Hive data is now 0, update the robot
            elif data['robot'] != self.str_index and data['robot'] != 0 and data['hive'] == 0:
                robo_mind.nodes[node_name]['data'] = copy.deepcopy(hive_mind_attr['data'])

            # 3) Two different robots are progressing the task (take the most recent data)
            # If both data in the Hive and robot exist and are different from one another
            elif data['hive'] != 0 and data['robot'] != 0 and data['hive'] != data['robot']:

                hive_node = self.blackboard.hive_mind.graph.nodes[node_name]
                robot_node = self.blackboard.robo_mind.graph.nodes[node_name]

                # Check timings - take  most recent data as correct
                if hive_node.get('time') > robot_node.get('time'):
                    robo_mind.nodes[node_name]['data'] = copy.deepcopy(hive_mind_attr['data'])
                else:
                    hive_mind.nodes[node_name]['data'] = copy.deepcopy(robo_mind_attr['data'])

        elif node_name.endswith('position'):
            # Update Hive Mind with robo_mind position attributes
            hive_mind.nodes[node_name]['data'] = copy.deepcopy(robo_mind_attr['data'])

        elif node_name.endswith('pheromone_map'):
            # Update Hive Mind with latest robo_mind pheromone attributes
            hive_mind.nodes[node_name]['data'] = copy.deepcopy(robo_mind_attr['data'])

        # elif node_type == 'robot':
        #     print(f'Robot differences - not implemented yet')

    def initialise(self):
        self.compare_robot_hive_graphs()

    def update(self):
        return py_trees.common.Status.SUCCESS

class Update_Robo_Mind(py_trees.behaviour.Behaviour):
    '''
    Behaviour class to update the data in the robot's observation space knowledge graph.
    '''
    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        self.str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(self.str_index))
        self.blackboard.register_key(key='boxes_seen', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='robo_mind', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='hive_mind', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_task_id', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_box', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='place_tol', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='w_rob_c', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='tick_clock', access=py_trees.common.Access.READ)

    def setup(self):
        self.logger.debug(f"Update_Robot_Mind::setup {self.name}")

    def initalise(self):
        self.logger.debug(f"Update_Robo_Mind::init {self.name}")

    def update_task_status(self):
        pass

    def update_position(self):
        self.blackboard.robo_mind.update_attribute(f'{self.str_index}_position', data = self.blackboard.w_rob_c[self.robot_index],
                                                   time=self.blackboard.tick_clock)

    def update_led_status(self):
        pass

    def update_speed(self):
        pass

    def update_heading(self):
        self.blackboard.robo_mind.update_attribute(f'{self.str_index}_heading', status = self.blackboard.w_rob_c[self.robot_index][2],
                                                   time=self.blackboard.tick_clock)

    def update_lifter_status(self):
        pass

    def update_pheromone_map(self):

        cell_size = 25  # Define hash cell size (e.g., 100 = 100 cm)
        # Access or initialize the pheromone map dictionary
        robo_pheromone_map = self.blackboard.robo_mind.graph.nodes[f'{self.str_index}_pheromone_map'].get('data')

        # Robot's current coordinates
        x = self.blackboard.w_rob_c[self.robot_index][0]
        y = self.blackboard.w_rob_c[self.robot_index][1]

        # Calculate the cell's centroid (used as the cell_id)
        cell_id_x = (math.floor(x / cell_size) * cell_size) + cell_size / 2
        cell_id_y = (math.floor(y / cell_size) * cell_size) + cell_size / 2
        cell_id = (cell_id_x, cell_id_y)

        # Increment the visit count for the cell_id
        if cell_id in robo_pheromone_map:
            robo_pheromone_map[cell_id] += 1
        else:
            robo_pheromone_map[cell_id] = 1

    def update(self):

        hive_mind_graph = self.blackboard.hive_mind.graph
        robot_graph = self.blackboard.robo_mind.graph

        self.update_task_status()
        self.update_position()
        self.update_led_status()
        self.update_speed()
        self.update_heading()
        self.update_pheromone_map()
        return py_trees.common.Status.SUCCESS

class Select_Action(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        self.str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(self.str_index))
        self.blackboard.register_key(key='action', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='heading', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="w_rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='robo_mind', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='hive_mind', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='minima_timer', access=py_trees.common.Access.WRITE)
        self.setup()

    def setup(self): 
        self.logger.debug(f"Select action::setup {self.name}")
        self.blackboard.action = 'ballistic_walk'

    def initialise(self):
        self.logger.debug(f"select action::initialise {self.name}")

    def update(self):

        if self.blackboard.action == 'random_walk':
            self.blackboard.minima_timer -= 1
            if self.blackboard.minima_timer <= 0:
                self.blackboard.action = 'ballistic_walk'

        if self.blackboard.action == 'ballistic_walk':
            pass

        # print(self.blackboard.action)

        return py_trees.common.Status.SUCCESS

class Check_Action_Random_Walk(py_trees.behaviour.Behaviour):

    def __init__(self,name, robot_index):
        super().__init__(name)
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key='action', access=py_trees.common.Access.WRITE)

    def setup(self):
        self.logger.debug(f"Select action::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"select action::initialise {self.name}")

    def update(self):
        if self.blackboard.action == 'random_walk':
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class Random_Walk(py_trees.behaviour.Behaviour):

    def __init__(self,name,robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="w_rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="w_boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_block', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="max_v", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="heading_change_rate", access=py_trees.common.Access.WRITE)
        self.setup()
        
    def setup(self):
        self.logger.debug(f"Random walk::setup {self.name}")
        self.counter = 100
        self.change_direction_counter = self.counter

    def initialise(self):
        self.logger.debug(f"Random walk::initialise {self.name}")

    def update(self):
        # Change direction every x seconds
        # Send heading to the path generator
        self.change_direction_counter -= 1
        if self.change_direction_counter == 0:
            # Set a random angle with a gaussian probability bias towards pi/2 - i.e. North in the warehouse
            mean = math.pi / 2
            std_dev = 3.0
            gaussian_angle = random.gauss(mean, std_dev)
            normalized_angle = gaussian_angle % (2 * math.pi)
            self.blackboard.w_rob_c[self.robot_index][2] = normalized_angle
            self.change_direction_counter = self.counter

        return py_trees.common.Status.SUCCESS

class Check_Action_Ballistic_Walk(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_index):
        super().__init__(name)
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key='action', access=py_trees.common.Access.WRITE)

    def setup(self):
        self.logger.debug(f"Select action::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"select action::initialise {self.name}")

    def update(self):
        if self.blackboard.action == 'ballistic_walk':
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class Ballistic_Walk(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="w_rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="max_v", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="map", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="action", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="hive_mind", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="repulsion_w", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="camera_sensor_range", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="minima_timer", access=py_trees.common.Access.WRITE)
        self.setup()

    def setup(self):
        self.logger.debug(f"Random walk::setup {self.name}")
        self.wall_threshold = 0.4
        self.pheromone_threshold = 0.5

        self.tick_timer = 0
        self.minima_threshold = 30
        self.minima_escape_time = 20
        self.current_cell = None
        self.blackboard.minima_timer = 0

        self.minimum_pheromone = 10000000
        self.min_cell_id = 0
        self.wall_counter = 0

    def initialise(self):
        self.logger.debug(f"Random walk::initialise {self.name}")

    def generate_pheromone_map(self):
        '''
        Generates pheromone map from Hive Mind data.
        :return: dictionary pheromone map
        '''
        swarm_pheromone_map = {}
        graph = self.blackboard.hive_mind.graph
        for node_name in graph:
            if node_name.endswith('pheromone_map'):
                # Use .get() to access the pheromone map from the node
                robot_pheromone_map = graph.nodes[node_name].get('data', {})
                # Merge into the swarm pheromone map
                swarm_pheromone_map.update(robot_pheromone_map)

        return swarm_pheromone_map

    def check_local_minima(self):
        '''
        Checks for local minima by recording if the robot has moved since last timestep.
        This is checked for self.minima_threshold ticks, if no moveemnt in this time, declare local minima
        :return: True if stuck in local minima
        '''

        # Robot's current coordinates
        x = self.blackboard.w_rob_c[self.robot_index][0]
        y = self.blackboard.w_rob_c[self.robot_index][1]

        # Calculate the cell's centroid (used as the cell_id)
        cell_size = 25
        cell_id_x = (math.floor(x / cell_size) * cell_size) + cell_size / 2
        cell_id_y = (math.floor(y / cell_size) * cell_size) + cell_size / 2
        cell_id = (cell_id_x, cell_id_y)

        if cell_id == self.current_cell:
            # Increase timer if we haven't moved since last tick
            self.tick_timer += 1
        else:
            # Reset timer to 0 if we have moved since last tick
            self.tick_timer = 0

        self.current_cell = cell_id

        # If above threshold, conform local minima
        if self.tick_timer > self.minima_threshold:
            self.tick_timer = 0
            return True
        else:
            return False

    def detect_pheromone_in_path(self, x, y, heading, pheromone_map):
        x = self.blackboard.w_rob_c[self.robot_index][0]
        y = self.blackboard.w_rob_c[self.robot_index][1]
        heading = self.blackboard.w_rob_c[self.robot_index][2]  # Robot's heading in radians
        cell_size = 25

        # Calculate the coordinates of the next cell in the robot's path
        next_x = x + cell_size * math.cos(heading)
        next_y = y + cell_size * math.sin(heading)

        # Determine the center of the next cell
        next_cell = (math.floor(next_x / cell_size) * cell_size + cell_size / 2,
                     math.floor(next_y / cell_size) * cell_size + cell_size / 2)

        # Check if there is a pheromone in the next cell
        pheromone_map = self.generate_pheromone_map()
        return pheromone_map.get(next_cell, 0) > 0

    def search_for_next_pheromone_free_cell(self, x, y, heading, pheromone_map, cell_size):
        search_radius = 200
        potential_cells = []

        for dx in range(-search_radius, search_radius + cell_size,cell_size):  # TODO: this is not super efficient as we are searching so many cells but then  generally only using the close ones - possibly stop search if pheromone = 0 ie cant improve on taht
            for dy in range(-search_radius, search_radius + cell_size, cell_size):

                # Skip cells outside boundaries
                if x + dx > 2500 or x + dx < 0 or y + dy > 2500 or y + dy < 0: # TODO: clearly not ideal...
                    continue

                # Calculate the cell center coordinates
                cell_x = (math.floor((x + dx) / cell_size) * cell_size) + cell_size / 2
                cell_y = (math.floor((y + dy) / cell_size) * cell_size) + cell_size / 2
                cell_id = (cell_x, cell_y)

                # Calculate distance and angle to the cell
                dist = math.sqrt(dx ** 2 + dy ** 2)
                angle = math.atan2(dy, dx) - heading

                # Normalize angle to range [-pi, pi]
                angle = (angle + np.pi) % (2 * np.pi) - np.pi

                # Add cell with its angle offset and distance
                potential_cells.append((cell_id, angle, dist))

        # We look for the cell with the minimum pheromone
        # Sort cells: prioritize first searching for cells with:
        # 1. small angle (forward from robot) and 2. short distance (close to robot)
        if random.random() > 0.5:
            potential_cells.sort(key=lambda cell: (abs(cell[2]), cell[1]))
        else:
            potential_cells.sort(key=lambda cell: (abs(cell[2]), -(cell[1])))

        # Search cells in the sorted order to find the minimum pheromone cell
        for cell_id, angle, dist in potential_cells:
            pheromone_level = pheromone_map.get(cell_id, 0)

            # Check pheromone level
            if pheromone_level < self.minimum_pheromone:
                self.minimum_pheromone = pheromone_level
                self.min_cell_id = cell_id
            # Update the minimum based on new information from pheromone map
            self.minimum_pheromone = pheromone_map.get(self.min_cell_id, 0)

        # Calculate angle to minimum pheromone cell
        angle_to_cell = math.atan2(self.min_cell_id[1] - y, self.min_cell_id[0] - x)

        return angle_to_cell

    def update(self):

        # Get wall repulsion force
        f_w = _generate_wall_avoidance_force(self.blackboard.w_rob_c, self.blackboard.map, self.robot_index,
                                             self.blackboard.repulsion_w)

        # If leaving a wall, continue moving until away from wall (unless we find another wall...)
        if self.wall_counter and abs(f_w)[0] < self.wall_threshold and abs(f_w)[1] < self.wall_threshold:
            self.wall_counter -= 1
            return py_trees.common.Status.SUCCESS

        # If pheromone data is available - get pheromone force
        min_pheromone = 0
        try:
            self.blackboard.hive_mind.graph.nodes[f'robot_{self.robot_index}_pheromone_map'].get('data')
        except KeyError:
            pass
        else:
            # Check if there is a pheromone in the next cell on robot's path
            x = self.blackboard.w_rob_c[self.robot_index][0]
            y = self.blackboard.w_rob_c[self.robot_index][1]
            heading = self.blackboard.w_rob_c[self.robot_index][2]  # Robot's heading in radians
            pheromone_map = self.generate_pheromone_map()
            cell_size = 25

            if self.detect_pheromone_in_path(x, y, heading, pheromone_map):
                min_pheromone = self.search_for_next_pheromone_free_cell(x, y, heading, pheromone_map, cell_size)

        # Set heading based on wall force or phereomone if available
        if abs(f_w)[0] > self.wall_threshold or abs(f_w)[1] > self.wall_threshold:
            heading = self.blackboard.w_rob_c[self.robot_index][2]
            self.blackboard.w_rob_c[self.robot_index][2] = heading + random.uniform(-np.pi / 2, np.pi / 2) + np.pi # random direction 90-270 degrees from current heading
            self.wall_counter = 20

        elif min_pheromone:
            self.blackboard.w_rob_c[self.robot_index][2] = min_pheromone

        return py_trees.common.Status.SUCCESS

class Send_Path(py_trees.behaviour.Behaviour):
    '''
    Send path behaviour uses potential field algorithm and adds up all forces from:
    - desired heading
    - repulsion from walls
    - repulsion from boxes
    - repulsion from other robots
    This gives a new heading force based on all of the above.

    Sometimes PFA algorithms can result in a local minima, as such a basic check has been implemented to avoid this.
    If over self.history_length timesteps, the difference between the first position and the last positition is less than'
    self.stuck_threshold, the robot will self detect that it is in a local minima.
    To escape, the robot will move in a random direction for self.escape_duration.
    The cooldown period is in place to stop re-evalation of local minima too quickly after escape.
    '''
    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="action", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="max_v", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="w_boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="w_rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="map", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="repulsion_w", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="repulsion_o", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="carrying_box", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_box", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="camera_sensor_range", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="hive_mind", access=py_trees.common.Access.READ)
        self.setup()

    def setup(self): 
        self.logger.debug(f"send path::setup {self.name}")

        # Escape local minima parameters
        self.position_history = []  # Store last 5 positions
        self.history_length = 5  # Number of timesteps to track for detecting stuck state
        self.stuck_threshold = 3.0  # Threshold for movement over the window of timesteps

        self.escaping = False
        self.escape_steps = 0
        self.escape_duration = 10  # Duration of escape routine
        self.cooldown = 0  # Cooldown period after escape
        self.cooldown_duration = 20  # Cooldown to prevent immediate re-evaluation

    def initialise(self):
            self.logger.debug(f"send path::initialise {self.name}")

    def _update_position_history(self, robot_x, robot_y):
        """
        Update the history of robot positions. Keep track of the last N positions (where N = self.history_length).
        """
        self.position_history.append((robot_x, robot_y))
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)  # Maintain fixed history length

    def _is_stuck_in_local_minima(self):
        """
        Determine if the robot is stuck in a local minimum by checking its total movement over the last N timesteps.
        If the total movement over N timesteps is below a threshold, the robot is considered stuck.
        """
        if len(self.position_history) < self.history_length:
            return False  # Not enough data yet to determine if stuck

        # Calculate displacement from start to end position in the history
        start_pos = np.array(self.position_history[0])
        end_pos = np.array(self.position_history[-1])
        displacement = np.linalg.norm(end_pos - start_pos)

        # If displacement is small and variance is low, the robot is stuck
        if displacement < self.stuck_threshold:
            return True
        return False

    def _escape_local_minima(self, robot_x, robot_y, max_v):
        """
        Escape local minima by introducing random perturbations in movement for a fixed duration.
        """
        if self.escape_steps < self.escape_duration:
            random_angle = np.random.uniform(0, 2 * np.pi)
            move_x = np.cos(random_angle) * max_v
            move_y = np.sin(random_angle) * max_v
            self.escape_steps += 1
        else:
            # Stop escape behavior after the duration and switch back to normal behavior
            self.escaping = False
            move_x = 0
            move_y = 0  # Let potential field take over after escape

        return move_x, move_y

    def update(self):
        # Set speed based on task (if picking or placing then use half speed)
        # Set speed based on task (if picking or placing then use half speed)

        max_v = self.blackboard.max_v

        # Compute forces on robot based on desired heading and nearby objects
        heading = self.blackboard.w_rob_c[self.robot_index][2]
        f_h = _generate_heading_force(heading)
        f_w = _generate_wall_avoidance_force(self.blackboard.w_rob_c, self.blackboard.map, self.robot_index, self.blackboard.repulsion_w)
        f_b, f_a = _generate_interobject_force(self.blackboard.w_boxes, self.blackboard.w_rob_c, self.robot_index, self.blackboard.action, self.blackboard.repulsion_o)

        # print(f'f_h {f_h}, f_w {f_w}, f_b {f_b}, f_a {f_a}, f_p {f_p}')

        # Total force from potential field
        F = f_h + f_w + f_b + f_a
        F_x = F[0]  # Total force in x
        F_y = F[1]  # Total force in y

        # Compute heading from potential field
        computed_heading = np.arctan2(F_y, F_x)
        move_x = np.cos(computed_heading) * max_v
        move_y = np.sin(computed_heading) * max_v

        x = self.blackboard.w_rob_c[self.robot_index][0]
        y = self.blackboard.w_rob_c[self.robot_index][1]
        next_x, next_y = x + math.cos(computed_heading)*max_v, y + math.sin(computed_heading)*max_v

        # Get robot's current position
        robot_x = self.blackboard.w_rob_c[self.robot_index][0]
        robot_y = self.blackboard.w_rob_c[self.robot_index][1]

        # No local minima used for this map
        # # Update position history
        # self._update_position_history(robot_x, robot_y)
        #
        # # Check if robot is stuck in a local minimum every 5 timesteps
        # if self.cooldown == 0 and self._is_stuck_in_local_minima():
        #     if not self.escaping:
        #         self.escaping = True
        #         self.escape_steps = 0  # Start escape routine
        #         self.cooldown = self.cooldown_duration  # Begin cooldown period
        #
        #     move_x, move_y = self._escape_local_minima(robot_x, robot_y, max_v)
        # else:
        #     if self.escaping:
        #         move_x, move_y = self._escape_local_minima(robot_x, robot_y, max_v)
        #     else:
        #         self.escaping = False  # Normal behavior if not stuck
        #
        # # Decrease cooldown timer if active
        # if self.cooldown > 0:
        #     self.cooldown -= 1

        # Update robot's position
        self.blackboard.w_rob_c[self.robot_index][0] += move_x
        self.blackboard.w_rob_c[self.robot_index][1] += move_y

        # Update box position if carrying
        if self.blackboard.carrying_box:
            self.blackboard.target_box.x = self.blackboard.w_rob_c[self.robot_index][0]
            self.blackboard.target_box.y = self.blackboard.w_rob_c[self.robot_index][1]

        return py_trees.common.Status.SUCCESS

def create_root(robot_index):
    str_index = 'robot_' + str(robot_index)

    root = py_trees.composites.Sequence(
        name=f'Pick Place DOTS: {str_index}',
        memory=False
    )

    # Sense
    sense = Sense(name='Sense', robot_index=robot_index)

    # Select action
    action = Select_Action(name='Select action', robot_index=robot_index)

    # Update local task log
    update_robo_mind = Update_Robo_Mind(name='Update robo mind', robot_index=robot_index)

    # Connect to HM local task log
    connect_to_hm = Connect_To_Hive_Mind(name='Connect to HM', robot_index=robot_index)

    # # Random path behaviour
    random_path = py_trees.composites.Sequence(name='Random walk', memory=False)
    random_path.add_child(Check_Action_Random_Walk(name='Check if random walk', robot_index=robot_index))
    random_path.add_child(Random_Walk(name='Random Walk', robot_index=robot_index))

    # Ballistic path behaviour
    ballistic_path = py_trees.composites.Sequence(name='Ballistic walk', memory=False)
    ballistic_path.add_child(Check_Action_Ballistic_Walk(name='Check if ballistic walk', robot_index=robot_index))
    ballistic_path.add_child(Ballistic_Walk(name='Ballistic Walk', robot_index=robot_index))

    # # Phermomone path behaviour
    # pheromone_path = py_trees.composites.Sequence(name='Pheromone walk', memory=False)
    # pheromone_path.add_child(Check_Action_Pheremone_Walk(name='Check if pheromone walk', robot_index=robot_index))
    # pheromone_path.add_child(Pheromone_Walk(name='Pheromone Walk', robot_index=robot_index))

    # Step 1: Sequence to execute initial actions first
    initial_actions = py_trees.composites.Sequence(name='Initial Actions', memory=False)
    initial_actions.add_child(sense)
    initial_actions.add_child(action)
    initial_actions.add_child(update_robo_mind)
    initial_actions.add_child(connect_to_hm)

    # Step 2: Robot actions after the initial actions
    DOTS_actions = py_trees.composites.Selector(name=f'DOTS Actions {str_index}', memory=True)
    Path_actions = py_trees.composites.Sequence(name='Path Actions', memory=False)

    # Step 3: Task actions
    DOTS_actions.add_child(ballistic_path)
    DOTS_actions.add_child(random_path)

    # Step 4: Path actions
    Path_actions.add_child(Send_Path(name='Send Path', robot_index=robot_index))

    # Combine: First run the initial actions, then move on to DOTS_actions
    root.add_child(initial_actions)
    root.add_child(Path_actions)
    root.add_child(DOTS_actions)

    return root
