import random
import math
from lib2to3.fixes.fix_metaclass import remove_trailing_newline

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
from scipy.spatial.distance import cdist
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

    # Boxes - only avoid ig task != pick
    if task != 'pick':
        box_dist = [euclidean_objects(w_rob_c[robot_index], coord) for coord in w_boxes]
        too_close_boxes = np.array([dist > 0 and dist <= repulsion for dist in box_dist]) # TRUE if agent is too close to a box (enable collision avoidance). Does not avoid box if agent does not have a box but this is considered later in the code (not_free*F_box)
        proximity_to_boxes = np.array([w_rob_c[robot_index] - [box.x, box.y, 0] for box in w_boxes])
        F_box = proximity_to_boxes[too_close_boxes, :2]                                      # Find which box vectors exhibit forces on the agents due to proximity 
        F_box = np.sum(F_box, axis=0)                                                    # Sum the vectors due to boxes on the agents 
   
    return F_box, F_agent

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
                    if key != 'time':
                        hive_value = hive_mind_attributes[key]
                        robot_value = robot_attributes[key]

                        # Handle comparison for different types
                        if isinstance(hive_value, np.ndarray) or isinstance(hive_value, list):
                            # Compare arrays or lists (ensure same shape/length for arrays)
                            if len(hive_value) == len(robot_value):  # Length check for lists or arrays
                                if not np.array_equal(hive_value, robot_value):
                                    data['hive'] = hive_mind_attributes[key]
                                    data['robot'] = robot_attributes.get(key)
                                    data['attribute'] = key

                        elif isinstance(hive_value, str):
                            # Compare strings
                            if hive_value != robot_value:
                                data['hive'] = hive_mind_attributes[key]
                                data['robot'] = robot_attributes.get(key)
                                data['attribute'] = key

                        elif isinstance(hive_value, (int, float, bool)):
                            # Compare integers or floats
                            if hive_value != robot_value:
                                data['hive'] = hive_mind_attributes[key]
                                data['robot'] = robot_attributes.get(key)
                                data['attribute'] = key
                        else:
                            # Handle any other types as needed
                            print(f"Unsupported type for {key}: {type(hive_value)}")

                # If there are differences, perform actions based on node type
                if data:
                    self.handle_update(node_name, data)

    def handle_update(self, node_name, data):
        robo_mind = self.blackboard.robo_mind.graph
        hive_mind = self.blackboard.hive_mind.graph
        robo_mind_attr = self.blackboard.robo_mind.graph.nodes[node_name]
        hive_mind_attr = self.blackboard.hive_mind.graph.nodes[node_name]

        if node_name.endswith('completion'):
            # If robot completion status is 1, update Hive Mind
            if data['robot'] == 1:
                nx.set_node_attributes(hive_mind, {node_name: robo_mind_attr})
            # If Hive completion status is 1, update the robot
            elif data['hive'] == 1:
                nx.set_node_attributes(robo_mind, {node_name: hive_mind_attr})

        elif node_name.endswith('progress'):
            # If robot progres status is robot name, update Hive Mind
            if data['robot'] and data['hive'] == 0:
                nx.set_node_attributes(hive_mind, {node_name: robo_mind_attr})

            # If Hive progress status is a robot name, update the robot
            elif data['robot'] == 0 and data['hive'] != 0:
                nx.set_node_attributes(robo_mind, {node_name: hive_mind_attr})

            # If both data in the Hive and robot exist and are different from one another
            elif data['hive'] != 0 and data['robot'] != 0 and data['hive'] != data['robot']:

                hive_node = self.blackboard.hive_mind.graph.nodes[node_name]
                robot_node = self.blackboard.robo_mind.graph.nodes[node_name]

                # Check timings - take  most recent data as correct
                if hive_node.get('time') > robot_node.get('time'):
                    nx.set_node_attributes(robo_mind, {node_name: hive_mind_attr})
                else:
                    nx.set_node_attributes(hive_mind, {node_name: robo_mind_attr})

        elif node_name.endswith('position'):
            # Update Hive Mind with robo_mind position attributes
            nx.set_node_attributes(hive_mind, {node_name: robo_mind_attr})

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
        '''
        This is used to check ONLY if the boxes that are relevant to the task we are working on.
        If it sees a box of the colour we are interested in, it check to see if that box has been placed.
        '''

        # Check we are working on a delivery
        if self.blackboard.target_task_id is not None:
            graph = self.blackboard.robo_mind.graph
            task_id = self.blackboard.target_task_id
            task_node = self.blackboard.robo_mind.find_node(task_id)

            # Get task attributes from the robo_mind knowledge graph
            for n in graph.successors(task_node):
                if graph.nodes[n].get('type') == 'box':
                    box_node = n
                if n == f'{task_id}_completion':
                    completion_status_node = n
                if graph.nodes[n].get('type') == 'dp':
                    dp_node = n

            for box in self.blackboard.boxes_seen:
                # Check if box has been placed
                if box != self.blackboard.target_box:
                    if box.colour == graph.nodes[box_node].get('colour'):
                        if euclidean_objects(graph.nodes[dp_node].get('coords'), box) < self.blackboard.place_tol:
                            graph.nodes[completion_status_node]['data'] = 1
                            graph.nodes[completion_status_node]['time'] = self.blackboard.tick_clock
                            break

    def update_position(self):
        self.blackboard.robo_mind.update_attribute(f'{self.str_index}_position', data = self.blackboard.w_rob_c[self.robot_index],
                                                   time=self.blackboard.tick_clock)

    def update_led_status(self):
        pass

    def update_speed(self):
        pass

    def update_heading(self):
        self.blackboard.robo_mind.update_attribute(f'{self.str_index}_heading', status = self.blackboard.rob_c[self.robot_index][2],
                                                   time=self.blackboard.tick_clock)

    def update_lifter_status(self):
        pass

    def update(self):
        self.update_task_status()
        self.update_position()
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
        self.blackboard.register_key(key="w_boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='boxes_seen', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='delivery_points', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_box', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_task_id', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='place_tol', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='carrying_box', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='abandon_complete', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='robo_mind', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='hive_mind', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='tick_clock', access=py_trees.common.Access.READ)
        self.setup()

    def setup(self): 
        self.logger.debug(f"Select action::setup {self.name}")
        self.blackboard.target_box = None
        self.blackboard.target_task_id = None
        self.blackboard.action = 'random_walk'
        self.blackboard.boxes_seen = []
        self.blackboard.abandon_complete = False

    def initialise(self):
        self.logger.debug(f"select action::initialise {self.name}")

    def check_task_status(self, task_id, type):
        '''
        Checks a task status for a given task from the internal task log.
        :param task_id: id for the task in question
        :param type: type of status either 'completion' or 'progress' as these are currently the only two task statuses implemented
        :return: True if task needs servicing. False if not.
        '''
        graph = self.blackboard.robo_mind.graph
        task_node = self.blackboard.robo_mind.find_node(task_id)
        if type == 'completion':
            completion_status_node = next((n for n in graph.successors(task_node) if n == f'{task_id}_completion'), None)
            return graph.nodes[completion_status_node].get('data')
        if type == 'progress':
            progress_status_node = next((n for n in graph.successors(task_node) if n == f'{task_id}_progress'), None)
            return graph.nodes[progress_status_node].get('data')
        else:
            print('Error: status type not recognised, cannot check task status')
            return None

    def handle_random_walk(self):
        if self.blackboard.boxes_seen:
            for box in self.blackboard.boxes_seen:
                if not box.action_status:
                    # Check if box colour needs picking using knowledge graph logic
                    graph = self.blackboard.robo_mind.graph
                    # self.blackboard.robo_mind.print_graph_mind()
                    box_node = next((n for n, attrs in graph.nodes(data=True) if attrs.get('type') == 'box' and attrs.get('colour') == box.colour), None)
                    task_node = next((n for n in graph.predecessors(box_node) if graph.nodes[n].get('type') == 'task'), None)
                    completion_status_node = next((n for n in graph.successors(task_node) if n == f'{task_node}_completion'), None)
                    completion_status = graph.nodes[completion_status_node].get('data')

                    # If it needs picking - set in blackboard
                    if not completion_status:
                        self.blackboard.target_box = box
                        self.blackboard.target_task_id = task_node
                        box.action_status = 1
                        break

        if self.blackboard.target_task_id is not None:
            self.blackboard.action = 'pick'

    def handle_pick(self):
        # Check if the task is completed or is currently being progressed by another agent
        task = self.blackboard.target_task_id
        if self.check_task_status(task, 'completion'):
            print('complete')
            self.blackboard.action = 'random_walk'
            self.blackboard.target_box = None
            self.blackboard.target_task_id = None

        if self.check_task_status(task, 'progress'):
            # Get task data for current task
            graph = self.blackboard.robo_mind.graph
            task_node = self.blackboard.target_task_id
            delivery_point = next((n for n in graph.successors(task_node) if graph.nodes[n].get('type') == 'dp'), None)
            desired_position = graph.nodes[delivery_point].get('coords')

            # Get existing progress robot position from hive_mind
            hive_progress_robot = self.check_task_status(task, 'progress')
            progress_node_name = f'{task_node}_progress'
            hive_robot_position = self.blackboard.hive_mind.graph.nodes[f'{hive_progress_robot}_position'].get('data')

            # Check if position data is available
            if hive_robot_position.any() != 999:

                # Compare to see which robot is closer to the delivery point
                if euclidean_agents(hive_robot_position, desired_position) > euclidean_agents(
                        self.blackboard.w_rob_c[self.robot_index], desired_position):
                    # If the Hive robot is further to the drop point, update local data
                    # self.blackboard.robo_mind.update_attribute(progress_node_name, data=self.str_index, time=self.blackboard.tick_clock)
                    pass
                else:
                    # If this robot is further to the drop point than the Hive robot, update local and abandon pick
                    self.blackboard.robo_mind.update_attribute(progress_node_name, data=hive_progress_robot, time=self.blackboard.tick_clock)
                    self.blackboard.action = 'random_walk'
                    self.blackboard.target_box = None
                    self.blackboard.target_task_id = None

        if self.blackboard.target_box is not None:
        # Check pick status
            box = self.blackboard.target_box
            box_distance = euclidean_objects(self.blackboard.w_rob_c[self.robot_index], box)
            if box_distance < self.blackboard.place_tol:
                # Set the target box
                self.blackboard.action = 'place'
                self.blackboard.carrying_box = True

                # Set progression status to robot string
                graph = self.blackboard.robo_mind.graph
                task_node = self.blackboard.target_task_id
                progress_status_node = next((n for n in graph.successors(task_node) if n == f'{self.blackboard.target_task_id}_progress'), None )
                self.blackboard.robo_mind.update_attribute(progress_status_node, data=self.str_index, time=self.blackboard.tick_clock)

    def handle_place(self):
        # Check if the task has been completed
        task = self.blackboard.target_task_id
        if self.check_task_status(task, 'completion'):
            self.blackboard.action = 'abandon'

        # Check if task is being completed by a different robot
        elif self.check_task_status(task, 'progress') != self.str_index:
            self.blackboard.action = 'abandon'

        # Check place status
        else:
            # Get task data # TODO: refactor the self.blackboard,task_id to be the same as node_name so for quicker task extraction
            graph = self.blackboard.robo_mind.graph
            task_node = next((n for n in graph.nodes if n == task), None)
            delivery_point = next((n for n in graph.successors(task_node) if graph.nodes[n].get('type') == 'dp'), None)
            desired_position = graph.nodes[delivery_point].get('coords')

            dx = desired_position[0] - self.blackboard.w_rob_c[self.robot_index][0]
            dy = desired_position[1] - self.blackboard.w_rob_c[self.robot_index][1]
            distance = math.sqrt(dx ** 2 + dy ** 2)

            if distance < self.blackboard.place_tol:
                # Block placed
                completion_status_node = next((n for n in graph.successors(task_node) if n == f'{task}_completion'), None)
                self.blackboard.robo_mind.update_attribute(completion_status_node, data=1, time=self.blackboard.tick_clock)

                dp_id = int(task_node[-1])-1 # TODO: hacky
                self.blackboard.delivery_points[dp_id].delivered = 1

                # Rest robot status
                self.blackboard.target_box = None
                self.blackboard.target_task_id = None
                self.blackboard.carrying_box = False
                self.blackboard.action = 'random_walk'

    def handle_abandon(self):
        # Return to randon walk if abandon complete
        if self.blackboard.abandon_complete:
            self.blackboard.target_box.action_status = 0
            self.blackboard.target_box = None
            self.blackboard.target_task_id = None
            self.blackboard.carrying_box = False
            self.blackboard.action = 'random_walk'

    def update(self):
        if self.blackboard.action == 'random_walk':
            self.handle_random_walk()

        elif self.blackboard.action == 'pick':
            self.handle_pick()

        elif self.blackboard.action == 'place':
            self.handle_place()

        elif self.blackboard.action == 'abandon':
            self.handle_abandon()

        # print(self.blackboard.action)

        # if self.robot_index == 1:
        #     self.blackboard.hive_mind.extract_tasks()

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

class Check_Action_Pick(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_index):
        super().__init__(name)
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key='action', access=py_trees.common.Access.READ)

    def setup(self):
        self.logger.debug(f"Select action::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"select action::initialise {self.name}")

    def update(self):
        if self.blackboard.action == 'pick':
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class Pick(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="w_rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="w_boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_box', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='robo_mind', access=py_trees.common.Access.WRITE)
        self.setup()

    def setup(self):
        self.logger.debug(f"Pick::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"Random walk::initialise {self.name}")

    def update(self):
        box = self.blackboard.target_box

        # Set heading to block location for send_path
        dx = box.x - self.blackboard.w_rob_c[self.robot_index][0]
        dy = box.y - self.blackboard.w_rob_c[self.robot_index][1]
        heading = math.atan2(dy, dx)
        self.blackboard.w_rob_c[self.robot_index][2] = heading

        return py_trees.common.Status.SUCCESS

class Check_Action_Place(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_index):
        super().__init__(name)
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key='action', access=py_trees.common.Access.READ)

    def setup(self):
        self.logger.debug(f"Select action::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"select action::initialise {self.name}")

    def update(self):
        if self.blackboard.action == 'place':
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class Place(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="w_rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="w_boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="target_box", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="target_task_id", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="robo_mind", access=py_trees.common.Access.WRITE)
        self.setup()

    def setup(self):
        self.logger.debug(f"Carry::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"Carry::initialise {self.name}")

    def update(self):

        # Calculate placement target position
        graph = self.blackboard.robo_mind.graph
        task_node = self.blackboard.target_task_id
        delivery_point = next((n for n in graph.successors(task_node) if graph.nodes[n].get('type') == 'dp'), None)
        desired_position = graph.nodes[delivery_point].get('coords')

        # Calculate heading and send to blackboard to send_path
        dx = desired_position[0] - self.blackboard.w_rob_c[self.robot_index][0]
        dy = desired_position[1] - self.blackboard.w_rob_c[self.robot_index][1]
        heading = math.atan2(dy, dx)
        self.blackboard.w_rob_c[self.robot_index][2] = heading

        return py_trees.common.Status.SUCCESS

class Check_Action_Abandon(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_index):
        super().__init__(name)
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key='action', access=py_trees.common.Access.READ)

    def setup(self):
        self.logger.debug(f"Select action::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"select action::initialise {self.name}")

    def update(self):
        if self.blackboard.action == 'abandon':
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class Abandon(py_trees.behaviour.Behaviour):
    '''
    If the delivery needs abandoning for any reason - e.g. delivery has already been completed.
    Carry the block away from teh delivery zone and abandon in a random location.
    Carrying away from the delivery zone prevents congestion of blocks in one area.
    '''

    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="w_rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="w_boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="target_box", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="target_task_id", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="abandon_complete", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="arena_size", access=py_trees.common.Access.WRITE)
        self.setup()

    def setup(self):
        self.logger.debug(f"Carry::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"Carry::initialise {self.name}")
        self.blackboard.abandon_complete = False

        # Move in random direction north, away from delivery zone
        heading = random.uniform(math.radians(170), math.radians(10))
        self.blackboard.w_rob_c[self.robot_index][2] = heading
        self.counts = 0
        self.abandon_counts = random.randint(200,500)

    def update(self):
        # Complete when number of counts is exceeded or when the robot is in the top half of the arena sufficiently far from drop zone
        if self.counts > self.abandon_counts or self.blackboard.w_rob_c[self.robot_index][1] > (self.blackboard.arena_size[1]/2):
            self.blackboard.abandon_complete = True
            return py_trees.common.Status.SUCCESS
        else:
            self.counts += 1
            return py_trees.common.Status.RUNNING

class Send_Path(py_trees.behaviour.Behaviour):

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
        self.setup()

    def setup(self): 
        self.logger.debug(f"send path::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"send path::initialise {self.name}")

    def update(self):
        # Set speed based on task (if picking or placing then use half speed)
        if self.blackboard.action == 'random_walk':
            max_v = self.blackboard.max_v
        else:
            max_v = self.blackboard.max_v / 2

        # Compute forces on robot based on desired heading and nearby objects
        # This is a basic implementation of the potential-field-algorithm
        heading = self.blackboard.w_rob_c[self.robot_index][2]
        f_h = _generate_heading_force(heading)
        f_w = _generate_wall_avoidance_force(self.blackboard.w_rob_c, self.blackboard.map, self.robot_index, self.blackboard.repulsion_w)
        f_b, f_a = _generate_interobject_force(self.blackboard.w_boxes, self.blackboard.w_rob_c, self.robot_index, self.blackboard.action, self.blackboard.repulsion_o)
        
        #print(f'fh {f_h}, fw {f_w}, fb {f_b}, fa {f_a}')
        F = f_h + f_w + f_b + f_a
        F_x = F[0] # total force in x
        F_y = F[1] # total force in y

        computed_heading = np.arctan2(F_y, F_x)
        move_x = np.cos(computed_heading) * max_v
        move_y = np.sin(computed_heading) * max_v

        self.blackboard.w_rob_c[self.robot_index][0] += move_x
        self.blackboard.w_rob_c[self.robot_index][1] += move_y

        # Also update box position if carrying
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
    update_robo_mind = Update_Robo_Mind(name='Update local task log', robot_index=robot_index)

    # Connect to HM local task log
    connect_to_hm = Connect_To_Hive_Mind(name='Connect to HM', robot_index=robot_index)

    # Random path behaviour
    random_path = py_trees.composites.Sequence(name='Random walk', memory=False)
    random_path.add_child(Check_Action_Random_Walk(name='Check if random walk', robot_index=robot_index))
    random_path.add_child(Random_Walk(name='Random Walk', robot_index=robot_index))

    # Pick box behaviour
    pick_box = py_trees.composites.Sequence(name='Pick Box Seq', memory=False)
    pick_box.add_child(Check_Action_Pick(name='Check if pick', robot_index=robot_index))
    pick_box.add_child(Pick(name='Pick Box', robot_index=robot_index))

    # Place block behaviour
    place_box = py_trees.composites.Sequence(name='Place Box', memory=False)
    place_box.add_child(Check_Action_Place(name='Check if place', robot_index=robot_index))
    place_box.add_child(Place(name='Place Box', robot_index=robot_index))

    # Abandon block behaviour
    abandon_box = py_trees.composites.Sequence(name='Abandon Box', memory=True)
    abandon_box.add_child(Check_Action_Abandon(name='Check if abandon', robot_index=robot_index))
    abandon_box.add_child(Abandon(name='Abandon Box', robot_index=robot_index))

    # Step 1: Sequence to execute initial actions first
    initial_actions = py_trees.composites.Sequence(name='Initial Actions', memory=False)
    initial_actions.add_child(sense)
    initial_actions.add_child(action)
    initial_actions.add_child(update_robo_mind)
    initial_actions.add_child(connect_to_hm)

    # Step 2: Robot actions after the initial actions
    DOTS_actions = py_trees.composites.Selector(name=f'DOTS Actions {str_index}', memory=True)
    Path_actions = py_trees.composites.Sequence(name='Path Actions', memory=False)

    # Step 3: Task actions (includes abandon_box)
    DOTS_actions.add_child(pick_box)
    DOTS_actions.add_child(random_path)
    DOTS_actions.add_child(place_box)
    DOTS_actions.add_child(abandon_box)

    # Step 4: Path actions
    Path_actions.add_child(Send_Path(name='Send Path', robot_index=robot_index))

    # Combine: First run the initial actions, then move on to DOTS_actions
    root.add_child(initial_actions)
    root.add_child(Path_actions)
    root.add_child(DOTS_actions)

    return root
