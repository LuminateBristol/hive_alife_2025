import random
import math
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
                                             
def _generate_wall_avoidance_force(rob_c, map, robot_index, repulsion_w): # input the warehouse map 
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
        distance_vec = distance_to_wall(rob_c[robot_index], wall)

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

def euclidean_agents(agent1, agent2):
    x1, y1 = agent1[0], agent1[1]   
    x2, y2 = agent2[0], agent2[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def euclidean_boxes(agent, box):
    x1, y1 = agent[0], agent[1]   
    x2, y2 = box.x, box.y
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Computes repulsion forces: a negative force means comes out as attraction
def _generate_interobject_force(boxes, rob_c, robot_index, task, repulsion_o, box_attraction=False):
    repulsion = repulsion_o

    # Agents - always avoid
    agent_dist = [euclidean_agents(rob_c[robot_index], coord) for coord in rob_c]
    too_close_agents = np.array([dist > 0 and dist <= repulsion for dist in agent_dist]) # TRUE if agent is too close to another agent (enable collision avoidance)
    proximity_to_agents = rob_c[robot_index] - rob_c
    F_agent = proximity_to_agents[too_close_agents, :2]                                  # Calc repulsion vector on agent due to proximity to other agents
    F_agent = np.sum(F_agent, axis =0)                                               # Sum the repulsion vectors

    # Boxes - only avoid ig task != pick
    if task != 'pick':
        box_dist = [euclidean_boxes(rob_c[robot_index], coord) for coord in boxes]
        too_close_boxes = np.array([dist > 0 and dist <= repulsion for dist in box_dist]) # TRUE if agent is too close to a box (enable collision avoidance). Does not avoid box if agent does not have a box but this is considered later in the code (not_free*F_box)
        proximity_to_boxes = np.array([rob_c[robot_index] - [box.x, box.y, 0] for box in boxes])
        F_box = proximity_to_boxes[too_close_boxes, :2]                                      # Find which box vectors exhibit forces on the agents due to proximity 
        F_box = np.sum(F_box, axis=0)                                                    # Sum the vectors due to boxes on the agents 
   
    return F_box, F_agent

class select_action(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key='blocks', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_block', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='action', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_position', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_angle', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='internal_task_log', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="boxes", access=py_trees.common.Access.WRITE)
        self.change_direction_counter = 10 # TODO: udpate this to align with config
        self.setup()

    def setup(self): 
        self.logger.debug(f"Select action::setup {self.name}")
        self.blackboard.target_block = None
        self.blackboard.action = 'random_walk'

    def initialise(self):
        self.logger.debug(f"select action::initialise {self.name}")

    def update(self):
        if self.blackboard.action == 'random_walk':
        
            return py_trees.common.Status.SUCCESS

class check_action_random_walk(py_trees.behaviour.Behaviour):

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

class random_walk(py_trees.behaviour.Behaviour):

    def __init__(self,name,robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_block', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="max_v", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="heading_change_rate", access=py_trees.common.Access.WRITE)
        self.setup()
        
    def setup(self):
        self.logger.debug(f"Random walk::setup {self.name}")
        self.change_direction_counter = 30

    def initialise(self):
        self.logger.debug(f"Random walk::initialise {self.name}")

    def update(self):
        # Change direction every x seconds
        # Send heading to the path generator
        self.change_direction_counter -= 1
        if self.change_direction_counter == 0:
            self.blackboard.rob_c[self.robot_index][2] = random.uniform(0, 2 * math.pi)
            self.change_direction_counter = self.blackboard.heading_change_rate 

        return py_trees.common.Status.SUCCESS

class send_path(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="action", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="max_v", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="map", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="repulsion_w", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="repulsion_o", access=py_trees.common.Access.READ)
        self.setup()

    def setup(self): 
        self.logger.debug(f"send path::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"send path::initialise {self.name}")

    def update(self):
        heading = self.blackboard.rob_c[self.robot_index][2]
        f_h = _generate_heading_force(heading)
        f_w = _generate_wall_avoidance_force(self.blackboard.rob_c, self.blackboard.map, self.robot_index, self.blackboard.repulsion_w)
        f_b, f_a = _generate_interobject_force(self.blackboard.boxes, self.blackboard.rob_c, self.robot_index, self.blackboard.action, self.blackboard.repulsion_o)
        
        #print(f'fh {f_h}, fw {f_w}, fb {f_b}, fa {f_a}')
        F = f_h + f_w + f_b + f_a
        F_x = F[0] # total force in x
        F_y = F[1] # total force in y

        computed_heading = np.arctan2(F_y, F_x)
        move_x = np.cos(computed_heading) * self.blackboard.max_v
        move_y = np.sin(computed_heading) * self.blackboard.max_v

        self.blackboard.rob_c[self.robot_index][0] += move_x
        self.blackboard.rob_c[self.robot_index][1] += move_y

        return py_trees.common.Status.SUCCESS        

def create_root(robot_index):
    str_index = 'robot_' + str(robot_index)

    root = py_trees.composites.Sequence(
        name    = f'Pick Place DOTS: {str_index}',
        memory  = False
    )

    # Select actiontalk_to_hive_mind(name='Talk to HM', robot_index=robot_index)
    action = select_action(name='Select action', robot_index=robot_index)
    action_behaviour = py_trees.decorators.Inverter(name='select_action', child=action)

    # Random path behaviour
    random_path = py_trees.composites.Sequence(name='Random walk', memory=False)
    random_path.add_child(check_action_random_walk(name='Check if random walk', robot_index=robot_index))
    random_path.add_child(random_walk(name='Random Walk', robot_index=robot_index))
    random_path.add_child(send_path(name='Send Path Random Walk', robot_index=robot_index))

    # Robot actions
    DOTS_actions = py_trees.composites.Selector(name    = f'DOTS Actions {str_index}', memory=False)
    DOTS_actions.add_child(action_behaviour)
    DOTS_actions.add_child(random_path)

    root.add_child(DOTS_actions)

    return root
