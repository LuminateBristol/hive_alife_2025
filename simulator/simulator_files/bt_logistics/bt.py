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
        self.blackboard.register_key(key='local_task_log', access=py_trees.common.Access.WRITE)

    def setup(self):
        self.logger.debug(f"Sense::setup {self.name}")

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

    def update(self):
        self.sense_boxes()
        return py_trees.common.Status.SUCCESS

class Update_Local_Task_Log(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key='boxes_seen', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='local_task_log', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_task_id', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='place_tol', access=py_trees.common.Access.WRITE)

    def setup(self):
        self.logger.debug(f"Update_Local_Task_Log::setup {self.name}")

    def initalise(self):
        self.logger.debug(f"Update_Local_Task_Log::init {self.name}")

    def check_if_box_placed(self):
        '''
        This is used to check ONLY if the boxes that are relevant to the task we are working on.
        If it sees a box of the colour we are interested in, it check to see if that box has been placed.
        '''

        # Check we are working on a delivery
        if self.blackboard.target_task_id is not None:
            task = self.blackboard.local_task_log[self.blackboard.target_task_id]

            for box in self.blackboard.boxes_seen:

                # Check if box has been placed
                if box.colour == task['colour']:
                    if euclidean_objects(task['target_c'], box) < self.blackboard.place_tol:
                        task['status'] = 1
                        break

    def update(self):
        self.check_if_box_placed()
        return py_trees.common.Status.SUCCESS

class Select_Action(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key='action', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='heading', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="w_rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="w_boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='boxes_seen', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='local_task_log', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='delivery_points', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_box', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_task_id', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='place_tol', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='carrying_box', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='abandon_complete', access=py_trees.common.Access.WRITE)
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

    def check_task_status(self, task_id):
        '''
        Checks a task status for a given task from the internal task log.
        :return: True if task needs servicing. False if not.
        '''
        task = self.blackboard.local_task_log[task_id]
        if task['status']:
            return False
        else:
            return True

    def handle_random_walk(self):
        # Check if any seen boxes need picking
        for box in self.blackboard.boxes_seen:
            # Check if box not already picked
            if not box.action_status:
                for task_id, value in self.blackboard.local_task_log.items():
                    # Check if box in task log
                    if box.colour == value['colour']:
                        # Check if box needs placing
                        if self.check_task_status(task_id):
                            self.blackboard.target_box = box
                            self.blackboard.target_task_id = task_id
                            box.action_status = 1
                        break

        if self.blackboard.target_task_id is not None:
            self.blackboard.action = 'pick'

    def handle_pick(self):
        # Check if the box we are picking still needs picking
        task = self.blackboard.target_task_id
        if not self.check_task_status(task):
            self.blackboard.action = 'abandon'

        # Check pick status
        else:
            box = self.blackboard.target_box
            box_distance = euclidean_objects(self.blackboard.w_rob_c[self.robot_index], box)
            if box_distance < self.blackboard.place_tol:
                # Set the target box
                self.blackboard.action = 'place'
                self.blackboard.carrying_box = True

    def handle_place(self):
        # Check if box needs placing
        task = self.blackboard.target_task_id
        if not self.check_task_status(task):
            self.blackboard.action = 'abandon'

        # Check place status
        else:
            desired_position = self.blackboard.local_task_log[task]['target_c']
            dx = desired_position[0] - self.blackboard.w_rob_c[self.robot_index][0]
            dy = desired_position[1] - self.blackboard.w_rob_c[self.robot_index][1]
            distance = math.sqrt(dx ** 2 + dy ** 2)
            if distance < self.blackboard.place_tol:
                # Block placed
                self.blackboard.local_task_log[task]['status'] = 1
                self.blackboard.delivery_points[task].delivered = 1
                self.blackboard.target_box = None
                self.blackboard.target_task_id = None
                self.blackboard.carrying_box = False
                self.blackboard.action = 'random_walk'

    def handle_abandon(self):
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
        self.blackboard.register_key(key="local_task_log", access=py_trees.common.Access.WRITE)
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
        self.blackboard.register_key(key="local_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="target_task_id", access=py_trees.common.Access.WRITE)
        self.setup()

    def setup(self):
        self.logger.debug(f"Carry::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"Carry::initialise {self.name}")

    def update(self):
        box = self.blackboard.target_box

        # Calculate placement target position
        desired_position = self.blackboard.local_task_log[self.blackboard.target_task_id]['target_c']

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
        self.blackboard.register_key(key="local_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="target_task_id", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="abandon_complete", access=py_trees.common.Access.WRITE)
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
        if self.counts > self.abandon_counts:
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
    update_local_task_log = Update_Local_Task_Log(name='Update local task log', robot_index=robot_index)

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
    initial_actions.add_child(update_local_task_log)

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
