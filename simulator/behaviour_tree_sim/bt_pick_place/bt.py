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

def _generate_wall_avoidance_force(rob_c, map, robot_index, repulsion_w): # input the warehouse map 
    '''
    Function to compute the force vector for wall-based collisions.
    1) compute robot distance from walls
    2) compute if robots are within safe distance from the walls
    3) compute exponential force vector - increases exponentially with closeness to the walls

    HH 1 - Reduced this down to work with a single robot by removing the rob_c element and using min/max values only
    HH 2 - Removed the 'interaction' calculation that checks if the robot is outside the limits of the walls as it is not needed
    HH note - Not sure if the map boxbound method is the most efficient approach for this but kept to keep in line with previous sim code
    '''
    ## 1) Distance from agent to walls
    # distance from the closest vertical wall to your agent
    difference_in_x = min(map.planeh-rob_c[robot_index][1], key=abs)
    # distance from the closest horizontal wall to your agent
    difference_in_y = min(map.planev-rob_c[robot_index][0], key=abs)

    ## 2) Compute exponential force vectors
    repulsion = repulsion_w

    Fy = np.exp(-2*abs(difference_in_x) + repulsion) # exponent calculation
    Fy = Fy*(difference_in_x*-1)                     # vector components * -1 to reverse the sign to move robot away from wall

    Fx = np.exp(-2*abs(difference_in_y) + repulsion) # exponent calculation
    Fx = Fx*(difference_in_y*-1)                     # vector components * -1 to reverse the sign to move robot away from wall
    
    # Combine to one vector variable
    F = np.array([Fx, Fy])
    return F

def _generate_heading_force(heading):
    '''
    Function to generate force based on a prespecified global heading
    '''
    # Force for movement according to new chosen heading 
    heading_x = 1*np.cos(heading) # move in x 
    heading_y = 1*np.sin(heading) # move in y
    return np.array([heading_x, heading_y])   

def euclidean_agents(agent1, agent2):
    '''
    Function to calculate euclidean distance between two agents or two coordinates
    '''
    x1, y1 = agent1[0], agent1[1]   
    x2, y2 = agent2[0], agent2[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def euclidean_boxes(agent, box):
    '''
    Function to calculate the euclidean distance between an agent or any other coordinate, and a box object
    '''
    x1, y1 = agent[0], agent[1]   
    x2, y2 = box.x, box.y
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# TODO: do euclidean angle generalised formula

def _generate_interobject_force(boxes, rob_c, robot_index, task, repulsion_o, target_box = None, box_attraction=False):
    '''
    Function to generate interobject forces between agents and other agents / boxes
    '''
    repulsion = repulsion_o

    # Agents 
    agent_dist = [euclidean_agents(rob_c[robot_index], coord) for coord in rob_c]
    too_close_agents = np.array([dist > 0 and dist <= repulsion for dist in agent_dist]) # TRUE if agent is too close to another agent (enable collision avoidance)
    proximity_to_agents = rob_c[robot_index] - rob_c
    F_agent = proximity_to_agents[too_close_agents, :2]                                  
    F_agent = np.sum(F_agent, axis =0)                                               

    # Boxes 
    box_dist = [euclidean_boxes(rob_c[robot_index], coord) for coord in boxes]
    too_close_boxes = np.array([dist > 0 and dist <= repulsion for dist in box_dist]) # TRUE if agent is too close to a box (enable collision avoidance). Does not avoid box if agent does not have a box but this is considered later in the code (not_free*F_box)  
    # Check if we are targeting a box - in which case we do not want to avoid it
    if target_box is not None:
        too_close_boxes[target_box] = False
    proximity_to_boxes = np.array([rob_c[robot_index] - [box.x, box.y, 0] for box in boxes])
    F_box = proximity_to_boxes[too_close_boxes, :2]                                    
    F_box = np.sum(F_box, axis=0)                                                 
   
    return F_box, F_agent

class select_action(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key='target_box', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='action', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='heading', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='local_task_log', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="carrying_box", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="pre_place_position", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="target_place_id", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="map", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="place_tol", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="abandon_tol", access=py_trees.common.Access.READ)
        self.setup()

    def setup(self): 
        self.logger.debug(f"Select action::setup {self.name}")
        self.blackboard.target_box = None
        self.blackboard.target_place_id = None
        self.blackboard.action = 'random_walk'
        self.blackboard.carrying_box = False

    def initialise(self):
        self.logger.debug(f"select action::initialise {self.name}")

    def update(self):
        if self.blackboard.action == 'random_walk':
            if self.blackboard.target_box is not None:
                self.blackboard.action = 'pick'
        
        # Pick block
        elif self.blackboard.action == 'pick':
            box = self.blackboard.boxes[self.blackboard.target_box]
            # Check if box has been placed since last tick
            for id, value in self.blackboard.local_task_log.items():
                # Check status of all boxes of that colour
                if value['colour'] == box.colour and value['status'] == 1:
                    # Box has been placed - revert to random walk
                    self.blackboard.action = 'random_walk'
                elif value['colour'] == box.colour and value['status'] != 1:
                    # Check if box has been picked (i.e. robot is under box centroid)
                    centroid_distance = euclidean_boxes(self.blackboard.rob_c[self.robot_index], box)
                    if centroid_distance < self.blackboard.place_tol:
                        # Set the target box
                        self.blackboard.action = 'pre_place'
                        self.blackboard.carrying_box = True
                        break
        
        # Pre-place block
        elif self.blackboard.action == 'pre_place':
            box = self.blackboard.boxes[self.blackboard.target_box]
            # Check if box has been placed in target position since last tick
            if self.blackboard.local_task_log[self.blackboard.target_place_id]['status'] == 1:
                for id, value in self.blackboard.local_task_log.items():
                    if value['colour'] == box.colour and value['status'] == 0:
                        # Box need placing elsewhere
                        self.blackboard.target_place_id = id
                        break
                    else:
                        self.blackboard.target_place_id = None
                if self.blackboard.target_place_id == None:
                    # Box colour has been placed - abandon
                    self.blackboard.action = 'abandon'
                    self.blackboard.rob_c[self.robot_index][2] = 0

            # Otherwise continue checking pre-place position
            else:
                desired_position = self.blackboard.pre_place_position
                dx = desired_position[0] - self.blackboard.rob_c[self.robot_index][0]
                dy = desired_position[1] - self.blackboard.rob_c[self.robot_index][1]
                distance = math.sqrt(dx ** 2 + dy ** 2)
                if distance < self.blackboard.place_tol:
                    self.blackboard.action = 'place'
                    self.blackboard.pre_place_position = None
    
        # Place block
        elif self.blackboard.action == 'place':
            box = self.blackboard.boxes[self.blackboard.target_box]
            # Check if box has been placed in target position since last tick
            if self.blackboard.local_task_log[self.blackboard.target_place_id]['status'] == 1:
                for id, value in self.blackboard.local_task_log.items():
                    if value['colour'] == box.colour and value['status'] == 0:
                        # Box need placing elsewhere
                        self.blackboard.target_place_id = id
                        break
                    else:
                        self.blackboard.target_place_id = None
                if self.blackboard.target_place_id == None:
                    # Box colour has been placed - abandon
                    self.blackboard.action = 'abandon'
                    self.blackboard.rob_c[self.robot_index][2] = 0

            # Otherwise continue checking place position
            else:
                desired_position = self.blackboard.local_task_log[self.blackboard.target_place_id]['target_c']
                dx = desired_position[0] - self.blackboard.rob_c[self.robot_index][0]
                dy = desired_position[1] - self.blackboard.rob_c[self.robot_index][1]
                distance = math.sqrt(dx ** 2 + dy ** 2)
                if distance < self.blackboard.place_tol:
                # Block placed
                    self.blackboard.action = 'random_walk'
                    self.blackboard.target_box = None
                    self.blackboard.target_place_id = None
                    self.blackboard.carrying_box = False
        
        # Abandon
        elif self.blackboard.action == 'abandon':
            box = self.blackboard.boxes[self.blackboard.target_box]
            # If within ABANDON_TOL pixels of the edge of the arena - revert back to explore
            # i.e. if close to edge of arena - drop the block, leave it there and carry on random walk
            if (
                self.blackboard.rob_c[self.robot_index][0] < self.blackboard.abandon_tol 
                or self.blackboard.rob_c[self.robot_index][0] > self.blackboard.map.width - self.blackboard.abandon_tol
                or self.blackboard.rob_c[self.robot_index][1] < self.blackboard.abandon_tol
                or self.blackboard.rob_c[self.robot_index][1] > self.blackboard.map.height - self.blackboard.abandon_tol
            ):
                self.blackboard.action = 'random_walk'
                self.blackboard.target_box = None
                self.blackboard.target_place_id = None
                box.action_status = 0
                self.blackboard.carrying_box = False
        
        # No target block - return to random walk
        elif self.blackboard.target_box is None:
            self.blackboard.action = 'random_walk'
            self.blackboard.carrying_box = False
            self.blackboard.target_place_id = None
        
        #print(f'Robot {self.robot_index} id {self.blackboard.target_place_id} action {self.blackboard.action} task log: {self.blackboard.local_task_log}')
        
        return py_trees.common.Status.SUCCESS

class look_for_blocks(py_trees.behaviour.Behaviour):
    '''
    Checks if there are any blocks within camera_sensor_range
    1) If block found that needs placing - make target block
    2) If block found that has been placed - update internal task log to reflect this
    3) If block found that has been placed - update the global task log so the simulator can track task completion progress
    '''
    def __init__(self,name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_box', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="max_v", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="camera_sensor_range", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="local_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="global_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="place_tol", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_place_id", access=py_trees.common.Access.READ)
        self.setup()

    def setup(self):
        self.logger.debug(f"Select action::setup {self.name}")
        self.blackboard.target_box = None
        
    def initialise(self):
        self.logger.debug(f"select action::initialise {self.name}")

    def update(self):
        for box in self.blackboard.boxes:
            # Exclude block that robot is carrying
            if self.blackboard.boxes.index(box) == self.blackboard.target_box:
                # Skip
                continue

            else:
                distance_to_block = euclidean_boxes(self.blackboard.rob_c[self.robot_index], box)

                # Check within threshold 
                if distance_to_block < self.blackboard.camera_sensor_range:
                    for id, value in self.blackboard.local_task_log.items():
                        # Check status in task log
                        if value['colour'] == box.colour and value['status'] != 1:
                            # Check if box has been placed:
                            target_place = self.blackboard.local_task_log[id]['target_c']
                            distance = euclidean_boxes(target_place, box)
                            # If placed
                            if distance < self.blackboard.place_tol:
                                self.blackboard.local_task_log[id]['status'] = 1
                                self.blackboard.global_task_log[id]['status'] = 1
                            # Else if we are not already targeting a box and box is free - target box
                            elif self.blackboard.target_box is None and box.action_status == 0:
                                self.blackboard.target_box = self.blackboard.boxes.index(box)
                                box.action_status = 1
            
        return py_trees.common.Status.SUCCESS

class update_hive_mind(py_trees.behaviour.Behaviour):
    '''
    '''
    def __init__(self,name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="local_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="global_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="use_hm", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='hm_session_id', access=py_trees.common.Access.WRITE)
        self.url = "http://127.0.0.1:8000"
        self.setup()

    def setup(self):
        self.blackboard.hm_session_id = 0
        self.logger.debug(f"Select update_hive_mind::setup {self.name}")
        
    def initialise(self):
        self.logger.debug(f"select update_hive_mind::initialise {self.name}")

    # Login to HM instance
    def login_pick_place_hm(self, username, password):
        http = httplib2.Http()
        # Set username and password
        payload = {
            "username": username,
            "password": password
        }

        # Send POST request
        response, content = http.request(
            f"{self.url}/login",
            method='POST',
            body=json.dumps(payload)
        )
        
        # Return response
        if response.status == 200:
            content_str = content.decode("utf-8")
            response_data = json.loads(content_str)
            self.blackboard.hm_session_id = response_data.get("session_id")
            print(f"Successsful login to Hive Mind instance, new session ID: {self.blackboard.hm_session_id}")
            return self.blackboard.hm_session_id
        else:
            raise Exception(f"Failed to login. Status code: {response.status}")

    def get_task_data(self):
        http = httplib2.Http()

        # Send GET request
        response, content = http.request(f"{self.url}/tasks/")

        if response.status == 200:
            # Decode the bytes to a string before parsing with json_tricks
            content_str = content.decode("utf-8")
            task_data = json.loads(json.loads(content_str))
            task_dict = [dict(item) for item in task_data] # TODO: this is a  bit hacky - there will be a way to pass dict directly instead of these steps!
            return task_dict
        else:
            raise Exception(f"Failed to get task data. Status code: {response.status}")

    def update_task(self, task_id, new_task_status):
        http = httplib2.Http()
        # Set headers and new status payload
        headers = {'Content-Type': 'application/json', 'session_id' : self.blackboard.hm_session_id}
        payload = {
            "new_status": new_task_status
        }

        # Send PUT request
        response, content = http.request(
            f"{self.url}/tasks/{task_id}",
            method='PUT',
            body=json.dumps(payload),
            headers=headers
        )

        # Return response
        if response.status == 200:
            content_str = content.decode("utf-8")
            updated_task = json.loads(content_str)
            print(f"Updated HM Task: {updated_task}")
            return updated_task
        else:
            raise Exception(f"Failed to update task. Status code: {response.status}")

    def get_task_data_fake(self):
        return self.blackboard.global_task_log
    
    def update_task_fake(self, colour, new_task_status):
        self.blackboard.global_task_log[colour]['status'] = new_task_status
        
    def update(self):
        # Using fake HM
        if self.blackboard.use_hm == 'fake_hm':
            task_data = self.get_task_data_fake()
            for colour in task_data:
                status = task_data[colour]['status']
                if self.blackboard.local_task_log[colour]['status'] != status:
                    if status == int(1):
                        # Update internal task log
                        self.blackboard.local_task_log[colour]['status'] = status
                    elif status == int(0):
                        # Update global task log (the fake HM)
                        self.update_task_fake(colour, 1)
                    else:
                        print(f'Error - HM status is not 1 or 0 it is: {status} of type {type(status)}')
        # Using real HM
        elif self.blackboard.use_hm == 'hm':
            if self.blackboard.hm_session_id:
                # Get the task data
                task_data = self.get_task_data() 
                #print(f'Global task log print: {task_data}')

                # Compare to internal task log
                for task in task_data:
                    id = task["task_id"]
                    colour = task["colour"]
                    status = task["status"]
                    if self.blackboard.local_task_log[colour]['status'] != status:
                        if status == int(1):
                            # Update internal log to reflect placed carrier that has not yet been seen
                            self.blackboard.local_task_log[colour]['status'] = status
                        if status == int(0):
                            # Update HM log to reflect seen placed carrier
                            self.update_task(id, 1)
                        else:
                            print("Error - HM status is not 1 or 0 it is: {status} of type {type(status)}")

            # Else login
            else:
                username = f'robot_{str(self.robot_index)}'
                password = 'hm-v1-1234'
                print(f"R {username}: Attempting to log in to HM, {username}, {password}")
                login_response = self.login_pick_place_hm(username, password)
                print(f'Logging in to HM {login_response}') 
        
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
        self.blackboard.register_key(key="target_box", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="carrying_box", access=py_trees.common.Access.READ)
        self.setup()

    def setup(self): 
        self.logger.debug(f"send path::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"send path::initialise {self.name}")

    def update(self):
        heading = self.blackboard.rob_c[self.robot_index][2]
        f_h = _generate_heading_force(heading)
        f_w = _generate_wall_avoidance_force(self.blackboard.rob_c, self.blackboard.map, self.robot_index, self.blackboard.repulsion_w)
        f_b, f_a = _generate_interobject_force(self.blackboard.boxes, self.blackboard.rob_c, self.robot_index, self.blackboard.action, self.blackboard.repulsion_o, self.blackboard.target_box)
        #print(f'fh {f_h}, fw {f_w}, fb {f_b}, fa {f_a}')
        F = f_h + f_w + f_b + f_a
        F_x = F[0] # total force in x
        F_y = F[1] # total force in y

        computed_heading = np.arctan2(F_y, F_x)

        # Set speed based on task (if picking or placing then use half speed)
        if self.blackboard.action == 'random_walk':
            max_v = self.blackboard.max_v
        else:
            max_v = self.blackboard.max_v / 2

        move_x = np.cos(computed_heading) * max_v
        move_y = np.sin(computed_heading) * max_v

        self.blackboard.rob_c[self.robot_index][0] += move_x
        self.blackboard.rob_c[self.robot_index][1] += move_y

        if self.blackboard.carrying_box:
            self.blackboard.boxes[self.blackboard.target_box].x = self.blackboard.rob_c[self.robot_index][0]
            self.blackboard.boxes[self.blackboard.target_box].y = self.blackboard.rob_c[self.robot_index][1]

        return py_trees.common.Status.SUCCESS        
       
class check_action_random_walk(py_trees.behaviour.Behaviour):

    def __init__(self,name, robot_index):
        super().__init__(name)
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key='action', access=py_trees.common.Access.READ)

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
        self.blackboard.register_key(key='target_box', access=py_trees.common.Access.WRITE)
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

class check_action_pick(py_trees.behaviour.Behaviour):

    def __init__(self,name, robot_index):
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

class pick(py_trees.behaviour.Behaviour):

    def __init__(self,name,robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_box', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="local_task_log", access=py_trees.common.Access.WRITE)
        self.setup()
        
    def setup(self):
        self.logger.debug(f"Pick::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"Random walk::initialise {self.name}")

    def update(self):
        box = self.blackboard.boxes[self.blackboard.target_box]

        # Set heading to block location for send_path
        dx = box.x - self.blackboard.rob_c[self.robot_index][0]
        dy = box.y - self.blackboard.rob_c[self.robot_index][1]
        heading = math.atan2(dy, dx)
        self.blackboard.rob_c[self.robot_index][2] = heading

        return py_trees.common.Status.SUCCESS

class check_action_pre_place(py_trees.behaviour.Behaviour):

    def __init__(self,name, robot_index):
        super().__init__(name)
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key='action', access=py_trees.common.Access.READ)

    def setup(self):
        self.logger.debug(f"Select action::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"select action::initialise {self.name}")

    def update(self):
        if self.blackboard.action == 'pre_place':
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE
        
class pre_place(py_trees.behaviour.Behaviour):

    def __init__(self,name,robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_box', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="local_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="pre_place_position", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="pre_place_delta", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_place_id", access=py_trees.common.Access.WRITE)
        self.setup()

    def setup(self):
        self.logger.debug(f"Carry::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"Carry::initialise {self.name}")

    def update(self):
        if self.blackboard.target_place_id is None:
            box = self.blackboard.boxes[self.blackboard.target_box]

            # Find closest placement position from the task log
            closest_id = None
            min_distance = float('inf')
            for id, value in self.blackboard.local_task_log.items():
                if value['colour'] == box.colour and value['status'] != 1:
                    distance = euclidean_agents(self.blackboard.rob_c[self.robot_index], value['target_c'])
                    if distance < min_distance:
                        min_distance = distance
                        closest_id = id
            self.blackboard.target_place_id = closest_id

        # Calcualte preplace position = delta cm left or right of the target position - pick closest
        desired_position = self.blackboard.local_task_log[self.blackboard.target_place_id]['target_c']
        delta = self.blackboard.pre_place_delta
        distance_1 = euclidean_agents(self.blackboard.rob_c[self.robot_index], [desired_position[0]+delta, desired_position[1]])
        distance_2 = euclidean_agents(self.blackboard.rob_c[self.robot_index], [desired_position[0]-delta, desired_position[1]])
        if distance_1 < distance_2:
            pass
        else:
            delta *= -1

        pre_place_x = desired_position[0]+delta
        pre_place_y = desired_position[1]
        self.blackboard.pre_place_position = [pre_place_x, pre_place_y]

        # Calculate heading and send to blackboard to send_path
        dx = pre_place_x - self.blackboard.rob_c[self.robot_index][0]
        dy = pre_place_y - self.blackboard.rob_c[self.robot_index][1]
        heading = math.atan2(dy, dx)
        self.blackboard.rob_c[self.robot_index][2] = heading
        
        return py_trees.common.Status.SUCCESS

class check_action_place(py_trees.behaviour.Behaviour):

    def __init__(self,name, robot_index):
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
        
class place(py_trees.behaviour.Behaviour):

    def __init__(self,name,robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_box', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="local_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="target_place_id", access=py_trees.common.Access.WRITE)
        self.setup()

    def setup(self):
        self.logger.debug(f"Carry::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"Carry::initialise {self.name}")

    def update(self):
        box = self.blackboard.boxes[self.blackboard.target_box]

        # Calculate placement target position
        desired_position = self.blackboard.local_task_log[self.blackboard.target_place_id]['target_c']

        # Calculate heading and send to blackboard to send_path
        dx = desired_position[0] - self.blackboard.rob_c[self.robot_index][0]
        dy = desired_position[1] - self.blackboard.rob_c[self.robot_index][1]
        heading = math.atan2(dy, dx)
        self.blackboard.rob_c[self.robot_index][2] = heading
        
        return py_trees.common.Status.SUCCESS

class check_action_abandon(py_trees.behaviour.Behaviour):

    def __init__(self,name, robot_index):
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
        
class abandon(py_trees.behaviour.Behaviour):

    def __init__(self,name,robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='target_box', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="local_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="map", access=py_trees.common.Access.READ)
        self.setup()

    def setup(self):
        self.logger.debug(f"Carry::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"Carry::initialise {self.name}")

    def update(self):
        # Calcualte abandon target angle depending on which side of the arena we are on - move towards wall + random angle
        if self.blackboard.rob_c[self.robot_index][2] == 0:
            if self.blackboard.rob_c[self.robot_index][0] > self.blackboard.map.width / 2:
                heading = random.uniform(-math.pi/3, math.pi/3)
            elif self.blackboard.rob_c[self.robot_index][0] <= self.blackboard.map.width / 2:
                heading = random.uniform(-math.pi/3, math.pi/3) + math.pi
            self.blackboard.rob_c[self.robot_index][2] = heading

        return py_trees.common.Status.SUCCESS

def create_root(robot_index):
    str_index = 'robot_' + str(robot_index)

    root = py_trees.composites.Sequence(
        name    = f'Pick Place DOTS: {str_index}',
    )

    # Look for blocks
    look_blocks = look_for_blocks(name='Look for blocks', robot_index=robot_index)

     # Update HM
    update_hm = update_hive_mind(name='Update Hive Mind', robot_index=robot_index)

    # Select actiontalk_to_hive_mind(name='Talk to HM', robot_index=robot_index)
    action = select_action(name='Select action', robot_index=robot_index)

    # Random path behaviour
    random_path = py_trees.composites.Sequence(name='Random walk')
    random_path.add_child(check_action_random_walk(name='Check if random walk', robot_index=robot_index))
    random_path.add_child(random_walk(name='Random Walk', robot_index=robot_index))
    random_path.add_child(send_path(name='Send Path Random Walk', robot_index=robot_index))

    # Pick box behaviour
    pick_box = py_trees.composites.Sequence(name='Pick Box Seq')
    pick_box.add_child(check_action_pick(name='Check if pick', robot_index=robot_index))
    pick_box.add_child(pick(name='Pick Box', robot_index=robot_index))
    pick_box.add_child(send_path(name='Send Path Pick', robot_index=robot_index))

    # Pre-place block behaviour
    pre_place_box = py_trees.composites.Sequence(name='Pre-Place Box')
    pre_place_box.add_child(check_action_pre_place(name='Check if pre-place', robot_index=robot_index))
    pre_place_box.add_child(pre_place(name='Pre-place Box', robot_index=robot_index))
    pre_place_box.add_child(send_path(name='Send Path Pre Place', robot_index=robot_index))

    # Place block behaviour
    place_box = py_trees.composites.Sequence(name='Place Box')
    place_box.add_child(check_action_place(name='Check if place', robot_index=robot_index))
    place_box.add_child(place(name='Place Box', robot_index=robot_index))
    place_box.add_child(send_path(name='Send Path Place', robot_index=robot_index))

    # Place block behaviour
    abandon_pick = py_trees.composites.Sequence(name='Abandon')
    abandon_pick.add_child(check_action_abandon(name='Check if abandon', robot_index=robot_index))
    abandon_pick.add_child(abandon(name='Abandon', robot_index=robot_index))
    abandon_pick.add_child(send_path(name='Send Path Abandon', robot_index=robot_index))

    # Robot actions
    DOTS_actions = py_trees.composites.Selector(name    = f'DOTS Actions {str_index}')
    DOTS_actions.add_child(py_trees.decorators.Inverter(action))
    DOTS_actions.add_child(py_trees.decorators.Inverter(look_blocks))
    DOTS_actions.add_child(py_trees.decorators.Inverter(update_hm))
    DOTS_actions.add_child(random_path)
    DOTS_actions.add_child(pick_box)
    DOTS_actions.add_child(pre_place_box)
    DOTS_actions.add_child(place_box)
    DOTS_actions.add_child(abandon_pick)

    root.add_child(DOTS_actions)

    return root
