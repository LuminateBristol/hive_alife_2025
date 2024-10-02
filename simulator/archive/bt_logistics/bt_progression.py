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

def check_for_nearby_boxes(agent, boxes, tol, carrying_box=None):
    '''
    Function to check if boxes are within a certain tolerance of the carrying agent.
    For example used in abandon to make sure boxes aren't dropped too close to one another.
    Returns True if box is within tolerance
    '''
    min_dis = float('inf')
    for box in boxes:
        if box == carrying_box:
            pass
        else:
            dis_to_box = euclidean_boxes(agent, box)
            if dis_to_box < min_dis:
                min_dis = dis_to_box

    if min_dis < tol:
        return True
    else:
        return False
    
class select_action(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        self.str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(self.str_index))
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
        self.blackboard.register_key(key="global_carry_to_zero", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="repulsion_o", access=py_trees.common.Access.WRITE)
        self.setup()

    def setup(self): 
        self.logger.debug(f"Select action::setup {self.name}")
        self.blackboard.target_box = None
        self.blackboard.target_place_id = None
        self.blackboard.action = 'random_walk'
        self.blackboard.carrying_box = False
        self.blackboard.global_carry_to_zero = []

    def initialise(self):
        self.logger.debug(f"select action::initialise {self.name}")

    def update(self):
        if self.blackboard.action == 'random_walk':
            if self.blackboard.target_box is not None:
                self.blackboard.action = 'pick'
        
        # Pick block
        elif self.blackboard.action == 'pick':
            box = self.blackboard.boxes[self.blackboard.target_box]

            # Check if same colous box has been placed (status==1) or if same colour box is carried (carry_status!=0)
            if all(item['status'] == 1 for item in self.blackboard.local_task_log.values() if item['colour'] == box.colour) or all(item['carry_status'] != 0 for item in self.blackboard.local_task_log.values() if item['colour'] == box.colour):            
                self.blackboard.action = 'random_walk'
                self.blackboard.target_box = None
                self.blackboard.carrying_box = False
            else:
                centroid_distance = euclidean_boxes(self.blackboard.rob_c[self.robot_index], box)
                if centroid_distance < self.blackboard.place_tol:
                    # Set the target box
                    self.blackboard.action = 'pre_place'
                    self.blackboard.carrying_box = True
        
        # Pre-place block
        elif self.blackboard.action == 'pre_place':
            box = self.blackboard.boxes[self.blackboard.target_box]
            id = self.blackboard.target_place_id
            # Check if box has been placed or is being carried by another agent
            if (self.blackboard.local_task_log[id]['status'] == 1 or self.blackboard.local_task_log[id]['carry_status'] not in {self.str_index, 0}):

                # If box hasbeen placed, update carry status 
                if self.blackboard.local_task_log[id]['status'] == 1:
                    self.blackboard.local_task_log[id]['carry_status'] = 0
                    self.blackboard.global_carry_to_zero.append(self.blackboard.target_place_id)

                # Look for new target place id for the same box colour (only applies if there are multiple delivery points per colour)
                for id, value in self.blackboard.local_task_log.items():
                    if value['colour'] == box.colour and value['status'] == 0 and value['carry_status'] == 0:
                        # Box need placing elsewhere - continue to carry
                        self.blackboard.target_place_id = id
                        self.blackboard.local_task_log[id]['carry_status'] = self.str_index
                        break
                    else:
                        self.blackboard.target_place_id = None

                # Box colour has been placed or is being carried by other agent - abandon
                if self.blackboard.target_place_id == None:
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
            id = self.blackboard.target_place_id

            # Check if box has been placed or is being carried by another agent
            if (self.blackboard.local_task_log[id]['status'] == 1 or self.blackboard.local_task_log[id]['carry_status'] not in {self.str_index, 0}):

                # If box hasbeen placed, update carry status
                if self.blackboard.local_task_log[id]['status'] == 1:
                    self.blackboard.local_task_log[id]['carry_status'] = 0
                    self.blackboard.global_carry_to_zero.append(self.blackboard.target_place_id)

                # Look for new target place id
                for id, value in self.blackboard.local_task_log.items():
                    if value['colour'] == box.colour and value['status'] == 0 and value['carry_status'] == 0:
                        # Box need placing elsewhere
                        self.blackboard.target_place_id = id
                        self.blackboard.local_task_log[id]['carry_status'] = self.str_index
                        break
                    else:
                        self.blackboard.target_place_id = None

                # Box colour has been placed or is being carried by other agent - abandon
                if self.blackboard.target_place_id == None:
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
                    # Update carry status for progression status
                    self.blackboard.local_task_log[self.blackboard.target_place_id]['carry_status'] = 0
                    self.blackboard.global_carry_to_zero.append(self.blackboard.target_place_id)
                    self.blackboard.target_place_id = None
                    self.blackboard.carrying_box = False
        
        # Abandon
        elif self.blackboard.action == 'abandon':
            box = self.blackboard.boxes[self.blackboard.target_box]
            # If within ABANDON_TOL pixels of the edge of the arena - revert back to explore
            # i.e. if close to edge of arena and not too close to other blocks - drop the block, leave it there and carry on random walk
            if (
                (self.blackboard.rob_c[self.robot_index][0] > self.blackboard.abandon_tol 
                and self.blackboard.rob_c[self.robot_index][0] < self.blackboard.map.width - self.blackboard.abandon_tol
                and self.blackboard.rob_c[self.robot_index][1] > self.blackboard.abandon_tol
                and self.blackboard.rob_c[self.robot_index][1] < self.blackboard.map.height - self.blackboard.abandon_tol)
                and not check_for_nearby_boxes(self.blackboard.rob_c[self.robot_index], self.blackboard.boxes, tol=30, carrying_box=box)
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
        
        #print(f'Robot {self.robot_index} loc {self.blackboard.rob_c[self.robot_index][0]} action {self.blackboard.action} task log: {self.blackboard.local_task_log}')

        #print(f'Robot {self.robot_index} id {self.blackboard.target_place_id} action {self.blackboard.action}')
        
        return py_trees.common.Status.SUCCESS

class update_carry_count(py_trees.behaviour.Behaviour):

    def __init__(self,name, robot_index):
        super().__init__(name)
        str_index = 'robot_' + str(robot_index)
        self.robot_index = robot_index
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key='carry_count', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='carrying_box', access=py_trees.common.Access.READ)

    def setup(self):
        self.logger.debug(f"Update_carry_count::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"Update_carry_count::initialise {self.name}")

    def update(self):
        if self.blackboard.carry_count is not None:
            if self.blackboard.carrying_box:
                self.blackboard.carry_count[self.robot_index] += 1
        
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
        self.blackboard.register_key(key="carrying_box", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="place_tol", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_place_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="use_global", access=py_trees.common.Access.READ)
        self.setup()

    def setup(self):
        self.logger.debug(f"Select action::setup {self.name}")
        self.blackboard.target_box = None
        self.blackboard.carrying_box = False
        
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
                                # Update global log if needed (ie if we need for exit_sim in sim.py):
                                if self.blackboard.use_global == 'None':
                                    self.blackboard.global_task_log[id]['status'] = 1
                            # Else if we are not already targeting a box and box is free - target box
                            elif self.blackboard.target_box is None and box.action_status == 0:
                                self.blackboard.target_box = self.blackboard.boxes.index(box)
                                box.action_status = 1

        return py_trees.common.Status.SUCCESS

class Msg:
    # Simple class to standardize message parameters
    def __init__(self, type, data):
        self.type = type
        self.data = data

class update_global(py_trees.behaviour.Behaviour): 
    '''
    '''
    def __init__(self, name, robot_index):
        super().__init__(name)
        self.robot_index = robot_index
        str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(str_index))
        self.blackboard.register_key(key="rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="local_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="global_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="use_global", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='hm_session_id', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="target_place_id", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="global_carry_to_zero", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='buffer_recv', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='buffer_send', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='drop_out_rate', access=py_trees.common.Access.WRITE)
        self.url = "http://127.0.0.1:8000"
        self.setup()

        # Create tick buffer lists
        # These are populated with all messages send in a single tick of this behaviour
        # These are then added to the global buffer list
        # Each tick, a tick buffer list is popped out of each of the global buffer lists
        # This allows us to apply a tick-wise latency to the entire system
        self.tick_buffer_recv = []
        self.tick_buffer_send = []

    def setup(self):
        self.blackboard.hm_session_id = 0
        self.blackboard.target_place_id = None
        self.logger.debug(f"Select update_global_mind::setup {self.name}")
        
    def initialise(self):
        self.logger.debug(f"select update_global_mind::initialise {self.name}")

    def get_task_data_fake(self):
        '''Requests data from fake global. Includes drop out rate. Sends data to the recv buffer.'''
        if self.handle_dropout():
            return None
        
        # If no drop out - return task log from global with latency applied
        else:
            self.add_to_tick_buffer('recv', type='global_data_recv', data=self.blackboard.global_task_log)
    
    def update_task_fake(self, id, new_task_status = None, new_carry_status = None):
        if self.handle_dropout():
            pass

        # If no latency or drop out - update the global
        else:
            if new_task_status is not None:
                self.add_to_tick_buffer('send', type='global_data_send', data={"id":id, "data_type":'status', "data": new_task_status})
            if new_carry_status is not None:
                self.add_to_tick_buffer('send', type='global_data_send', data={"id":id, "data_type":'carry_status', "data": new_carry_status})
    
    def add_to_tick_buffer(self, dir, type, data):
        # Add messages to and from global to respective buffers
        msg = Msg(type, data)

        if dir == 'send':
            self.tick_buffer_send.insert(0, msg)
        elif dir == 'recv':
            self.tick_buffer_recv.insert(0, msg)
             
    def handle_latency_recv(self):
        # Add all recv_messages from this timestep to the recv buffer, if none received, add 0
        if self.tick_buffer_recv:
            self.blackboard.buffer_recv.insert(0, self.tick_buffer_recv)
        else:
            self.blackboard.buffer_recv.insert(0, 0)

        # Output incoming recv_message from the buffer for this timestep
        msg_recv = self.blackboard.buffer_recv.pop()

        return msg_recv
    
    def handle_latency_send(self):
        # Add all send messages from this timestep to the recv buffer, if none sent, add 0
        if self.tick_buffer_send:
            self.blackboard.buffer_send.insert(0, self.tick_buffer_send)
        else:
            self.blackboard.buffer_send.insert(0, 0)

        # Latency of outgoing messages
        msg_send = self.blackboard.buffer_send.pop()

        return msg_send

    def handle_dropout(self):
        #Implement drop out rate as a probability using random number between 0 and 1
        # Compare the random number to the actual drop out rate and only send if within range
        # Example: if dop our rate =  10%, then 90% of messages get through, drop out message only if random num < 0.1
        # Return true if message is dropped
        if random.random() < self.blackboard.drop_out_rate:
            return True
        else:
            return False
        
    def update(self):
        # Using fake global
        if self.blackboard.use_global == 'global':

            # Reset buffer message each timestep
            self.tick_buffer_recv = []
            self.tick_buffer_send = []

            # 1. Request data from the global
            self.get_task_data_fake()
            recv_msg = self.handle_latency_recv()

            # Get task data from buffer msg if available
            if recv_msg != 0:
                for msg in recv_msg:
                    if msg.type == 'global_data_recv':
                        task_data = msg.data
            else:
                task_data = None

            # Check if drop out or latency
            if task_data is None or task_data == 0:
                pass
            
            else:
                # We have received data succesfully so update internal and global task logs if needed
                for id in task_data:
                    global_status = task_data[id]['status']
                    global_carry_status = task_data[id]['carry_status']

                    # Completion status updates - either 0 or 1 depending if box has been delivered yet
                    if self.blackboard.local_task_log[id]['status'] != global_status:

                        if global_status == int(1):
                            # Update internal task log
                            self.blackboard.local_task_log[id]['status'] = global_status

                        elif global_status == int(0):
                            # Send update to global task log (the fake global) 
                            self.update_task_fake(id, new_task_status=1)

                        else:
                            print(f'Error - global status is not 1 or 0 it is: {global_status} of type {type(global_status)}')

                    # Progression status updates
                    # If we are targetting this box id - robot name of the robot that has first stated to the global that they are carrying a box
                    # Note that with latency it is possible for two robots to both be carrying the same coloured box
                    # In which case the robot that was first to upload their name to carry_status on the global, has priority
                    if self.blackboard.target_place_id == id:
                        
                        # Check if global aligns with local task log
                        if global_carry_status != self.blackboard.local_task_log[id]['carry_status']:

                            # Upate local global if box being carried by a different robot
                            if global_carry_status != 0:
                                self.blackboard.local_task_log[id]['carry_status'] = global_carry_status
                            
                            # Update global global if box not being carried by a different robot to say we are carrying now
                            elif global_carry_status == 0:
                                self.update_task_fake(id, new_carry_status=self.blackboard.local_task_log [id]['carry_status']) 
                    
                    # Else if not target box but needs setting to None (ie we no longer are carrying box)
                    elif id in self.blackboard.global_carry_to_zero:
                        self.update_task_fake(id, new_carry_status=0) 
                    
                    # Else if we are not targeting box, then update local task log based on carry status   
                    elif global_carry_status != self.blackboard.local_task_log [id]['carry_status']:
                            self.blackboard.local_task_log[id]['carry_status'] = global_carry_status    
                     
                    # All ids have been checked and reverted to zero where needed so reset blackboard variable
                    self.blackboard.global_carry_to_zero = []   

                    # 2. Send data to the global
                    # Returns a list of messages to send in this tick (based on send requests & tick based latency)
                    buffer_msg_send = self.handle_latency_send()

                    if buffer_msg_send != 0:
                        # In this implementation, we only have one message type we are interested in: global task updates
                        # There is a global side check here to make sure that the update is ok:
                        # - if 0 then take robot name
                        # - else if robot name alreay in place then reject request
                        # This is to take account of latency and prevent overwriting of data
                        for msg in buffer_msg_send:
                            if msg.type == 'global_data_send':
                                id = msg.data['id']
                                data_type = msg.data['data_type']
                                data = msg.data['data']
                                # If incoming data is 0 - i.e. resetting status to 0:
                                if data == 0:
                                    self.blackboard.global_task_log[id][data_type] = data
                                # If existing data is 0 - i.e. status / carry_status is ready to be updated
                                elif self.blackboard.global_task_log[id][data_type] == 0:
                                    self.blackboard.global_task_log[id][data_type] = data
                            
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
        self.str_index = 'robot_' + str(robot_index)
        self.blackboard = self.attach_blackboard_client(name=name, namespace=str(self.str_index))
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
        # If no target_place_id has been set yet:
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
            # Set carry status locally for progression information sharing
            self.blackboard.local_task_log[closest_id]['carry_status'] = self.str_index
        
        if self.blackboard.target_place_id is None:
            print('Error - no target place id found for the target block despite moving into pre-place')
            print(f'R{self.robot_index}, task_log {self.blackboard.local_task_log}, box picked id {box.id} colour {box.colour}')
        
        # Check a target_place_id has been found
        #if self.blackboard.target_place_id is not None:
        # Calcualte preplace position = delta cm left or right of the target position - pick closest
        desired_position = self.blackboard.local_task_log[self.blackboard.target_place_id]['target_c'] # TODO: this causes error occasionally of a None type KeyError - temporary hacky fix with the if statement above
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
        self.blackboard.register_key(key="heading_change_rate", access=py_trees.common.Access.READ)
        self.setup()

    def setup(self):
        self.logger.debug(f"Carry::setup {self.name}")
        self.change_direction_counter = 30

    def initialise(self):
        self.logger.debug(f"Carry::initialise {self.name}")

    def update(self):
        # Random walk until we are in suitable location to abandon box (i.e. far from walls where drop points are)
        # Change direction every x seconds
        # Send heading to the path generator
        self.change_direction_counter -= 1
        if self.change_direction_counter == 0:
            self.blackboard.rob_c[self.robot_index][2] = random.uniform(0, 2 * math.pi)
            self.change_direction_counter = self.blackboard.heading_change_rate 

        return py_trees.common.Status.SUCCESS

def create_root(robot_index):
    str_index = 'robot_' + str(robot_index)

    root = py_trees.composites.Sequence(
        name    = f'Pick Place DOTS: {str_index}',
        memory=False
    )

    # Look for blocks
    look_blocks = look_for_blocks(name='Look for blocks', robot_index=robot_index)

     # Update global
    update_global_server = update_global(name='Update global', robot_index=robot_index)

    # Select actiontalk_to_global_mind(name='Talk to global', robot_index=robot_index)
    action = select_action(name='Select action', robot_index=robot_index)

    # Update carry counter - this tracks how many timesteps the robots spend carrying boxes
    # Used for post-processing data analysis only
    carry_count = update_carry_count(name='Update Carry Count', robot_index=robot_index)

    # Random path behaviour
    random_path = py_trees.composites.Sequence(name='Random walk', memory=False)
    random_path.add_child(check_action_random_walk(name='Check if random walk', robot_index=robot_index))
    random_path.add_child(random_walk(name='Random Walk', robot_index=robot_index))
    random_path.add_child(send_path(name='Send Path Random Walk', robot_index=robot_index))

    # Pick box behaviour
    pick_box = py_trees.composites.Sequence(name='Pick Box Seq', memory=False)
    pick_box.add_child(check_action_pick(name='Check if pick', robot_index=robot_index))
    pick_box.add_child(pick(name='Pick Box', robot_index=robot_index))
    pick_box.add_child(send_path(name='Send Path Pick', robot_index=robot_index))

    # Pre-place block behaviour
    pre_place_box = py_trees.composites.Sequence(name='Pre-Place Box', memory=False)
    pre_place_box.add_child(check_action_pre_place(name='Check if pre-place', robot_index=robot_index))
    pre_place_box.add_child(pre_place(name='Pre-place Box', robot_index=robot_index))
    pre_place_box.add_child(send_path(name='Send Path Pre Place', robot_index=robot_index))

    # Place block behaviour
    place_box = py_trees.composites.Sequence(name='Place Box', memory=False)
    place_box.add_child(check_action_place(name='Check if place', robot_index=robot_index))
    place_box.add_child(place(name='Place Box', robot_index=robot_index))
    place_box.add_child(send_path(name='Send Path Place', robot_index=robot_index))

    # Place block behaviour
    abandon_pick = py_trees.composites.Sequence(name='Abandon', memory=False)
    abandon_pick.add_child(check_action_abandon(name='Check if abandon', robot_index=robot_index))
    abandon_pick.add_child(abandon(name='Abandon', robot_index=robot_index))
    abandon_pick.add_child(send_path(name='Send Path Abandon', robot_index=robot_index))

    # Robot actions
    DOTS_actions = py_trees.composites.Selector(name    = f'DOTS Actions {str_index}', memory=False)
    DOTS_actions.add_child(py_trees.decorators.Inverter(look_blocks))
    DOTS_actions.add_child(py_trees.decorators.Inverter(update_global_server))
    DOTS_actions.add_child(py_trees.decorators.Inverter(action))
    DOTS_actions.add_child(py_trees.decorators.Inverter(carry_count))
    DOTS_actions.add_child(random_path)
    DOTS_actions.add_child(pick_box)
    DOTS_actions.add_child(pre_place_box)
    DOTS_actions.add_child(place_box)
    DOTS_actions.add_child(abandon_pick)

    root.add_child(DOTS_actions)

    return root