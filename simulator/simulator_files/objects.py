import numpy as np
import py_trees
import copy
import importlib

from . import bt_setup

class Robot:
    # max_v: max speed, assume robot moves at max speed if healthy
    # camera_sensor: assume camera range is 360deg (may be multiple cameras)
    def __init__(self, radius, max_v, camera_sensor_range, place_tol=None, abandon_tol=None, pre_place_delta=None, lifter_state=1): 
        # Setup behaviour tree and share variables
        self.robot_index = None # String index
        self.root = None
        self.robot_tree = None
        self.blackboard = None

        self.radius = radius
        self.max_v = max_v
        self.camera_sensor_range = camera_sensor_range
        self.place_tol = place_tol
        self.abandon_tol = abandon_tol
        self.lifter_state = lifter_state
        self.pre_place_delta = pre_place_delta
    
    def setup_bb(self, width, height, heading_change_rate, repulsion_o, repulsion_w, task_log, use_hm,hm_mode,  carry_count = []):
        self.blackboard.register_key(key="rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="radius", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="max_v", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="heading_change_rate", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="camera_sensor_range", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="lifter_state", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="arena_size", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="repulsion_w", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="repulsion_o", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="place_tol", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="abandon_tol", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="pre_place_delta", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="local_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="global_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="use_hm", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="carry_count", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="hm_mode", access=py_trees.common.Access.WRITE)

        self.blackboard.rob_c = []
        self.blackboard.boxes = []
        self.blackboard.radius = self.radius
        self.blackboard.max_v = self.max_v
        self.blackboard.heading_change_rate = heading_change_rate
        self.blackboard.camera_sensor_range = self.camera_sensor_range
        self.blackboard.lifter_state = self.lifter_state
        self.blackboard.arena_size = [width, height]
        self.blackboard.repulsion_w = repulsion_w
        self.blackboard.repulsion_o = repulsion_o
        self.blackboard.place_tol = self.place_tol
        self.blackboard.abandon_tol = self.abandon_tol
        self.blackboard.pre_place_delta = self.pre_place_delta
        self.blackboard.local_task_log = copy.deepcopy(task_log) # Use deepcopy so each has a unique task_log object
        self.blackboard.global_task_log = task_log # No deepcopy used here so the same global object will be used for all robots (this is used to track the status of progress of the task)
        self.blackboard.use_hm = use_hm
        self.blackboard.hm_mode = hm_mode
        self.blackboard.carry_count = carry_count
        
    def add_map(self, map):
        self.blackboard.register_key(key="map", access=py_trees.common.Access.WRITE)
        self.blackboard.map = map
        
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
        self.counter = 0
        self.F_heading = None
        self.agent_dist = None
    
    def add_agents(self, agent_obj, number, width, height, bt_controller, task_log=None, use_hm=False, hm_mode='task', carry_count = None, print_bt = False):
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

            # Setup behavuour tree
            bt_module = bt_setup.behaviour_trees[bt_controller] # Get controller from the setup file
            ag.root = bt_module.create_root(robot_index = ag.robot_index)
            if print_bt:
                py_trees.display.render_dot_tree(ag.root) # Uncomment to print png of behaviour tree
            ag.robot_tree = py_trees.trees.BehaviourTree(ag.root)

            # Setup blackboard
            name    = f'Pick Place DOTS: {str_index}'
            namespace = str_index
            ag.blackboard = py_trees.blackboard.Client(name=name, namespace=namespace)
            ag.setup_bb(width, height, self.heading_change_rate, self.repulsion_o, self.repulsion_w, task_log, use_hm, hm_mode, carry_count)
            self.number_of_agents += 1   

    def add_map(self, map):
        for ag in self.agents:
            ag.add_map(map)

    def iterate(self, rob_c, boxes):
        rob_c_new = rob_c
        boxes_new = boxes
        for ag in self.agents:
            # Update behaviour tree robot positions and boxes
            ag.blackboard.rob_c = rob_c_new
            ag.blackboard.boxes = boxes_new
            # Tick behaviour tree
            ag.robot_tree.tick()
            # Update robot and box positions
            rob_c_new = ag.blackboard.rob_c
            boxes_new = ag.blackboard.boxes

        return rob_c_new, boxes_new



















    # TODO: need to replace all the below with behaviour tree implementations
    # TODO: 1) random walk - get rob_d (vector?) from the behaviour tree blackboard with rob_c as an input - called in warehouse.iterate
    # TODO: 2) collision avoidance - to do this, each robot needs to know where every other robot is located - perhaps we can have this stored as part of this swarm class?
    #                              - option 1: input box and robot positions into each behaviour tree somehow
    #                              - option 2: workout these aspects outside of the BT and add callbacks like in ROS



    
    # rob_c: robot center coordinates
    # box_c: box center coordinates
    def random_walk(self, rob_c, box_c, is_box_in_transit, map, heading_bias=False, box_attraction=False):
        self.F_heading = self._generate_heading_force(heading_bias)

        # Compute euclidean (cdist) distance between agents and other agents
        self.agent_dist = cdist(rob_c, rob_c)
        F_box, F_agent = self._generate_interobject_force(box_c, rob_c, is_box_in_transit, box_attraction)

        # Compute distance to wall segments
        self.wall_dist = cdist(rob_c, map.wall_divisions)

        # Force on agent due to proximity to walls calculated elsewhere
        F_wall_avoidance = self._generate_wall_avoidance_force(rob_c, map)

        # Movement vectors summed
        F_agent += F_wall_avoidance + self.F_heading + F_box.T
        F_x = F_agent.T[0] # Repulsion vector in x
        F_y = F_agent.T[1] # in y 
        
        # New movement due to vectors
        t = self.counter%10
        self.computed_heading_prev[t] = self.computed_heading.tolist()

        # new heading due to vectors: this is actually the heading of the repelling force
        self.computed_heading = np.arctan2(F_y, F_x)
        move_x = np.multiply(self.robot_v, np.cos(self.computed_heading)) # Movement in x 
        move_y = np.multiply(self.robot_v, np.sin(self.computed_heading)) # Movement in y 
        
        # Total change in movement of agent (robot deviation)
        rob_d = -np.array([[move_x[n], move_y[n]] for n in range(0, self.number_of_agents)]) # Negative to avoid collisions
        return rob_d

    def update_hook(self):
        # Allow for updates to variables of swarm
        return

    def set_agent_box_state(self, agent_index, state):
        self.agent_has_box[agent_index] = state
        return True

    def dropoff_box(self, warehouse, active_boxes):
        boxes_in_dropoff = warehouse.box_c.T[0] > warehouse.width - warehouse.exit_width - warehouse.radius
        return boxes_in_dropoff*active_boxes
		
    ## Avoidance behaviour for avoiding the warehouse walls ##		
    def _generate_wall_avoidance_force(self, rob_c, map): # input the warehouse map 
        ## distance from agents to walls ##
        # distance from the vertical walls to your agent (horizontal distance between x coordinates)
        difference_in_x = np.array([map.planeh-rob_c[n][1] for n in range(self.number_of_agents)])
        # distance from the horizontal walls to your agent (vertical distance between y coordinates)
        difference_in_y = np.array([map.planev-rob_c[n][0] for n in range(self.number_of_agents)])
        
        # x coordinates of the agent's centre coordinate
        agentsx = rob_c.T[0]
        # y coordinates  
        agentsy = rob_c.T[1]

        ## Are the agents within the limits of the warehouse? 
        x_lower_wall_limit = agentsx[:, np.newaxis] >= map.limh.T[0] # limh is for horizontal walls. x_lower is the bottom of the square
        x_upper_wall_limit = agentsx[:, np.newaxis] <= map.limh.T[1] # x_upper is the top bar of the warehouse square 
        # Interaction combines the lower and upper limit information to give a TRUE or FALSE value to the agents depending on if it is IN/OUT the warehouse boundaries 
        interaction = x_upper_wall_limit*x_lower_wall_limit
            
        # Fy is repulsion vector on the agent in y direction due to proximity to the horziontal walls 
        # This equation was designed to be very high when the agent is close to the wall and close to 0 otherwise
        # repulsion = np.minimum(self.repulsion_w, self.camera_sensor_range_V[0]) # @TODO figure out how wall avoidance works: what is planeh ?
        repulsion = self.repulsion_w
        Fy = np.exp(-2*abs(difference_in_x) + repulsion)
        # The repulsion vector is zero if the interaction is FALSE meaning that the agent is safely within the warehouse boundary
        Fy = Fy*difference_in_x*interaction	

        # Same as x boundaries but now in y
        y_lower_wall_limit = agentsy[:, np.newaxis] >= map.limv.T[0] # limv is vertical walls 
        y_upper_wall_limit = agentsy[:, np.newaxis] <= map.limv.T[1]
        interaction = y_lower_wall_limit*y_upper_wall_limit
        Fx = np.exp(-2*abs(difference_in_y) + repulsion)
        Fx = Fx*difference_in_y*interaction
        
        # For each agent the repulsion in x and y is the sum of the repulsion vectors from each wall
        Fx = np.sum(Fx, axis=1)
        Fy = np.sum(Fy, axis=1)
        # Combine to one vector variable
        F = np.array([[Fx[n], Fy[n]] for n in range(self.number_of_agents)])
        return F

    def _generate_heading_force(self, heading_bias=False):
        if self.counter % self.heading_change_rate == 0:
            # Add noise to the heading 
            noise = 0.01*np.random.randint(-50, 50, (self.number_of_agents))
            self.heading += noise

        # Force for movement according to new chosen heading 
        heading_x = 1*np.cos(self.heading) # move in x 
        heading_y = 1*np.sin(self.heading) # move in y

        if heading_bias:
            carriers = self.agent_has_box == 1 
            heading_x = heading_x + carriers*heading_bias # bias on heading if carrying a box

        return -np.array(list(zip(heading_x, heading_y)))
         
    # Computes repulsion forces: a negative force means comes out as attraction
    def _generate_interobject_force(self, box_c, rob_c, is_box_in_transit, box_attraction=False):
        repulsion = self.repulsion_o#np.minimum(self.repulsion_d, self.camera_sensor_range_V) @TODO allow for collision behaviour
        self.too_close = self.agent_dist < repulsion # TRUE if agent is too close to another agent (enable collision avoidance)
        
        # Compute euclidean (cdist) distance between boxes and agents
        self.box_dist = cdist(box_c, rob_c) # distance between all the boxes and all the agents
        self.too_close_boxes = self.box_dist < repulsion # TRUE if agent is too close to a box (enable collision avoidance). Does not avoid box if agent does not have a box but this is considered later in the code (not_free*F_box)

        proximity_to_robots = rob_c[:, :, np.newaxis] - rob_c.T[np.newaxis, :, :] # Compute vectors between agents
        proximity_to_boxes = box_c[:, :, np.newaxis] - rob_c.T[np.newaxis, :, :] # Computer vectors between agents and boxes 
        
        F_box = self.too_close_boxes[:,np.newaxis,:]*proximity_to_boxes # Find which box vectors exhibit forces on the agents due to proximity 
        F_box = np.sum(F_box, axis=0) # Sum the vectors due to boxes on the agents
        
        not_free = self.agent_has_box == 1
        F_box_occupied = [not_free,not_free]*F_box 	# Only be repelled by boxes if you already have a box 
        
        if box_attraction:
            # Compute box attraction between free robots and free boxes
            too_close_free_boxes = (self.box_dist < self.camera_sensor_range_V) & np.transpose(np.tile((is_box_in_transit == 0), (self.number_of_agents, 1))) # attraction force from free boxes
            F_box_free = too_close_free_boxes[:,np.newaxis,:]*proximity_to_boxes
            F_box_free = np.sum(F_box_free, axis=0)
            free_agents = (self.agent_has_box == 1)
            noise = 0.0005*np.random.randint(0, 200, (2, self.number_of_agents))
            noise = np.add(np.ones((2, self.number_of_agents)), noise)
            F_box_free = [free_agents,free_agents]*F_box_free*noise
            F_box_total = F_box_occupied - F_box_free
        else:
            F_box_total = F_box_occupied
            
        F_agent = self.too_close[:,np.newaxis,:]*proximity_to_robots # Calc repulsion vector on agent due to proximity to other agents
        F_agent = np.sum(F_agent, axis =0).T # Sum the repulsion vectors
        
        return (F_box_total, F_agent)

    # Check if a collision has occurred and in those cases, generate a rebound force
    # Call before random walk (which generates random movement)
    def check_collision(self, warehouse, respect_physics=False):
        try:
            no_agents = self.number_of_agents
            no_boxes = warehouse.number_of_boxes
            box_ag_dist = self.box_dist # shape (box_no, agent_no)
            ag_ag_dist = self.agent_dist
            ag_r = self.robot_r
            box_r = warehouse.radius

            # if interobject distance is < obj1_r+obj2_r, then we have a collision
            tile_box_ag = np.tile(ag_r, (no_boxes, 1)) + np.transpose(np.tile(box_r, (no_agents, no_boxes)))
            tile_ag_ag = np.tile(ag_r, (no_agents, 1))*2
            
            col_box_ag = box_ag_dist < tile_box_ag
            col_ag_ag = ag_ag_dist < tile_ag_ag
            
            box_collisions = np.sum(col_box_ag, axis=1)
            agent_collisions = np.sum(col_ag_ag, axis=1) - 1
            
            if not respect_physics:
                return box_collisions, agent_collisions, warehouse.rob_d
            
            # remember to take into account picking up boxes
            # warehouse.rob_d
            # warehouse.rob_c
            # warehouse.box_c
            # cdist(box_c, rob_c)
            
            # print(ag_ag_dist)
            # print(self.counter)
            # time.sleep(600)
            # exit()
            # print('ag', b.shape)
        except Exception as e:
            print(e)

        # returns new rob_d
        # rebound force and heading - will depend on how many collision objects and their forces and headings!