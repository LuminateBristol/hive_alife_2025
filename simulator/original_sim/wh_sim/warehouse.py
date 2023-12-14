import numpy as np
import random
from scipy.spatial.distance import cdist, pdist, euclidean

class Warehouse:

	RANDOM_OBJ_POS = 0
	OBJ_POS_1 = 1		# Boxes aligned along left wall, agents aligned along right 
	OBJ_POS_TEST = 2

	def __init__(self, width, height, number_of_boxes, box_radius, swarm, exit_width,
		init_object_positions=RANDOM_OBJ_POS,
		check_collisions=False):

		self.width = width
		self.height = height
		self.box_range = box_radius*2.0#box_range # range at which a box can be picked up 
		self.number_of_boxes = number_of_boxes
		self.radius = box_radius # physical radius of the box (approximated to a circle even though square in animation)
		if exit_width is None:
			self.exit_width = int(0.05*self.width) # if it is too small then it will avoid the wall and be less likely to reach the exit zone 
		else:
			self.exit_width = exit_width

		self.check_collisions = check_collisions
		self.box_is_free = np.ones(self.number_of_boxes) # Box states set to 1 = Free (not on a robots), if = 0 = Not free (on a robot)
		self.delivered = 0 # Number of boxes that have been delivered 
		self.counter = 0 # time starts at 0s or time step = 0 

		# Initialise the positions of boxes and robots
		self.box_c = [] # centre coordinates of boxes starts as an empty list 
		self.swarm = swarm
		self.map = Map(width, height)

		self.rob_c = [] # robot centre coordinates
		self.rob_c_prev = {}
		self.rob_d = np.zeros((self.swarm.number_of_agents, 2)) # robot centre cooridinate deviation (how much the robot moves in one time step)
		self.rob_delivery_count = np.zeros(self.swarm.number_of_agents, dtype='int')
		self.generate_object_positions(int(init_object_positions))
		
		self.box_c = np.array(self.box_c) # convert list to array
		self.rob_c = np.array(self.rob_c) # convert list to array 
		self.box_d = np.zeros((self.number_of_boxes, 2)) # box centre coordinate deviation (how far the box moves in one time step)
		self.to_be_delivered = np.ones(self.number_of_boxes) # is the box still awaiting delivery or is it "gone" from warehouse
		self.robot_carrier = np.full((self.number_of_boxes), -1) # Value at index = box number is the robot number that is currently moving that box
		self.boxes_in_dropoff = np.zeros(self.number_of_boxes) # Boolean list of boxes that are over the delivery boundary (1 is in the delivery area)

	def generate_object_positions(self, conf):
		if conf == self.RANDOM_OBJ_POS:
			possible_x = int((self.width)/(self.radius*2)) # number of positions possible on the x axis
			possible_y = int((self.height)/(self.radius*2)) # number of positions possible on the y axis
			list_n = [] # empty list of possible positions 
			for x in range(possible_x):
				for y in range(possible_y):
					list_n.append([x,y]) # list of possible positions in the warehouse
			
			N = self.number_of_boxes + self.swarm.number_of_agents # total number of units to assign positions to
			XY_idx = np.random.choice(len(list_n),N,replace=False) # select N unique coordinates at random from the list of possible positions
			XY = np.array(list_n)[XY_idx]
			
			c_select = [] # central coordinates (empty list) 
			for j in range(N): #for the total number of units 
				c_select.append([self.radius + ((self.radius*2))*XY[j][0], self.radius + ((self.radius*2))*XY[j][1]]) # assign a central coordinate to unit j (can be a box or an agent) based on the unique randomly selected list, XY

			for b in range(self.number_of_boxes):
				self.box_c.append(c_select[b]) # assign initial box positions
			for r in range(self.swarm.number_of_agents):
				self.rob_c.append(c_select[r+self.number_of_boxes]) # assign initial robot positions

		elif conf == self.OBJ_POS_1:
			possible_x_half = int((self.width/2)/(self.radius*2)) # number of positions possible on the x axis in one side of warehosue
			possible_y = int((self.height)/(self.radius*2)) # number of positions possible on the y axis
			list_n_box = [] # empty list of possible positions 
			list_n_agent = [] # empty list of possible positions 
			for x in range(possible_x_half):
				for y in range(possible_y):
					list_n_box.append([x,y])
			for x in range(possible_x_half, 2*possible_x_half):
				for y in range(possible_y):
					list_n_agent.append([x,y])

			XY_idx_box = np.random.choice(len(list_n_box),self.number_of_boxes,replace=False) # select N unique coordinates at random from the list of possible positions
			XY_box = np.array(list_n_box)[XY_idx_box]
			XY_idx_agent = np.random.choice(len(list_n_agent),self.swarm.number_of_agents,replace=False) # select N unique coordinates at random from the list of possible positions
			XY_agent = np.array(list_n_agent)[XY_idx_agent]
			
			c_select = [] # central coordinates (empty list) 
			for j in range(self.number_of_boxes): #for the total number of units 
				c_select.append([self.radius + ((self.radius*2))*XY_box[j][0], self.radius + ((self.radius*2))*XY_box[j][1]])
			for j in range(self.swarm.number_of_agents): #for the total number of units 
				c_select.append([self.radius + ((self.radius*2))*XY_agent[j][0], self.radius + ((self.radius*2))*XY_agent[j][1]])

			for b in range(self.number_of_boxes):
				self.box_c.append(c_select[b]) # assign initial box positions
			for r in range(self.swarm.number_of_agents):
				self.rob_c.append(c_select[r+self.number_of_boxes]) # assign initial robot positions

		else:
			raise Exception("Object position not valid")

	def iterate(self, heading_bias=False, box_attraction=False): # moves the robot and box positions forward in one time step
		dist_rob_to_box = cdist(self.box_c, self.rob_c) # calculates the euclidean distance from every robot to every box (centres)
		is_closest_rob_in_range = np.min(dist_rob_to_box, 1) < self.box_range # if the minimum distance box-robot is less than the pick up sensory range, then qu_close_box = 1
		closest_rob_id = np.argmin(dist_rob_to_box, 1)	
		boxes_to_pickup = self.is_box_free()*is_closest_rob_in_range
		to_pickup = np.argwhere(boxes_to_pickup==1)

		# needs to be a loop (rather than vectorised) in case two robots are close to the same box
		for box_id in to_pickup:
			closest_r = closest_rob_id[box_id][0]
			is_robot_carrying_box = self.is_robot_carrying_box(closest_r)
			# Check if robot is already carrying a box: if not, then set robot to "lift" the box (may fail if faulty)
			if is_robot_carrying_box == 0 and self.swarm.set_agent_box_state(closest_r, 1):
				self.box_is_free[box_id] = 0 # change box state to 0 (not free, on a robot)
				self.box_c[box_id] = self.rob_c[closest_r] # change the box centre so it is aligned with its robot carrier's centre
				self.robot_carrier[box_id] = closest_r # set the robot_carrier for box b to that robot ID

		self.rob_d = self.swarm.iterate(
			self.rob_c, 
			self.box_c, 
			self.box_is_free, 
			self.map, 
			heading_bias,
			box_attraction) # the robots move using the random walk function which generates a new deviation (rob_d)
		
		if self.check_collisions:
			b_collisions, r_collisions, self.rob_d = self.swarm.check_collision(self)

		t = self.counter%10
		self.rob_c_prev[t] = self.rob_c # Save a record of centre coordinates before update
		self.rob_c = self.rob_c + self.rob_d # robots centres change as they move
		active_boxes = self.box_is_free == 0 # boxes which are on a robot
		self.box_d = np.array((active_boxes,active_boxes)).T*self.rob_d[self.robot_carrier] # move the boxes by the amount equal to the robot carrying them 
		self.box_c = self.box_c + self.box_d
		self.boxes_in_dropoff = self.swarm.dropoff_box(self, active_boxes) # TODO does not include boxes which are inactive (already spawned in the drop off zone)
		if any(self.boxes_in_dropoff) == 1: # if any boxes have been delivered
			self.delivered = self.delivered + np.sum(self.boxes_in_dropoff) # add to the number of deliveries made
			box_n = np.argwhere(self.boxes_in_dropoff == 1) # box IDs that have been delivered
			rob_n = self.robot_carrier[box_n] # robot IDs that have delivered a box just now
			self.rob_delivery_count[rob_n] += 1
			self.box_is_free[box_n] = 1 # mark boxes as free again
			self.swarm.agent_has_box[rob_n] = 0 # mark robots as free again
			self.box_c.T[0,box_n] = self.box_c.T[0,box_n] - self.width-600 # remove boxes from the warehouse
		
		self.counter += 1
		self.swarm.counter = self.counter

	# wrapper functions to help with managing variables
	def is_box_free(self, bid=None):
		if bid is None:
			return self.box_is_free
		
		return self.box_is_free[bid]

	def is_robot_carrying_box(self, rid=None):
		if rid is None:
			return self.swarm.agent_has_box
		
		return self.swarm.agent_has_box[rid]

class WallBounds:

    def __init__(self):
        self.start = np.array([0,0])
        self.end = np.array([0,0])
        self.width = 1
        self.hitbox = []

class BoxBounds:
	'''
	Class which contains definitions for building a bounding box.
	'''
	def __init__(self, h, w, mid_point):
		self.height = h
		self.width = w
		self.walls = []

		self.walls.append(WallBounds())
		self.walls[0].start = [mid_point[0]-(0.5*w), mid_point[1]+(0.5*h)]; self.walls[0].end = [mid_point[0]+(0.5*w), mid_point[1]+(0.5*h)]
		self.walls.append(WallBounds())
		self.walls[1].start = [mid_point[0]-(0.5*w), mid_point[1]-(0.5*h)]; self.walls[1].end = [mid_point[0]+(0.5*w), mid_point[1]-(0.5*h)]
		self.walls.append(WallBounds())
		self.walls[2].start = [mid_point[0]-(0.5*w), mid_point[1]+(0.5*h)]; self.walls[2].end = [mid_point[0]-(0.5*w), mid_point[1]-(0.5*h)]
		self.walls.append(WallBounds())
		self.walls[3].start = [mid_point[0]+(0.5*w), mid_point[1]+(0.5*h)]; self.walls[3].end = [mid_point[0]+(0.5*w), mid_point[1]-(0.5*h)]

class Map:

	def __init__(self, width, height, wall_divisions=10):
		self.width = width
		self.height = height
		self.obstacles = [] # contains a list of all walls that make up an environment
		self.walls = np.array([]) # same as obsticales variable but as a numpy array
		self.wallh = np.array([]) # a list of only horizontal walls
		self.wallv = np.array([]) # a list of only vertical walls
		self.planeh = np.array([]) # a list of horizontal avoidance planes formed by walls
		self.planev = np.array([]) # a list of horizontal vertical planes formed by walls

		self.generate()
		self.generate_wall_divisions(wall_divisions)
	
	# @TODO could wall generation be refactored?
	def generate(self):		
		map_bounds = BoxBounds(self.height, self.width, [self.width/2, self.height/2]); 
		[self.obstacles.append(map_bounds.walls[x]) for x in range(0, len(map_bounds.walls))]
			
		self.walls = np.zeros((2*len(self.obstacles), 2))
		self.wallh = np.zeros((2*len(self.obstacles), 2))
		self.wallv = np.zeros((2*len(self.obstacles), 2))
		self.planeh = np.zeros(len(self.obstacles))
		self.planev = np.zeros(len(self.obstacles))
		self.limh = np.zeros((len(self.obstacles), 2))
		self.limv = np.zeros((len(self.obstacles), 2))

		for n in range(0, len(self.obstacles)):
			if self.obstacles[n].start[0] == self.obstacles[n].end[0]:
				self.wallv[2*n] = np.array([self.obstacles[n].start[0], self.obstacles[n].start[1]])
				self.wallv[2*n+1] = np.array([self.obstacles[n].end[0], self.obstacles[n].end[1]])

				self.planev[n] = self.wallv[2*n][0]
				self.limv[n] = np.array([np.min([self.obstacles[n].start[1], self.obstacles[n].end[1]])-0.5, np.max([self.obstacles[n].start[1], self.obstacles[n].end[1]])+0.5])

			# if wall is horizontal
			if self.obstacles[n].start[1] == self.obstacles[n].end[1]:
				self.wallh[2*n] = np.array([self.obstacles[n].start[0], self.obstacles[n].start[1]])
				self.wallh[2*n+1] = np.array([self.obstacles[n].end[0], self.obstacles[n].end[1]])

				self.planeh[n] = self.wallh[2*n][1]
				self.limh[n] = np.array([np.min([self.obstacles[n].start[0], self.obstacles[n].end[0]])-0.5, np.max([self.obstacles[n].start[0], self.obstacles[n].end[0]])+0.5])

			self.walls[2*n] = np.array([self.obstacles[n].start[0], self.obstacles[n].start[1]])
			self.walls[2*n+1] = np.array([self.obstacles[n].end[0], self.obstacles[n].end[1]])
	
	def generate_wall_divisions(self, divisions=10):
		wall_divisions = np.array([])
		
		# Generate vertical walls
		division_size = self.height/divisions
		d = np.arange(0, self.height, division_size)
		d += division_size/2
		d_ = np.tile(d, 2)
		x = np.concatenate([np.zeros(len(d)), np.ones(len(d))*self.width])
		wall_divisions = np.stack((x, d_), axis=-1)
		
		# Generate horizontal walls
		division_size = self.width/divisions
		d = np.arange(0, self.width, division_size)
		d += division_size/2
		d_ = np.tile(d, 2)
		y = np.concatenate([np.zeros(len(d)), np.ones(len(d))*self.height])
		wall_divisions = np.concatenate([wall_divisions, np.stack((d_, y), axis=-1)])
		self.wall_divisions = wall_divisions