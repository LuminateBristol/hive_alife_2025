# TODO - update the generate positions - split out to boxes and robots seperately and add config for biases of where to put each

import numpy as np
import copy
from scipy.spatial.distance import cdist, pdist, euclidean
from .objects import Box

class Warehouse:

	RANDOM_OBJ_POS = 0
	OBJ_POS_1 = 1		# Boxes aligned along left wall, agents aligned along right 
	OBJ_POS_2 = 2	    # Boxes and robots spread out only in the outer 2/3rds of the arena - central area is empty
	OBJ_POS_TEST = 3

	def __init__(self, width, height, boxes, box_radius, swarm, exit_width,
		init_object_positions=RANDOM_OBJ_POS,
		check_collisions=False):

		self.width = width
		self.height = height
		self.box_range = box_radius*2.0#box_range # range at which a box can be picked up 
		self.radius = box_radius # physical radius of the box (approximated to a circle even though square in animation)
		if exit_width is None:
			self.exit_width = int(0.05*self.width) # if it is too small then it will avoid the wall and be less likely to reach the exit zone 
		else:
			self.exit_width = exit_width

		self.check_collisions = check_collisions
		self.delivered = 0 # Number of boxes that have been delivered 
		self.counter = 0 # time starts at 0s or time step = 0 

		# Initialise the positions of boxes and robots
		self.boxes = [] # centre coordinates of boxes starts as an empty list
		for colour in boxes:
			for i in range(boxes[colour]):
				  self.boxes.append(Box(colour=colour))
		
		self.number_of_boxes = len(self.boxes)
		self.map = Map(width, height)
		self.swarm = swarm
		swarm.add_map(self.map)

		self.rob_c = [] 
		self.generate_object_positions(int(init_object_positions))
		
		self.rob_c = np.array(self.rob_c) # convert list to array 

	def generate_object_positions(self, conf):
		if conf == self.RANDOM_OBJ_POS:
			possible_x = int((self.width)/(self.radius*2)) # number of positions possible on the x axis
			possible_y = int((self.height)/(self.radius*2)) # number of positions possible on the y axis
			list_n = [] # empty list of possible positions 
			h = 0 # initiate all headings as 0 - random heading will be calculated in the behaviour tree
			for x in range(possible_x):
				for y in range(possible_y):
					list_n.append([x,y,h]) # list of possible positions in the warehouse
			
			N = self.number_of_boxes + self.swarm.number_of_agents # total number of units to assign positions to
			XY_idx = np.random.choice(len(list_n),N,replace=False) # select N unique coordinates at random from the list of possible positions
			XY = np.array(list_n)[XY_idx]
			
			c_select = [] # central coordinates (empty list) 
			for j in range(N): #for the total number of units 
				c_select.append([self.radius + ((self.radius*2))*XY[j][0], self.radius + ((self.radius*2))*XY[j][1], 0]) # assign a central coordinate to unit j (can be a box or an agent) based on the unique randomly selected list, XY

			for b in range(self.number_of_boxes):
				self.boxes[b].x = c_select[b][0]
				self.boxes[b].y = c_select[b][1]
			for r in range(self.swarm.number_of_agents):
				self.rob_c.append(c_select[r+self.number_of_boxes]) # assign initial robot positions

		elif conf == self.OBJ_POS_1:
			
			possible_x_half = int((self.width/2)/(self.radius*2)) # number of positions possible on the x axis in one side of warehosue
			possible_y = int((self.height)/(self.radius*2)) # number of positions possible on the y axis
			list_n_box = [] # empty list of possible positions 
			list_n_agent = [] # empty list of possible positions 
			for x in range(possible_x_half):
				for y in range(possible_y):
					list_n_box.append([x,y,h])
			for x in range(possible_x_half, 2*possible_x_half):
				for y in range(possible_y):
					list_n_agent.append([x,y])

			XY_idx_box = np.random.choice(len(list_n_box),self.number_of_boxes,replace=False) # select N unique coordinates at random from the list of possible positions
			XY_box = np.array(list_n_box)[XY_idx_box]
			XY_idx_agent = np.random.choice(len(list_n_agent),self.swarm.number_of_agents,replace=False) # select N unique coordinates at random from the list of possible positions
			XY_agent = np.array(list_n_agent)[XY_idx_agent]

			c_select = [] # central coordinates (empty list) 
			for j in range(self.number_of_boxes): #for the total number of units: x, y
				c_select.append([self.radius + ((self.radius*2))*XY_box[j][0], self.radius + ((self.radius*2))*XY_box[j][1], 0])
			for j in range(self.swarm.number_of_agents): #for the total number of units: x, y, heading
				c_select.append([self.radius + ((self.radius*2))*XY_agent[j][0], self.radius + ((self.radius*2))*XY_agent[j][1], 0])

			for b in range(self.number_of_boxes):
				self.boxes[b].x = c_select[b][0]
				self.boxes[b].y = c_select[b][1]
			for r in range(self.swarm.number_of_agents):
				self.rob_c.append(c_select[r+self.number_of_boxes]) # assign initial robot positions
		
		elif conf == self.OBJ_POS_2:
			divider_obj = self.radius*3
			divider_wh = 1
			possible_x = int((self.width)/(divider_obj)) # number of positions possible on the x axis
			possible_y = int((self.height)/(divider_obj)) # number of positions possible on the y axis
			list_n = [] # empty list of possible positions 
			h = 0 # initiate all headings as 0 - random heading will be calculated in the behaviour tree
			
			for x in range(possible_x):
				for y in range(possible_y):		
					if x <= divider_wh * possible_x or x >= (1-divider_wh) * possible_x or y <= divider_wh * possible_y or y >= (1-divider_wh)* possible_y:
							list_n.append([x, y, h])	# list of possible positions in the warehouse
			
			N = self.number_of_boxes + self.swarm.number_of_agents # total number of units to assign positions to
			XY_idx = np.random.choice(len(list_n),N,replace=False) # select N unique coordinates at random from the list of possible positions
			XY = np.array(list_n)[XY_idx]
			
			c_select = [] # central coordinates (empty list) 
			for j in range(N): #for the total number of units 
				c_select.append([self.radius + ((divider_obj))*XY[j][0], self.radius + ((divider_obj))*XY[j][1], 0]) # assign a central coordinate to unit j (can be a box or an agent) based on the unique randomly selected list, XY

			for b in range(self.number_of_boxes):
				self.boxes[b].x = c_select[b][0]
				self.boxes[b].y = c_select[b][1]
			for r in range(self.swarm.number_of_agents):
				self.rob_c.append(c_select[r+self.number_of_boxes]) # assign initial robot positions

		else:
			raise Exception("Object position not valid")

	def iterate(self, heading_bias=False, box_attraction=False): # moves the robot and box positions forward in one time step
		self.rob_c, self.boxes = self.swarm.iterate(self.rob_c, self.boxes) # the robots move using the random walk function which generates a new deviation (rob_d)

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
		[self.obstacles.append(map_bounds.walls[x]) for x in range(0, len(map_bounds.walls))] # Create obstacles objects as a list of WallBounds objects for each wall

		# Setup generalised parameters for the vertical and horizontal walls
		# We do not know which is which in the obstacles object so large general arrays are used
		self.walls = np.zeros((2*len(self.obstacles), 2))
		self.wallh = np.zeros((2*len(self.obstacles), 2))
		self.wallv = np.zeros((2*len(self.obstacles), 2))
		self.planeh = np.zeros(len(self.obstacles))
		self.planev = np.zeros(len(self.obstacles))
		self.limx = np.zeros((len(self.obstacles), 2))
		self.limy = np.zeros((len(self.obstacles), 2))

		for n in range(0, len(self.obstacles)):
			# if wall is vertical
			if self.obstacles[n].start[0] == self.obstacles[n].end[0]:
				self.wallv[2*n] = np.array([self.obstacles[n].start[0], self.obstacles[n].start[1]])
				self.wallv[2*n+1] = np.array([self.obstacles[n].end[0], self.obstacles[n].end[1]])

				# planev: plane of vertical wall defined by two extents [x1, x2]
				self.planev[n] = self.wallv[2*n][0]
				# limy: limits of the horizontal wall i.e. y limits over which the robot is beyond the length of the wall
				self.limy[n] = np.array([np.min([self.obstacles[n].start[1], self.obstacles[n].end[1]])-0.5, np.max([self.obstacles[n].start[1], self.obstacles[n].end[1]])+0.5])

			# if wall is horizontal
			if self.obstacles[n].start[1] == self.obstacles[n].end[1]:
				self.wallh[2*n] = np.array([self.obstacles[n].start[0], self.obstacles[n].start[1]])
				self.wallh[2*n+1] = np.array([self.obstacles[n].end[0], self.obstacles[n].end[1]])

				# planeh: plane of horizontal wall defined by two extents [y1, y2]
				self.planeh[n] = self.wallh[2*n][1]
				# limx: limits of the horizontal wall i.e. x limits over which the robot is beyond the length of the wall
				self.limx[n] = np.array([np.min([self.obstacles[n].start[0], self.obstacles[n].end[0]])-0.5, np.max([self.obstacles[n].start[0], self.obstacles[n].end[0]])+0.5])

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