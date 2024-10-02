import numpy as np
import copy
from scipy.spatial.distance import cdist, pdist, euclidean
from .objects import Box, DeliveryPoint

class Warehouse:

	RANDOM_OBJ_POS = 0
	AVOID_DROP_ZONE = 1		# Boxes only dropped outside of (>) predefined drop zone limit

	def __init__(self, width, height, boxes, box_radius, swarm, exit_width, wallsh, wallsv, depot=False, drop_zone_limit = None, init_object_positions=RANDOM_OBJ_POS, check_collisions=False):

		self.width = width
		self.height = height
		self.init_boxes = boxes
		self.box_range = box_radius*2.0	#box_range # range at which a box can be picked up 
		self.radius = box_radius # physical radius of the box (approximated to a circle even though square in animation)

		if exit_width is None:
			self.exit_width = int(0.05*self.width) # if it is too small then it will avoid the wall and be less likely to reach the exit zone 
		else:
			self.exit_width = exit_width

		self.check_collisions = check_collisions
		self.delivered = 0 # Number of boxes that have been delivered
		self.counter = 0 # time starts at 0s or time step = 0 

		# Initialise boxes
		self.boxes = [] # centre coordinates of boxes starts as an empty list
		for colour in boxes:
			for i in range(boxes[colour]):
				  self.boxes.append(Box(colour=colour))
		self.number_of_boxes = len(self.boxes)

		# Intiate depot
		print(type(depot))
		if depot == True:
			self.depot = True
		else:
			self.depot = False

		# Initialise map and swarm objects
		self.map = Map(width, height, wallsh, wallsv)
		self.swarm = swarm
		swarm.add_map(self.map)

		# Initiate robots
		self.rob_c = []
		self.drop_zone_limit = drop_zone_limit
		self.generate_object_positions(int(init_object_positions))
		
		self.rob_c = np.array(self.rob_c) # convert list to array

	def generate_object_positions(self, conf):
		if conf == self.RANDOM_OBJ_POS:

			# Calculate a list of possible x-y coordinates that objects can be placed into, this prevents objects being spawned over one another
			possible_x = int((self.width)/(self.radius*2)) # number of positions possible on the x axis
			possible_y = int((self.height)/(self.radius*2)) # number of positions possible on the y axis
			list_n = [] # empty list of possible positions 
			h = 0 # initiate all headings as 0 - random heading will be calculated in the behaviour tree

			print(possible_x, possible_y)

			for x in range(possible_x):
				for y in range(possible_y):
					list_n.append([x,y,h]) # list of possible positions in the warehouse
			
			N = self.number_of_boxes + self.swarm.number_of_agents # total number of units to assign positions to
			XY_idx = np.random.choice(len(list_n),N,replace=False) # select N unique coordinates at random from the list of possible positions
			XY = np.array(list_n)[XY_idx]
			
			c_select = [] # central coordinates (empty list) 
			for j in range(N): #for the total number of units 
				c_select.append([self.radius + ((self.radius*2))*XY[j][0], self.radius + ((self.radius*2))*XY[j][1], 0]) # assign a central coordinate to unit j (can be a box or an agent) based on the unique randomly selected list, XY

			for r in range(self.swarm.number_of_agents):
				self.rob_c.append(c_select[r]) # assign initial robot positions

			if self.depot:
				for b in range(self.number_of_boxes):
					index = list(self.init_boxes.keys()).index(self.boxes[b].colour) + 1
					if index % 2 == 0:
						self.boxes[b].x = self.width / 2 + (index * self.radius)
					else:
						self.boxes[b].x = self.width / 2 - (index * self.radius)
					self.boxes[b].y = self.height - 2* self.radius
			else:
				for b in range(self.number_of_boxes):
					self.boxes[b].x = c_select[b + self.swarm.number_of_agents][0]
					self.boxes[b].y = c_select[b + self.swarm.number_of_agents][1]

		elif conf == self.AVOID_DROP_ZONE:

			# Calculate a list of possible x-y coordinates that objects can be placed into, this prevents objects being spawned over one another
			possible_x = int((self.width) / (self.radius * 2))  # number of positions possible on the x axis
			possible_y = int((self.height-self.drop_zone_limit) / (self.radius * 2))  # number of positions possible on the y axis
			list_n = []  # empty list of possible positions
			h = 0  # initiate all headings as 0 - random heading will be calculated in the behaviour tree

			for x in range(possible_x):
				for y in range(possible_y):
					list_n.append([x, y, h])  # list of possible positions in the warehouse

			N = self.number_of_boxes + self.swarm.number_of_agents  # total number of units to assign positions to
			XY_idx = np.random.choice(len(list_n), N,
									  replace=False)  # select N unique coordinates at random from the list of possible positions
			XY = np.array(list_n)[XY_idx]

			c_select = []  # central coordinates (empty list)
			for j in range(N):  # for the total number of units
				c_select.append(
					[self.radius + ((self.radius * 2)) * XY[j][0], self.drop_zone_limit + self.radius + ((self.radius * 2)) * XY[j][1],
					 0])  # assign a central coordinate to unit j (can be a box or an agent) based on the unique randomly selected list, XY

			for r in range(self.swarm.number_of_agents):
				self.rob_c.append(c_select[r])  # assign initial robot positions

			if self.depot:
				for b in range(self.number_of_boxes):
					index = list(self.init_boxes.keys()).index(self.boxes[b].colour) + 1
					if index % 2 == 0:
						self.boxes[b].x = self.width / 2 + (index * self.radius)
					else:
						self.boxes[b].x = self.width / 2 - (index * self.radius)
					self.boxes[b].y = self.height - 2 * self.radius
			else:
				for b in range(self.number_of_boxes):
					self.boxes[b].x = c_select[b + self.swarm.number_of_agents][0]
					self.boxes[b].y = c_select[b + self.swarm.number_of_agents][1]


		else:
			raise Exception("Object position not valid")

	def iterate(self, heading_bias=False, box_attraction=False): # moves the robot and box positions forward in one time step
		self.rob_c, self.boxes = self.swarm.iterate(self.rob_c, self.boxes) # the robots move using the random walk function which generates a new deviation (rob_d)

		self.counter += 1
		self.swarm.counter = self.counter

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

	def __init__(self, width, height, wallsh, wallsv, wall_divisions=10):
		self.width = width
		self.height = height
		self._wallsh = wallsh
		self._wallsv = wallsv
		self.walls = np.array([]) # a list of all walls
		self.wallsh = np.array([]) # a list of only horizontal walls
		self.wallsv = np.array([]) # a list of only vertical walls
		self.planeh = np.array([]) # a list of horizontal avoidance planes formed by walls
		self.planev = np.array([]) # a list of horizontal vertical planes formed by walls

		self.generate()
		self.generate_wall_divisions(wall_divisions)
	
	def generate(self):		
		# Updated map generation:
		# For more complex map builds, the map can be described wall by wall (note it is possible to automate the wall generation from coordinates but this is not done here)
		
		# Horizontal walls
		for wall in range(len(self._wallsh)):
			self.wallsh = np.append(self.wallsh, WallBounds())
			self.wallsh[wall].start = self._wallsh[wall][0]; self.wallsh[wall].end = self._wallsh[wall][1]

		# Vertical walls
		for wall in range(len(self._wallsv)):
			self.wallsv = np.append(self.wallsv, WallBounds())
			self.wallsv[wall].start = self._wallsv[wall][0]; self.wallsv[wall].end = self._wallsv[wall][1]

		# All Walls
		self.walls = np.append(self.walls, self.wallsh)
		self.walls = np.append(self.walls, self.wallsv)
	
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