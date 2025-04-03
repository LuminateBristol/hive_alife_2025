import numpy as np
import math
import random
from .objects import Box, DeliveryPoint

class Warehouse:
	"""
    Represents a warehouse environment where robots and objects are positioned.
    Handles the initialization of objects, robots, and the pheromone mapping.
    """

	def __init__(self, gen_cfg, exp_cfg, map_cfg, swarm, hive_mind = None):
		"""
        Initializes the warehouse with dimensions, walls, boxes, and other relevant settings.

        Args:
            config (Config): Configuration object with warehouse parameters.
            swarm (class object): The swarm operating within the warehouse - initated in sim.py - see objects.py for definition
            hive_mind (class object): Hive knowledge graph 0 initiated in sim.py - see hive_mind.py for definition
        """
		self.gen_cfg = gen_cfg
		self.exp_cfg = exp_cfg
		self.width = gen_cfg.get('warehouse', 'width')
		self.height = gen_cfg.get('warehouse' , 'height')
		self.wallsh = map_cfg.get(exp_cfg.get('map'), 'wallsh')
		self.wallsv = map_cfg.get(exp_cfg.get('map'), 'wallsv')
		self.init_boxes = exp_cfg.get('warehouse', 'boxes')
		self.box_range = gen_cfg.get('warehouse', 'box_radius')*2.0	#box_range # range at which a box can be picked up
		self.radius = gen_cfg.get('warehouse', 'box_radius') # physical radius of the box (approximated to a circle even though square in animation)
		self.pheromone_map = {}
		self.hive_mind = hive_mind

		self.delivered = 0 # Number of boxes that have been delivered
		self.counter = 0 # time starts at 0s or time step = 0 

		# LOGISTICS: Initialise boxes
		self.boxes = [] # centre coordinates of boxes starts as an empty list
		for colour in exp_cfg.get('warehouse', 'boxes'):
			for i in range(exp_cfg.get('warehouse', 'boxes')[colour]):
				self.boxes.append(Box(colour=colour))
		self.number_of_boxes = len(self.boxes)

		# LOGISTICS: Initiate depot
		if exp_cfg.get('warehouse', 'depot') == True:
			self.depot = True
		else:
			self.depot = False

		# Initialise map and hive_mind objects
		self.map = Map(self.width, self.height, self.wallsh, self.wallsv)
		self.swarm = swarm
		swarm.add_map(self.map)
		if self.hive_mind is not None:
			swarm.add_hive_mind(self.hive_mind)

		# Initiate robots
		self.rob_c = []
		self.drop_zone_limit = exp_cfg.get('warehouse', 'drop_zone_limit')
		self.generate_object_positions(int(exp_cfg.get('warehouse', 'object_position')))
		
		self.rob_c = np.array(self.rob_c) # convert list to array

	def generate_object_positions(self, object_position):
		"""
        Generates initial positions for objects and robots within the warehouse based on the chosen configuration.

        Args:
            object_position (int): Configuration mode defining initial object placement rules - defined in exp_setup.yaml
        """

		# Set initial position integers
		self.RANDOM_OBJ_POS = 0
		self.AVOID_DROP_ZONE = 1  # Boxes only dropped outside of (>) predefined drop zone limit


		if object_position == self.RANDOM_OBJ_POS:

			# Calculate a list of possible x-y coordinates that objects can be placed into, this prevents objects being spawned over one another
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

		elif object_position == self.AVOID_DROP_ZONE:

			# Calculate a list of possible x-y coordinates that objects can be placed into, this prevents objects being spawned over one another
			possible_x = int((self.width) / (self.radius * 4))  # number of positions possible on the x axis
			possible_y = int((self.height-self.drop_zone_limit) / (self.radius * 4))  # number of positions possible on the y axis
			list_n = []  # empty list of possible positions

			for x in range(possible_x):
				for y in range(possible_y):
					h = 0
					list_n.append([x, y, h])  # list of possible positions in the warehouse

			N = self.number_of_boxes + self.swarm.number_of_agents  # total number of units to assign positions to
			XY_idx = np.random.choice(len(list_n), N, replace=False)  # select N unique coordinates at random from the list of possible positions
			XY = np.array(list_n)[XY_idx]

			c_select = []  # central coordinates (empty list)
			for j in range(N):  # for the total number of units
				c_select.append(
					[self.radius + ((self.radius * 4)) * XY[j][0], self.drop_zone_limit + self.radius + ((self.radius * 4)) * XY[j][1],
					 random.random() * 2 * np.pi])  # assign a central coordinate to unit j (can be a box or an agent) based on the unique randomly selected list, XY

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

	def update_pheromone_map(self):
		"""
        Updates the pheromone map based on the robots' positions at each time step.
        This is based on a hash grid with a set resolution (cell_size).

        The robots all use the same hash function to convert theit x-y position into a cell_id.
        The pheromone map stores the cell_ids in a dictionary where the key is cell id and the value is number of
        visits.

        This warehouse level map adds all numbers of visits together to get a total phereomone map.
        """

		cell_size = self.exp_cfg.get('warehouse', 'cell_size') # TODO: add to config - robot size
		for robot in self.rob_c:
			# Robot's current coordinates
			x = robot[0]
			y = robot[1]

			# Calculate the cell's centroid (now used as the ID)
			cell_id_x = (math.floor(x / cell_size) * cell_size) + cell_size / 2
			cell_id_y = (math.floor(y / cell_size) * cell_size) + cell_size / 2
			cell_id = (cell_id_x, cell_id_y)

			# Increment the visit count for the cell_id
			if cell_id in self.pheromone_map:
				self.pheromone_map[cell_id] += 1
			else:
				self.pheromone_map[cell_id] = 1

	def iterate(self):
		"""
        Moves the simulation forward by one time step, updating robot and box positions.

        Args:
            pheromones (bool, optional): If True, updates the pheromone map. Defaults to False.
        """

		# Run setup command if this is the first iteration of the BT
		if self.counter == 0:
			self.rob_c, self.boxes = self.swarm.iterate(self.rob_c, self.boxes, init=1)
		else:
			self.rob_c, self.boxes = self.swarm.iterate(self.rob_c, self.boxes, init=0)

			# Update pheromones
			if self.exp_cfg.get('warehouse', 'pheromones'):
				self.update_pheromone_map()

			# Run Hive Mind cleanup
			# (note in reality this would be handled by the Hive server but since this is hosted in the warehouse, we run here)
			if self.hive_mind is not None:
				self.hive_mind.cleanup_hive_mind()

		self.counter += 1
		self.swarm.counter = self.counter

class WallBounds:
	"""
    Defines the standard format for walls in the warehouse, specifying height, width, and walls.
    """

	def __init__(self):
		"""
        Initializes the wall at zero values
        """
		self.start = np.array([0,0])
		self.end = np.array([0,0])
		self.width = 1
		self.hitbox = []

class BoxBounds:
	"""
    Defines the bounding box for objects in the warehouse, specifying height, width, and walls.
    """
	def __init__(self, h, w, mid_point):
		"""
		Initializes the bounding box with given dimensions and midpoint.

		Args:
		    h (float): Height of the box.
		    w (float): Width of the box.
		    mid_point (tuple): The midpoint coordinates of the box.
		"""
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
	"""
    Represents the warehouse map, containing walls and partitions.
    Generates walls and divides the map into structured areas for swarm navigation.
    See map.yaml for config
    """

	def __init__(self, width, height, wallsh, wallsv, wall_divisions=10):
		"""
        Initializes the map with specified dimensions, walls, and optional subdivisions.

        Args:
            width (float): The width of the warehouse.
            height (float): The height of the warehouse.
            wallsh (list): List of horizontal walls.
            wallsv (list): List of vertical walls.
            wall_divisions (int, optional): Number of divisions for structuring walls. Defaults to 10.
        """
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
		"""
        Generates walls for the warehouse based on predefined configurations.
        """
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
		"""
        Creates subdivisions within the map to aid navigation and object placement.

        Args:
            divisions (int, optional): Number of subdivisions for wall placement. Defaults to 10.
        """
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