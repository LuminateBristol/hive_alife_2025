from copy import deepcopy

import numpy as np
import py_trees
import copy

from . import bt_setup, GraphMind

class Robot:
    """
    Represents a robot in the simulation with movement, sensors, and interaction capabilities.
    """
    def __init__(self, gen_cfg, exp_cfg):
        """
        Initializes the robot with given physical and sensor properties.
        See default.yaml and exp_setup.yaml for config info

        Args:
            radius (float): Radius of the robot.
            max_v (float): Maximum velocity of the robot.
            camera_sensor_range (float): Range of the camera sensor.
            place_tol (float, optional): Placement tolerance for objects. Defaults to None.
        """
        self.robot_index = None # String index
        self.root = None
        self.robot_tree = None
        self.blackboard = None

        self.radius = gen_cfg.get('robot', 'radius')
        self.max_v = gen_cfg.get('robot', 'max_v')
        self.camera_sensor_range = gen_cfg.get('robot', 'camera_sensor_range')
        self.communication_range = gen_cfg.get('robot', 'communication_range')
        self.place_tol = exp_cfg.get('tolerances', 'place_tol')

        # Set observations ready for sending to the Hive Mind
        self.observation_space = []

    def add_observations(self):
        """
        Adds self-observable information of the robot to the observation space.
        These observations include position, heading, speed, and various status attributes.

        The observation lists here are set up for GraphMind class in hive_mind.py. The format is as follows:
        [parent_name, node_name, edge, **attributes]
        """
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_position',      'has_status',   {'type':'robot_position',        'weight':0, 'robot_name':f'robot_{self.robot_index}',  'data':np.array([999, 999, 999]),   'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_led_status',    'has_status',   {'type':'robot_led_status',      'weight':0, 'robot_name':f'robot_{self.robot_index}',  'data':[0.0, 0.0, 0.0],                          'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_lifter_status', 'has_status',   {'type':'robot_lifter_status',   'weight':0, 'robot_name':f'robot_{self.robot_index}',  'data':False,                       'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_heading',       'has_status',   {'type':'robot_heading',         'weight':0, 'robot_name':f'robot_{self.robot_index}',  'data':0,                           'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_speed',         'has_status',   {'type':'robot_speed_status',    'weight':0, 'robot_name':f'robot_{self.robot_index}',  'data':2.0,                           'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_task_id',       'in_progress',  {'type':'robot_task_id',         'weight':0, 'robot_name':f'robot_{self.robot_index}',  'data':'task_x',                           'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_pheromone_map', 'has_status',   {'type':'robot_pheromone_map',   'weight':0, 'robot_name':f'robot_{self.robot_index}',  'data':{},                          'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_chosen_door',   'has_status',   {'type':'chosen_door',           'weight':0, 'robot_name':f'robot_{self.robot_index}',  'data':'door_x',                           'time':0} ])
        self.observation_space.append([f'robot_{self.robot_index}', f'robot_{self.robot_index}_waypoint',      'in_progress',  {'type':'waypoint',              'weight':0, 'robot_name':f'robot_{self.robot_index}',  'data':[0.0, 0.0, 0.0],                           'time':0} ])
    
    def setup_bb(self, width, height, heading_change_rate, repulsion_o, repulsion_w, task_log, delivery_points, traffic_score, latency):
        """
        Sets up the robot's blackboard, registering keys and initializing parameters.

        Args:
            width: width of warehouse
            height: height of warehouse
            heading_change_rate (float): Heading change rate of the robot
            repulsion_o (float): Repulsion odometer from other objects
            repulsion_w (float): Repulsion wall from walls
            task_log (float): Task log
            delivery_points (dict): Delivery points dictionary
            traffic_score (float): Traffic score
        """
        self.blackboard.register_key(key="w_rob_c", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="w_boxes", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="carrying_box", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="radius", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="max_v", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="heading_change_rate", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="camera_sensor_range", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="communication_range", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="place_tol", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="arena_size", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="repulsion_w", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="repulsion_o", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="local_task_log", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="delivery_points", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="traffic_score", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="latency", access=py_trees.common.Access.WRITE)

        self.blackboard.w_rob_c = []
        self.blackboard.w_boxes = []
        self.blackboard.carrying_box = False
        self.blackboard.radius = self.radius
        self.blackboard.max_v = self.max_v
        self.blackboard.heading_change_rate = heading_change_rate
        self.blackboard.camera_sensor_range = self.camera_sensor_range
        self.blackboard.communication_range = self.communication_range
        self.blackboard.place_tol = self.place_tol
        self.blackboard.arena_size = [width, height]
        self.blackboard.repulsion_w = repulsion_w
        self.blackboard.repulsion_o = repulsion_o
        self.blackboard.local_task_log = copy.deepcopy(task_log) # Use deepcopy so that each robot has a unique copy
        self.blackboard.delivery_points = delivery_points
        self.blackboard.traffic_score = traffic_score
        self.blackboard.latency = latency
        
    def add_map(self, map):
        """
        Adds the map to the robots blackboard.
        This is done seperately as it needs to be done after the warehouse is fully initialized.

        Args:
            map (class object): Map object - see objects.py
        """
        self.blackboard.register_key(key="map", access=py_trees.common.Access.WRITE)
        self.blackboard.map = map

    def build_robo_mind(self, entities, tasks, map=None):
        """
        Sets up the robot's internal knowledge graph.

        Args:
            entities (dict): Entities - see exp_setup.yaml
            tasks (dict): Tasks - see exp_setup.yaml
            map (class object): Map object
        """
        self.add_observations()
        self.robo_mind = GraphMind()

        if entities is not None:
            for entity in entities:
                self.robo_mind.add_information_node(entity[0], entity[1], entity[2], **entity[3])
        if tasks is not None:
            for task in tasks:
                self.robo_mind.add_information_node(task[0], task[1], task[2], **task[3])
        if map is not None:
            for wp in map:
                self.robo_mind.add_information_node(wp[0], wp[1], wp[2], **wp[3])

        self.robo_mind.add_edge('robots', f'robot_{self.robot_index}')
        for observation in self.observation_space:
            self.robo_mind.add_information_node(observation[0], observation[1], observation[2], **observation[3])

        # Register the robo mind in blackboard
        self.blackboard.register_key(key="robo_mind", access=py_trees.common.Access.WRITE)
        self.blackboard.robo_mind = self.robo_mind

    def add_hive_mind(self, hive_mind): # TODO: I think i can just put this in setup blackboard but need to set up for hive/dist/cent - it is simply pointing to the object so does not matter when it is setup
        """
        Sets up the link to the Hive knowledge graph within the robot's blackboard

        Args:
            hive_mind (class object): Hive mind object - see hive_mind.py and initalised in sim.py
        """
        self.blackboard.register_key(key="hive_mind", access=py_trees.common.Access.WRITE)
        self.blackboard.hive_mind = hive_mind

class Box:
    """
    Represents a box object in the warehouse with attributes like position, color, and status.
    """

    def __init__(self, colour=None, id=None):
        """
        Initializes the box with optional color and ID attributes.

        Args:
            colour (str, optional): Color of the box. Defaults to None.
            id (int, optional): Unique identifier for the box. Defaults to None.
        """
        self.x = None
        self.y = None
        self.colour = colour
        self.id = id
        self.action_status = 0 # Set to 1 if being carried or if placed so other robots ignore
        
class Swarm:
    """
    Represents a swarm of robots in the simulation, handling coordination and movement.
    """
    def __init__(self, gen_config, exp_config):
        """
        Initializes the swarm with repulsion parameters and heading change rate.

        Args:
            repulsion_o (class object): Repulsion object - see default.yaml
            repulsion_w (class object): Repulsion object - see default.yaml
            heading_change_rate (int): Heading change rate - see default.yaml
        """
        self.agents = [] # turn this into a dictionary to make it accessible later for heterogeneous swarms?
        self.number_of_agents = 0
        self.gen_cfg = gen_config
        self.exp_cfg = exp_config
        self.F_heading = None
        self.agent_dist = None

    def add_agents(self, agent_obj, hive_mind=None, processed_delivery_points=None, traffic_score=None):
        """
        Adds a specified number of robot agents to the swarm, initializing their behavior trees and blackboards.

        Args:
            agent_obj (class object): Agent object - see objects.py
            number (int): Number of agents to add - defined per experiment - see exp_setup.yaml
            width (int): Width of the warehouse
            height (int): Height of the warehouse
            bt_controller (class object): Chosen behaviour tree controller object - see exp_setup.yaml
            print_bt (bool): print the behaviour tree for each agent if true
            task_log (class object): Task object - see exp_setup.yaml
            delivery_points (dict): Delivery points dictionary - see exp_setup.yaml
            traffic_score (dict): Traffic score object - only used for traffic scenario experiments
        """

        # Get swarm parameters from config
        number              = self.exp_cfg.get('number_of_agents')
        width               = self.gen_cfg.get('warehouse', 'width')
        height              = self.gen_cfg.get('warehouse', 'height')
        bt_controller       = self.exp_cfg.get('behaviour_tree')
        print_bt            = self.gen_cfg.get('robot', 'print_bt')
        repulsion_o         = self.gen_cfg.get('warehouse', 'repulsion_object')
        repulsion_w         = self.gen_cfg.get('warehouse', 'repulsion_wall')
        heading_change_rate = self.gen_cfg.get('heading_change_rate')

        # Create swarm
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

            # Set up behaviour tree
            bt_module = bt_setup.behaviour_trees[bt_controller] # Get controller from the setup file
            ag.root = bt_module.create_root(robot_index = ag.robot_index)
            if print_bt and ag.robot_index == 0:
                py_trees.display.render_dot_tree(ag.root) # Uncomment to print png of behaviour tree
            ag.robot_tree = py_trees.trees.BehaviourTree(ag.root)

            # Set up blackboard
            name            = f'Pick Place DOTS: {str_index}'
            namespace       = str_index
            task_log        = self.exp_cfg.get('task_log')
            latency         = self.exp_cfg.get('latency')
            ag.blackboard   = py_trees.blackboard.Client(name=name, namespace=namespace)

            ag.setup_bb(width, height, heading_change_rate, repulsion_o, repulsion_w, task_log, processed_delivery_points, traffic_score, latency)

            # Set up Robo Mind and Hive Mind
            # Add entities from config
            try:
                entities = self.exp_cfg.get('entities')
            except:
                print('No entities defined in config')
                entities = None

            # Add tasks from config
            try:
                tasks = self.exp_cfg.get('tasks')
            except:
                print('No tasks defined in config')
                tasks = None

            # Build
            ag.build_robo_mind(entities, tasks)
            if hive_mind is not None:
                hive_mind.add_robot_observation_space(ag.observation_space)
            ag.add_hive_mind(hive_mind)

            self.number_of_agents += 1

    def add_map(self, map):
        """
        Adds the map to each agent in the swarm - see Robot class object

        Args:
            map (class object): Map object
        """
        for ag in self.agents:
            ag.add_map(map)

    def iterate(self, rob_c, boxes, init=0):
        """
        Advances the simulation by one step, updating robot and box positions.

        Args:
            rob_c (list): Current positions of robots.
            boxes (list): Current positions of boxes.
            init (int, optional): If set to 1, runs initial setup for behavior trees. Defaults to 0.

        Returns:
            tuple: Updated positions of robots and boxes.
        """
        rob_c_new = rob_c
        boxes_new = boxes

        for ag in self.agents:
            # Run setup() method for all behaviours on first run of the behaviour tree
            if init:
                ag.robot_tree.setup()

            # Update behaviour tree robot positions and boxes after last tick of all robots
            ag.blackboard.w_rob_c = rob_c_new
            ag.blackboard.w_boxes = boxes_new
            # Tick behaviour tree
            ag.robot_tree.tick()
            # py_trees.display.unicode_tree(ag.robot_tree.root)
            # Update robot and box positions
            rob_c_new = ag.blackboard.w_rob_c
            boxes_new = ag.blackboard.w_boxes

        return rob_c_new, boxes_new

class Swarm_Centralised:
    """
    Represents a centralised multi-robot system in which case all robots are controlled here from this object directly.
    Control is passed to the robots by accessing their blackboard variables to update individual actions throughout the experiment.
    """

    def __init__(self, gen_config, exp_config):
        """
        Initializes the swarm with repulsion parameters and heading change rate.

        Args:
            task ('string): describes which task is being controlled by the centralised controller
            repulsion_o (class object): Repulsion object - see default.yaml
            repulsion_w (class object): Repulsion object - see default.yaml
            heading_change_rate (int): Heading change rate - see default.yaml
        """
        self.agents = []  # turn this into a dictionary to make it accessible later for heterogeneous swarms?
        self.number_of_agents = 0
        self.gen_cfg = gen_config
        self.exp_cfg = exp_config
        self.task = self.exp_cfg.get('task')
        self.repulsion_o = self.gen_cfg.get('warehouse','repulsion_object')  # repulsion distance between agents-objects
        self.repulsion_w = self.gen_cfg.get('warehouse', 'repulsion_wall')  # repulsion distance between agents-walls
        self.heading_change_rate = self.gen_cfg.get('heading_change_rate')
        self.F_heading = None
        self.agent_dist = None

    def add_agents(self, agent_obj, hive_mind=None, processed_delivery_points=None, traffic_score=None):
        """
        Adds a specified number of robot agents to the swarm, initializing their behavior trees and blackboards.

        Args:
            agent_obj (class object): Agent object - see objects.py
            number (int): Number of agents to add - defined per experiment - see exp_setup.yaml
            width (int): Width of the warehouse
            height (int): Height of the warehouse
            bt_controller (class object): Chosen behaviour tree controller object - see exp_setup.yaml
            print_bt (bool): print the behaviour tree for each agent if true
            task_log (class object): Task object - see exp_setup.yaml
            delivery_points (dict): Delivery points dictionary - see exp_setup.yaml
            traffic_score (dict): Traffic score object - only used for traffic scenario experiments
        """

        # Get swarm parameters from config
        number = self.exp_cfg.get('number_of_agents')
        width = self.gen_cfg.get('warehouse', 'width')
        height = self.gen_cfg.get('warehouse', 'height')
        bt_controller = self.exp_cfg.get('behaviour_tree')
        print_bt = self.gen_cfg.get('robot', 'print_bt')
        repulsion_o = self.gen_cfg.get('warehouse', 'repulsion_object')
        repulsion_w = self.gen_cfg.get('warehouse', 'repulsion_wall')
        heading_change_rate = self.gen_cfg.get('heading_change_rate')

        # Create swarm
        for num in range(number):
            ag = copy.deepcopy(agent_obj)  # Use deepcopy soy that each robot is a unique agent object
            self.agents.append(ag)

        num = 0
        # Each agent obj is of class 'Robot' above
        for ag in self.agents:
            # Add robot to swarm
            ag.robot_index = num
            str_index = 'robot_' + str(ag.robot_index)
            num += 1

            # Set up behaviour tree
            bt_module = bt_setup.behaviour_trees[bt_controller]  # Get controller from the setup file
            ag.root = bt_module.create_root(robot_index=ag.robot_index)
            if print_bt and ag.robot_index == 0:
                py_trees.display.render_dot_tree(ag.root)  # Uncomment to print png of behaviour tree
            ag.robot_tree = py_trees.trees.BehaviourTree(ag.root)

            # Set up blackboard
            name = f'Pick Place DOTS: {str_index}'
            namespace = str_index
            task_log = self.exp_cfg.get('task_log')
            latency = self.exp_cfg.get('latency')
            ag.blackboard = py_trees.blackboard.Client(name=name, namespace=namespace)

            ag.setup_bb(width, height, heading_change_rate, repulsion_o, repulsion_w, task_log,
                        processed_delivery_points, traffic_score, latency)

            # Set up Robo Mind and Hive Mind
            # Add entities from config
            try:
                entities = self.exp_cfg.get('entities')
            except:
                print('No entities defined in config')
                entities = None

            # Add tasks from config
            try:
                tasks = self.exp_cfg.get('tasks')
            except:
                print('No tasks defined in config')
                tasks = None

            ag.build_robo_mind(entities, tasks)
            if hive_mind is not None:
                hive_mind.add_robot_observation_space(ag.observation_space)
            ag.add_hive_mind(hive_mind)

            self.number_of_agents += 1

    def add_map(self, map):
        """
        Adds the map to each agent in the swarm - see Robot class object

        Args:
            map (class object): Map object
        """
        for ag in self.agents:
            ag.add_map(map)

    def iterate_logistics_task(self, rob_c, boxes, init):
        """
        Centralised iterations of the robots happens here in this class.
        For centralised we assume we have perfect information - we then send the exact actions for each robot based on that.

        For logistics:

        1) Identify closest correctly coloured box for each delivery point
        2) Identify closest robot to each of those boxes
        3) Seto those robots target box directly ready for delivery
        4) Complete task

        # In this case, all the centralised decision-making happens at the start of the task
        # We run this method once, then let the robots complete the task once deliveries are all assigned
        """

        # At the start of the program - find the closest box for each delivery point
        if init:
            self.closest_boxes = []
            for delivery_point in self.delivery_points:
                if not delivery_point.status:
                    min_distance = float('inf')
                    closest_box = None

                    for box in boxes:
                        if box.colour == delivery_point.colour and box.action_status == 0:
                            distance = np.sqrt((box.x - delivery_point.x) ** 2 + (box.y - delivery_point.y) ** 2)
                            if distance < min_distance:
                                min_distance = distance
                                closest_box = box

                    self.closest_boxes.append(closest_box)

        # Each timestep - find available agents
        available_agents = []
        for agent in self.agents:
            agent.blackboard.register_key(key="action", access=py_trees.common.Access.WRITE)
            if agent.blackboard.action == 'random_walk':
                available_agents.append(agent)

        # If agents are available - send them to their nearest closest box - i.e. the nearest box from self.closest_boxes
        if available_agents:
            for box in self.closest_boxes:
                min_distance = float('inf')
                closest_agent = None

                # Agent is available so set delivery point status here to 1 and remove from cloests boxes
                # TODO: I don't think it gets updated within the behaviour tree and we need it for centralised control so we do it here
                #       on the assumption that the robot will definitely deliver the box - should be done on confirmation of delivery instead
                col = box.colour
                for delivery_point in self.delivery_points:
                    if delivery_point.colour == col:
                        delivery_point.status = 1
                self.closest_boxes.remove(box)

                # Find closest available agent
                for ag in available_agents:
                    ag_coord = rob_c[ag.robot_index]
                    distance = np.sqrt((box.x - ag_coord[0]) ** 2 + (box.y - ag_coord[1]) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_agent = ag

                # Assign agent with target box
                if closest_agent is not None:
                    available_agents.remove(closest_agent)
                    closest_agent.blackboard.register_key(key="target_box", access=py_trees.common.Access.WRITE)
                    closest_agent.blackboard.target_box = box

    def iterate_area_cov_task(self, rob_c, boxes, init):
        """
        Centralised iterations of the robots happens here in this class.
        For centralised we assume we have perfect information - we then send the exact actions for each robot based on that.

        For area coverage:

        1) Split map into number hash grid cells - same grid size used during the distributed implementation
        2) Split the cells into groups, number of groups == number of robots
        3) Assign robots to their cloest group
        4) Robots moe through their assigned cells one by one
        """
        if init:
            # Configuration
            cell_size = 25  # TODO: add to config
            width = self.width
            height = self.height
            num_agents = len(self.agents)

            # Step 1: Split the map into n x n square cells
            num_cols = int(np.ceil(width / cell_size))
            num_rows = int(np.ceil(height / cell_size))

            # Generate centroids for each square cell
            cells = []
            for row in range(num_rows):
                for col in range(num_cols):
                    centroid_x = (col + 0.5) * cell_size
                    centroid_y = (row + 0.5) * cell_size
                    # Ensure centroids stay within bounds
                    centroid_x = min(centroid_x, width - cell_size / 2)
                    centroid_y = min(centroid_y, height - cell_size / 2)
                    cells.append((centroid_x, centroid_y))

            # Step 2: Divide cells into chunks for each robot
            cell_group_size = int(np.ceil(len(cells) / num_agents))
            cell_groups = [cells[i:i + cell_group_size] for i in range(0, len(cells), cell_group_size)]

            # Step 3: Assign each robot to the closest chunk based on the last cell in each chunk
            for agent in self.agents:
                min_dis = np.inf
                assigned_cells = None

                for cell_group in cell_groups:
                    start_position = cell_group[-1]  # The last cell in the chunk
                    agent_position = np.array([rob_c[agent.robot_index][0], rob_c[agent.robot_index][1]])
                    dis = np.linalg.norm(np.array(start_position) - agent_position)

                    if dis < min_dis:
                        min_dis = dis
                        assigned_cells = cell_group

                # Remove the chosen chunk from the list of chunks
                cell_groups.remove(assigned_cells)

                # Set to init position on blackboard
                agent.blackboard.register_key(key="search_area", access=py_trees.common.Access.WRITE)
                agent.blackboard.search_area = assigned_cells

    def iterate_traffic(self, rob_c, boxes, init):
        """
        Centralised iterations of the robots happens here in this class.
        For centralised we assume we have perfect information - we then send the exact actions for each robot based on that.

        For the traffic task:

        1) Identify the best path between the two tasks (this is done manually by us, not algorithmically)
        2) Send path to all robots
        3) Robots go to closest WP in path and then continue to loop path
        """
        if init:
            ideal_path = ['task_1', 'door_A', 'task_2', 'door_B']
            for agent in self.agents:
                agent.blackboard.register_key(key="assigned_path", access=py_trees.common.Access.WRITE)
                agent.blackboard.assigned_path = ideal_path

    def iterate(self, rob_c, boxes, init=0):
        """
        Advances the simulation by one step, updating robot and box positions.

        Args:
            rob_c (list): Current positions of robots.
            boxes (list): Current positions of boxes.
            init (int, optional): If set to 1, runs initial setup for behavior trees. Defaults to 0.

        Returns:
            tuple: Updated positions of robots and boxes.
        """

        # Run setup() method for all behaviours on first run of the behaviour tree
        for ag in self.agents:
            if init:
                ag.robot_tree.setup()

        # Run centralised control
        if self.task == 'logistics':
            self.iterate_logistics_task(rob_c, boxes, init)
        if self.task == 'area_coverage':
            self.iterate_area_cov_task(rob_c, boxes, init)
        if self.task == 'traffic':
            self.iterate_traffic(rob_c, boxes, init)

        # Update behaviour trees
        rob_c_new = rob_c
        boxes_new = boxes
        for ag in self.agents:
            # Update behaviour tree robot positions and boxes after last tick of all robots
            ag.blackboard.w_rob_c = rob_c_new
            ag.blackboard.w_boxes = boxes_new
            # Tick behaviour tree
            ag.robot_tree.tick()
            # py_trees.display.unicode_tree(ag.robot_tree.root)
            # Update robot and box positions
            rob_c_new = ag.blackboard.w_rob_c
            boxes_new = ag.blackboard.w_boxes

        return rob_c_new, boxes_new

class DeliveryPoint:
    """
    Represents a delivery point where boxes are delivered in the warehouse.
    Standard format - only used for certain experiments.
    """
    def __init__(self, x, y, colour, delivered, id):
        """
        Initializes a delivery point with coordinates, color, and delivery status.

        Args:
            x (float): X-coordinate of the delivery point.
            y (float): Y-coordinate of the delivery point.
            colour (str): Color associated with the delivery point.
            delivered (bool): Whether the delivery point has been served.
            id (int): Unique identifier for the delivery point.
        """
        self.x = x
        self.y = y
        self.colour = colour
        self.id = id
        self.delivered = delivered