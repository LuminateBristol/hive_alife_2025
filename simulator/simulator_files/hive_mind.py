import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

class LocalGraph:
    '''
    A generalised knowledge graph format to be inherited for formalisation of the Hive Mind KG
    '''
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node_name):
        '''
        Adds a node in the graph.
        :param node_name: Name of the node
        '''
        self.graph.add_node(node_name)

    def add_edge(self, node1, node2, edge_type=None):
        '''
        Adds an edge between two nodes in the graph.
        '''
        self.graph.add_edge(node1, node2, edge_type=edge_type)

    def add_edges(self, edges):
        '''
        Add multiple edges between existing nodes.
        :param edges: A list containing nodes between which edges will be added e.g. [(1, 2), (2, 3), (3, 4)]
        '''
        self.graph.add_edges_from(edges)

    def display_graph(self):
        '''
        Displays teh graph nodes and edges
        :return:
        '''
        print("Nodes in the graph:")
        print(self.graph.nodes(data=True))
        print("\nEdges in the graph:")
        print(self.graph.edges(data=True))

    def check_for_node(self, node_name):
        '''
        Checks if node exists
        :return: True if node exists, otherwise False.
        '''
        if node_name in self.graph.nodes:
            return True
        else:
            return False

    def update_attribute(self, node_name, **attributes):
        """
        Updates attributes for a specific node.
        :param node_id: Node whose attributes need to be updated
        :param attributes: Updated key-value pairs (e.g. battery status)
        """
        if node_name in self.graph.nodes:
            # Update the node's attributes directly
            self.graph.nodes[node_name].update(attributes)
        else:
            print(f"Node {node_name} does not exist.")

class GraphMind(LocalGraph):
    '''
    Knowledge graph specifically for the Graph Mind.
    '''

    def __init__(self):
        super().__init__()

    def add_robot_observation_space(self, robot_observation_space):
        '''
        :param: robot_observation_space
            Provide list of lists for possible observations that the robot can contribute to the Graph Mind
            Inner list format: [observation, edge_type, **attributes]
        '''
        for observation in robot_observation_space:
            self.add_information_node(observation[0], observation[1], observation[2], **observation[3])

    def add_information_node(self, parent_node, info_node, edge_type=None, **attributes):
        """
        Add new information to the Graph Mind.
        :param parent_node: Name of the parent node
        :param info_node: Name of the new node
        :param edge_type: Type of the edge connecting the nodes
        :param attributes: Any attributed data to be added to the Node - in key-pair format e.g., {'id': task_id, 'type': 'task'}
        """
        try:
            self.check_for_node(parent_node)
        except NodeNotFoundError as e:
            print(f"Error: Original node '{parent_node}' does not exist. Details: {str(e)}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            raise
        else:
            # Add the node with the attributes directly
            self.graph.add_node(info_node, **attributes)
            self.add_edge(parent_node, info_node, edge_type)

        return info_node

    def find_node(self, node_name):
        for node, data in self.graph.nodes(data=True):
            if node == node_name:
                return node
        print(f'Cannot find node {node_name}')
        return None

    def update_weight_for_node_name(self, node_name, weight):
        """
        Update the needs_weight for all nodes with the same name.
        :param node_name: Name of the node to update the weight for
        :param weight: The weight to assign
        """
        for node, data in self.graph.nodes(data=True):
            if node == node_name:
                self.update_attribute(node, needs_weight=weight)

    def extract_tasks(self):
        """
        Extract and print all nodes with the attribute 'type' equal to 'task'.
        """
        tasks = []
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'task':
                successors = self.graph.successors(node) if self.graph.is_directed() else self.graph.neighbors(node)

                # Iterate through successors and print their attributes
                for successor in successors:
                    successor_attributes = self.graph.nodes[successor]
                    print(f"  Successor: {successor}, Attributes: {successor_attributes}")

    def extract_task_id(self, id):
        """
        Extract and print all nodes with the attribute 'type' equal to 'task'.
        """
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'task' and data.get('id') == id:
                return node, data

    def extract_informational_needs(self):
        """
        Extract and print all nodes with an attribute 'in_weight' equal to 1.
        """
        weighted_nodes = []
        for node, data in self.graph.nodes(data=True):
            if data.get('in_need') == 0:
                weighted_nodes.append((node, data))

        return weighted_nodes

    def print_graph_mind(self, attribute_filter=None):
        """
        Visualize nodes in the graph based on the specified attribute filter.

        :param attribute_filter: A dictionary containing the attributes to filter nodes by (e.g., {'weight': 1})
        """
        plt.figure(figsize=(12, 8))
        pos = nx.arf_layout(self.graph)  # or another layout

        # Color mapping for node types
        color_map = {
            'task': '#ff9999',  # Example color for task
            'entity': '#66b3ff',  # Example color for entity
            'status': '#ffcc99',  # Example color for status
            'task_status': '#99ff99',  # Example color for task_status
        }

        filtered_nodes = []  # List to store nodes that meet the filter criteria
        node_colors = []  # List to store colors for filtered nodes

        # Iterate through nodes and filter based on the attribute filter
        for node, attributes in self.graph.nodes(data=True):
            # Check if the node matches the filter
            if attribute_filter is None or all(attributes.get(key) == value for key, value in attribute_filter.items()):
                filtered_nodes.append(node)  # Keep track of filtered nodes
                node_type = attributes.get('type', None)
                # Assign color based on the type
                if node_type in color_map:
                    node_colors.append(color_map[node_type])
                else:
                    node_colors.append('#d9d9d9')  # Default color for nodes without a valid 'type'

        # Create a subgraph of filtered nodes
        filtered_graph = self.graph.subgraph(filtered_nodes)

        # Draw the filtered graph, ensuring colors correspond to the nodes in the filtered graph
        nx.draw(filtered_graph, pos, with_labels=True, node_color=node_colors[:len(filtered_graph)], font_size=6)
        edge_labels = nx.get_edge_attributes(filtered_graph, 'edge_type')
        nx.draw_networkx_edge_labels(filtered_graph, pos, edge_labels=edge_labels, font_size=3)
        plt.title('Filtered Graph Mind Visualization')
        plt.show()

    def plot_node_tree(self, node):
        """
        Plots the specified node and all of its successors (i.e., all nodes reachable from it).

        Parameters:
        graph (networkx.DiGraph): The directed graph containing the nodes.
        node: The node to plot along with all its successors.
        """
        if not self.graph.has_node(node):
            print(f"Node {node} does not exist in the graph.")
            return

        # Get all nodes reachable from the specified node (i.e., all successors)
        all_successors = nx.descendants(self.graph, node)

        # Include the node itself in the subgraph
        subgraph_nodes = all_successors | {node}
        subgraph = self.graph.subgraph(subgraph_nodes)

        # Generate positions for the nodes
        pos = nx.spring_layout(subgraph)

        # Plot the nodes
        plt.figure(figsize=(8, 6))
        nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=12, font_weight='bold')

        # Highlight the main node (the node passed in)
        nx.draw_networkx_nodes(subgraph, pos, nodelist=[node], node_color='lightgreen', node_size=4000)

        # Draw the edge labels (if the graph has edge labels)
        edge_labels = nx.get_edge_attributes(subgraph, 'label')
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels)

        plt.title(f"Node '{node}' and all its successors", fontsize=16)
        plt.show()

# class OptimiseHiveMind(): # TODO: update this so it takes in the Hive Mind as an input somewhere and returns new weights - the hive mind this time will be a sim.hive_mind object (move to run file maybe?)
#     def __init__(self, robot_observation_space, tasks):
#         self.robot_observation_space = robot_observation_space
#         self.tasks = tasks
#
#     def cost_function(self, task_time, messages, w_T=1, w_M=1):
#         return w_T * task_time + w_M * messages
#
#     def build_hive_mind(self):
#         self.Hive_Mind = HiveMind()
#         self.Hive_Mind.build_hive_mind(self.robot_observation_space, self.tasks)
#
#     def simulate_task(self, hive_mind):
#         # You already have the simulator integrated
#         task_time, messages = run_simulation(hive_mind)
#         return task_time, messages
#
#     def greedy_optimization(self, simulate_task, num_simulations=10):
#         best_cost = float('inf')
#         best_weights = {}
#
#         # Gather all unique node names (e.g., 'battery_status', 'task_status')
#         unique_node_names = set(observation[2] for observation in self.robot_observation_space)
#
#         for node_name in unique_node_names:
#             # Test weight = 1 for all nodes with the same name
#             print(f"Testing node: {node_name} with weight=1")
#             self.build_hive_mind()
#             self.Hive_Mind.update_weight_for_node_name(node_name, weight=1)
#
#             costs = []
#             for _ in range(num_simulations):
#                 task_time, messages = simulate_task(self.Hive_Mind)
#                 current_cost = self.cost_function(task_time, messages)
#                 costs.append(current_cost)
#
#             avg_cost_weight_1 = sum(costs) / num_simulations
#
#             # Test weight = 0 for all nodes with the same name
#             print(f"Testing node: {node_name} with weight=0")
#             self.build_hive_mind()
#             self.Hive_Mind.update_weight_for_node_name(node_name, weight=0)
#
#             costs = []
#             for _ in range(num_simulations):
#                 task_time, messages = simulate_task(self.Hive_Mind)
#                 current_cost = self.cost_function(task_time, messages)
#                 costs.append(current_cost)
#
#             avg_cost_weight_0 = sum(costs) / num_simulations
#
#             # Keep the best weight (0 or 1)
#             if avg_cost_weight_1 < avg_cost_weight_0:
#                 print(f"Node {node_name} weight=1 performs better with cost: {avg_cost_weight_1}")
#                 best_weights[node_name] = 1
#             else:
#                 print(f"Node {node_name} weight=0 performs better with cost: {avg_cost_weight_0}")
#                 best_weights[node_name] = 0
#
#         return best_weights