import networkx as nx
import matplotlib.pyplot as plt
import copy
from networkx.drawing.nx_pydot import graphviz_layout

class LocalGraph:
    '''
    A generalised knowledge graph format to be inherited for formalisation of the Hive Mind KG
    '''
    def __init__(self):
        """
        Initializes an empty MultiDiGraph.
        This makes for a directional graph.
        """
        self.graph = nx.MultiDiGraph()

    def add_node(self, node_name):
        """
        Adds a node to the graph.

        Args:
            node_name (str): Name of the node to add.
        """
        self.graph.add_node(node_name)

    def add_edge(self, node1, node2, edge_type=None, direction=True):
        """
        Adds an edge between two nodes in the graph.

        Args:
            node1 (str): Source node.
            node2 (str): Destination node.
            edge_type (str, optional): Type of relationship between nodes. Defaults to None.
            direction (bool, optional): Indicates if the edge is directed. Defaults to True.
        """
        self.graph.add_edge(node1, node2, edge_type=edge_type, direction=direction)

    def add_edges(self, edges):
        """
        Adds multiple edges between existing nodes.

        Args:
            edges (list of tuples): List of tuples representing edges (e.g., [(1, 2), (2, 3)]).
        """
        self.graph.add_edges_from(edges)

    def display_graph(self):
        """
        Displays the graph nodes and edges in the console.
        """
        print("Nodes in the graph:")
        print(self.graph.nodes(data=True))
        print("\nEdges in the graph:")
        print(self.graph.edges(data=True))

    def check_for_node(self, node_name):
        """
        Checks if a node exists in the graph.

        Args:
            node_name (str): Name of the node to check.

        Returns:
            bool: True if the node exists, otherwise False.
        """
        if node_name in self.graph.nodes:
            return True
        else:
            return False

    def update_attribute(self, node_name, **attributes):
        """
        Updates attributes for a specific node.

        Args:
            node_name (str): Node whose attributes need to be updated.
            **attributes: Key-value pairs representing updated attributes.
        """
        if node_name in self.graph.nodes:
            # Update the node's attributes directly
            self.graph.nodes[node_name].update(attributes)
        else:
            print(f"Node {node_name} does not exist.")

class GraphMind(LocalGraph):
    """
    Knowledge graph specifically for the Graph Mind.
    """

    def __init__(self):
        """
        Initializes the GraphMind by inheriting from LocalGraph.
        """
        super().__init__()

    def add_robot_observation_space(self, robot_observation_space):
        """
        Adds robot observations to the Graph Mind.
        This is specifically setup for the format so those observations must be provided in that format.
        Observations are defined in objects.py

        Args:
            robot_observation_space (list): List of lists containing robot observations.
        """
        for observation in robot_observation_space:
            attributes = copy.deepcopy(observation[3])
            self.add_information_node(parent_node=observation[0], info_node=observation[1], edge_type=observation[2], direction=True, **attributes)

    def add_information_node(self, parent_node, info_node, edge_type=None, direction=True, **attributes):
        """
        Adds new information to the Graph Mind.

        Args:
            parent_node (str): Name of the parent node.
            info_node (str): Name of the new node.
            edge_type (str, optional): Type of the edge. Defaults to None.
            direction (bool, optional): Indicates if the edge is directed. Defaults to True.
            **attributes: Additional attributes for the node.

        Returns:
            str: The name of the added info node.
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
            self.add_edge(parent_node, info_node, edge_type, direction)

        return info_node

    def find_node(self, node_name):
        """
        Finds and returns a node if it exists.

        Args:
            node_name (str): The name of the node to search for.

        Returns:
            str or None: Node name if found, otherwise None.
        """
        for node, data in self.graph.nodes(data=True):
            if node == node_name:
                return node
        print(f'Cannot find node {node_name}')
        return None

    def update_weight_for_node_name(self, node_name, weight):
        """
        Updates the weight for all nodes with the same name.

        Args:
            node_name (str): Name of the node.
            weight (int): New weight to be assigned.
        """
        for node, data in self.graph.nodes(data=True):
            if node == node_name:
                self.update_attribute(node, needs_weight=weight)

    def print_graph_mind(self, attribute_filter=None):
        """
        Visualizes nodes in the graph based on an optional attribute filter.

        Args:
            attribute_filter (dict, optional): Dictionary containing attributes to filter nodes (e.g., {'weight': 1}). Defaults to None.
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
        Plots a node and all of its successors in a hierarchical format.

        Args:
            node (str): The node to plot along with its successors.
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

    def cleanup_hive_mind(self):
        """
        Cleans up the Hive Mind by removing nodes with a weight of 0.
        """
        # nodes_to_remove = [n for n, attr in self.graph.nodes(data=True) if attr.get('weight') == 0]
        # self.graph.remove_nodes_from(nodes_to_remove)

        # For optimisatio purposes - this has been removed
        # TODO: review this part of the code for each task and workout its use / integration to optimisation
        pass
