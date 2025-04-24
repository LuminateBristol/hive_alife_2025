from . import Simulator
from matplotlib import pyplot as plt, animation
import numpy as np


class VizSim(Simulator):
    """
    A visualisation class for the simulator that extends the Simulator class.
    This class is responsible for rendering the simulation environment, plotting walls,
    agent positions, movement, and other visual elements.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the VizSim class.

        Args:
            *args: Variable-length argument list passed to the Simulator class.
            **kwargs: Arbitrary keyword arguments passed to the Simulator class.
        """
        super().__init__(*args, **kwargs)
        self.snapshot_s = [1]  # [50,1250,2250]
        self.verbose = True

    def plot_walls(self, ax):
        """
        Plots the walls of the simulation environment on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which to plot the walls.
        """

        # Plot horizontal walls
        for wall in self.map_cfg.get(self.exp_cfg.get('map'), 'wallsh'):
            start, end = wall
            x0, y0 = start
            x1, y1 = end
            ax.plot([x0, x1], [y0, y1], 'k-')  # 'k-' means black solid line

        # Plot vertical walls
        for wall in self.map_cfg.get(self.exp_cfg.get('map'), 'wallsv'):
            start, end = wall
            x0, y0 = start
            x1, y1 = end
            ax.plot([x0, x1], [y0, y1], 'k-')

    def generate_dot_positional_data(self):
        """
        Generates positional data for agents as dots on the visualisation.

        Returns:
            tuple: A tuple containing lists of x-coordinates, y-coordinates, and marker styles.
        """
        agent_range = range(self.exp_cfg.get('number_of_agents'))
        x_data = [
            [self.warehouse.rob_c[i, 0] for i in agent_range]
        ]
        y_data = [
            [self.warehouse.rob_c[i, 1] for i in agent_range]
        ]
        marker = ['ko']
        return (x_data, y_data, marker)

    def generate_dot_heading_arrow(self):
        """
        Generates arrow vectors to indicate the heading direction of each agent.

        Returns:
            tuple: Lists of x-coordinates and y-coordinates for the arrows.
        """
        length = 20
        steps = 20
        agents = self.swarm.number_of_agents
        x_vec = []
        y_vec = []
        for i in range(agents):
            start_x = self.warehouse.rob_c[i, 0]
            end_x = start_x + length * -np.cos(self.warehouse.rob_c[i, 2])
            start_y = self.warehouse.rob_c[i, 1]
            end_y = start_y + length * -np.sin(self.warehouse.rob_c[i, 2])
            x_vec.append(np.linspace(start_x, end_x, steps).tolist())
            y_vec.append(np.linspace(start_y, end_y, steps).tolist())
        return x_vec, y_vec

    def get_marker_size_in_data_units(self, cell_size, ax, scale_factor=0.75):
        """
        Computes the size of a marker in data units to maintain a scaled appearance in the visualiser.

        Args:
            cell_size (float): The size of the cell in data units.
            ax (matplotlib.axes.Axes): The axis on which to compute the marker size.
            scale_factor (float, optional): Scaling factor to adjust marker size. Defaults to 0.75.

        Returns:
            float: The computed marker size in points.
        """
        # Get the limits of the axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Get the size of the figure in inches and the DPI (dots per inch)
        fig = ax.get_figure()
        fig_width_inch, fig_height_inch = fig.get_size_inches()
        dpi = fig.dpi

        # Calculate the size of the data units in points
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Width and height of the axes in inches
        ax_width_inch = fig_width_inch * ax.get_position().width
        ax_height_inch = fig_height_inch * ax.get_position().height

        # Convert cell size from data units to points, with a slight scaling to avoid overlap
        marker_size_in_points_x = (cell_size / x_range) * (ax_width_inch * dpi)
        marker_size_in_points_y = (cell_size / y_range) * (ax_height_inch * dpi)

        # Take the minimum of x and y sizes for a uniform circular marker, apply scale factor
        marker_size_in_points = min(marker_size_in_points_x, marker_size_in_points_y) * scale_factor

        return marker_size_in_points

    def iterate(self, frame, dot=None, boxes=None, h_line=None, cam_range=None, snapshot=False):
        """
        Advances the simulation by one step and updates the visualisation.

        Args:
            frame (int): The current frame number.
            dot (list, optional): List of plotted agent markers.
            boxes (list, optional): List of plotted boxes.
            h_line (list, optional): List of heading arrows.
            cam_range (matplotlib plot object, optional): Camera range visualisation.
            snapshot (bool, optional): If True, captures a snapshot. Defaults to False.

        Returns:
            list: Updated plot elements.
        """
        self.warehouse.iterate()
        counter = self.warehouse.counter

        dot, boxes, h_line, cam_range = self.animate(frame, counter, dot, boxes, h_line, cam_range)

        if self.verbose:
            if self.warehouse.counter == 1:
                print("Progress |", end="", flush=True)
            if self.warehouse.counter % 100 == 0:
                print("=", end="", flush=True)

        self.exit_sim(counter=counter)

        dot = list(dot.values())
        h_line = list(h_line.values())

        return dot + h_line + boxes + [cam_range]

    def animate(self, i, counter, dot=None, boxes=None, h_line=None, cam_range=None): 
        """
        Animates the visualisation.

        Args:
            frame (int): The current frame number.
            dot (list, optional): List of plotted agent markers.
            boxes (list, optional): List of plotted boxes.
            h_line (list, optional): List of heading arrows.
            cam_range (matplotlib plot object, optional): Camera range visualisation.
            snapshot (bool, optional): If True, captures a snapshot. Defaults to False.

        Returns:
            list: Updated plot elements.
        """
        cam_range.set_data(
            [self.warehouse.rob_c[i, 0] for i in range(self.exp_cfg.get('number_of_agents'))],
            [self.warehouse.rob_c[i, 1] for i in range(self.exp_cfg.get('number_of_agents'))]
        )

        # Re-plot pheromone cells based on updated pheromone_map
        # Clear existing pheromone markers before re-plotting
        for p in self.pheromone_markers:
            p.remove()
        self.pheromone_markers.clear()

        # Plot each pheromone cell with varying opacity based on visit count
        for cell_id, visit_count in self.warehouse.pheromone_map.items():
            x, y = cell_id  # Cell centroid coordinates
            # Scale alpha from 0.1 (light) to 1.0 (dark) based on visit count
            alpha = min(0.01 + (visit_count * 0.01), 1.0)  # Limits alpha to a max of 1.0
            marker, = self.ax.plot(x, y, 's', color='blue', markersize=self.pheromone_marker_size, alpha=alpha)
            self.pheromone_markers.append(marker)  # Store the marker for clearing in the next iteration

        # Update robot data
        x_data, y_data, _ = self.generate_dot_positional_data()
        for i in range(len(dot)):
            dot[i].set_data(x_data[i], y_data[i])

        if boxes is not None:
            # Update box data
            for box, wbox in zip(boxes, self.warehouse.boxes):
                box.set_data([wbox.x], [wbox.y])

        # Update heading arrow data
        h_x_vec, h_y_vec = self.generate_dot_heading_arrow()
        for i in range(self.swarm.number_of_agents):
            h_line[i].set_data(h_x_vec[i], h_y_vec[i])

        return dot, boxes, h_line, cam_range

    def exit_sim(self, counter=None):
        """
        Sets the exit criteria depending on the selected experiment - see exp_setup.yaml

        Args:
            counter (int, optional): Timestep that the simulation is currently on
        """
        if self.task == 'counter':
            if counter > self.cfg.get('time_limit'):
                if self.verbose:
                    print("in", counter, "seconds")

                if self.gen_cfg.get('animate'):
                    exit()

        elif self.task == 'logistics':

            if all(dp.delivered for dp in self.processed_delivery_points):
                if self.verbose:
                    print("All boxes delivered in", counter, "seconds")

                if self.gen_cfg.get('animate'):
                    exit()

            if counter > self.gen_cfg.get('time_limit'):
                if self.verbose:
                    print("in", counter, "seconds")

                if self.gen_cfg.get('animate'):
                    exit()

        elif self.task == 'area_coverage':
            if counter > self.gen_cfg.get('time_limit'):
                total_cells = (self.gen_cfg.get('warehouse', 'width') * self.gen_cfg.get('warehouse', 'height')) / self.gen_cfg.get('warehouse', 'cell_size') ** 2
                print(self.warehouse.pheromone_map)
                percent_explored = (len(self.warehouse.pheromone_map) / total_cells) * 100
                print(f'{counter} counts reached - Time limit expired - Percentage explored: {percent_explored}%')
                exit()

        elif self.task == 'traffic':
            # print(counter, self.traffic_score)
            if self.traffic_score['score'] >= 200:
                print(f"{counter} counts reached - Time limit expired. Traffic score: {self.traffic_score['score']}")
                self.exit_threads = True
                self.exit_run = True
                exit()

    def run(self, iteration=0):
        """
        Starts the simulation and visualisation.

        Args:
            iteration (int, optional): Iteration number. Defaults to 0.
        """
        if self.verbose:
            print("Running")

        self.init_animate()
        plt.show()

        if self.verbose:
            print("\n")

    def init_animate(self):
        """
        Initializes the animation environment and starts the visualisation.
        """
        self.fig = plt.figure(figsize=(8, 8))
        plt.rcParams['font.size'] = '16'
        self.ax = plt.axes(xlim=(0, self.gen_cfg.get('warehouse', 'width')), ylim=(0, self.gen_cfg.get('warehouse', 'height')))
        self.cell_size = 1  # Set cell size based on configuration


        # Get marker sizes
        robot_size = self.gen_cfg.get('robot', 'radius')  # Size in data units
        box_size = self.gen_cfg.get('warehouse', 'box_radius')
        camera_sensor_range = self.gen_cfg.get('robot', 'camera_sensor_range')

        # Initialize all potential pheromone markers (set low opacity)
        self.pheromone_marker_size = self.get_marker_size_in_data_units(robot_size, self.ax) # TODO: cell_size in config / 2
        self.pheromone_markers = []

        # Plot walls
        self.plot_walls(self.ax)

        # Scale marker sizes to data units
        cam_range_marker_size = self.get_marker_size_in_data_units(camera_sensor_range, self.ax)
        cam_range, = self.ax.plot(
            [self.warehouse.rob_c[i, 0] for i in range(self.exp_cfg.get('number_of_agents'))],
            [self.warehouse.rob_c[i, 1] for i in range(self.exp_cfg.get('number_of_agents'))],
            'ko',
            markersize=cam_range_marker_size,
            color="#f2f2f2",
            fillstyle='none'
        )

        # LOGISTICS GRAPHICS
        # Plot delivery points as squares
        for dp in self.processed_delivery_points:
            box_marker_size = self.get_marker_size_in_data_units(box_size, self.ax)

            # Plot each delivery point as a square
            self.ax.plot(dp.x, dp.y, marker='s', markersize=box_marker_size * 1.8,
                    markeredgecolor='black',
                    markerfacecolor=dp.colour,  # Solid fill with the same color as edge
                    linewidth=2,
                    alpha=0.35)

        # Plot dropzone
        self.ax.fill_between(np.linspace(0, self.gen_cfg.get('warehouse', 'width'), 100), 0, self.exp_cfg.get('warehouse', 'drop_zone_limit'), color='lightgrey', alpha=0.2)
        self.ax.axhline(y=self.exp_cfg.get('warehouse', 'drop_zone_limit'), color='black', linewidth=1)

        # TRAFFIC GRAPHICS
        # Plot the target points as squares

        # Plot DOTS robots
        x_data, y_data, marker = self.generate_dot_positional_data()
        dot = {}
        for i in range(len(x_data)):
            robot_marker_size = self.get_marker_size_in_data_units(robot_size, self.ax)
            dot[i], = self.ax.plot(x_data[i], y_data[i], marker[i], markersize=robot_marker_size, fillstyle='none')

        # Plot boxes
        boxes = []
        for boxi in self.warehouse.boxes:
            box_marker_size = self.get_marker_size_in_data_units(box_size, self.ax)
            box, = self.ax.plot(boxi.x, boxi.y, marker='s', color=boxi.colour, markersize=box_marker_size)
            boxes.append(box)

        h_x_vec, h_y_vec = self.generate_dot_heading_arrow()
        h_line = {}
        for i in range(self.swarm.number_of_agents):
            h_line[i], = self.ax.plot(h_x_vec[i], h_y_vec[i], linestyle="dashed", color="#4CB580")

        self.anim = animation.FuncAnimation(self.fig, self.iterate, frames=10000, interval=0.1, blit=False,
                                            fargs=(dot, boxes, h_line, cam_range))

        plt.show()

