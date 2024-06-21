from . import Simulator
from matplotlib import pyplot as plt, animation
import numpy as np

class VizSim(Simulator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_s = [1]#[50,1250,2250]
        self.verbose = True

    def plot_walls(self, ax):
        # Plot horizontal walls
        for wall in self.cfg.get('wallsh'):
            start, end = wall
            x0, y0 = start
            x1, y1 = end
            ax.plot([x0, x1], [y0, y1], 'k-')  # 'k-' means black solid line

        # Plot vertical walls 
        for wall in self.cfg.get('wallsv'):
            start, end = wall
            x0, y0 = start
            x1, y1 = end
            ax.plot([x0, x1], [y0, y1], 'k-')


    def generate_dot_positional_data(self, faulty=False):

        agent_range = range(self.cfg.get('warehouse', 'number_of_agents'))
        x_data = [
            [self.warehouse.rob_c[i,0] for i in agent_range]
        ]
        y_data = [
            [self.warehouse.rob_c[i,1] for i in agent_range]
        ]
        marker = ['ko']

        return (x_data, y_data, marker)

    def generate_dot_heading_arrow(self):
        length = 20
        steps = 20
        agents = self.swarm.number_of_agents
        x_vec = []
        y_vec = []
        for i in range(agents):
            start_x = self.warehouse.rob_c[i,0]
            end_x = start_x + length * -np.cos(self.warehouse.rob_c[i,2])
            start_y = self.warehouse.rob_c[i,1]
            end_y = start_y + length * -np.sin(self.warehouse.rob_c[i,2])
            x_vec.append(np.linspace(start_x, end_x, steps).tolist())
            y_vec.append(np.linspace(start_y, end_y, steps).tolist())
        
        return x_vec, y_vec
        

    # iterate method called once per timestep
    def iterate(self, frame, dot=None, boxes=None, h_line=None, cam_range=None, snapshot=False):
        self.warehouse.iterate(self.cfg.get('heading_bias'), self.cfg.get('box_attraction'))
        counter = self.warehouse.counter

        dot, boxes, h_line, cam_range = self.animate(frame, counter, dot, boxes, h_line, cam_range)

        if self.verbose:
            if self.warehouse.counter == 1:
                print("Progress |", end="", flush=True)
            if self.warehouse.counter%100 == 0:
                print("=", end="", flush=True)

        self.exit_sim(counter=counter)

        dot = list(dot.values())
        h_line = list(h_line.values())

        return dot + h_line + boxes + [cam_range]

    def animate(self, i, counter, dot=None, boxes=None, h_line=None, cam_range=None):
        cam_range.set_data(
            [self.warehouse.rob_c[i,0] for i in range(self.cfg.get('warehouse', 'number_of_agents'))],
            [self.warehouse.rob_c[i,1] for i in range(self.cfg.get('warehouse', 'number_of_agents'))]
        )
        
        x_data, y_data, _ = self.generate_dot_positional_data()
        for i in range(len(dot)):
            dot[i].set_data(x_data[i], y_data[i])
        
        for box, wbox in zip(boxes, self.warehouse.boxes):
            box.set_data([wbox.x, wbox.y])

        h_x_vec, h_y_vec = self.generate_dot_heading_arrow()
        for i in range(self.swarm.number_of_agents):
            h_line[i].set_data(h_x_vec[i], h_y_vec[i])

        return dot, boxes, h_line, cam_range
  

    def exit_sim(self,counter=None):
        if  counter > self.cfg.get('time_limit'):
            if self.verbose:
                print("in", counter, "seconds")

            if self.cfg.get('animate'):
                exit()

    def run(self):
        if self.verbose:
            print("Running with seed: %d"%self.random_seed)

        self.init_animate()
        plt.show()
        
        if self.verbose:
            print("\n")

    def init_animate(self):
        self.fig = plt.figure()
        plt.rcParams['font.size'] = '16'
        ax = plt.axes(xlim=(0, self.cfg.get('warehouse', 'width')), ylim=(0, self.cfg.get('warehouse', 'height')))

        self.plot_walls(ax)

        # assume all swarm radius same
        marker_size = 12.5
        robot_r = self.cfg.get('robot', 'radius')
        camera_sensor_range = self.cfg.get('robot', 'camera_sensor_range')
        cam_range_marker_size = marker_size/robot_r*camera_sensor_range
        cam_range, = ax.plot(
            [self.warehouse.rob_c[i,0] for i in range(self.cfg.get('warehouse', 'number_of_agents'))],
            [self.warehouse.rob_c[i,1] for i in range(self.cfg.get('warehouse', 'number_of_agents'))], 
            'ko', 
            markersize = cam_range_marker_size,
            # linestyle=":",
            color="#f2f2f2",
            fillstyle='none'
        )

        x_data, y_data, marker = self.generate_dot_positional_data()

        dot = {}
        for i in range(len(x_data)):
            dot[i], = ax.plot(x_data[i], y_data[i], marker[i],
                markersize = marker_size, fillstyle = 'none')

        boxes=[]
        for boxi in self.warehouse.boxes:
            box, = ax.plot(boxi.x, boxi.y, marker='s', color=boxi.colour, markersize=marker_size-5)
            boxes.append(box)

        h_x_vec, h_y_vec = self.generate_dot_heading_arrow()
        h_line = {}
        for i in range(self.swarm.number_of_agents):
            h_line[i], = ax.plot(h_x_vec[i], h_y_vec[i], linestyle="dashed", color="#4CB580")
        
        self.anim = animation.FuncAnimation(self.fig, self.iterate, frames=10000, interval=0.1, blit=True,
                                            fargs=(dot, boxes, h_line, cam_range))

        plt.show()