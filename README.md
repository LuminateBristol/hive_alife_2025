
# DOTS 2D Simulator - Python

This simulator is designed to be used as a basic experimentation and setup tool for the DOTS robots. It allows users to test behaviour based controllers on robots at high speed and with basic physics. Written in Python for ease of use.

This simulator is based on previous project work which can be found here: https://bitbucket.org/suet_lee/metric_extraction_ddmefd/src/master/



## Installation

Install the simulator by simply cloning this repository. Note that some dependent libraries will need to be installed. All code is written in Python.


    
## Documentation

### Overview
This simulator has been built for use with behaviour tree controllers. The file system currently includes two examples, a random walk and a logistics task.

### Modifying the Swarm Behaviours 
Behaviour trees are currently implemented using the PyTrees library. This is very well documented here: https://py-trees.readthedocs.io/en/devel/

Users are encouraged to create new or modify existing behaviour trees for their own simulations using the same structure given in this repo. The behaviour tree can be split into multiple files as long as the root of the tree is called in the same way.

### File structure
The object-oriented approach of this simulator means that there is an interlinking relationship between different objects in the file structure. The diagram below gives and overview as to how each part fits together.

![File Structure](images/dot_2d_flow.jpg)


#### Simulator Files
| bt_logistics   | Logistics task behaviour tree                                                              |
|----------------|--------------------------------------------------------------------------------------------|
| bt_random_walk | Random walk behaviour tree                                                                 |
| bt_setup.py    | File convention setup for behaviour trees                                                  |
| faults.py      | Used for fault analysis only                                                               |
| objects.py     | Contains the classes for all objects (robots, swarm and boxes)                             |
| sim.py         | Contains the classes and functions for running the simulator                               |
| viz_sim.py     | Contains the classes and functions for running the simulator with matplotlib visualisation |
| warehouse.py   | Contains the classes and functions for settup up the warehouse                             |

#### Config Files (cfg)

| Config File    | Description                                                          |
|----------------|----------------------------------------------------------------------|
| default.yaml   | Sets up the general parameters of the simulation including:          |
|                | - robot parameters (size, speed, sensor range)                       |
|                | - warehouse parameters (size, physics, object size)                  |
|                | - simulator parameters (timeout, save options)                       |
| exp_setup.yaml | Sets up the specific experimental parameters for each run including: |
|                | - number of agents                                                   |
|                | - beheviour tree controller                                          |
|                | - object setup                                                       |
|                | - tolerances                                                         |

#### Library Files (lib)

| Config File | Description                                                    |
|-------------|----------------------------------------------------------------|
| Config.py   | Functions for parsing the config files into useable parameters |
| Redis.py    | Data storage - not used in this implementation                 |
| Save_to.py  | Data saving - not used in this implementation                  |


## Contributing

Contributions are always welcome!

As every user is likely to have a slightly different simulation setup, we encourage users to create a branch for their own use. If you develop a new behaviour tree that may be useful to others, we can add it to the examples or keep it as an aptly named seperate branch.

Enjoy!
