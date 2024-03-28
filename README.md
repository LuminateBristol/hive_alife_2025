### Overview

The code in this repository is based on an existing 2D DOTS simulator created for previous projects: https://bitbucket.org/suet_lee/metric_extraction_ddmefd/src/master/

This new version has been modified to accept behaviour tree controllers. Much of the project specific work from the previous simulator has also been removed and Hive Mind specific code has been added. In time the main branch of this simulator will be modified to be a generalised 2D dots simulator and all subsequenct branches will be project specific with only generalised code allowed for merge into main.


### Scripts

Simulation run scripts:

| Name | Description |
| ----------- | ----------- |
| run_ex.py | Run the experiment with given configuration, visualisation optional | 
| run_ex_multipp.py | Multiprocessing runs of the experiment, no visualisation |

Simulator scripts:

| Name | Description |
| ----------- | ----------- |
| sim.py | Sets up the simulator based on the provided config files - calls wearehouse.py|
| warehouse.py | Sets up the warehouse including walls, box locations and robot locations - calls various from objects.py |
| objects.py | Sets up the indivudual boxes, robots and swarm objects |
| bt_logistics.py | Behaviour tree controller for the logistics task |
| bt_random_walk.py | Behaviour tree controller for random walk |

