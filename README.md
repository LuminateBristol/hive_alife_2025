### Overview

The code in this repository simulates an intralogistics task for a robot swarm and in particular, provides a pipeline for data generation and processing for a swarm which may have faulty robots. The robots operate in a 500 by 500 centimeter bounded arena, referred to as the warehouse: the task is to retrieve and deliver them to the drop-off zone, a 25cm-wide vertical strip extending along the length of the right-hand wall.

### Scripts

Simulation run scripts:

| Name | Description |
| ----------- | ----------- |
| run_ex.py | Run the experiment with given configuration, no visualization | 
| run_ex_multic.py | Run the experiment with given configuration, no visualization, multicore processing option available |
| run_ex_viz.py | Run the experiment with Matplotlib visualization | 
| run_redis.py | Run experiment and export visualization data to redis: requires redis to be running | 
| run_thresh.py | Run experiment with fault detection | 

General scripts:

| Name | Description |
| ----------- | ----------- |
| gui.sh | Start the flask web application for visualization |
| redis.sh | Start redis with set configuration |


Data processing scripts:

| Name | Description |
| ----------- | ----------- |
| run_sampler.py | Processes standard data samples (generated from DataModel class) | 
| run_min_sampler.py | Processes mimal data samples (generated from MinimalDataModel and ExtremeMinDataModel classes) | 
| run_stats.py | Computes effect size and thresholds for metrics for given experiment | 

### Data generation and analysis pipeline

Standard pipeline:  

1. Generate data by running a simulation script for n iterations and a range of faults.  
2. Process generated data with the appropriate data processing script (dependend on the DataModel class used).  
3. Compute statistics: effect size and thresholds.
