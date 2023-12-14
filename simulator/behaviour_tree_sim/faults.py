from .objects import Swarm
import numpy as np

class FaultySwarm(Swarm):

    FAILED_BOX_PICKUP = 1
    ALTER_AGENT_SPEED = 2
    STOP_AT_RANDOM = 3
    REDUCED_CAMERA_RANGE = 4
    FAILED_BOX_DROPOFF = 5
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.faults = {
            self.FAILED_BOX_PICKUP: {},
            self.ALTER_AGENT_SPEED: {},
            self.REDUCED_CAMERA_RANGE: {},
            self.FAILED_BOX_DROPOFF: 0
        }

    def add_fault(self, ftype, **kwargs):
        if ftype == self.FAILED_BOX_PICKUP:
            self.add_failed_box_pickup(**kwargs)
        if ftype == self.ALTER_AGENT_SPEED:
            self.add_alter_agent_speed(**kwargs)
        if ftype == self.REDUCED_CAMERA_RANGE:
            self.add_reduced_camera_range(**kwargs)
        if ftype == self.FAILED_BOX_DROPOFF:
            self.add_failed_box_dropoff(**kwargs)
    
    def update_hook(self):
        self.update_agent_speed()

    # time: start time t for failure occurrences
    # failure is certain to occur past time t
    def add_failed_box_pickup(self, lookup):
        self.faults[self.FAILED_BOX_PICKUP] = lookup

    # Simulate fault in picking up boxes
    def set_agent_box_state(self, robot_id, state):
        lookup = self.faults[self.FAILED_BOX_PICKUP]        

        if robot_id in list(lookup.keys()) and lookup[robot_id] <= self.counter:
            return False

        self.agent_has_box[robot_id] = state
        return True

    def add_alter_agent_speed(self, lookup):
        self.faults[self.ALTER_AGENT_SPEED] = lookup

    def update_agent_speed(self):
        lookup = self.faults[self.ALTER_AGENT_SPEED]
        if len(lookup) == 0:
            return

        unaltered = []
        for i, it in enumerate(lookup):
            time, robot_id, speed = it
            
            if time <= self.counter:
                self.robot_v[robot_id] *= speed
            else:
                unaltered.append(it)

        self.faults[self.ALTER_AGENT_SPEED] = unaltered

    def add_reduced_camera_range(self, lookup):
        self.faults[self.REDUCED_CAMERA_RANGE] = lookup
        self.set_reduced_camera_range()

    def set_reduced_camera_range(self):
        lookup = self.faults[self.REDUCED_CAMERA_RANGE] # (robot_id, reduced range) key, value pairs
        for robot_id, r_range in lookup.items():
            r0 = self.camera_sensor_range_V[robot_id]
            self.camera_sensor_range_V[robot_id] = r_range*r0

    def add_failed_box_dropoff(self, lookup):
        faulty = len(lookup)
        # faulty_arr = [0]*faulty + [1]*(self.number_of_agents-faulty)
        self.faults[self.FAILED_BOX_DROPOFF] = faulty

    def dropoff_box(self, warehouse, active_boxes):
        is_box_in_dropoff = warehouse.box_c.T[0] > warehouse.width - warehouse.exit_width - warehouse.radius
        faulty = self.faults[self.FAILED_BOX_DROPOFF]
        box_ready_for_dropoff = is_box_in_dropoff*active_boxes
        if faulty == 0:
            return box_ready_for_dropoff
        
        box_ids = box_ready_for_dropoff.nonzero()
        carrier_ids = warehouse.robot_carrier # robots carrying ready boxes
        check = (carrier_ids >= faulty)        
        return box_ready_for_dropoff*check
