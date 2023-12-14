from .bt_random_walk import bt as bt_random_walk
from .bt_pick_place import bt as bt_pick_place

behaviour_trees = {'random_walk': bt_random_walk,
                   'pick_place': bt_pick_place}