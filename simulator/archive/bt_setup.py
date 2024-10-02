from .bt_random_walk import bt as bt_random_walk
from .bt_logistics import bt_completion as bt_hm_completion
from .bt_logistics import bt_progression as bt_hm_progression
from .bt_logistics import bt_completion as bt_hm_completion_local
from .bt_logistics import bt_progression as bt_hm_progression_local

behaviour_trees = {'random_walk': bt_random_walk,
                   'hm_completion': bt_hm_completion,
                   'hm_progression':bt_hm_progression,
                   'hm_completion_local':bt_hm_completion_local,
                   'hm_progression_local':bt_hm_progression_local}