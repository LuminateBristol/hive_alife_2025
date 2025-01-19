from .bt_random_walk import bt as bt_random_walk
from .bt_logistics import bt as bt_logistics
from .bt_area_coverage import bt as bt_area_coverage
from .bt_traffic import bt_no_astr as bt_traffic

behaviour_trees = {'random_walk':     bt_random_walk,
                   'logistics':       bt_logistics,
                   'area_coverage':   bt_area_coverage,
                   'traffic':         bt_traffic}