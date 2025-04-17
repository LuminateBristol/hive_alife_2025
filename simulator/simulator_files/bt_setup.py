from .bt_random_walk import bt as bt_random_walk
from .bt_logistics import bt as bt_logistics
from .bt_area_coverage import bt as bt_area_coverage
from .bt_traffic import bt_hive_latency as bt_traffic
from .bt_traffic import bt_centralised as bt_traffic_centralised
from .bt_traffic import bt_dist as bt_traffic_distributed

behaviour_trees = {'random_walk':           bt_random_walk,
                   'logistics':             bt_logistics,
                   'area_coverage':         bt_area_coverage,
                   'traffic_hive':          bt_traffic,
                   'traffic_centralised':   bt_traffic_centralised,
                   'traffic_distributed':   bt_traffic_distributed}