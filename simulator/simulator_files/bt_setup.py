
from .bt_traffic import bt_hive as bt_traffic
from .bt_traffic import bt_centralised as bt_traffic_centralised

behaviour_trees = {'traffic_hive':          bt_traffic,
                   'traffic_centralised':   bt_traffic_centralised}