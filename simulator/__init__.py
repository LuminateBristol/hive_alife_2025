import inspect, os.path

filename = inspect.getframeinfo(inspect.currentframe()).filename
path     = os.path.dirname(os.path.abspath(filename))

CFG_FILES = {
    'default': os.path.join(path, 'cfg', 'default.yaml'),
    'test': os.path.join(path, 'cfg', 'test.yaml'),
    'ex_1': os.path.join(path, 'cfg', 'ex_1.yaml'),
    'ex_2': os.path.join(path, 'cfg', 'ex_2.yaml'),
    'exp_setup': os.path.join(path, 'cfg', 'exp_setup.yaml'),
    'map': os.path.join(path, 'cfg', 'map.yaml')
}

ROOT_DIR = "/home/dots_native_2/dots_system_devel/src/hm_swarm_v1_suet_sim/suet_sim"
MODEL_ROOT = os.path.join(ROOT_DIR, 'models')
STATS_ROOT = os.path.join(ROOT_DIR, 'stats')