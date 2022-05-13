import inspect, os.path

filename = inspect.getframeinfo(inspect.currentframe()).filename
path     = os.path.dirname(os.path.abspath(filename))

CFG_FILES = {
    'default': os.path.join(path, 'cfg', 'default.yaml'),
    'ex_1': os.path.join(path, 'cfg', 'ex_1.yaml'),
    'ex_2': os.path.join(path, 'cfg', 'ex_2.yaml')
}

