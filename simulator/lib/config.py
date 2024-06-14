import yaml
import os
from pathlib import Path


class Config(dict):

    def __init__(self, cfg_path, default_cfg_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        try:
            cfg_base = yaml.safe_load(open(default_cfg_path))
        except:
            cfg_base = {}
            print("No default cfg specified")

        cfg_ex = yaml.safe_load(open(cfg_path))        
        cfg_override = {}

        if 'config_override' in kwargs:
            cfg_override = kwargs['config_override']
        
        if 'ex_id' in kwargs:
            ex_id = kwargs['ex_id']
            cfg_ex = cfg_ex[ex_id]
        
        if 'map' in kwargs:
            map = kwargs['map']
            map_file = yaml.safe_load(open(map))  

        if len(cfg_base) == 0 and len(cfg_override) == 0:
            self.__dict__.update(cfg_ex)
            return

        try:
            cfg = self._overwrite(cfg_base, cfg_ex)
        except:
            pass
        
        try:
            cfg = self._overwrite(cfg, cfg_override)
        except:
            pass
        
        try:
            cfg = self._overwrite(cfg, map_file)
        except:
            print('Map not loaded')
            pass

        self.__dict__.update(cfg)

    def _overwrite(self, base, overwrite):
        for k, v in overwrite.items():
            try:
                for k_, v_ in v.items():
                    if k not in base:
                        base[k] = {}
                    base[k][k_] = v_
            except:
                if k not in base:
                    base[k] = {}

                base[k] = v

        return base

    def has_key(self, key):
        return key in self.__dict__

    def get_key(self, key):
        return self.__dict__.get(key)

    def get(self, *args):
        key = args[0]
        cfg = self.get_key(key)      
        for arg in args[1:]:
            try:
                cfg = cfg[arg]
            except:
                raise Exception('Config key not valid')
                # return cfg
        return cfg

    def set(self, key, value):
        keys = key.split('.')
        cfg = self.__dict__
        
        for k in keys[:-1]:
            try:
                cfg = cfg[k]
            except (KeyError, TypeError):
                raise Exception(f'Config key "{key}" not valid')

        # Update the value for the specified key
        try:
            cfg[keys[-1]] = value
        except (KeyError, TypeError):
            raise Exception(f'Config key "{key}" not valid')

    def print(self):
        print(self.__dict__)

