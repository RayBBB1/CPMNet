import os
import yaml
import datetime
from tqdm import tqdm
from typing import Optional, Dict, Any
from os.path import splitext
from importlib import import_module

def init_seed(seed: int):
    import torch
    import numpy as np
    import random
    
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
        
def get_local_time_in_taiwan() -> datetime.datetime:
    utc_now = datetime.datetime.utcnow()
    taiwan_now = utc_now + datetime.timedelta(hours=8) # Taiwan in UTC+8
    return taiwan_now

def get_local_time_str_in_taiwan() -> str:
    cur_time = get_local_time_in_taiwan()
    timestamp = "[%d-%02d-%02d-%02d%02d]" % (cur_time.year, cur_time.month, cur_time.day, cur_time.hour, cur_time.minute)
    return timestamp

def get_progress_bar(identifer: str, total_steps: int, initial_step: int = 0) -> tqdm:
    """Get the progress bar
    """
    progress_bar = tqdm(initial = initial_step,
                        total = total_steps, 
                        desc = "{:10s}".format(identifer), 
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    return progress_bar

def write_yaml(save_path: str,
                config: dict,
                default_flow_style: str = ''):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style = default_flow_style)
        
def load_yaml(path: str, 
            overwrite_config:Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """Load the yaml path
    """
    with open(path, 'r') as f:
        settings = yaml.safe_load(f)
    if overwrite_config != None:
        for key, value in overwrite_config:
            if settings.get(key) != None:
                settings[key] = value
    return settings

def build_class(template: str) -> Any:
    """Build a classs based on give template
    """
    class_name = splitext(template)[1].strip('.')
    module_path = splitext(template)[0]
    module = import_module(module_path)
    cls = getattr(module, class_name)

    return cls