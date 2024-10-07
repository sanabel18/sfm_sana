import yaml
import numpy as np


def safe_yaml_dump(obj, file_path):
    with open(file_path,'w') as fp:
        yaml.dump(obj, fp, default_flow_style=False, sort_keys=False)
    fp.close()

def safe_yaml_load(fp: str) -> dict:
    with open(fp, 'r') as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict    
