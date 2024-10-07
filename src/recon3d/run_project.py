#!/opt/conda/bin/python

import os
import sys
import shutil

from project import PreprocProj, ReconProj, LocProj
from utils.safe_yaml import safe_yaml_load


if __name__ == "__main__":
    cfg = safe_yaml_load(sys.argv[1])

    if cfg['project']['type'] == 'LocProj':
        prj = LocProj(cfg)
    elif cfg['project']['type'] == 'ReconProj':
        prj = ReconProj(cfg)
    elif cfg['project']['type'] == 'PreprocProj':
        prj = PreprocProj(cfg)

    prj.run_pipeline()
