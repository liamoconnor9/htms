"""
Usage:
    make_dirs.py <config_file>
"""
import os
path = os.path.dirname(os.path.abspath(__file__))
from configparser import ConfigParser
from docopt import docopt
from pathlib import Path
import sys

config = ConfigParser()
try:
    args = docopt(__doc__)
    filename = Path(args['<config_file>'])
except:
    filename = path + '/config.cfg'

config.read(str(filename))
write_suffix = str(config.get('parameters', 'suffix'))

dir_lst = [write_suffix]
dir_lst += [write_suffix + '/checkpoints']

if config.getboolean('parameters', 'show'):
    dir_lst += [write_suffix + '/snapshots_forward']
    dir_lst += [write_suffix + '/snapshots_backward']
    dir_lst += [write_suffix + '/frames_forward']
    dir_lst += [write_suffix + '/frames_backward']
    dir_lst += [write_suffix + '/frames_target']
# dir_lst += [write_suffix + '/movies_forward']
# dir_lst += [write_suffix + '/movies_backward']
# dir_lst += [write_suffix + '/frames_error']
# dir_lst += [write_suffix + '/movies_error']

for dir_name in dir_lst:
    if (not os.path.isdir(dir_name)):
        os.makedirs(dir_name)
